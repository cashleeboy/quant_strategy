#!/usr/bin/env python3
"""Frozen single-account ablation. Offline only; does not write paper accounts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from analysis.single_account_adaptive import (
    prior_weights, update_ic_weights, family_score, history_risk, choose_portfolio,
    estimated_budget, budget_change_allowed,
)
from analysis.holdings_count import metrics, replay_batch
from backtest.research_replay import replay
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.run_six_strategy_ic_comparison import load_prices
from strategies.holding_buffer import ordered_scores

PROTOCOL = ROOT / "research_protocols/20261020_single_account_adaptive_v1.json"
OUT = ROOT / "output/single_account_adaptive_20261020"


def account_hashes():
    root = ROOT / "output/six_strategy_forward_20260928"
    return {str(p.relative_to(root)): sha(p) for folder in ["initial", "plans", "days"]
            for p in sorted((root / folder).rglob("*")) if p.is_file()}


def freeze(out):
    p = json.loads(PROTOCOL.read_text())
    source = ROOT / p["factor_source"]
    files = {}
    for folder in sorted(source.glob("inputs_*")):
        manifest = json.loads((folder / "complete.json").read_text())
        for name in ["BASE_V2_MULTI_FACTOR_weekly.csv.gz", "BASE_V2_MULTI_FACTOR_ic.csv",
                     "BASE_V2_MULTI_FACTOR_fixed_top100.csv.gz"]:
            path = folder / name
            digest = sha(path)
            if digest != manifest[name]:
                raise ValueError(f"source changed: {path}")
            files[str(path.relative_to(ROOT))] = digest
    for name in ["calendar.json", "frozen.json"]:
        files[str((source / name).relative_to(ROOT))] = sha(source / name)
    for name in [p["family_protocol"], p["prior_replay"] + "/prices.npz",
                 "analysis/single_account_adaptive.py", "scripts/run_single_account_adaptive.py",
                 "backtest/research_replay.py", "factors/preprocess.py",
                 "strategies/holding_buffer.py", "analysis/holdings_count.py"]:
        files[name] = sha(ROOT / name)
    stamp = dict(protocol=sha(PROTOCOL), files=files)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen.json").exists() and json.loads((out / "frozen.json").read_text()) != stamp:
        raise ValueError("frozen inputs or code changed; choose new --output")
    write_json(out / "frozen.json", stamp)
    write_json(out / "protocol.json", p)
    if not (out / "forward_before.json").exists():
        write_json(out / "forward_before.json", account_hashes())
    return p


def prepare(p, prices, out):
    source = ROOT / p["factor_source"]
    families = json.loads((ROOT / p["family_protocol"]).read_text())["candidate_families"]
    prior = prior_weights(families)
    parts = [pd.read_csv(f / "BASE_V2_MULTI_FACTOR_ic.csv", parse_dates=["date", "outcome_end"])
             for f in sorted(source.glob("inputs_*"))]
    all_ic = pd.concat(parts).set_index("date").sort_index()
    if all_ic.index.has_duplicates:
        raise ValueError("duplicate IC dates")
    ic, end = all_ic[prior.index], all_ic["outcome_end"]
    weights, month, weight_log, ranks, parity = prior.copy(), None, [], [], []
    groups, risk_cache = {}, {}
    returns = prices.pct_change(fill_method=None)
    for folder in sorted(source.glob("inputs_*")):
        panel = pd.read_csv(folder / "BASE_V2_MULTI_FACTOR_weekly.csv.gz", parse_dates=["date"])
        original = pd.read_csv(folder / "BASE_V2_MULTI_FACTOR_fixed_top100.csv.gz", parse_dates=["date"])
        for date, rows in panel.groupby("date", sort=True):
            # Warm weights through 2011; do not initialize from future evidence.
            if date.to_period("M") != month:
                weights, log = update_ic_weights(ic, end, date, weights, p["ic"])
                weight_log.extend(log)
                month = date.to_period("M")
            if date < pd.Timestamp(p["evaluation_start"]) or date > pd.Timestamp(p["evaluation_end"]):
                continue
            x = rows.set_index(["date", "symbol"])
            fixed = family_score(x, families, prior).droplevel(0)
            strict_adaptive = family_score(x, families, weights).droplevel(0).reindex(fixed.index)
            adaptive = family_score(x, families, weights, require_all=False).droplevel(0).reindex(fixed.index).dropna()
            # Never let adaptive missingness silently change the fixed comparison pool.
            if len(adaptive) != len(fixed):
                raise ValueError(f"active-factor missing scores changed common sample on {date}")
            fixed = ordered_scores(fixed)
            adaptive = ordered_scores(adaptive)
            expected = original[original.date.eq(date)].sort_values(["rank", "symbol"]).symbol.tolist()
            if fixed.head(100).index.tolist() != expected:
                raise AssertionError(f"fixed production ranking parity failed: {date}")
            parity.append(dict(date=date, fixed_top100_matches=True, candidates=len(fixed),
                               adaptive_available_family_renormalized=int(strict_adaptive.isna().sum())))
            industry = rows.set_index("symbol").industry
            groups[date] = dict(fixed=fixed, ic=adaptive, industry=industry)
            candidates = adaptive.head(p["selection"]["candidate_limit"]).index
            risk_cache[date] = history_risk(returns, date, candidates, p["selection"])
            for variant, score in [("fixed", fixed), ("ic", adaptive)]:
                for rank, (symbol, value) in enumerate(score.head(100).items(), 1):
                    ranks.append(dict(date=date, variant=variant, symbol=symbol, score=value, rank=rank))
        print(f"Prepared {folder.name}: {len(groups)} evaluation weeks", flush=True)
    pd.DataFrame(weight_log).to_csv(out / "factor_weights.csv", index=False)
    pd.DataFrame(ranks).to_csv(out / "rankings.csv.gz", index=False)
    pd.DataFrame(parity).to_csv(out / "fixed_score_parity.csv", index=False)
    return groups, risk_cache, returns


class Account:
    def __init__(self, arm, groups, risk_cache, returns, p):
        self.arm, self.groups, self.risk_cache = arm, groups, risk_cache
        self.returns, self.p = returns, p
        self.base = pd.Series(dtype=float)
        self.budget, self.initialized, self.last_change = 1., False, -100000
        self.logs, self.k_trials = [], []

    def __call__(self, date, held):
        weekly = date in self.groups
        if weekly:
            group = self.groups[date]
            if self.arm.startswith("FIXED"):
                k = 50 if self.arm == "FIXED50" else 15
                self.base = pd.Series(1/k, index=group["fixed"].head(k).index)
                self.k = k
            elif self.arm == "IC15":
                self.k = 15
                self.base = pd.Series(1/15, index=group["ic"].head(15).index)
            else:
                corr, cov = self.risk_cache[date]
                self.base, trials = choose_portfolio(group["ic"], group["industry"], held,
                    corr, cov, self.p["selection"], "DYNAMIC" in self.arm)
                self.k = trials[-1]["requested_k"]
                self.k_trials.extend(dict(date=date, **r) for r in trials)
        if not weekly and not self.arm.endswith("_RISK"):
            return None
        desired, vol, observations = None, float("nan"), 0
        changed = False
        if self.arm.endswith("_RISK"):
            desired, vol, observations = estimated_budget(self.returns, date, self.base, self.p["risk"])
            pos = self.returns.index.get_loc(date)
            changed = budget_change_allowed(self.budget, desired, pos-self.last_change,
                                            self.p["risk"], self.initialized)
            if changed:
                self.budget, self.last_change, self.initialized = desired, pos, True
        trade = weekly or changed
        target = self.base * self.budget
        self.logs.append(dict(date=date, weekly=weekly, risk_changed=changed, action=trade,
            requested_k=getattr(self, "k", 0), selected=len(self.base), budget=self.budget,
            desired_budget=desired, estimated_vol=vol, risk_observations=observations,
            target_equity=float(target.sum()), insufficient_risk_history=desired is None))
        return target if trade else None


def summarize(result, arm, fee):
    daily, ex = result["daily"], result["executions"]
    nav = daily.post_nav
    m = {k: float(v[0]) for k, v in metrics(nav).items()}
    years = (len(nav)-1)/252
    actions = ex[ex.action.ne("HOLD")]
    return dict(arm=arm, fee=fee, **m,
        annual_two_way_turnover=ex.two_way_turnover.sum()/years,
        average_equity=(1-daily.cash/daily.post_nav).mean(),
        average_execution_holdings=actions.holdings.mean(),
        action_dates=len(actions), annual_action_dates=len(actions)/years,
        trade_legs=len(result["trades"]),
        missing_target_events=int(actions.missing_targets.sum()),
        blocked_exit_events=int(actions.blocked_exit_value.gt(1e-10).sum()))


def run(out):
    p = freeze(out)
    full_prices = load_prices(p).loc[:p["evaluation_end"]]
    groups, risk_cache, returns = prepare(p, full_prices, out)
    first = min(groups)
    prices = full_prices.loc[first:]
    schedule = pd.DataFrame({"execution_date": prices.index[1:]}, index=prices.index[:-1])
    navs, summary, periods, annual, decisions = {}, [], [], [], []
    validation = dict(first_signal=str(first.date()), last_valuation=str(prices.index[-1].date()),
                      weekly_signals=len(groups), fixed_batch_parity={}, costs=[*p["fees"]])
    for fee in p["fees"]:
        fee_name = f"{int(round(fee*10000))}bp"
        for arm in p["arms"]:
            account = Account(arm, groups, risk_cache, returns, p)
            result = replay(prices, schedule, account, fee=fee)
            key = f"{arm}_{fee_name}"
            folder = out / key
            folder.mkdir(exist_ok=True)
            for name, frame in result.items():
                frame.to_csv(folder / f"{name}.csv.gz", index=(name == "daily"))
            pd.DataFrame(account.logs).to_csv(folder / "decisions.csv", index=False)
            pd.DataFrame(account.k_trials).to_csv(folder / "k_trials.csv", index=False)
            row = summarize(result, arm, fee)
            summary.append(row)
            nav = result["daily"].post_nav
            navs[key] = nav
            for year, series in nav.groupby(nav.index.year):
                position = nav.index.get_loc(series.index[0])
                start_nav = nav.iloc[max(0,position-1)]
                annual.append(dict(arm=arm, fee=fee, year=year, return_=series.iloc[-1]/start_nav-1))
            for start, stop in [("2012","2015"),("2016","2020"),("2021","2026")]:
                series = nav.loc[start:stop]
                pos = nav.index.get_loc(series.index[0])
                if pos > 0:
                    series = pd.concat([nav.iloc[pos-1:pos],series])
                periods.append(dict(arm=arm, fee=fee, period=f"{start}-{stop}",
                                    **{k:float(v[0]) for k,v in metrics(series).items()}))
            if fee == p["fees"][0] and arm in ["FIXED15","FIXED50"]:
                k = int(arm.removeprefix("FIXED"))
                signals = sorted(groups)
                plans = np.array([prices.columns.get_indexer(groups[d]["fixed"].head(k).index)
                                  for d in signals])[:,None,:]
                indices = np.array([prices.index.get_loc(d)+1 for d in signals])
                other, _, _ = replay_batch(prices.to_numpy(), indices, plans, fee)
                error = float(np.max(np.abs(other[:,0]-nav.to_numpy())))
                if error > 1e-9:
                    raise AssertionError(f"independent replay mismatch {arm}: {error}")
                validation["fixed_batch_parity"][arm] = error
            decisions.extend(dict(arm=arm, fee=fee, **r) for r in account.logs)
            print(f"{key}: CAGR={row['cagr']:.2%}, DD={row['max_drawdown']:.2%}, "
                  f"Sharpe={row['sharpe']:.3f}, turnover={row['annual_two_way_turnover']:.1f}x",flush=True)
    pd.DataFrame(navs).to_csv(out / "nav.csv")
    pd.DataFrame(summary).to_csv(out / "performance.csv",index=False)
    pd.DataFrame(periods).to_csv(out / "subperiods.csv",index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv",index=False)
    pd.DataFrame(decisions).to_csv(out / "decisions.csv.gz",index=False)
    before = json.loads((out / "forward_before.json").read_text())
    validation["forward_unchanged"] = account_hashes() == before
    if not validation["forward_unchanged"]:
        raise AssertionError("forward ledger changed")
    write_json(out / "validation.json", validation)
    write_json(out / "completed.json", {str(f.relative_to(out)):sha(f)
        for f in sorted(out.rglob("*")) if f.is_file() and f.name != "completed.json"})
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    run(parser.parse_args().output)
