#!/usr/bin/env python3
"""Complete one missing cell without changing the sealed parent experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_single_account_adaptive import Account, prepare, summarize, account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.validate_single_account_adaptive import validate
from analysis.holdings_count import metrics
from backtest.research_replay import replay

PROTOCOL = ROOT / "research_protocols/20261021_fixed15_risk_extension_v1.json"
OUT = ROOT / "output/single_account_fixed15_risk_20261021"


def resolved_protocol(extension, parent):
    """Only change experiment arms; all numerical rules remain parent rules."""
    if extension["new_arm"] != "IC_CONSTRAINED15_RISK":
        raise ValueError("unexpected extension arm")
    if extension["arms"] != ["IC_CONSTRAINED15", "IC_CONSTRAINED15_RISK", "IC_DYNAMIC_RISK", "FIXED15_RISK"]:
        raise ValueError("unexpected extension comparison")
    return {**parent, "arms": extension["arms"]}


def freeze(out):
    extension = json.loads(PROTOCOL.read_text())
    parent = ROOT / extension["parent_output"]
    seal = json.loads((parent / "frozen.json").read_text())
    parent_protocol = ROOT / extension["parent_protocol"]
    if sha(parent_protocol) != seal["protocol"]:
        raise ValueError("parent protocol changed")
    for name, digest in seal["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"parent source changed: {name}")
    for name, digest in json.loads((parent / "completed.json").read_text()).items():
        if sha(parent / name) != digest:
            raise ValueError(f"parent result changed: {name}")
    source_parent = json.loads(parent_protocol.read_text())
    if source_parent != json.loads((parent / "protocol.json").read_text()):
        raise ValueError("parent resolved protocol mismatch")
    p = resolved_protocol(extension, source_parent)
    files = dict(seal["files"])
    for path in [PROTOCOL, parent_protocol, parent / "completed.json", parent / "frozen.json",
                 Path(__file__), ROOT / "scripts/validate_single_account_adaptive.py"]:
        files[str(path.relative_to(ROOT))] = sha(path)
    # 'protocol' explicitly identifies the inherited protocol; extension records the only override.
    stamp = dict(protocol=seal["protocol"], extension_protocol=sha(PROTOCOL), files=files)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen.json").exists() and json.loads((out / "frozen.json").read_text()) != stamp:
        raise ValueError("frozen extension changed; choose a new output directory")
    write_json(out / "frozen.json", stamp)
    write_json(out / "protocol.json", p)
    write_json(out / "extension_protocol.json", extension)
    if not (out / "forward_before.json").exists():
        write_json(out / "forward_before.json", account_hashes())
    return p, extension, parent


def run(out=OUT):
    p, extension, parent = freeze(out)
    prices_all = load_prices(p).loc[:p["evaluation_end"]]
    groups, risk_cache, returns = prepare(p, prices_all, out)
    for name in ["factor_weights.csv", "rankings.csv.gz", "fixed_score_parity.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(parent / name))
    prices = prices_all.loc[min(groups):]
    schedule = pd.DataFrame({"execution_date": prices.index[1:]}, index=prices.index[:-1])
    rows, navs, periods, annual, decisions, comparisons = [], {}, [], [], [], []
    for fee in p["fees"]:
        for arm in p["arms"]:
            account = Account(arm, groups, risk_cache, returns, p)
            result = replay(prices, schedule, account, fee=fee)
            key = f"{arm}_{int(round(fee*10000))}bp"
            folder = out / key
            folder.mkdir(exist_ok=True)
            for name, frame in result.items():
                file = folder / f"{name}.csv.gz"
                frame.to_csv(file, index=(name == "daily"))
                if arm != extension["new_arm"]:
                    old = pd.read_csv(parent / key / file.name)
                    new = pd.read_csv(file)
                    pd.testing.assert_frame_equal(new, old, rtol=1e-12, atol=1e-12)
            log = pd.DataFrame(account.logs)
            log.to_csv(folder / "decisions.csv", index=False)
            pd.DataFrame(account.k_trials).to_csv(folder / "k_trials.csv", index=False)
            if arm == extension["new_arm"]:
                assert log.requested_k.eq(15).all(), "new arm accidentally became dynamic K"
                assert not log.risk_changed.eq(False).all(), "risk budget never activated"
            else:
                old_nav = pd.read_csv(parent / key / "daily.csv.gz").post_nav.to_numpy()
                comparisons.append(dict(arm=arm, fee=fee, max_daily_nav_error=float(
                    np.abs(old_nav-result["daily"].post_nav.to_numpy()).max()),
                    daily_targets_executions_trades_match=True))
            row = summarize(result, arm, fee)
            rows.append(row)
            nav = result["daily"].post_nav
            navs[key] = nav
            for start, stop in [("2012", "2015"), ("2016", "2020"), ("2021", "2026")]:
                series = nav.loc[start:stop]
                pos = nav.index.get_loc(series.index[0])
                if pos:
                    series = pd.concat([nav.iloc[pos-1:pos], series])
                periods.append(dict(arm=arm, fee=fee, period=f"{start}-{stop}",
                    **{k:float(v[0]) for k,v in metrics(series).items()}))
            for year, series in nav.groupby(nav.index.year):
                pos = nav.index.get_loc(series.index[0])
                annual.append(dict(arm=arm, fee=fee, year=year,
                                   return_=series.iloc[-1]/nav.iloc[max(0,pos-1)]-1))
            decisions.extend(dict(arm=arm, fee=fee, **r) for r in account.logs)
            print(f"{key}: total={row['total_return']:.2%}, CAGR={row['cagr']:.2%}, "
                  f"vol={row['volatility']:.2%}, Sharpe={row['sharpe']:.3f}, "
                  f"DD={row['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(rows).to_csv(out / "performance.csv", index=False)
    pd.DataFrame(navs).to_csv(out / "nav.csv")
    pd.DataFrame(periods).to_csv(out / "subperiods.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "annual.csv", index=False)
    pd.DataFrame(decisions).to_csv(out / "decisions.csv.gz", index=False)
    pd.DataFrame(comparisons).to_csv(out / "control_parity.csv", index=False)
    assert account_hashes() == json.loads((out / "forward_before.json").read_text())
    validation = dict(first_signal=str(prices.index[0].date()),last_valuation=str(prices.index[-1].date()),
        weekly_signals=len(groups),same_factor_weights_rankings=True,control_parity=comparisons,
        forward_unchanged=True,parent_results_preserved=True)
    write_json(out / "validation.json", validation)
    write_json(out / "completed.json", {str(f.relative_to(out)):sha(f)
        for f in sorted(out.rglob("*")) if f.is_file() and f.name not in
        ["completed.json", "audit_summary.csv", "independent_validation.json"]})
    validate(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    run(parser.parse_args().output)
