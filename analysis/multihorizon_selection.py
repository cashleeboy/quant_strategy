"""Predeclared multi-horizon weights and one-rule-at-a-time selection diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.single_account_adaptive import (
    budget_change_allowed,
    capped_positive_weights,
    estimated_budget,
    family_score,
    update_ic_weights,
)
from strategies.holding_buffer import ordered_scores


def mixed_horizon_ic(ic_by_horizon, mapping, factors):
    """Build one IC panel without mixing a factor's predeclared evaluation horizon."""
    out = {}
    for factor in factors:
        horizon = next((h for h, names in mapping.items() if factor in names), None)
        if horizon is None:
            raise ValueError(f"No predeclared horizon for {factor}")
        out[factor] = ic_by_horizon[str(horizon)][factor]
    return pd.DataFrame(out)


def impose_trend_floor(weights, factors=("MOMENTUM", "MOMENTUM_60D"), floor=.10):
    """Each named trend signal receives a fixed nominal floor; preserve other proportions."""
    result = weights.astype(float).copy()
    required = {name: max(float(result.get(name, 0)), floor) for name in factors}
    mass = sum(required.values())
    if mass >= 1:
        raise ValueError("trend floor leaves no mass")
    other = result.drop(list(factors))
    if other.sum() <= 0:
        raise ValueError("no nontrend mass")
    other *= (1-mass)/other.sum()
    result.loc[other.index] = other
    for name, value in required.items():
        result.loc[name] = value
    result /= result.sum()
    return result


def select_with_ablation(scores, industry, held, corr, spec, ablation):
    """Original fixed15 selector with exactly one named rule removed."""
    if ablation not in {"FULL", "NO_BUFFER", "NO_CORRELATION", "NO_INDUSTRY_CAP", "NO_TECH_CAP"}:
        raise ValueError(ablation)
    k = 15
    ranked = ordered_scores(scores).head(spec["candidate_limit"])
    exit_multiple = 1 if ablation == "NO_BUFFER" else spec["buffer_exit_multiple"]
    retained = [s for s in ranked.head(k*exit_multiple).index if s in set(held)]
    order = retained + [s for s in ranked.index if s not in set(retained)]
    industries = industry.fillna("UNKNOWN").replace("", "UNKNOWN")
    industry_limit = k if ablation == "NO_INDUSTRY_CAP" else int(np.floor(k*spec["industry_cap"]+1e-9))
    theme_limit = k if ablation == "NO_TECH_CAP" else int(np.floor(k*spec["theme_cap"]+1e-9))
    chosen, counts, themes = [], {}, 0
    for symbol in order:
        sector = industries.get(symbol, "UNKNOWN")
        is_theme = sector in spec["theme_industries"]
        if counts.get(sector, 0) >= industry_limit or (is_theme and themes >= theme_limit):
            continue
        if ablation != "NO_CORRELATION":
            if symbol not in corr.index or not np.isfinite(corr.loc[symbol, symbol]):
                continue
            if chosen:
                pairs = corr.loc[symbol, chosen]
                if pairs.isna().any() or (pairs > spec["correlation_cap"]).any():
                    continue
        chosen.append(symbol)
        counts[sector] = counts.get(sector, 0)+1
        themes += int(is_theme)
        if len(chosen) == k:
            break
    return pd.Series(1/k, index=chosen, dtype=float)


class StudyAccount:
    """Same fixed15/risk state machine; selector is explicit for leave-one-out research."""
    def __init__(self, groups, risk_cache, returns, protocol, ablation="FULL"):
        self.groups, self.risk_cache, self.returns = groups, risk_cache, returns
        self.p, self.ablation = protocol, ablation
        self.base = pd.Series(dtype=float)
        self.budget, self.initialized, self.last_change = 1., False, -100000
        self.logs = []

    def __call__(self, date, held):
        weekly = date in self.groups
        if weekly:
            group = self.groups[date]
            corr, _ = self.risk_cache[date]
            self.base = select_with_ablation(group["ic"], group["industry"], held, corr,
                                              self.p["selection"], self.ablation)
        desired, vol, observations = estimated_budget(self.returns, date, self.base, self.p["risk"])
        pos = self.returns.index.get_loc(date)
        changed = budget_change_allowed(self.budget, desired, pos-self.last_change,
                                        self.p["risk"], self.initialized)
        if changed:
            self.budget, self.last_change, self.initialized = desired, pos, True
        trade = weekly or changed
        target = self.base*self.budget
        self.logs.append(dict(date=date, weekly=weekly, risk_changed=changed, action=trade,
            requested_k=15, selected=len(self.base), budget=self.budget,
            desired_budget=desired, estimated_vol=vol, risk_observations=observations,
            target_equity=float(target.sum()), insufficient_risk_history=desired is None,
            ablation=self.ablation))
        return target if trade else None


def build_monthly_weights(ic, outcome_end, update_dates, initial, spec):
    weights, rows, snapshots = initial.copy(), [], {}
    for date in update_dates:
        weights, log = update_ic_weights(ic, outcome_end, date, weights, spec)
        snapshots[date] = weights.copy()
        rows.extend(log)
    return pd.DataFrame(snapshots).T, pd.DataFrame(rows)


def build_mixed_monthly_weights(ic, outcome_ends, update_dates, initial, spec):
    """Maturity is checked factor by factor because label horizons differ."""
    previous, rows, snapshots = initial.copy(), [], {}
    for signal in update_dates:
        means, counts, windows, latest = {}, {}, {}, {}
        for factor in previous.index:
            end = pd.to_datetime(outcome_ends[factor].reindex(ic.index))
            known = ic[factor].loc[(ic.index < signal) & end.lt(signal)].tail(spec["lookback"])
            means[factor] = known.mean()
            counts[factor] = known.count()
            windows[factor] = (known.index.min(), known.index.max())
            latest[factor] = end.reindex(known.index).max()
        mean, count = pd.Series(means), pd.Series(counts)
        positive = mean.where(count >= spec["minimum"], 0).fillna(0).clip(lower=0)
        reason = "positive_rank_ic"
        if positive.gt(0).sum()*spec["factor_cap"] < 1-1e-12:
            result, reason = previous.copy(), "retain_previous_insufficient_positive_evidence"
        else:
            desired = capped_positive_weights(positive, spec["factor_cap"])
            delta = desired-previous
            scale = min(1., spec["maximum_monthly_weight_change"]/max(delta.abs().max(), 1e-15))
            result = (previous+scale*delta).clip(lower=0)
            result /= result.sum()
        for factor in previous.index:
            rows.append(dict(date=signal, factor=factor, weight=result[factor],
                previous_weight=previous[factor], mean_rank_ic=mean[factor],
                valid_ic_days=int(count[factor]), reason=reason,
                latest_outcome_end=latest[factor], window_start=windows[factor][0],
                window_end=windows[factor][1]))
        previous = result
        snapshots[signal] = result.copy()
    return pd.DataFrame(snapshots).T, pd.DataFrame(rows)


def score_panel(panel, families, weights, fixed_pool):
    score = family_score(panel, families, weights, require_all=False).droplevel(0)
    score = score.reindex(fixed_pool).dropna()
    if len(score) != len(fixed_pool):
        raise ValueError("Weight rule changed the frozen comparison pool")
    return ordered_scores(score)
