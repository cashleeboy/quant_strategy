"""Causal research-only portfolio rules. Never imported by paper/live defaults."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.preprocess import cross_sectional_zscore
from strategies.holding_buffer import ordered_scores


def prior_weights(families):
    return pd.Series({c: 1 / len(families) / len(cols)
                      for cols in families.values() for c in cols}, dtype=float)


def capped_positive_weights(positive, cap):
    """Proportional water filling; never allocate to nonpositive evidence."""
    x = positive.clip(lower=0).fillna(0).astype(float)
    if (x.gt(0).sum() * cap < 1 - 1e-12) or not 0 < cap <= 1:
        raise ValueError("not enough positive components for cap")
    result = x * 0
    remaining = x[x > 0]
    mass = 1.
    while len(remaining):
        allocation = remaining / remaining.sum() * mass
        over = allocation > cap + 1e-14
        if not over.any():
            result.loc[remaining.index] = allocation
            break
        names = allocation.index[over]
        result.loc[names] = cap
        mass -= len(names) * cap
        remaining = remaining.drop(names)
    return result


def update_ic_weights(ic, outcome_end, signal, previous, spec):
    """Caller invokes only at a monthly update; same-day label ends excluded."""
    end = pd.to_datetime(outcome_end.reindex(ic.index))
    known = ic.loc[(ic.index < signal) & end.lt(signal)].tail(spec["lookback"])
    counts = known.count().reindex(previous.index, fill_value=0)
    mean = known.mean().reindex(previous.index)
    positive = mean.where(counts >= spec["minimum"], 0).fillna(0).clip(lower=0)
    reason = "positive_rank_ic"
    if positive.gt(0).sum() * spec["factor_cap"] < 1 - 1e-12:
        result, reason = previous.copy(), "retain_previous_insufficient_positive_evidence"
    else:
        desired = capped_positive_weights(positive, spec["factor_cap"])
        delta = desired - previous
        scale = min(1., spec["maximum_monthly_weight_change"] / max(delta.abs().max(), 1e-15))
        result = previous + scale * delta
        result = result.clip(lower=0)
        result /= result.sum()
    records = [dict(date=signal, factor=c, weight=result[c], previous_weight=previous[c],
                    mean_rank_ic=mean[c], valid_ic_days=int(counts[c]), reason=reason,
                    latest_outcome_end=end.reindex(known.index).max(),
                    window_start=known.index.min(), window_end=known.index.max())
               for c in previous.index]
    return result, records


def family_score(panel, families, weights, *, require_all=True):
    """Fixed prior is exactly production's five-family equal-budget score."""
    composites, budgets = {}, {}
    for name, cols in families.items():
        w = weights.reindex(cols)
        budget = w.sum()
        if budget <= 1e-14:
            continue
        x = panel[cols]
        denominator = x.notna().astype(float).dot(w)
        composites[name] = x.mul(w).sum(axis=1, min_count=1) / denominator.where(denominator > 0)
        budgets[name] = budget
    standardized = cross_sectional_zscore(pd.DataFrame(composites))
    budget = pd.Series(budgets)
    denominator = standardized.notna().astype(float).dot(budget)
    value = standardized.mul(budget).sum(axis=1, min_count=len(budgets) if require_all else 1)
    if not require_all:
        value = value / denominator.where(denominator > 0)
    return value.dropna().rename("score")


def history_risk(returns, signal, candidates, spec):
    """Slice first, estimate second. Missing return pairs never become zero risk."""
    x = returns.loc[:signal, candidates].tail(spec["history_sessions"])
    good = x.columns[(x.count() >= spec["minimum_name_observations"]) & x.std().gt(0)]
    x = x[good]
    corr = x.corr(min_periods=spec["minimum_pair_observations"])
    cov = x.cov(min_periods=spec["minimum_pair_observations"])
    # Covariance is only used on fully known selected submatrices below.
    return corr, cov


def diversification_squared(cov, names):
    if not names:
        return float("nan")
    matrix = cov.loc[names, names].to_numpy()
    if not np.isfinite(matrix).all():
        return float("nan")
    diagonal = np.diag(np.diag(matrix))
    matrix = .8 * matrix + .2 * diagonal
    vals, vecs = np.linalg.eigh(matrix)
    matrix = (vecs * np.maximum(vals, 1e-12)) @ vecs.T
    variance = float(matrix.mean())
    return float(np.sqrt(np.diag(matrix)).mean() ** 2 / variance) if variance > 0 else float("nan")


def constrained_selection(scores, industry, held, k, corr, spec):
    ranked = ordered_scores(scores).head(spec["candidate_limit"])
    retained = [s for s in ranked.head(k * spec["buffer_exit_multiple"]).index if s in set(held)]
    order = retained + [s for s in ranked.index if s not in set(retained)]
    chosen, counts, theme_count = [], {}, 0
    industries = industry.fillna("UNKNOWN").replace("", "UNKNOWN")
    industry_limit = int(np.floor(k * spec["industry_cap"] + 1e-9))
    theme_limit = int(np.floor(k * spec["theme_cap"] + 1e-9))
    for symbol in order:
        sector = industries.get(symbol, "UNKNOWN")
        theme = sector in spec["theme_industries"]
        if counts.get(sector, 0) >= industry_limit or (theme and theme_count >= theme_limit):
            continue
        if symbol not in corr.index or not np.isfinite(corr.loc[symbol, symbol]):
            continue
        if chosen:
            pairs = corr.loc[symbol, chosen]
            if pairs.isna().any() or (pairs > spec["correlation_cap"]).any():
                continue
        chosen.append(symbol)
        counts[sector] = counts.get(sector, 0) + 1
        theme_count += int(theme)
        if len(chosen) == k:
            break
    return chosen


def choose_portfolio(scores, industry, held, corr, cov, spec, dynamic):
    choices = spec["k_candidates"] if dynamic else [15]
    diagnostics = []
    for k in choices:
        chosen = constrained_selection(scores, industry, held, k, corr, spec)
        dr = diversification_squared(cov, chosen)
        meets = len(chosen) == k and np.isfinite(dr) and dr >= spec["diversification_ratio_squared_target"]
        diagnostics.append(dict(requested_k=k, selected=len(chosen), diversification_squared=dr, meets_target=meets))
        if not dynamic or meets:
            break
    # If every candidate fails, the final K is the predeclared largest K, not the best historical performer.
    return pd.Series(1 / k, index=chosen, dtype=float), diagnostics


def estimated_budget(returns, signal, base, spec):
    if base.empty:
        return None, float("nan"), 0
    x = returns.loc[:signal, base.index].tail(spec["lookback"]).dropna(how="any")
    if len(x) < spec["minimum_complete_observations"]:
        return None, float("nan"), len(x)
    basket = x.dot(base / base.sum())
    vol = float(basket.std(ddof=1) * np.sqrt(252))
    if not np.isfinite(vol) or vol <= 0:
        return None, vol, len(x)
    return float(np.clip(spec["target_annual_volatility"] / vol,
                         spec["minimum_budget"], spec["maximum_budget"])), vol, len(x)


def budget_change_allowed(current, desired, elapsed_sessions, spec, initialized):
    return desired is not None and (not initialized or
        (elapsed_sessions >= spec["minimum_sessions_between_budget_changes"] and
         abs(desired - current) >= spec["adjustment_threshold"] - 1e-12))
