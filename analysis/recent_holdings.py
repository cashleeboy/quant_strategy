"""Actual-share mark-to-market attribution; isolated retrospective research."""
import numpy as np
import pandas as pd


def attribute(prices, daily, trades):
    """Reconstruct holdings from fills, NOT targets. Attribute close-to-close PNL.

    A trade at today's close earns no preceding intraday/overnight return. Missing
    quotes retain stale valuation but are never synthesized into executable fills.
    Currency PNL includes unrealized changes, realized changes and proportional fees.
    """
    names = prices.columns
    marks = prices.ffill().fillna(0).to_numpy(float)
    shares = np.zeros(len(names))
    by_date = {d: g for d, g in trades.groupby('date')}
    records = []
    for i, d in enumerate(prices.index):
        before = shares.copy()
        gross = before * (marks[i] - marks[i-1]) if i else np.zeros(len(names))
        fees = np.zeros(len(names))
        if d in by_date:
            t = by_date[d]
            if t.symbol.duplicated().any():
                raise ValueError('multiple fills per symbol/day require ordered aggregation')
            positions = names.get_indexer(t.symbol)
            if (positions < 0).any():
                raise ValueError('unknown symbol')
            np.testing.assert_allclose(before[positions], t.shares_before, atol=1e-12)
            shares[positions] = t.shares_after.to_numpy()
            fees[positions] = t.fee.to_numpy()
        value = shares * marks[i]
        nav = daily.loc[d, 'post_nav']
        prev_nav = daily.post_nav.iloc[i-1] if i else 1.
        np.testing.assert_allclose(value.sum() + daily.loc[d, 'cash'], nav, atol=1e-10)
        np.testing.assert_allclose((gross-fees).sum(), nav-prev_nav, atol=1e-10)
        for j in np.flatnonzero((before*marks[i] > 1e-10) | (value > 1e-10) | (fees > 0)):
            records.append(dict(date=d, symbol=names[j], shares=shares[j], value=value[j],
                                weight=value[j]/nav, gross_pnl=gross[j], fee=fees[j],
                                net_pnl=gross[j]-fees[j], prior_weight=(before[j]*marks[i-1,j]/prev_nav if i else 0.)))
    return pd.DataFrame(records)


def contribution(ledger, nav, start, end):
    """Additive percentage-point contribution to the period's cumulative return."""
    dates = nav.loc[start:end].index
    if not len(dates):
        raise ValueError('empty period')
    pos = nav.index.get_loc(dates[0])
    initial = nav.iloc[pos-1] if pos else 1.
    x = ledger[ledger.date.isin(dates)].copy()
    result = x.groupby('symbol').agg(gross_pnl=('gross_pnl','sum'), fee=('fee','sum'),
        net_pnl=('net_pnl','sum'), weight_sum=('weight','sum'),
        held_days=('value',lambda v: int(v.gt(1e-10).sum())))
    result['contribution'] = result.net_pnl/initial
    result['average_weight'] = result.weight_sum/len(dates)
    np.testing.assert_allclose(result.contribution.sum(), nav.loc[dates[-1]]/initial-1, atol=1e-9)
    return result.sort_values('contribution')
