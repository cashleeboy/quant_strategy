"""Research-only market sensitivity; no registration or production weights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_market_sensitivity(prices, market_returns, window=252, minimum=200):
    """OLS slope with intercept on pairwise-valid adjacent-session raw returns.

    Caller supplies a full market calendar. Missing prices are NOT forward filled.
    Includes the signal-close observation; only suitable for later execution.
    Volatility uses precisely the same paired observations as beta.
    """
    if not 2 <= minimum <= window:
        raise ValueError('invalid window/minimum')
    if (not prices.index.equals(market_returns.index) or prices.index.has_duplicates
            or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates):
        raise ValueError('aligned unique sorted axes required')
    if np.isinf(prices.to_numpy()).any() or (prices <= 0).any().any():
        raise ValueError('invalid stock prices')
    if np.isinf(market_returns.to_numpy()).any():
        raise ValueError('invalid market returns')
    y = prices.pct_change(fill_method=None)
    y = y.where(np.isfinite(y)).where(market_returns.notna(), axis=0)
    x = pd.DataFrame(np.broadcast_to(market_returns.to_numpy()[:, None], y.shape),
                     index=y.index, columns=y.columns).where(y.notna())
    n = y.rolling(window, min_periods=1).count()
    sx, sy = x.rolling(window, min_periods=minimum).sum(), y.rolling(window, min_periods=minimum).sum()
    xx = x.pow(2).rolling(window, min_periods=minimum).sum() - sx*sx/n
    yy = y.pow(2).rolling(window, min_periods=minimum).sum() - sy*sy/n
    xy = (x*y).rolling(window, min_periods=minimum).sum() - sx*sy/n
    beta = (xy/xx).where(xx > 1e-14).where(n >= minimum)
    vol = np.sqrt((yy/(n-1)).clip(lower=0)*252).where(n >= minimum)
    return beta, vol, n


def execution_return(prices, signal, horizon):
    """Next-close entry, h-session hold; require all interval quotes for labels."""
    pos = prices.index.get_loc(signal)
    if pos+1+horizon >= len(prices):
        return pd.Series(np.nan, index=prices.columns)
    path = prices.iloc[pos+1:pos+2+horizon]
    return (path.iloc[-1]/path.iloc[0]-1).where(path.notna().all())


def market_regression(returns, market_returns):
    """Descriptive raw-return OLS; intercept NOT certified independent alpha."""
    d = pd.concat([returns.rename('y'), market_returns.rename('x')], axis=1).dropna()
    if len(d) < 3 or d.x.var() <= 1e-14:
        return dict(beta=np.nan, annual_intercept=np.nan, r_squared=np.nan, observations=len(d))
    beta = d.y.cov(d.x)/d.x.var()
    return dict(beta=float(beta), annual_intercept=float((d.y.mean()-beta*d.x.mean())*252),
                r_squared=float(d.y.corr(d.x)**2), observations=len(d))


def controlled_beta(frame):
    """Contemporaneous cross-sectional residual, not fitted to future returns."""
    cols = ['beta', 'vol252', 'total_mv_yuan', 'industry_l1']
    f = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    f = f[f.total_mv_yuan.gt(0) & f.industry_l1.ne('')].copy()
    result = pd.Series(np.nan, index=frame.index)
    if len(f) < 30:
        return result
    x = pd.DataFrame({'vol': f.vol252, 'size': np.log(f.total_mv_yuan)}, index=f.index)
    x = (x-x.mean())/x.std().replace(0, 1)
    x = pd.concat([pd.Series(1., index=f.index, name='intercept'), x,
                   pd.get_dummies(f.industry_l1, drop_first=True, dtype=float)], axis=1)
    if len(x) <= x.shape[1]+5:
        return result
    coef = np.linalg.lstsq(x.to_numpy(), f.beta.to_numpy(), rcond=None)[0]
    result.loc[f.index] = f.beta-x.to_numpy()@coef
    return result
