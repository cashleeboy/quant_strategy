"""Calendar-only frequency experiment helpers; no live defaults."""
import numpy as np
import pandas as pd

def schedules(sessions,weekly,start,end):
    sessions=pd.DatetimeIndex(sessions).sort_values()
    weekly=pd.DatetimeIndex(weekly).sort_values()
    first=pd.Timestamp(start);last=pd.Timestamp(end)
    weekly=weekly[(weekly>=first)&(weekly<last)]
    if len(weekly)==0 or weekly[0]!=first:raise ValueError('bootstrap must match first weekly signal')
    anchor=first.to_period('W-FRI').ordinal
    fortnight=weekly[np.array([(d.to_period('W-FRI').ordinal-anchor)%2==0 for d in weekly])]
    months=pd.Series(sessions,index=sessions).groupby(sessions.to_period('M')).max()
    months=months[(months.index.end_time.normalize()<=last)&(months>=first)&(months<last)]
    monthly=pd.DatetimeIndex(months).union(pd.DatetimeIndex([first]))
    return {'WEEKLY':weekly,'FORTNIGHTLY':fortnight,'MONTHLY':monthly}

def asof_weights(weights,date):
    known=weights.loc[:date]
    if known.empty:raise ValueError('No matured historical weights')
    return known.iloc[-1],known.index[-1]

def classify_executions(executions,decisions,bootstrap):
    d=decisions.set_index('date')
    out=executions.copy()
    categories=[]
    for r in out.itertuples():
        if r.action=='HOLD':category='HOLD'
        else:
            log=d.loc[r.signal_date]
            if r.signal_date==bootstrap:category='BOOTSTRAP'
            elif log.weekly and log.risk_changed:category='SCHEDULED_AND_RISK'
            elif log.weekly:category='SCHEDULED_ONLY'
            elif log.risk_changed:category='RISK_ONLY'
            else:raise AssertionError('Unexplained trade action')
        categories.append(category)
    out['trigger']=categories
    return out

def index_nav(levels,dates):
    """Common next-close entry, no fee or initial-day phantom index return."""
    v=levels.reindex(dates)
    if v.isna().any() or (v<=0).any():raise ValueError('Incomplete index calendar')
    nav=v/v.iloc[1]
    nav.iloc[0]=1.
    return nav
