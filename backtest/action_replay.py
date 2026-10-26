"""Asset-aware v2 research replay; use for new controlled experiments.

The legacy research_replay remains byte-for-byte frozen for old-result parity.
This entry requires explicit corporate-action evidence and delisting metadata.
It preserves legacy no-action trading math, but never silently carries a known
terminated asset forever. It is not the broker/paper-account execution engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtest.corporate_actions import AssetTransitions


def replay(prices,schedule,target_factory,*,actions,adjustment_factors,delist_dates,fee=.0003):
    if (not isinstance(prices.index,pd.DatetimeIndex) or prices.empty or prices.index.hasnans or
        prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates or prices.columns.isna().any()):
        raise ValueError('invalid price axes')
    px=prices.to_numpy(float)
    if np.isinf(px).any() or (px[np.isfinite(px)]<=0).any() or not np.isfinite(fee) or not 0<=fee<1:
        raise ValueError('invalid prices or fee')
    schedule=schedule.copy();schedule.index=pd.to_datetime(schedule.index,errors='raise')
    schedule['execution_date']=pd.to_datetime(schedule.execution_date,errors='raise')
    if schedule.index.has_duplicates or schedule.index.hasnans or schedule.execution_date.isna().any() or schedule.execution_date.duplicated().any():
        raise ValueError('duplicate/missing schedule')
    for signal,row in schedule.iterrows():
        pos=prices.index.get_indexer([signal])[0]
        if pos<0 or pos+1>=len(prices) or row.execution_date!=prices.index[pos+1]:raise ValueError('execution must be next market day')
    assets=AssetTransitions(prices,adjustment_factors,actions,delist_dates)
    marks=prices.ffill().fillna(0).to_numpy(float);names=prices.columns
    shares,cash=np.zeros(len(names)),1.
    pending,daily,executions,trades,plans,positions={},[],[],[],[],[]
    for day,date in enumerate(prices.index):
        quote,mark=px[day],marks[day];valid=np.isfinite(quote);safe=np.where(valid,quote,0.)
        before_actions=float(cash+shares@mark)
        action_pnl=assets.apply(date,shares,mark)
        pre=float(cash+shares@mark)
        if not np.isclose(pre,before_actions+action_pnl,atol=1e-10,rtol=1e-10):raise AssertionError('action NAV mismatch')
        if date in pending:
            signal,weights=pending.pop(date);assets.validate_target(date,weights)
            before,old_cash=shares.copy(),cash
            buy,sell=np.zeros(len(names)),np.zeros(len(names))
            action='HOLD' if weights is None else ('CASH' if weights.empty else 'TARGET')
            target=np.zeros(len(names));loc=np.array([],dtype=int) if weights is None else names.get_indexer(weights.index)
            if weights is not None:target[loc]=weights.to_numpy(float)
            all_missing=action=='TARGET' and not valid[loc].any()
            if action!='HOLD' and not all_missing:
                dollars=pre*target
                sell=np.maximum(shares*safe-dollars,0.);sell[sell<=1e-9]=0.
                cash+=sell.sum()*(1-fee)
                shares-=np.divide(sell,safe,out=np.zeros_like(sell),where=safe>0)
                need=np.maximum(dollars-shares*safe,0.)*valid;need[need<=1e-9]=0.
                total=need.sum()*(1+fee);buy=need*(min(1.,cash/total) if total>0 else 1.)
                cash-=buy.sum()*(1+fee)
                shares+=np.divide(buy,safe,out=np.zeros_like(buy),where=safe>0)
            paid=float((buy.sum()+sell.sum())*fee);post=float(cash+shares@mark)
            if cash < -1e-9 or not np.isclose(post,pre-paid,rtol=1e-10,atol=1e-10):raise AssertionError('NAV reconciliation failed')
            if not np.isclose(cash,old_cash+sell.sum()-buy.sum()-paid,atol=1e-10):raise AssertionError('cash reconciliation failed')
            blocked=(before*mark>1e-10)&~valid&(target==0)
            executions.append(dict(date=date,signal_date=signal,action=action,pre_nav=pre,post_nav=post,cash=cash,
                buy_notional=float(buy.sum()),sell_notional=float(sell.sum()),fees=paid,two_way_turnover=float((buy.sum()+sell.sum())/pre),
                cash_fraction=cash/post,holdings=int((shares*mark>1e-10).sum()),all_targets_missing_hold=all_missing,
                missing_targets=int((~valid[loc]).sum()),blocked_exit_value=float((before*mark*blocked).sum())))
            for j in np.flatnonzero(buy+sell>0):
                trades.append(dict(date=date,symbol=names[j],side='BUY' if buy[j] else 'SELL',notional=float(buy[j]+sell[j]),
                    fee=float((buy[j]+sell[j])*fee),shares_before=float(before[j]),shares_after=float(shares[j]),price=float(quote[j]),pre_nav=pre))
        post=float(cash+shares@mark)
        daily.append(dict(date=date,pre_nav=pre,post_nav=post,cash=cash,unquoted_value=float((shares*mark*~valid).sum()),action_pnl=action_pnl))
        for j in np.flatnonzero(shares*mark>1e-10):
            positions.append(dict(date=date,symbol=names[j],units=float(shares[j]),value=float(shares[j]*mark[j]),weight=float(shares[j]*mark[j]/post),quoted=bool(valid[j])))
        if date in schedule.index:
            held=list(names[shares*mark>1e-10]);target=target_factory(date,held)
            if target is not None:
                target=pd.Series(target,dtype=float)
                if (target.index.has_duplicates or target.index.isna().any() or not target.index.isin(names).all() or
                    not np.isfinite(target).all() or (target<=0).any() or target.sum()>1+1e-12):raise ValueError('invalid target')
            assets.validate_target(date,target)
            execution=schedule.loc[date,'execution_date'];pending[execution]=(date,target)
            for symbol,weight in ([] if target is None else target.items()):
                plans.append(dict(date=date,execution_date=execution,symbol=symbol,weight=weight,was_actually_held=symbol in held))
    return dict(daily=pd.DataFrame(daily).set_index('date'),executions=pd.DataFrame(executions),
        trades=pd.DataFrame(trades,columns=['date','symbol','side','notional','fee','shares_before','shares_after','price','pre_nav']),
        targets=pd.DataFrame(plans,columns=['date','execution_date','symbol','weight','was_actually_held']),
        positions=pd.DataFrame(positions,columns=['date','symbol','units','value','weight','quoted']),asset_actions=assets.frame())
