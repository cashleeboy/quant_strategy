"""Research diagnostic primitives, with ex-ante selection and explicit missingness."""
from __future__ import annotations
import numpy as np
import pandas as pd


def horizon_returns(prices, signal, next_signal, horizons):
    pos=prices.index.get_loc(signal)+1
    result={}; ends={}
    for h in horizons:
        end=pos+h
        ends[str(h)]=prices.index[end] if end<len(prices) else pd.NaT
        result[str(h)]=(prices.iloc[end]/prices.iloc[pos]-1 if end<len(prices)
                        else pd.Series(np.nan,index=prices.columns))
    end=prices.index.get_loc(next_signal)+1 if next_signal is not None else len(prices)
    ends['NEXT']=prices.index[end] if end<len(prices) else pd.NaT
    result['NEXT']=(prices.iloc[end]/prices.iloc[pos]-1 if end<len(prices)
                    else pd.Series(np.nan,index=prices.columns))
    return pd.DataFrame(result).replace([np.inf,-np.inf],np.nan),ends


def cross_section_diagnostic(scores, future, common):
    """Never rank only stocks with known future prices: future changes coverage, not picks."""
    scores=scores.replace([np.inf,-np.inf],np.nan).dropna().sort_index()
    ranked=scores.sort_values(ascending=False,kind='stable')
    n=len(ranked)
    if n<30:
        return None
    y=future.reindex(ranked.index).where(common.reindex(ranked.index,fill_value=False))
    paired=pd.concat([ranked.rename('x'),y.rename('y')],axis=1).dropna()
    ic=paired.x.corr(paired.y,method='spearman') if len(paired)>=30 and paired.x.nunique()>1 and paired.y.nunique()>1 else np.nan
    decile=(np.arange(n)*10//n)+1  # Q1 highest, Q10 lowest; all groups selected before outcomes.
    means=[float(y.iloc[decile==q].mean()) for q in range(1,11)]
    row=dict(n_candidates=n,n_evaluable=int(y.notna().sum()),coverage=float(y.notna().mean()),
             ic=ic,pool_return=y.mean(),top15_return=y.iloc[:15].mean(),
             top15_coverage=float(y.iloc[:15].notna().mean()),top15_intended=15,
             top_decile_return=means[0],top_decile_coverage=float(y.iloc[decile==1].notna().mean()),
             high_minus_pool=means[0]-y.mean(),top15_minus_pool=y.iloc[:15].mean()-y.mean(),
             top15_minus_decile=y.iloc[:15].mean()-means[0],high_minus_low=means[0]-means[-1])
    row.update({f'q{q}':means[q-1] for q in range(1,11)})
    row.update({f'q{q}_coverage':float(y.iloc[decile==q].notna().mean()) for q in range(1,11)})
    return row


def hac_mean(values,lags=12):
    """Pointwise normal HAC interval for mean, not a multiple-test pass/fail gate."""
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if n<3:
        return dict(mean=float(np.mean(x)) if n else np.nan,n=n,se=np.nan,low=np.nan,high=np.nan)
    mean=float(x.mean());z=x-mean
    variance=float(z@z/n)
    for lag in range(1,min(lags,n-1)+1):
        variance+=2*(1-lag/(lags+1))*float(z[lag:]@z[:-lag]/n)
    se=np.sqrt(max(variance,0)/n)
    return dict(mean=mean,n=n,se=se,low=mean-1.96*se,high=mean+1.96*se)


def remove_family(weights,columns):
    w=weights.copy();w.loc[columns]=0
    if w.sum()<=1e-14:
        raise ValueError('all remaining evidence zero; do not invent fallback')
    return w/w.sum()


def compact_replay(prices,schedule,factory,fee):
    """Same fractional sell-first engine without millions of per-stock trade log rows.

    Retains daily accounting and aggregate executions. Validated against research_replay
    and the sealed full-period BASE_RISK at both nonzero fees before report release.
    """
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates:
        raise ValueError('invalid axes')
    px=prices.to_numpy(float); marks=prices.ffill().fillna(0).to_numpy(float)
    if np.isinf(px).any() or (px[np.isfinite(px)]<=0).any() or not 0<=fee<1:
        raise ValueError('invalid price or fee')
    dates=prices.index;names=prices.columns;signal_set=set(schedule.index)
    for s,r in schedule.iterrows():
        j=dates.get_indexer([s])[0]
        if j<0 or j+1>=len(dates) or r.execution_date!=dates[j+1]:
            raise ValueError('must execute next session')
    shares=np.zeros(len(names));cash=1.;pending={};daily=[];executions=[]
    for day,date in enumerate(dates):
        quote=px[day];mark=marks[day];valid=np.isfinite(quote);safe=np.where(valid,quote,0.)
        pre=float(cash+shares@mark)
        if date in pending:
            signal,target=pending.pop(date)
            action='HOLD' if target is None else ('CASH' if target.empty else 'TARGET')
            sell=np.zeros(len(names));buy=sell.copy();old_cash=cash;before=shares.copy()
            positions=np.array([],dtype=int) if target is None else names.get_indexer(target.index)
            desired=np.zeros(len(names))
            if target is not None: desired[positions]=target.to_numpy()
            all_missing=action=='TARGET' and not valid[positions].any()
            if action!='HOLD' and not all_missing:
                dollars=pre*desired
                sell=np.maximum(shares*safe-dollars,0);sell[sell<=1e-9]=0
                cash+=sell.sum()*(1-fee)
                shares-=np.divide(sell,safe,out=np.zeros_like(sell),where=safe>0)
                need=np.maximum(dollars-shares*safe,0)*valid;need[need<=1e-9]=0
                cost=need.sum()*(1+fee)
                buy=need*(min(1.,cash/cost) if cost>0 else 1.)
                cash-=buy.sum()*(1+fee)
                shares+=np.divide(buy,safe,out=np.zeros_like(buy),where=safe>0)
            paid=float((sell.sum()+buy.sum())*fee);post=float(cash+shares@mark)
            if cash < -1e-9 or not np.isclose(post,pre-paid,rtol=1e-10,atol=1e-10) or not np.isclose(cash,old_cash+sell.sum()-buy.sum()-paid,atol=1e-10):
                raise AssertionError('cash/fee reconciliation')
            blocked=(before*mark>1e-10)&~valid&(desired==0)
            executions.append(dict(date=date,signal_date=signal,action=action,pre_nav=pre,post_nav=post,cash=cash,
                buy_notional=buy.sum(),sell_notional=sell.sum(),fees=paid,two_way_turnover=(buy.sum()+sell.sum())/pre,
                cash_fraction=cash/post,holdings=int((shares*mark>1e-10).sum()),all_targets_missing_hold=all_missing,
                missing_targets=int((~valid[positions]).sum()),blocked_exit_value=float((before*mark*blocked).sum()),
                trade_legs=int((buy+sell>0).sum())))
        daily.append(dict(date=date,pre_nav=pre,post_nav=float(cash+shares@mark),cash=cash,
                          unquoted_value=float((shares*mark*~valid).sum())))
        if date in signal_set:
            target=factory(date,list(names[shares*mark>1e-10]))
            if target is not None:
                target=pd.Series(target,dtype=float)
                if target.index.has_duplicates or not target.index.isin(names).all() or not np.isfinite(target).all() or (target<=0).any() or target.sum()>1+1e-12:
                    raise ValueError('invalid target')
            pending[schedule.loc[date,'execution_date']]=(date,target)
    return dict(daily=pd.DataFrame(daily).set_index('date'),executions=pd.DataFrame(executions))
