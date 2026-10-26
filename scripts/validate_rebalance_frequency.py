#!/usr/bin/env python3
"""Independently audit saved frequency schedules, labels, fills and index entry."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_rebalance_frequency import OUT
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices

def validate():
    for name,digest in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/name)==digest,name
    for name,digest in json.loads((OUT/'frozen.json').read_text())['files'].items():assert sha(ROOT/name)==digest,name
    q=json.loads((OUT/'protocol.json').read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    prices=load_prices(p).loc[:q['evaluation_end']];dates=prices.loc[q['evaluation_start']:].index
    saved={k:pd.DatetimeIndex(v) for k,v in json.loads((OUT/'schedules.json').read_text()).items()}
    monthly=[]
    for month in pd.period_range(dates[0],dates[-1],freq='M'):
        if month.end_time.normalize()>dates[-1]:continue
        d=dates[dates.to_period('M')==month]
        if len(d):monthly.append(d[-1])
    assert saved['MONTHLY'].equals(pd.DatetimeIndex([dates[0]]+monthly).unique().sort_values())
    expected=[d for d in saved['WEEKLY'] if ((d.to_period('W-FRI').start_time-dates[0].to_period('W-FRI').start_time).days//7)%2==0]
    assert saved['FORTNIGHTLY'].equals(pd.DatetimeIndex(expected))
    assert all(v[0]==dates[0] for v in saved.values())
    usage=pd.read_csv(OUT/'weight_usage.csv',parse_dates=['date','weight_update'])
    assert usage.weight_update.le(usage.date).all()
    weight_log=pd.read_csv(ROOT/'output/single_account_fixed15_risk_20261021/factor_weights.csv',parse_dates=['date','latest_outcome_end'])
    known=weight_log.latest_outcome_end.notna()
    assert weight_log.loc[known,'latest_outcome_end'].lt(weight_log.loc[known,'date']).all()
    assert weight_log.loc[~known,'valid_ic_days'].eq(0).all()
    assert weight_log.loc[~known,'reason'].eq('retain_previous_insufficient_positive_evidence').all()
    updates=pd.DatetimeIndex(sorted(weight_log.date.unique()))
    for r in usage.itertuples():assert r.weight_update==updates[updates<=r.date][-1]
    cost=pd.read_csv(OUT/'cost_attribution.csv');nav=pd.read_csv(OUT/'nav.csv',index_col='date',parse_dates=True)
    checks=0
    for frequency,signals in saved.items():
        for bp in [0,3,13]:
            folder=OUT/f'{frequency}_{bp}bp'
            ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
            d=pd.read_csv(folder/'decisions.csv.gz',parse_dates=['date']).set_index('date')
            t=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date']);daily=pd.read_csv(folder/'daily.csv.gz')
            actual=pd.DatetimeIndex(d.index[d.scheduled_reselection])
            assert actual.equals(signals)
            np.testing.assert_allclose(ex.post_nav,ex.pre_nav-ex.fees,atol=1e-11)
            np.testing.assert_allclose(ex.fees,(ex.buy_notional+ex.sell_notional)*bp/10000,atol=1e-11)
            np.testing.assert_allclose(ex.two_way_turnover,(ex.buy_notional+ex.sell_notional)/ex.pre_nav,atol=1e-11)
            assert daily.cash.ge(-1e-10).all()
            for r in ex.itertuples():
                assert r.date==dates[dates.get_loc(r.signal_date)+1]
                if r.trigger=='RISK_ONLY':assert r.signal_date not in signals and d.loc[r.signal_date,'risk_changed']
            previous=None
            for date,g in t.groupby('date',sort=True):
                assert len(g)<=15 and g.symbol.is_unique
                np.testing.assert_allclose(g.weight,d.loc[date,'budget']/15,atol=1e-12)
                assert g.weight.sum()<=1+1e-12
                current=set(g.symbol)
                if date not in signals:assert current==previous
                previous=current
                checks+=1
            aggregate=cost[cost.arm.eq(frequency)&np.isclose(cost.fee,bp/10000)]
            assert aggregate.action_dates.sum()==len(ex)
            assert np.isclose(aggregate.paid_fees_initial_nav_units.sum(),ex.fees.sum())
            assert ex[ex.action.ne('HOLD')].signal_date.tolist()==d.index[d.action].tolist()
    index_checks=[]
    for r in json.loads((OUT/'index_validation.json').read_text()):
        if r['status']!='COMPLETE':continue
        code=r['code'];f=pd.read_csv(OUT/'market'/f'{code}.csv',dtype={'trade_date':str})
        f.index=pd.to_datetime(f.trade_date,format='%Y%m%d');levels=f.close.reindex(dates)
        assert levels.notna().all() and levels.gt(0).all()
        expected=levels/levels.iloc[1];expected.iloc[0]=1
        np.testing.assert_allclose(nav[code],expected,atol=1e-12)
        for path in (OUT/'market').glob(f'{code}_*.json'):
            meta=json.loads(path.read_text());assert meta['code']==code and sha(path.with_suffix('.csv'))==meta['sha256']
        index_checks.append(code)
    # Independent provenance cross-check against the prior beta study's CSI300 cache.
    old=pd.read_csv(ROOT/'output/beta_research_20261019/market/000300.SH.csv',dtype={'trade_date':str}).set_index('trade_date')
    new=pd.read_csv(OUT/'market/000300.SH.csv',dtype={'trade_date':str}).set_index('trade_date')
    np.testing.assert_allclose(new.close,old.close.reindex(new.index),atol=1e-9)
    parity=pd.read_csv(OUT/'parent_parity.csv');assert parity.nav_error.max()<1e-10 and len(parity)==3
    for folder in OUT.glob('month_inputs_*'):
        assert json.loads((folder/'parity.json').read_text())['status']=='PASS'
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    result=dict(status='PASS',target_checks=checks,month_end_calendar=True,holiday_phase_preserved=True,
        same_matured_weights=True,all_nine_fee_and_cash_checks=True,benchmark_codes=index_checks,
        csi300_previous_download_matches=True,weekly_all_three_costs_matches_parent=True,forward_unchanged=True)
    write_json(OUT/'independent_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':validate()
