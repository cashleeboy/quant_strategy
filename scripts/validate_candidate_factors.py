#!/usr/bin/env python3
"""Audit sealed candidate experiments from saved schedules, targets and fills."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_candidate_factors import OUT
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from analysis.candidate_factors import CANDIDATES
from universe.strategy import point_in_time_industry

def validate():
    for name,digest in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/name)==digest,name
    for name,digest in json.loads((OUT/'frozen.json').read_text())['files'].items():assert sha(ROOT/name)==digest,name
    q=json.loads((OUT/'protocol.json').read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    src=json.loads((ROOT/q['source_protocol']).read_text());prices=load_prices(p);dates=prices.loc[q['evaluation_start']:q['evaluation_end']].index
    saved={k:pd.DatetimeIndex(v) for k,v in json.loads((OUT/'schedules.json').read_text()).items()}
    parent=json.loads((ROOT/q['parent_output']/'schedules.json').read_text())
    assert all(v.equals(pd.DatetimeIndex(parent[k])) for k,v in saved.items())
    w=pd.read_csv(OUT/'factor_weights.csv',parse_dates=['date','latest_outcome_end','window_end'])
    known=w.latest_outcome_end.notna();assert w.loc[known,'latest_outcome_end'].lt(w.loc[known,'date']).all()
    assert w.loc[~known,'valid_ic_days'].eq(0).all()
    assert w.weight.between(-1e-12,.2+1e-12).all()
    assert (w.weight-w.previous_weight).abs().le(.05+1e-12).all()
    np.testing.assert_allclose(w.groupby(['arm','date']).weight.sum(),1,atol=1e-12)
    ic=pd.read_csv(OUT/'daily_ic.csv',parse_dates=['date','outcome_end']).set_index('date')
    for r in ic.itertuples():
        loc=prices.index.get_loc(r.Index)
        if pd.notna(r.outcome_end):assert r.outcome_end==prices.index[loc+6]
    usage=pd.read_csv(OUT/'rankings.csv.gz',parse_dates=['date','weight_update'])
    assert usage.weight_update.le(usage.date).all()
    rankgroups={(a,d):set(g.symbol) for (a,d),g in usage.groupby(['arm','date'])}
    members=pd.read_csv(ROOT/src['data_source']/'index_member_all.csv.gz')
    industry_index=pd.MultiIndex.from_frame(usage[['date','symbol']].drop_duplicates())
    all_sectors=point_in_time_industry(industry_index,members).fillna('UNKNOWN').replace('','UNKNOWN')
    sector_cache={};target_checks=0;selection_checks=0
    for arm in q['arms']:
        for frequency,signals in saved.items():
            for bp in [0,3,13]:
                folder=OUT/f'{arm}_{frequency}_{bp}bp'
                ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
                daily=pd.read_csv(folder/'daily.csv.gz')
                d=pd.read_csv(folder/'decisions.csv.gz',parse_dates=['date']).set_index('date')
                t=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date'])
                assert pd.DatetimeIndex(d.index[d.scheduled_reselection]).equals(signals)
                np.testing.assert_allclose(ex.post_nav,ex.pre_nav-ex.fees,atol=1e-10)
                np.testing.assert_allclose(ex.fees,(ex.buy_notional+ex.sell_notional)*bp/10000,atol=1e-10)
                np.testing.assert_allclose(ex.two_way_turnover,(ex.buy_notional+ex.sell_notional)/ex.pre_nav,atol=1e-10)
                assert daily.cash.ge(-1e-10).all()
                assert ex[ex.action.ne('HOLD')].signal_date.tolist()==d.index[d.action].tolist()
                for r in ex.itertuples():assert r.date==dates[dates.get_loc(r.signal_date)+1]
                previous=None
                for date,g in t.groupby('date',sort=True):
                    assert len(g)<=15 and g.symbol.is_unique
                    np.testing.assert_allclose(g.weight,d.loc[date,'budget']/15,atol=1e-12)
                    assert g.weight.sum()<=1+1e-12
                    current=set(g.symbol)
                    if date not in signals:assert current==previous
                    else:
                        assert current.issubset(rankgroups[(arm,date)])
                        if bp==3:
                            key=(date,tuple(sorted(current)))
                            if key not in sector_cache:
                                idx=pd.MultiIndex.from_arrays([[date]*len(g),g.symbol],names=['date','symbol'])
                                sectors=all_sectors.reindex(idx)
                                assert sectors.notna().all()
                                assert sectors.value_counts().max()<=3
                                assert sectors.isin(p['selection']['theme_industries']).sum()<=5
                                sector_cache[key]=True
                            selection_checks+=1
                    previous=current;target_checks+=1
    coverage=pd.read_csv(OUT/'coverage.csv');assert coverage.complete.le(coverage.pool).all()
    rawrows=0
    for folder in OUT.glob('inputs_*'):
        raw=pd.read_csv(folder/'raw.csv.gz',parse_dates=['date','latest_disclosure','dividend_snapshot'])
        assert (raw.latest_disclosure.isna()|raw.latest_disclosure.lt(raw.date)).all()
        known=raw.dividend_snapshot.notna()
        assert (raw.loc[known,'date']-raw.loc[known,'dividend_snapshot']).dt.days.between(0,7).all()
        z=pd.read_csv(folder/'standard.csv.gz');assert (z[CANDIDATES].notna().to_numpy()==raw[CANDIDATES].notna().to_numpy()).all()
        for date,g in raw.groupby('date'):
            valid=set(g.loc[g[CANDIDATES].notna().all(axis=1),'symbol'])
            for arm in ['COMMON_BASE','COMMON_ALL5']:
                if (arm,date) in rankgroups:assert rankgroups[(arm,date)].issubset(valid)
        rawrows+=len(raw)
    parity=pd.read_csv(OUT/'parent_parity.csv');assert len(parity)==6 and parity.max_nav_error.max()<1e-10
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    result={'status':'PASS','accounts':54,'target_checks':target_checks,'industry_theme_selection_checks':selection_checks,
            'saved_raw_rows_checked':rawrows,'mature_labels':True,'weight_limits':True,'all_fee_cash_next_close_checks':True,
            'baseline_weekly_monthly_parity':True,'forward_unchanged':True}
    write_json(OUT/'independent_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':validate()
