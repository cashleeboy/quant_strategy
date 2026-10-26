#!/usr/bin/env python3
"""Audit saved diagnostic outputs and independently recompute selected observations."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_factor_diagnosis import OUT


def validate(out=OUT):
    for name,digest in json.loads((out/'completed.json').read_text()).items():
        assert sha(out/name)==digest,name
    for name,digest in json.loads((out/'frozen.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest,name
    q=json.loads((out/'protocol.json').read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    prices=load_prices(p).loc[:q['evaluation_end']]
    weights=pd.read_csv(out/'weights.csv.gz',parse_dates=['date','latest_outcome_end'])
    assert weights.latest_outcome_end.dropna().lt(weights.loc[weights.latest_outcome_end.notna(),'date']).all()
    assert weights.weight.between(-1e-12,.2+1e-12).all()
    assert (weights.weight-weights.previous_weight).abs().max()<=.05+1e-12
    assert np.allclose(weights.groupby(['date','window']).weight.sum(),1)
    lag=pd.read_csv(out/'lag_prediction.csv.gz')
    assert lag.loc[~lag.future_month_complete,'next_month_ic'].isna().all()
    horizon=pd.read_csv(out/'horizon_weekly.csv.gz',parse_dates=['date','outcome_end'])
    assert horizon.outcome_end.le(prices.index[-1]).all()
    assert horizon.outcome_end.gt(horizon.date).all()
    for col in ['coverage','top15_coverage','top_decile_coverage']:
        assert horizon[col].dropna().between(0,1).all()
    assert horizon.n_evaluable.le(horizon.n_candidates).all()
    common=horizon[horizon['sample'].eq('common_all_horizons')]
    assert common.groupby(['date','factor']).n_evaluable.nunique().eq(1).all()
    assert common.groupby(['date','factor']).horizon.nunique().eq(5).all()
    # Recompute sparse fixed observations directly, without diagnostic helper functions.
    checks=0
    observed_dates=sorted(horizon.date.unique())
    sample_dates=observed_dates[::100]+observed_dates[-1:]
    source=ROOT/p['factor_source']
    for dt in sample_dates:
        date=pd.Timestamp(dt)
        panel=pd.read_csv(source/f'inputs_{date.to_period("Q")}'/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date'])
        panel=panel[panel.date.eq(date)].set_index('symbol')
        # Use valid common universe from BASE fixed families, independently reconstructed.
        from analysis.single_account_adaptive import family_score,prior_weights
        families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
        x=panel.reset_index().set_index(['date','symbol'])
        eligible=family_score(x,families,prior_weights(families)).droplevel(0).index
        for factor in ['PE','ROE','MOMENTUM','VOLATILITY']:
            scores=panel[factor].reindex(eligible).dropna().sort_index().sort_values(ascending=False,kind='stable')
            for horizon_days in [1,5,20,60]:
                entry=prices.index.get_loc(date)+1;exit_pos=entry+horizon_days
                if exit_pos>=len(prices):continue
                y=(prices.iloc[exit_pos]/prices.iloc[entry]-1).reindex(scores.index)
                expected=scores.corr(y,method='spearman')
                row=horizon[horizon.date.eq(date)&horizon.factor.eq(factor)&horizon.horizon.eq(str(horizon_days))&horizon['sample'].eq('native')].iloc[0]
                assert np.isclose(expected,row.ic,equal_nan=True,atol=1e-12)
                assert np.isclose(y.iloc[:15].mean(),row.top15_return,equal_nan=True,atol=1e-12)
                checks+=1
    parent=ROOT/q['parent_output']
    parity=pd.read_csv(out/'parent_parity.csv');assert parity.max_nav_error.max()<1e-10
    for fee in [0,3,13]:
        base=pd.read_csv(out/f'BASE_RISK_{fee}bp/decisions.csv',parse_dates=['date']).set_index('date')
        bench=pd.read_csv(out/f'COMMON_EQ_MATCHED_{fee}bp/executions.csv.gz',parse_dates=['date','signal_date'])
        actual_actions=bench[bench.action.ne('HOLD')].signal_date.tolist()
        assert actual_actions==base.index[base.action].tolist()
        for arm in ['BASE_RISK','COMMON_EQ','COMMON_EQ_MATCHED','RAW_TOP15','TOP_DECILE','CONSTRAINED15','ORIGINAL_EQ','WINDOW126']+[
            'DROP_VALUE','DROP_QUALITY','DROP_GROWTH','DROP_MOMENTUM_REVERSAL','DROP_RISK']:
            folder=out/f'{arm}_{fee}bp'
            ex=pd.read_csv(folder/'executions.csv.gz');daily=pd.read_csv(folder/'daily.csv.gz')
            assert np.allclose(ex.pre_nav-ex.fees,ex.post_nav)
            assert np.allclose(ex.fees,(ex.buy_notional+ex.sell_notional)*fee/10000)
            assert daily.cash.ge(-1e-9).all() and daily.post_nav.gt(0).all()
            if (folder/'selections.csv.gz').exists():
                selections=pd.read_csv(folder/'selections.csv.gz')
                assert selections.weight.gt(0).all() and selections.groupby('date').weight.sum().le(1+1e-12).all()
    # No future-data construction in the random reference: membership valid at every signal.
    plans=np.load(out/'random_plans.npz');assert plans['plans'].shape[1:]==(50,15)
    last_quarter=None
    for date,plan in zip(pd.to_datetime(plans['signals']),plans['plans']):
        quarter=date.to_period('Q')
        if quarter!=last_quarter:
            panel=pd.read_csv(source/f'inputs_{quarter}'/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date'])
            last_quarter=quarter
        x=panel[panel.date.eq(date)].set_index(['date','symbol'])
        eligible=family_score(x,families,prior_weights(families)).droplevel(0).index
        assert prices.columns[plan.ravel()].isin(eligible).all()
        assert (np.diff(np.sort(plan,axis=1),axis=1)>0).all()
    for name,digest in json.loads((parent/'completed.json').read_text()).items():assert sha(parent/name)==digest
    assert account_hashes()==json.loads((out/'forward_before.json').read_text())
    result=dict(status='PASS',independent_horizon_checks=checks,common_horizon_sample_verified=True,
        compact_engine_parent_parity=True,accounting_all_39_replays=True,matched_benchmark_action_dates=True,
        random_membership_verified=True,parent_results_unchanged=True,forward_unchanged=True,
        limits='Statistical intervals are pointwise diagnostics, not independent discovery or production certification.')
    write_json(out/'independent_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':validate()
