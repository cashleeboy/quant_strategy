#!/usr/bin/env python3
"""Independent numerical and replay checks for the isolated beta experiment."""
import json
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_beta_research import OUTPUT, forward_hashes
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.run_six_strategy_ic_comparison import load_prices
from analysis.holdings_count import metrics, replay_batch


def validate(out=OUTPUT):
    completed=json.loads((out/'completed.json').read_text())
    for name,digest in completed['files'].items():
        if sha(out/name)!=digest:raise AssertionError('changed output '+name)
    frozen=json.loads((out/'frozen.json').read_text())
    for name,digest in frozen['code'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('changed research code '+name)
    for code,digest in frozen['market'].items():
        if sha(out/'market'/f'{code}.csv')!=digest:raise AssertionError('changed market '+code)
    if forward_hashes()!=json.loads((out/'forward_before.json').read_text()):
        raise AssertionError('changed forward ledger')
    p=json.loads((out/'protocol.json').read_text());px=load_prices(p)
    f=pd.read_csv(out/'market'/f'{p["market"]}.csv',dtype={'trade_date':str})
    f.index=pd.to_datetime(f.trade_date,format='%Y%m%d');market=f.close.reindex(px.index).pct_change(fill_method=None)
    beta=pd.read_csv(out/'beta252_weekly.csv.gz',index_col='date',parse_dates=True)
    rng=np.random.default_rng(20261019);errors=[]
    for d in beta.index[np.linspace(0,len(beta)-1,32,dtype=int)]:
        candidates=beta.loc[d].dropna().index
        for s in rng.choice(candidates,size=min(4,len(candidates)),replace=False):
            pos=px.index.get_loc(d)
            y=px[s].pct_change(fill_method=None).iloc[max(0,pos-251):pos+1]
            pair=pd.concat([y.rename('y'),market.reindex(y.index).rename('x')],axis=1).dropna()
            coef=np.linalg.lstsq(np.column_stack([np.ones(len(pair)),pair.x]),pair.y,rcond=None)[0][1]
            errors.append(abs(coef-beta.loc[d,s]))
    if max(errors)>1e-9:raise AssertionError('beta OLS mismatch')
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True)
    perf=pd.read_csv(out/'performance.csv')
    for _,row in perf.iterrows():
        key=row.variant if np.isclose(row.fee,p['fees'][0]) else row.variant+'__STRESS'
        for k,v in metrics(nav[key]).items():
            if not np.isclose(row[k],v[0],atol=1e-10):raise AssertionError('metrics mismatch')
    # Reconstruct one complete 50-name path using the separately tested batch engine.
    targets=pd.read_csv(out/'targets_LOW_BETA50.csv.gz',parse_dates=['date','execution_date'])
    prices=px.reindex(nav.index);plans=[];ex=[]
    for d,g in targets.groupby('date',sort=True):
        if len(g)!=50 or not np.allclose(g.weight,1/50):raise AssertionError('invalid K50 targets')
        ex.append(prices.index.get_loc(g.execution_date.iloc[0]));plans.append(prices.columns.get_indexer(g.symbol))
    ref,logs,stranded=replay_batch(prices.to_numpy(),ex,np.asarray(plans)[:,None,:],fee=p['fees'][0])
    replay_error=float(np.max(np.abs(ref[:,0]-nav.LOW_BETA50.to_numpy())))
    if replay_error>1e-9:raise AssertionError('independent replay mismatch')
    ic=pd.read_csv(out/'ic.csv',parse_dates=['date'])
    maturity_checks=0
    for _,row in ic.iterrows():
        pos=px.index.get_loc(row.date)
        if pos+1+row.horizon>=len(px):
            if not pd.isna(row.rank_ic):raise AssertionError('immature IC label')
            maturity_checks+=1
    write_json(out/'delivery_validation.json',dict(status='PASS',all_sealed_files_verified=True,
        source_code_and_market_verified=True,forward_unchanged=True,independent_beta_checks=len(errors),
        maximum_beta_error=max(errors),independent_full_path_maximum_nav_error=replay_error,
        all_metrics_recomputed=True,immature_ic_rows_checked=maturity_checks))
    print((out/'delivery_validation.json').read_text())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT)
    validate(ap.parse_args().output)
