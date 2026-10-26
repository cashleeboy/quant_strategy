#!/usr/bin/env python3
"""Explain frozen quintile membership in numeric beta units; no new backtest."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_beta_research import OUTPUT
from scripts.run_mvp_holdings_count import sha, write_json


def describe(out=OUTPUT):
    sealed=json.loads((out/'completed.json').read_text())['files']
    names=['beta252_weekly.csv.gz','targets_COMMON.csv.gz']+[f'targets_BETA_Q{i}.csv.gz' for i in range(1,6)]
    for name in names:
        if sha(out/name)!=sealed[name]:raise ValueError('changed source '+name)
    beta=pd.read_csv(out/'beta252_weekly.csv.gz',index_col='date',parse_dates=True)
    targets=pd.read_csv(out/'targets_COMMON.csv.gz',parse_dates=['date'])
    archived={}
    for i in range(1,6):
        f=pd.read_csv(out/f'targets_BETA_Q{i}.csv.gz',parse_dates=['date'])
        archived[i]={d:set(g.symbol) for d,g in f.groupby('date')}
    rows=[];values=[];examples=[]
    for d,g in targets.groupby('date',sort=True):
        scores=beta.loc[d].reindex(sorted(g.symbol)).sort_values(kind='stable')
        if scores.isna().any():raise ValueError('missing common beta')
        values.append(scores.to_numpy())
        for q,part in enumerate(np.array_split(np.arange(len(scores)),5),1):
            s=scores.iloc[part]
            if set(s.index)!=archived[q][d]:raise AssertionError('quintile membership mismatch')
            rows.append(dict(date=d,group=f'Q{q}',names=len(s),minimum=s.min(),maximum=s.max(),
                             mean=s.mean(),median=s.median()))
        for label,symbol in [('minimum',scores.index[0]),('maximum',scores.index[-1])]:
            examples.append(dict(date=d,kind=label,symbol=symbol,beta=float(scores[symbol])))
    frame=pd.DataFrame(rows);v=np.concatenate(values)
    frame.to_csv(out/'beta_group_ranges.csv',index=False)
    summary=dict(scope='common sample stock-signal observations, repeated stocks across dates',
                 observations=len(v),signals=len(values),minimum=float(v.min()),maximum=float(v.max()),
                 quantiles={str(q):float(np.quantile(v,q)) for q in [.01,.05,.5,.95,.99]},
                 negative_fraction=float((v<0).mean()),above_one_fraction=float((v>1).mean()),
                 latest_signal=str(frame.date.max().date()),
                 mean_of_weekly_group_means=frame.groupby('group')['mean'].mean().to_dict(),
                 mean_group_counts=frame.groupby('group').names.mean().to_dict(),
                 membership_verified=True,source_hashes={n:sha(out/n) for n in names})
    extremes=pd.DataFrame(examples)
    summary['minimum_observation']=extremes.loc[extremes.beta.idxmin()].to_dict()
    summary['maximum_observation']=extremes.loc[extremes.beta.idxmax()].to_dict()
    write_json(out/'beta_group_range_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,default=str,indent=2))
    print(frame[frame.date.eq(frame.date.max())].to_string(index=False))
    print(frame[frame.date.eq(frame.date.min())].to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT)
    describe(ap.parse_args().output)
