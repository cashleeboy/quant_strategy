#!/usr/bin/env python3
"""Matched-sample diagnostic; does not fit, select, or change any portfolio."""
import json
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_beta_research import OUTPUT
from scripts.run_six_strategy_ic_comparison import load_prices
from analysis.beta_research import execution_return


def diagnose(out=OUTPUT):
    p=json.loads((out/'protocol.json').read_text());px=load_prices(p)
    beta=pd.read_csv(out/'beta252_weekly.csv.gz',index_col='date',parse_dates=True)
    targets=pd.read_csv(out/'targets_COMMON.csv.gz',parse_dates=['date'])
    common={d:set(g.symbol) for d,g in targets.groupby('date')}
    original=pd.read_csv(out/'ic.csv',parse_dates=['date']).set_index(['date','horizon','factor'])
    rows=[]
    for path in sorted((ROOT/p['prior_replay']).glob('inputs_*/universe.csv.gz')):
        f=pd.read_csv(path,usecols=['date','symbol','industry_l1','total_mv_yuan'],parse_dates=['date'])
        for d,g in f.groupby('date'):
            if d not in common:continue
            g=g[g.symbol.isin(common[d])].set_index('symbol')
            g=g[g.industry_l1.notna()&g.industry_l1.ne('')&g.total_mv_yuan.gt(0)&np.isfinite(g.total_mv_yuan)]
            score=-beta.loc[d].reindex(g.index)
            for h in [5,20]:
                y=execution_return(px,d,h).reindex(g.index)
                ok=y.notna()&score.notna();old=original.loc[(d,h,'controlled_beta')]
                if int(ok.sum())!=int(old.names):raise AssertionError('controlled sample mismatch')
                v=score[ok].corr(y[ok],method='spearman') if ok.sum()>=30 and y[ok].nunique()>1 else np.nan
                rows.append(dict(date=d,horizon=h,raw_low_beta_ic_same_controls_sample=v,
                                 controlled_low_beta_ic=old.rank_ic,names=int(ok.sum())))
    pd.DataFrame(rows).to_csv(out/'matched_control_ic.csv',index=False)
    print(pd.DataFrame(rows).groupby('horizon')[['raw_low_beta_ic_same_controls_sample','controlled_low_beta_ic']].mean().to_string())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT)
    diagnose(ap.parse_args().output)
