#!/usr/bin/env python3
"""Frozen, resumable ML scoring comparison on the asset-aware fixed15 account."""
from pathlib import Path
import argparse
import copy
import importlib.metadata
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.fixed15_ml import training_sample, fit_model, predict
from analysis.ic_weight_comparison import execution_labels
from analysis.single_account_adaptive import family_score, prior_weights, history_risk
from analysis.holdings_count import metrics
from analysis.rebalance_frequency import classify_executions
from backtest.action_replay import replay
from scripts.run_single_account_adaptive import Account, account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.run_rebalance_frequency import verify_sealed
from strategies.holding_buffer import ordered_scores

OUT=ROOT/'output/fixed15_ml_20261024'
PROTOCOL=ROOT/'research_protocols/20261024_fixed15_ml_v1.json'


def freeze():
    q=json.loads(PROTOCOL.read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    parent=ROOT/q['repaired_parent']
    files=dict(json.loads((parent/'frozen.json').read_text())['files'])
    for path,digest in files.items():
        if sha(ROOT/path)!=digest:raise ValueError('Changed inherited input '+path)
    for path,digest in json.loads((parent/'completed.json').read_text()).items():
        if sha(parent/path)!=digest:raise ValueError('Changed repaired result '+path)
        files[str((parent/path).relative_to(ROOT))]=digest
    for path in [PROTOCOL,Path(__file__),ROOT/'analysis/fixed15_ml.py',ROOT/'factors/factor_ml.py',
                 ROOT/'requirements-ml-research.txt',ROOT/q['parent_protocol']]:
        files[str(path.relative_to(ROOT))]=sha(path)
    versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost']}
    stamp=dict(files=files,versions=versions)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text())!=stamp:
        raise ValueError('Frozen run changed; use a new output directory')
    if (OUT/'completed.json').exists():raise ValueError('Completed run is immutable')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():write_json(OUT/'forward_before.json',account_hashes())
    return p,q


def prepare(p,q,prices):
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
    features=[c for cols in families.values() for c in cols];prior=prior_weights(families)
    parts=[];coverage=[]
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        path=folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz'
        if sha(path)!=json.loads((folder/'complete.json').read_text())[path.name]:raise ValueError('Changed factor panel')
        frame=pd.read_csv(path,parse_dates=['date']).set_index(['date','symbol'])
        for date,rows in frame.groupby(level='date'):
            eligible=family_score(rows,families,prior).index
            parts.append(rows.loc[eligible,features+['industry']])
            coverage.append(dict(date=date,raw=len(rows),eligible=len(eligible)))
    panel=pd.concat(parts).sort_index()
    if panel.index.has_duplicates:raise ValueError('Duplicate weekly panel')
    industry=panel.pop('industry');panel=panel.astype(float)
    labels,end=execution_labels(prices,5)
    labels.index.name='date';labels.columns.name='symbol'
    # Stack only weekly dates to avoid a full daily multi-index duplicate.
    labels=labels.reindex(panel.index.get_level_values('date').unique()).stack().reindex(panel.index)
    plans=pd.DatetimeIndex(json.loads((ROOT/'output/candidate_factors_20261022/schedules.json').read_text())['WEEKLY'])
    assert plans.min()==pd.Timestamp(q['start']) and plans.max()<=pd.Timestamp(q['end'])
    assert plans.isin(panel.index.get_level_values('date')).all()
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    themes=p['selection']['theme_industries']
    for backend in ['lightgbm','catboost']:
        for month in plans.to_period('M').unique():
            dates=plans[plans.to_period('M')==month];refit=dates[0]
            dest=OUT/'training'/backend/str(month)
            if (dest/'complete.json').exists():verify_sealed(dest);continue
            x,y,log=training_sample(panel,labels,end,prices.index,refit,q)
            model,medians=fit_model(backend,x,y)
            ranks=[];diagnostics=[]
            for date in dates:
                frame=panel.xs(date,level='date');ind=industry.xs(date,level='date')
                score=ordered_scores(predict(model,frame,medians))
                truth=labels.xs(date,level='date').reindex(score.index)
                for area in q['areas']:
                    s=score if area=='BROAD' else score[ind.reindex(score.index).isin(themes)]
                    pairs=pd.concat([s.rename('prediction'),truth.rename('label')],axis=1).reindex(s.index).dropna()
                    rankic=pairs.prediction.corr(pairs.label,method='spearman') if len(pairs)>=30 else np.nan
                    diagnostics.append(dict(date=date,area=area,backend=backend,candidates=len(s),
                        label_pairs=len(pairs),rank_ic=rankic,raw_top15_return=truth.reindex(s.head(15).index).mean(),
                        pool_return=truth.reindex(s.index).mean(),refit_date=refit))
                    for rank,(symbol,value) in enumerate(s.head(100).items(),1):
                        ranks.append(dict(date=date,area=area,backend=backend,symbol=symbol,score=value,
                                          rank=rank,industry=ind.loc[symbol],refit_date=refit))
            dest.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(ranks).to_csv(dest/'rankings.csv.gz',index=False)
            pd.DataFrame(diagnostics).to_csv(dest/'diagnostics.csv',index=False)
            pd.DataFrame([log]).to_csv(dest/'training.csv',index=False)
            pd.DataFrame({'median':medians,'importance':model.feature_importances_}).to_csv(dest/'features.csv')
            write_json(dest/'parameters.json',model.get_params())
            write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
            print('Trained',backend,str(month),log['train_rows'],'rows',flush=True)
    checkpoints=sorted((OUT/'training').glob('*/*'))
    for name in ['rankings.csv.gz','diagnostics.csv','training.csv']:
        frames=[]
        for folder in checkpoints:
            f=pd.read_csv(folder/name)
            if 'backend' not in f:f['backend']=folder.parent.name
            frames.append(f)
        pd.concat(frames,ignore_index=True).to_csv(OUT/name,index=False)


def accounts(p,q,full):
    parent=ROOT/q['repaired_parent'];prices=full.loc[q['start']:q['end']]
    returns=full.pct_change(fill_method=None)
    schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    events=json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']
    adj=pd.read_csv(parent/'event_adjustments.csv.gz',index_col='date',parse_dates=True)
    dead=pd.to_datetime(pd.read_csv(parent/'delist_dates.csv',index_col=0).iloc[:,0])
    rankings=pd.read_csv(OUT/'rankings.csv.gz',parse_dates=['date'])
    references={};rows=[];navs={};annual=[]
    for backend in q['arms']:
        for area in q['areas']:
            if backend!='BASE':
                group={}
                for date,f in rankings[rankings.backend.eq(backend)&rankings.area.eq(area)].groupby('date'):
                    f=f.sort_values('rank').set_index('symbol');group[date]=dict(ic=f.score,fixed=f.score,industry=f.industry)
                risk={d:history_risk(returns,d,g['ic'].index,p['selection']) for d,g in group.items()}
            for fee in q['fees']:
                key=f'{backend}_{area}_{round(fee*10000)}bp'
                if backend=='BASE':
                    oldkey='BASE_'+('TECH15_' if area=='TECH15' else '')+f'WEEKLY_{round(fee*10000)}bp'
                    dest=parent/oldkey;verify_sealed(dest)
                else:
                    dest=OUT/key
                    if (dest/'complete.json').exists():verify_sealed(dest)
                    else:
                        pp=copy.deepcopy(p)
                        if area=='TECH15':pp['selection']['theme_cap']=1.
                        account=Account('IC_CONSTRAINED15_RISK',group,risk,returns,pp)
                        result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,fee=fee)
                        result['decisions']=pd.DataFrame(account.logs)
                        result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0])
                        dest.mkdir(exist_ok=True)
                        for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
                        write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
                references[key]=str(dest.relative_to(ROOT))
                daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True)
                ex=pd.read_csv(dest/'executions.csv.gz',parse_dates=['date'])
                nav=daily.post_nav;navs[key]=nav
                for period,start,endyear in [('FULL','2012','2026'),('RECENT','2021','2026')]:
                    v=nav.loc[start:endyear];first=nav.index.get_loc(v.index[0]);base=nav.iloc[first-1:first] if first else nav.iloc[:0]
                    value=pd.concat([base,v]);m={k:float(z[0]) for k,z in metrics(value).items()}
                    n=len(value)-1;e=ex[ex.date.between(v.index[0],v.index[-1])];d=daily.loc[v.index]
                    rows.append(dict(key=key,backend=backend,area=area,fee=fee,period=period,**m,
                        average_equity=float((1-d.cash/d.post_nav).mean()),
                        annual_two_way_turnover=float(e.two_way_turnover.sum()/(n/252)),
                        annual_action_dates=float(e.action.ne('HOLD').sum()/(n/252)),
                        terminal_unquoted_fraction=float(d.unquoted_value.iloc[-1]/d.post_nav.iloc[-1])))
                for year in range(2012,2027):
                    v=nav.loc[str(year)];loc=nav.index.get_loc(v.index[0]);base=nav.iloc[loc-1] if loc else 1.
                    annual.append(dict(key=key,year=year,return_value=v.iloc[-1]/base-1))
                print('Account complete',key,flush=True)
    write_json(OUT/'accounts.json',references)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv')
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completed.json'})


def run(stage):
    p,q=freeze();full=load_prices(p).loc[:q['end']]
    if stage in ['all','train']:prepare(p,q,full)
    if stage in ['all','replay']:accounts(p,q,full)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','train','replay'],default='all')
    run(parser.parse_args().stage)
