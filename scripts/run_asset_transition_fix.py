#!/usr/bin/env python3
"""Re-evaluate unchanged strategies with explicit mandatory asset transitions."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from backtest.action_replay import replay
from analysis.single_account_adaptive import history_risk
from analysis.holdings_count import metrics
from analysis.rebalance_frequency import classify_executions
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_rebalance_frequency import verify_sealed
from scripts.run_mvp_holdings_count import sha,write_json
from universe.strategy import point_in_time_industry

OUT=ROOT/'output/asset_transition_fix_20261024'
OLD=ROOT/'output/recent_holdings_tech_20261023'
CAND=ROOT/'output/candidate_factors_20261022'
PROTOCOL=ROOT/'research_protocols/20261024_asset_transition_fix_v1.json'


def tasks():
    items=[]
    for area in ['BROAD','TECH15']:
        for arm in ['BASE','ALL5']:
            for frequency in ['WEEKLY','MONTHLY']:
                for fee in [.0003,.0013]:
                    key=f'{arm}_'+('TECH15_' if area=='TECH15' else '')+f'{frequency}_{int(round(fee*10000))}bp'
                    folder=(OLD if area=='TECH15' else CAND)/key
                    items.append(dict(key=key,area=area,arm=arm,frequency=frequency,fee=fee,old=folder))
    for arm,fee in [('DIVIDEND_TTM',0),('DIVIDEND_TTM',.0003),('DIVIDEND_TTM',.0013),('ALL5',0)]:
        key=f'{arm}_WEEKLY_{int(round(fee*10000))}bp'
        items.append(dict(key=key,area='BROAD',arm=arm,frequency='WEEKLY',fee=fee,old=CAND/key))
    return items


def freeze():
    q=json.loads(PROTOCOL.read_text());p=json.loads((ROOT/'research_protocols/20261020_single_account_adaptive_v1.json').read_text())
    files=dict(json.loads((OLD/'frozen.json').read_text())['files'])
    for path,h in files.items():
        if sha(ROOT/path)!=h:raise ValueError('changed inherited input '+path)
    for parent in [OLD,CAND]:
        for f,h in json.loads((parent/'completed.json').read_text()).items():
            if sha(parent/f)!=h:raise ValueError('changed frozen result '+str(parent/f))
        files[str((parent/'completed.json').relative_to(ROOT))]=sha(parent/'completed.json')
    for path in [PROTOCOL,Path(__file__),ROOT/q['registry'],ROOT/'backtest/action_replay.py',ROOT/'backtest/corporate_actions.py']+list((ROOT/'data/corporate_actions_evidence').glob('*')):
        files[str(path.relative_to(ROOT))]=sha(path)
    stamp=dict(files=files,protocol=sha(PROTOCOL));OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text())!=stamp:raise ValueError('frozen run changed')
    if (OUT/'completed.json').exists():raise ValueError('completed run immutable')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():write_json(OUT/'forward_before.json',account_hashes())
    return p,q


def inputs(p,q,prices):
    raw=ROOT/'data/all_a_history_20100101_20261015'
    events=json.loads((ROOT/q['registry']).read_text())['events']
    symbols={e[k] for e in events for k in ['source','destination']}
    parts=[]
    for chunk in pd.read_csv(raw/'adj_factor.csv.gz',usecols=['ts_code','trade_date','adj_factor'],dtype={'trade_date':str},chunksize=500000):
        parts.append(chunk[chunk.ts_code.isin(symbols)])
    a=pd.concat(parts);a['date']=pd.to_datetime(a.trade_date,format='%Y%m%d')
    if a.duplicated(['date','ts_code']).any():raise ValueError('duplicate raw adjustment')
    adj=a.pivot(index='date',columns='ts_code',values='adj_factor').sort_index()
    adj.to_csv(OUT/'event_adjustments.csv.gz')
    b=pd.read_csv(raw/'stock_basic.csv.gz',dtype=str)
    b['delist']=pd.to_datetime(b.delist_date,format='%Y%m%d',errors='coerce')
    if b.ts_code.duplicated().any():raise ValueError('duplicate stock metadata')
    missing=b.delist_date.notna()&b.delist.isna()
    if missing.any():raise ValueError('unparsed delisting date')
    dead=b.set_index('ts_code').delist.reindex(prices.columns)
    assert prices.columns.isin(b.ts_code).all()
    dead.to_csv(OUT/'delist_dates.csv')
    # Validate the adjustment convention against independent raw closes at every event endpoint.
    needed=[]
    for e in events:
        d=pd.Timestamp(e['effective_date']);last=prices.loc[:e['record_date'],e['source']].last_valid_index()
        needed += [(last,e['source']),(d,e['destination'])]
    rawparts=[]
    for chunk in pd.read_csv(raw/'daily.csv.gz',usecols=['ts_code','trade_date','close'],dtype={'trade_date':str},chunksize=500000):
        rawparts.append(chunk[chunk.ts_code.isin(symbols)])
    rawclose=pd.concat(rawparts);rawclose['date']=pd.to_datetime(rawclose.trade_date,format='%Y%m%d')
    rawclose=rawclose.set_index(['date','ts_code']).close
    checks=[]
    for date,s in needed:
        val=float(rawclose.loc[(date,s)]);f=float(adj.loc[date,s]);np.testing.assert_allclose(val*f,prices.loc[date,s],atol=1e-8)
        checks.append(dict(date=date,symbol=s,raw_close=val,adjustment=f,adjusted_close=prices.loc[date,s]))
    pd.DataFrame(checks).to_csv(OUT/'unit_convention_checks.csv',index=False)
    return events,adj,dead


def groups(p):
    plans={k:pd.DatetimeIndex(v) for k,v in json.loads((CAND/'schedules.json').read_text()).items()}
    broad=pd.read_csv(CAND/'rankings.csv.gz',parse_dates=['date'])
    broad=broad[broad.arm.isin(['BASE','ALL5','DIVIDEND_TTM'])].copy()
    idx=pd.MultiIndex.from_frame(broad[['date','symbol']].drop_duplicates())
    membership=pd.read_csv(ROOT/'data/all_a_history_20100101_20261015/index_member_all.csv.gz')
    ind=point_in_time_industry(idx,membership)
    broad['industry']=ind.reindex(pd.MultiIndex.from_frame(broad[['date','symbol']])).to_numpy()
    tech=pd.read_csv(OLD/'rankings.csv.gz',parse_dates=['date'])
    result={}
    for area,x in [('BROAD',broad),('TECH15',tech)]:
        for a,g in x.groupby('arm'):
            result[area,a]={}
            for d,frame in g.groupby('date'):
                f=frame.sort_values('rank').set_index('symbol');score=f.score
                result[area,a][d]=dict(ic=score,fixed=score,industry=f.industry)
    return result,plans


def run():
    p,q=freeze();full=load_prices(p).loc[:q['end']];prices=full.loc[q['start']:]
    schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    events,adj,dead=inputs(p,q,full)
    g,plans=groups(p);returns=full.pct_change(fill_method=None)
    rows=[];annual=[];navs={};parities=[];all_actions=[];cache={}
    for task in tasks():
        key=task['key'];dest=OUT/key;groupkey=(task['area'],task['arm'])
        if (dest/'complete.json').exists():verify_sealed(dest)
        else:
            if groupkey not in cache:
                cache={groupkey:{d:history_risk(returns,d,x['ic'].index,p['selection']) for d,x in g[groupkey].items()}}
            pp=copy.deepcopy(p)
            if task['area']=='TECH15':pp['selection']['theme_cap']=1.
            selected={d:g[groupkey][d] for d in plans[task['frequency']]}
            account=Account('IC_CONSTRAINED15_RISK',selected,cache[groupkey],returns,pp)
            result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,fee=task['fee'])
            result['decisions']=pd.DataFrame(account.logs)
            result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0])
            dest.mkdir(exist_ok=True)
            for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
            write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='complete.json'})
        daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True)
        old=pd.read_csv(task['old']/'daily.csv.gz',index_col='date',parse_dates=True)
        actions=pd.read_csv(dest/'asset_actions.csv.gz',parse_dates=['date'])
        until=actions.date.min() if len(actions) else pd.Timestamp.max
        prior=daily.index<until
        error=float((daily.loc[prior,'post_nav']-old.loc[prior,'post_nav']).abs().max())
        np.testing.assert_allclose(daily.loc[prior,old.columns],old.loc[prior],atol=1e-10,rtol=1e-10)
        parities.append(dict(key=key,pre_action_max_nav_error=error,events=len(actions),no_event_full_parity=not len(actions)))
        if len(actions):all_actions.append(actions.assign(key=key))
        for period,start in [('FULL','2012'),('RECENT','2021')]:
            vals={}
            for version,frame in [('old',old),('new',daily)]:
                v=frame.post_nav;x=v.loc[start:];loc=v.index.get_loc(x.index[0])
                if loc:x=pd.concat([v.iloc[loc-1:loc],x])
                vals.update({version+'_'+k:float(z[0]) for k,z in metrics(x).items()})
            rows.append(dict(key=key,period=period,fee=task['fee'],**vals,terminal_unquoted_fraction=float(daily.unquoted_value.iloc[-1]/daily.post_nav.iloc[-1])))
        for y in range(2012,2027):
            rec=dict(key=key,year=y)
            for version,frame in [('old',old),('new',daily)]:
                x=frame.post_nav.loc[str(y)];pos=frame.index.get_loc(x.index[0]);base=frame.post_nav.iloc[pos-1] if pos else 1.
                rec[version+'_return']=x.iloc[-1]/base-1
            annual.append(rec)
        navs[key]=daily.post_nav
        print('Completed',key,'actions',len(actions),'pre-event parity',error,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(parities).to_csv(OUT/'legacy_parity.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv')
    pd.concat(all_actions,ignore_index=True).to_csv(OUT/'all_asset_actions.csv',index=False)
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completed.json'})
    print('20-account asset repair complete; old outputs and paper accounts preserved.',flush=True)

if __name__=='__main__':run()
