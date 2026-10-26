#!/usr/bin/env python3
"""Continue frozen scores with a new, explicitly versioned OTC asset ledger."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from backtest.delisting_replay_v2 import replay,DelistingBridge
from analysis.single_account_adaptive import history_risk
from analysis.holdings_count import metrics
from analysis.rebalance_frequency import classify_executions
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_asset_transition_fix import groups as baseline_groups
from scripts.run_rebalance_frequency import verify_sealed
from scripts.run_mvp_holdings_count import sha,write_json

OUT=ROOT/'output/fixed15_ml_delisting_v3_20261024'
PROTOCOL=ROOT/'research_protocols/20261024_fixed15_ml_delisting_v3.json'
PARENT=ROOT/'output/asset_transition_fix_20261024'


def freeze():
    q=json.loads(PROTOCOL.read_text());train=ROOT/q['score_parent'];audit=ROOT/q['audit_parent']
    p=json.loads((ROOT/'research_protocols/20261020_single_account_adaptive_v1.json').read_text())
    files=dict(json.loads((train/'frozen.json').read_text())['files'])
    for name,h in files.items():assert sha(ROOT/name)==h,name
    for base,manifest in [(ROOT/x,'complete.json') for x in q['market_data'].values()]+[(audit,'completed.json')]:
        for name,h in json.loads((base/manifest).read_text()).items():
            assert sha(base/name)==h,name;files[str((base/name).relative_to(ROOT))]=h
    for path in [PROTOCOL,Path(__file__),ROOT/q['registry'],ROOT/'backtest/delisting_replay_v2.py',train/'rankings.csv.gz',
                 ROOT/'scripts/fetch_delisted_000982.py',ROOT/'scripts/fetch_delisted_600297.py',ROOT/'tests/test_delisting_replay_v2.py',train/'forward_before.json']:
        files[str(path.relative_to(ROOT))]=sha(path)
    stamp=dict(files=files);OUT.mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():assert json.loads((OUT/'frozen.json').read_text())==stamp
    if (OUT/'completed.json').exists():raise ValueError('Completed version immutable')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    return p,q


def inputs(full,q):
    events=json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']
    transfers=json.loads((ROOT/q['registry']).read_text())['events']
    adj=pd.read_csv(PARENT/'event_adjustments.csv.gz',index_col='date',parse_dates=True)
    path=OUT/'adjustments.csv.gz'
    if path.exists():adj=pd.read_csv(path,index_col='date',parse_dates=True)
    else:
        chunks=[];symbols=[e['source'] for e in transfers]
        for x in pd.read_csv(ROOT/'data/all_a_history_20100101_20261015/adj_factor.csv.gz',usecols=['ts_code','trade_date','adj_factor'],dtype={'trade_date':str},chunksize=500000):
            chunks.append(x[x.ts_code.isin(symbols)])
        x=pd.concat(chunks);x['date']=pd.to_datetime(x.trade_date,format='%Y%m%d')
        if x.duplicated(['date','ts_code']).any():raise ValueError('duplicate raw adjustment')
        adj=adj.join(x.pivot(index='date',columns='ts_code',values='adj_factor'),how='outer').sort_index();adj.to_csv(path)
    dead=pd.to_datetime(pd.read_csv(PARENT/'delist_dates.csv',index_col=0).iloc[:,0])
    bars={}
    for e in transfers:
        b=pd.read_csv(ROOT/q['market_data'][e['destination']]/(e['destination'].replace('.','_')+'_daily_raw.csv'),index_col='date',parse_dates=True)
        bars[e['destination']]=b.loc[e['listing_date']:e.get('coverage_end',q['end'])]
    bridge=DelistingBridge(full,adj,dead,transfers,bars,q['confirmations'])
    write_json(OUT/'market_checks.json',{e['source']:dict(rows=len(bars[e['destination']]),
        first=str(bars[e['destination']].index.min().date()),last=str(bars[e['destination']].index.max().date()),
        source_factor=float(adj.loc[e['last_source_quote'],e['source']]),
        raw_close=e['last_raw_close'],source_adjusted_close=float(full.loc[e['last_source_quote'],e['source']]),
        first_day_sellable=bool(bridge.sell_masks[e['destination']].loc[e['listing_date']])) for e in transfers})
    return events,transfers,adj,dead,bars


def run():
    p,q=freeze();full=load_prices(p).loc[:q['end']];prices=full.loc[q['start']:];returns=full.pct_change(fill_method=None)
    events,transfers,adj,dead,bars=inputs(full,q)
    schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    baseline,plans=baseline_groups(p)
    ranking=pd.read_csv(ROOT/q['score_parent']/'rankings.csv.gz',parse_dates=['date'])
    refs={};navs={};summary=[];parities=[];annual=[];eventrows=[];exits=[]
    auditrefs=json.loads((ROOT/q['audit_parent']/'accounts.json').read_text())
    for model in ['BASE','lightgbm','catboost']:
        for area in ['BROAD','TECH15']:
            if model=='BASE':group={d:baseline[area,'BASE'][d] for d in plans['WEEKLY']}
            else:
                group={}
                for d,g in ranking[ranking.backend.eq(model)&ranking.area.eq(area)].groupby('date'):
                    x=g.sort_values('rank').set_index('symbol');group[d]=dict(ic=x.score,fixed=x.score,industry=x.industry)
            risk={d:history_risk(returns,d,g['ic'].index,p['selection']) for d,g in group.items()}
            for fee in q['fees']:
                key=f'{model}_{area}_{round(fee*10000)}bp';dest=OUT/key
                if (dest/'complete.json').exists():verify_sealed(dest)
                else:
                    pp=copy.deepcopy(p)
                    if area=='TECH15':pp['selection']['theme_cap']=1.
                    account=Account('IC_CONSTRAINED15_RISK',group,risk,returns,pp)
                    result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,
                                  transfers=transfers,otc_bars=bars,confirmations=q['confirmations'],fee=fee)
                    result['decisions']=pd.DataFrame(account.logs)
                    result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0])
                    dest.mkdir(exist_ok=True)
                    for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
                    write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
                refs[key]=str(dest.relative_to(ROOT));daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True)
                old=pd.read_csv(ROOT/auditrefs[key]/'daily.csv.gz',index_col='date',parse_dates=True)
                common=old.index if model=='BASE' else old.loc[:'2024-08-09'].index
                np.testing.assert_allclose(daily.loc[common,old.columns],old.loc[common],atol=1e-10,rtol=1e-10)
                parities.append(dict(key=key,until=str(common[-1].date()),max_nav_error=float((daily.loc[common,'post_nav']-old.post_nav.loc[common]).abs().max())))
                n=daily.post_nav;navs[key]=n;ex=pd.read_csv(dest/'executions.csv.gz',parse_dates=['date'])
                for period,start in [('FULL','2012'),('RECENT','2021')]:
                    v=n.loc[start:];i=n.index.get_loc(v.index[0]);v=pd.concat([n.iloc[i-1:i],v]) if i else v
                    m={k:float(z[0]) for k,z in metrics(v).items()}
                    e=ex[ex.date.ge(start)];d=daily.loc[start:]
                    summary.append(dict(key=key,model=model,area=area,fee=fee,period=period,**m,
                        average_equity=(1-d.cash/d.post_nav).mean(),annual_two_way_turnover=e.two_way_turnover.sum()/((len(v)-1)/252),
                        terminal_unquoted_fraction=d.unquoted_value.iloc[-1]/d.post_nav.iloc[-1]))
                for year in range(2012,2027):
                    v=n.loc[str(year)];i=n.index.get_loc(v.index[0]);base=n.iloc[i-1] if i else 1.
                    annual.append(dict(key=key,year=year,return_value=v.iloc[-1]/base-1))
                actions=pd.read_csv(dest/'asset_actions.csv.gz');eventrows.append(actions.assign(key=key))
                t=pd.read_csv(dest/'trades.csv.gz');exits.append(t[t.symbol.isin([e['destination'] for e in transfers])].assign(key=key))
                print('Complete',key,'OTC exits',len(exits[-1]),flush=True)
    write_json(OUT/'accounts.json',refs);pd.DataFrame(navs).to_csv(OUT/'nav.csv')
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(parities).to_csv(OUT/'parity.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.concat(eventrows).to_csv(OUT/'all_asset_actions.csv',index=False);pd.concat(exits).to_csv(OUT/'otc_exits.csv',index=False)
    assert account_hashes()==json.loads((ROOT/q['score_parent']/'forward_before.json').read_text())
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completed.json'})
    print('COMPLETE all12 accounts',flush=True)


if __name__=='__main__':run()
