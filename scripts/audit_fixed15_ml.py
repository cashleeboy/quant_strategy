#!/usr/bin/env python3
"""Attempt every pre-registered account; preserve rejected prefixes, never bypass guards.

This audit exists because the strict original runner stopped at an unknown
terminated holding. No scoring, prices, fees or accounting rules are modified.
Exception-local inspection is limited to our replay frame and exports its ledger.
"""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from backtest.action_replay import replay
from backtest.corporate_actions import AssetTransitionError
from analysis.single_account_adaptive import history_risk
from analysis.holdings_count import metrics
from analysis.rebalance_frequency import classify_executions
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_rebalance_frequency import verify_sealed
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_fixed15_ml import OUT as TRAIN,PROTOCOL

OUT=ROOT/'output/fixed15_ml_20261024_audit'


def rejected_prefix(error):
    tb=error.__traceback__;state=None
    while tb:
        if tb.tb_frame.f_code is replay.__code__:state=tb.tb_frame.f_locals
        tb=tb.tb_next
    if state is None:raise RuntimeError('No replay diagnostic frame')
    last=pd.DataFrame(state['daily']).set_index('date')
    assets=state['assets'];date=state['date'];mark=state['mark'];units=state['shares'];names=state['names']
    expired=assets.delist.notna()&assets.delist.le(date)
    bad=expired.to_numpy()&(units*mark>1e-10)
    nav=state['cash']+units@mark
    blocked=[]
    for i in np.flatnonzero(bad):
        symbol=names[i];quotes=state['prices'].loc[:date,symbol].dropna()
        blocked.append(dict(symbol=symbol,units=float(units[i]),stale_value=float(units[i]*mark[i]),
            stale_weight=float(units[i]*mark[i]/nav),last_quote=quotes.index[-1],last_mark=float(mark[i]),delist_date=assets.delist.loc[symbol]))
    log=dict(status='BLOCKED',error=str(error),failure_date=date,last_valid_date=last.index[-1],blocked_assets=blocked)
    return dict(daily=last,executions=pd.DataFrame(state['executions']),trades=pd.DataFrame(state['trades']),
                targets=pd.DataFrame(state['plans']),positions=pd.DataFrame(state['positions']),asset_actions=assets.frame()),log


def run():
    q=json.loads(PROTOCOL.read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    for path,h in json.loads((TRAIN/'frozen.json').read_text())['files'].items():assert sha(ROOT/path)==h,path
    for folder in sorted((TRAIN/'training').glob('*/*')):verify_sealed(folder)
    OUT.mkdir(exist_ok=True)
    files={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),PROTOCOL,TRAIN/'frozen.json',TRAIN/'rankings.csv.gz',TRAIN/'training.csv',TRAIN/'diagnostics.csv']}
    stamp=dict(files=files,original_protocol_sha=sha(PROTOCOL),purpose='All accounts attempted; failed prefixes excluded from full-period comparison')
    if (OUT/'frozen.json').exists():assert json.loads((OUT/'frozen.json').read_text())==stamp
    if (OUT/'completed.json').exists():raise ValueError('Immutable completed audit')
    write_json(OUT/'frozen.json',stamp)
    full=load_prices(p).loc[:q['end']];prices=full.loc[q['start']:];returns=full.pct_change(fill_method=None)
    schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    parent=ROOT/q['repaired_parent'];events=json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']
    adj=pd.read_csv(parent/'event_adjustments.csv.gz',index_col='date',parse_dates=True)
    dead=pd.to_datetime(pd.read_csv(parent/'delist_dates.csv',index_col=0).iloc[:,0])
    rankings=pd.read_csv(TRAIN/'rankings.csv.gz',parse_dates=['date'])
    references={};states={};rows=[];navs={};annual=[]
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
                    dest=parent/('BASE_'+('TECH15_' if area=='TECH15' else '')+f'WEEKLY_{round(fee*10000)}bp')
                    verify_sealed(dest);status={'status':'COMPLETE','baseline_reference':True}
                else:
                    dest=OUT/key
                    if (dest/'complete.json').exists():verify_sealed(dest);status=json.loads((dest/'status.json').read_text())
                    else:
                        pp=copy.deepcopy(p)
                        if area=='TECH15':pp['selection']['theme_cap']=1.
                        account=Account('IC_CONSTRAINED15_RISK',group,risk,returns,pp)
                        try:
                            result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,fee=fee)
                            status={'status':'COMPLETE'}
                        except AssetTransitionError as error:
                            result,status=rejected_prefix(error)
                        result['decisions']=pd.DataFrame(account.logs)
                        result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0])
                        dest.mkdir(exist_ok=True)
                        for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
                        write_json(dest/'status.json',status)
                        write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
                references[key]=str(dest.relative_to(ROOT));states[key]=status
                print(key,json.dumps(status,default=str),flush=True)
                if status['status']!='COMPLETE':continue
                daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True)
                ex=pd.read_csv(dest/'executions.csv.gz',parse_dates=['date']);nav=daily.post_nav;navs[key]=nav
                for period,start in [('FULL','2012'),('RECENT','2021')]:
                    v=nav.loc[start:];first=nav.index.get_loc(v.index[0]);value=pd.concat([nav.iloc[first-1:first],v]) if first else v
                    m={k:float(z[0]) for k,z in metrics(value).items()};e=ex[ex.date.between(v.index[0],v.index[-1])];d=daily.loc[v.index]
                    rows.append(dict(key=key,backend=backend,area=area,fee=fee,period=period,**m,
                        average_equity=float((1-d.cash/d.post_nav).mean()),annual_two_way_turnover=e.two_way_turnover.sum()/((len(value)-1)/252),
                        annual_action_dates=e.action.ne('HOLD').sum()/((len(value)-1)/252),
                        terminal_unquoted_fraction=float(d.unquoted_value.iloc[-1]/d.post_nav.iloc[-1])))
                for year in range(2012,2027):
                    v=nav.loc[str(year)];loc=nav.index.get_loc(v.index[0]);base=nav.iloc[loc-1] if loc else 1.
                    annual.append(dict(key=key,year=year,return_value=v.iloc[-1]/base-1))
    write_json(OUT/'accounts.json',references);write_json(OUT/'status.json',states)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv');pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    assert account_hashes()==json.loads((TRAIN/'forward_before.json').read_text())
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completed.json'})
    print('AUDIT COMPLETE; full-period account count',len(navs),flush=True)


if __name__=='__main__':run()
