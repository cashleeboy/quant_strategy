#!/usr/bin/env python3
"""Holdings audit and frozen same-rule technology pool counterfactual."""
from pathlib import Path
import sys, json, copy, argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from analysis.recent_holdings import attribute, contribution
from analysis.single_account_adaptive import prior_weights, family_score, history_risk
from analysis.rebalance_frequency import asof_weights, classify_executions
from analysis.candidate_factors import CANDIDATES
from analysis.holdings_count import metrics
from backtest.research_replay import replay
from scripts.run_single_account_adaptive import Account, account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_rebalance_frequency import verify_sealed
from scripts.run_mvp_holdings_count import sha, write_json
from strategies.holding_buffer import ordered_scores
from universe.strategy import point_in_time_industry

OUT = ROOT/'output/recent_holdings_tech_20261023'
PROTOCOL = ROOT/'research_protocols/20261023_recent_holdings_tech_v1.json'
PARENT = ROOT/'output/candidate_factors_20261022'


def freeze():
    q = json.loads(PROTOCOL.read_text())
    c = json.loads((PARENT/'protocol.json').read_text())
    p = json.loads((ROOT/c['parent_protocol']).read_text())
    src = json.loads((ROOT/c['source_protocol']).read_text())
    files = dict(json.loads((PARENT/'frozen.json').read_text())['files'])
    for f, digest in files.items():
        if sha(ROOT/f) != digest:
            raise ValueError('Changed parent input '+f)
    for f, digest in json.loads((PARENT/'completed.json').read_text()).items():
        if sha(PARENT/f) != digest:
            raise ValueError('Changed parent result '+f)
    for f in [PROTOCOL, Path(__file__), ROOT/'analysis/recent_holdings.py', PARENT/'completed.json',
              ROOT/'backtest/research_replay.py']:
        files[str(f.relative_to(ROOT))] = sha(f)
    for f in ['nav.csv','benchmark_nav.csv','stress_nav.csv']:
        path = ROOT/p['factor_source']/f
        files[str(path.relative_to(ROOT))] = sha(path)
    stamp = dict(files=files, protocol=sha(PROTOCOL))
    OUT.mkdir(exist_ok=True, parents=True)
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text()) != stamp:
        raise ValueError('Frozen experiment changed')
    write_json(OUT/'frozen.json',stamp)
    write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():
        write_json(OUT/'forward_before.json',account_hashes())
    return p, q, c, src


def saved_factory(folder):
    decisions = pd.read_csv(folder/'decisions.csv.gz',parse_dates=['date']).set_index('date')
    targets = pd.read_csv(folder/'targets.csv.gz',parse_dates=['date'])
    groups = {d:g.set_index('symbol').weight for d,g in targets.groupby('date')}
    def f(d, held):
        if bool(decisions.loc[d,'action']):
            return groups.get(d,pd.Series(dtype=float))
        return None
    return f


def save_result(dest, result, logs=None):
    dest.mkdir(exist_ok=True)
    if logs is not None:
        result['decisions'] = pd.DataFrame(logs)
        ex = classify_executions(result['executions'],result['decisions'],result['daily'].index[0])
        result['executions'] = ex
    for key, frame in result.items():
        frame.to_csv(dest/(key+'.csv.gz'),index=(key=='daily'))
    write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='complete.json'})


def restore_parents(prices, schedule):
    for arm in ['BASE','ALL5']:
        for freq in ['WEEKLY','MONTHLY']:
            key=f'{arm}_{freq}_3bp';dest=OUT/key;old=PARENT/key
            if (dest/'complete.json').exists():
                verify_sealed(dest);continue
            result=replay(prices,schedule,saved_factory(old),fee=.0003)
            expected=pd.read_csv(old/'daily.csv.gz',index_col='date',parse_dates=True)
            pd.testing.assert_frame_equal(result['daily'],expected,check_dtype=False,atol=1e-11,rtol=1e-11)
            ex=pd.read_csv(old/'executions.csv.gz',parse_dates=['date','signal_date'])
            pd.testing.assert_frame_equal(result['executions'],ex[result['executions'].columns],check_dtype=False,atol=1e-11,rtol=1e-11)
            result['executions']=ex
            save_result(dest,result)
            print('Exact parent NAV and execution parity:',key,flush=True)


def tech_groups(p,q,c):
    plans={k:pd.DatetimeIndex(v) for k,v in json.loads((PARENT/'schedules.json').read_text()).items()}
    needed=plans['WEEKLY'].union(plans['MONTHLY'])
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
    prior=prior_weights(families)
    raw_weights=pd.read_csv(PARENT/'factor_weights.csv',parse_dates=['date'])
    weights={a:raw_weights[raw_weights.arm.eq(a)].pivot(index='date',columns='factor',values='weight') for a in q['recipes']}
    oldranks=pd.read_csv(PARENT/'rankings.csv.gz',parse_dates=['date'])
    expected={(a,d):g.sort_values('rank') for (a,d),g in oldranks[oldranks.arm.isin(q['recipes'])].groupby(['arm','date'])}
    groups={a:{} for a in q['recipes']};pools={};ranks=[];parity=[]
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        quarter=folder.name.removeprefix('inputs_')
        x=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        mf=ROOT/c['parent_output']/f'month_inputs_{quarter}'/'panel.csv.gz'
        if mf.exists():
            m=pd.read_csv(mf,parse_dates=['date']).set_index(['date','symbol'])
            x=pd.concat([x,m.loc[~m.index.get_level_values(0).isin(x.index.get_level_values(0).unique())]])
        x=x.loc[x.index.get_level_values(0).isin(needed)]
        if x.empty:continue
        new=pd.read_csv(PARENT/f'inputs_{quarter}'/'standard.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        for d,g in x.groupby(level=0):
            pool=family_score(g,families,prior).droplevel(0).index
            merged=g.join(new.reindex(g.index)[CANDIDATES].fillna(0))
            ind=g.industry.droplevel(0)
            tech=ind.reindex(pool).isin(q['technology_definition'])
            pools[d]=pool[tech]
            if len(pools[d])<15:raise ValueError('Tech universe unexpectedly small')
            for arm in q['recipes']:
                recipe=copy.deepcopy(families)
                if arm=='ALL5':
                    for f in CANDIDATES:recipe[c['candidate_families'][f]].append(f)
                w,updated=asof_weights(weights[arm],d)
                full=ordered_scores(family_score(merged,recipe,w,require_all=False).droplevel(0).reindex(pool))
                e=expected[arm,d]
                assert full.head(100).index.tolist()==e.symbol.tolist()
                np.testing.assert_allclose(full.head(100),e.score,atol=1e-10,rtol=1e-10)
                score=full.loc[full.index.isin(pools[d])].head(100)
                groups[arm][d]=dict(ic=score,fixed=score,industry=ind)
                parity.append(dict(arm=arm,date=d,full_pool_top100_parity=True,pool=len(pools[d])))
                for rank,(s,v) in enumerate(score.items(),1):
                    ranks.append(dict(arm=arm,date=d,symbol=s,rank=rank,score=v,industry=ind[s],weight_update=updated))
        print('Technology scores',quarter,flush=True)
    pd.DataFrame(ranks).to_csv(OUT/'rankings.csv.gz',index=False)
    pd.DataFrame(parity).to_csv(OUT/'score_parity.csv',index=False)
    return groups,pools,plans


def run_tech(p,q,c,allprices,prices,schedule):
    keys=[f'{a}_{v}_{f}_{int(round(fee*10000))}bp' for a in q['recipes'] for v in q['variants'] for f in q['frequencies'] for fee in q['fees']]
    keys += [f'TECH_EQ_{f}_3bp' for f in q['frequencies']]
    if all((OUT/key/'complete.json').exists() for key in keys):
        for key in keys:verify_sealed(OUT/key)
        return
    groups,pools,plans=tech_groups(p,q,c)
    returns=allprices.pct_change(fill_method=None)
    for arm in q['recipes']:
        cache={d:history_risk(returns,d,g['ic'].index,p['selection']) for d,g in groups[arm].items()}
        print('Technology risk cache ready',arm,flush=True)
        for variant,cap in q['variants'].items():
            params=copy.deepcopy(p);params['selection']['theme_cap']=cap
            for freq in q['frequencies']:
                selected={d:groups[arm][d] for d in plans[freq]}
                for fee in q['fees']:
                    key=f'{arm}_{variant}_{freq}_{int(round(fee*10000))}bp';dest=OUT/key
                    if (dest/'complete.json').exists():verify_sealed(dest);continue
                    a=Account('IC_CONSTRAINED15_RISK',selected,cache,returns,params)
                    result=replay(prices,schedule,a,fee=fee)
                    log=pd.DataFrame(a.logs)
                    assert log.selected.max() <= (5 if variant=='TECH_LITERAL' else 15)
                    assert result['daily'].cash.min()>-1e-9
                    save_result(dest,result,a.logs)
                    print('Completed',key,flush=True)
    for freq in q['frequencies']:
        dest=OUT/f'TECH_EQ_{freq}_3bp'
        if (dest/'complete.json').exists():verify_sealed(dest);continue
        dates=set(plans[freq])
        def factory(d,held):
            return pd.Series(1/len(pools[d]),index=pools[d]) if d in dates else None
        result=replay(prices,schedule,factory,fee=.0003)
        save_result(dest,result)
        print('Completed technology pool equal weight',freq,flush=True)


def summarize(q,p):
    rows=[];annual=[];navs={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'daily.csv.gz').exists():continue
        daily=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date'])
        navs[folder.name]=daily.post_nav
        for label,start in [('FULL','2012'),('RECENT','2021')]:
            nav=daily.post_nav;x=nav.loc[start:];loc=nav.index.get_loc(x.index[0])
            if loc:x=pd.concat([nav.iloc[loc-1:loc],x])
            dx=daily.loc[start:];e=ex[ex.date.ge(dx.index[0])]
            rows.append(dict(key=folder.name,period=label,**{k:float(v[0]) for k,v in metrics(x).items()},
                average_equity=float((1-dx.cash/dx.post_nav).mean()),
                average_holdings=float(e.holdings.mean()),
                annual_turnover=float(e.two_way_turnover.sum()/((len(x)-1)/252))))
        for year in range(2021,2027):
            nav=daily.post_nav;x=nav.loc[str(year)];pos=nav.index.get_loc(x.index[0]);base=nav.iloc[pos-1] if pos else 1.
            annual.append(dict(key=folder.name,year=year,return_=x.iloc[-1]/base-1))
    references={}
    old=pd.read_csv(ROOT/p['factor_source']/'nav.csv',index_col='date',parse_dates=True)
    references['OLD_TECH_FIXED50']=old['TECH_GROWTH_THEME_MULTI_FACTOR__FIXED__K50']
    bm=pd.read_csv(ROOT/p['factor_source']/'benchmark_nav.csv',index_col='date',parse_dates=True)
    references['OLD_TECH_POOL_EQ']=bm['TECH_GROWTH_THEME_MULTI_FACTOR']
    idx=pd.read_csv(ROOT/'output/rebalance_frequency_20261022/nav.csv',index_col='date',parse_dates=True)
    for k in ['H00300.CSI','H00016.CSI','POOL_EQ_3bp']:references[k]=idx[k]
    for k,nav in references.items():
        nav=nav.reindex(next(iter(navs.values())).index)
        assert nav.notna().all();navs[k]=nav
        for label,start in [('FULL','2012'),('RECENT','2021')]:
            x=nav.loc[start:];pos=nav.index.get_loc(x.index[0])
            if pos:x=pd.concat([nav.iloc[pos-1:pos],x])
            rows.append(dict(key=k,period=label,**{n:float(v[0]) for n,v in metrics(x).items()}))
        for year in range(2021,2027):
            x=nav.loc[str(year)];pos=nav.index.get_loc(x.index[0]);base=nav.iloc[pos-1] if pos else 1.
            annual.append(dict(key=k,year=year,return_=x.iloc[-1]/base-1))
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv')


def holdings(prices,src,q):
    basic=pd.read_csv(ROOT/src['data_source']/'stock_basic.csv.gz',usecols=['ts_code','name']).drop_duplicates('ts_code').set_index('ts_code').name
    membership=pd.read_csv(ROOT/src['data_source']/'index_member_all.csv.gz')
    keys=[f'{a}_{f}_3bp' for a in q['recipes'] for f in q['frequencies']]
    keys += [f'{a}_TECH15_{f}_3bp' for a in q['recipes'] for f in q['frequencies']]
    industries=[];exposures=[];checks=[]
    for key in keys:
        folder=OUT/key
        if not (folder/'daily.csv.gz').exists():continue
        nav=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        dest=OUT/('holdings_'+key);dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():verify_sealed(dest)
        else:
            trades=pd.read_csv(folder/'trades.csv.gz',parse_dates=['date'])
            ledger=attribute(prices,nav,trades)
            ledger=ledger[ledger.date.ge(q['focus_start'])].copy()
            # Returns belong to the industry known at the beginning of the holding day.
            previous=pd.Series(prices.index[:-1],index=prices.index[1:])
            idx=pd.MultiIndex.from_arrays([ledger.date.map(previous),ledger.symbol],names=['date','symbol'])
            ledger['industry']=point_in_time_industry(idx,membership).to_numpy()
            ledger['name']=ledger.symbol.map(basic).fillna('')
            ledger.to_csv(dest/'daily_holdings.csv.gz',index=False)
            for label,start,end in [('RECENT','2021','2026')]+[(str(y),str(y),str(y)) for y in range(2021,2027)]:
                tab=contribution(ledger,nav.post_nav,start,end)
                tab['name']=tab.index.map(basic);tab.to_csv(dest/(label+'_stocks.csv'))
            trades.loc[trades.date.ge(q['focus_start'])].assign(name=lambda x:x.symbol.map(basic)).to_csv(dest/'trades_recent.csv.gz',index=False)
            last=ledger[ledger.date.eq(ledger.date.max()) & ledger.value.gt(1e-10)]
            last.to_csv(dest/'last_holdings.csv',index=False)
            write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='complete.json'})
        ledger=pd.read_csv(dest/'daily_holdings.csv.gz',parse_dates=['date'])
        for label,start,end in [('RECENT','2021','2026')]+[(str(y),str(y),str(y)) for y in range(2021,2027)]:
            dates=nav.loc[start:end].index;pos=nav.index.get_loc(dates[0]);base=nav.post_nav.iloc[pos-1] if pos else 1.
            x=ledger[ledger.date.isin(dates)]
            total=x.net_pnl.sum()/base;expected=nav.post_nav.loc[dates[-1]]/base-1
            assert abs(total-expected)<1e-9
            checks.append(dict(key=key,period=label,attribution_error=abs(total-expected)))
            for ind,g in x.groupby('industry',dropna=False):
                industries.append(dict(key=key,period=label,industry=ind,net_contribution=g.net_pnl.sum()/base,
                    gross_contribution=g.gross_pnl.sum()/base,fees=g.fee.sum()/base,average_weight=g.prior_weight.sum()/len(dates)))
            tech=x[x.industry.isin(q['technology_definition'])]
            exposures.append(dict(key=key,period=label,technology_average_weight=tech.prior_weight.sum()/len(dates),
                technology_contribution=tech.net_pnl.sum()/base,fees=x.fee.sum()/base,
                positive_stock_contribution=x.groupby('symbol').net_pnl.sum().clip(lower=0).sum()/base,
                negative_stock_contribution=x.groupby('symbol').net_pnl.sum().clip(upper=0).sum()/base,
                unique_names=x.loc[x.value.gt(1e-10),'symbol'].nunique()))
        print('Actual holdings reconciled:',key,flush=True)
    pd.DataFrame(industries).to_csv(OUT/'industry_attribution.csv',index=False)
    pd.DataFrame(exposures).to_csv(OUT/'exposure_attribution.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'attribution_checks.csv',index=False)


def run(stage):
    p,q,c,src=freeze();allprices=load_prices(p).loc[:q['end']]
    prices=allprices.loc[q['historical_start']:]
    schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    restore_parents(prices,schedule)
    if stage!='parents':run_tech(p,q,c,allprices,prices,schedule)
    holdings(prices,src,q)
    if stage!='parents':summarize(q,p)
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    if stage!='parents':
        write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='completed.json'})
    print('Done; forward accounts unchanged.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['parents','all'],default='all')
    run(ap.parse_args().stage)
