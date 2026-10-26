#!/usr/bin/env python3
"""Four diagnostic audits on the frozen fixed15-risk strategy; no live changes."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.factor_diagnosis import horizon_returns,cross_section_diagnostic,hac_mean,remove_family,compact_replay
from analysis.single_account_adaptive import prior_weights,update_ic_weights,family_score,history_risk,capped_positive_weights
from analysis.holdings_count import metrics,replay_batch
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import sha,write_json
from strategies.holding_buffer import ordered_scores

PROTOCOL=ROOT/'research_protocols/20261021_factor_diagnosis_v1.json'
OUT=ROOT/'output/factor_diagnosis_20261021'
PERIODS={'FULL':('2012','2026'),'EARLY':('2012','2015'),'MID':('2016','2020'),'RECENT':('2021','2026')}


def freeze(out):
    q=json.loads(PROTOCOL.read_text());parent=ROOT/q['parent_output']
    parent_seal=json.loads((parent/'frozen.json').read_text())
    for name,digest in parent_seal['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'parent input changed: {name}')
    for name,digest in json.loads((parent/'completed.json').read_text()).items():
        if sha(parent/name)!=digest:raise ValueError(f'parent result changed: {name}')
    p=json.loads((ROOT/q['parent_protocol']).read_text())
    files=dict(parent_seal['files'])
    for f in [PROTOCOL,Path(__file__),ROOT/'analysis/factor_diagnosis.py',
              ROOT/'analysis/single_account_adaptive.py',ROOT/'analysis/ic_weight_comparison.py',
              parent/'completed.json',ROOT/p['factor_source']/'benchmark_members.csv.gz']:
        files[str(f.relative_to(ROOT))]=sha(f)
    seal=dict(protocol=sha(PROTOCOL),files=files)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists() and json.loads((out/'frozen.json').read_text())!=seal:
        raise ValueError('frozen diagnostic changed; choose a new output directory')
    write_json(out/'frozen.json',seal);write_json(out/'protocol.json',q)
    if not (out/'forward_before.json').exists():write_json(out/'forward_before.json',account_hashes())
    return p,q,parent


def prepare(p,q,out,prices):
    source=ROOT/p['factor_source']
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
    prior=prior_weights(families);factors=list(prior.index)
    cache=pd.concat([pd.read_csv(d/'BASE_V2_MULTI_FACTOR_ic.csv',parse_dates=['date','outcome_end'])
                     for d in sorted(source.glob('inputs_*'))]).set_index('date').sort_index()
    ic=cache[factors];ends=cache.outcome_end
    signals=pd.DatetimeIndex(json.loads((source/'calendar.json').read_text())['signals'])
    signals=signals[(signals>=q['evaluation_start'])&(signals<=q['evaluation_end'])]
    next_signal=dict(zip(signals,list(signals[1:])+[None]))
    monthly=pd.read_csv(ROOT/q['parent_output']/'factor_weights.csv',parse_dates=['date'])
    original_weights=monthly.pivot(index='date',columns='factor',values='weight')
    weights={252:prior.copy(),126:prior.copy()};month=None
    logs=[];lag=[];diagnostics=[];redundancy=[];exposures=[];groups={};common_pools={}
    variants=['BASE_RISK','WINDOW126']+[f'DROP_{f}' for f in families]
    for folder in sorted(source.glob('inputs_*')):
        panel=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date'])
        for date,rows in panel.groupby('date',sort=True):
            x=rows.set_index(['date','symbol'])
            if date.to_period('M')!=month:
                month=date.to_period('M')
                for window in [252,126]:
                    spec={**p['ic'],'lookback':window}
                    weights[window],records=update_ic_weights(ic,ends,date,weights[window],spec)
                    if window==252:
                        np.testing.assert_allclose(weights[window],original_weights.loc[date,factors],rtol=1e-12,atol=1e-12)
                    known=ic.loc[(ic.index<date)&ends.lt(date)].tail(window)
                    positive=known.mean().where(known.count()>=spec['minimum'],0).fillna(0).clip(lower=0)
                    desired=capped_positive_weights(positive,spec['factor_cap']) if positive.gt(0).sum()*spec['factor_cap']>=1-1e-12 else prior*np.nan
                    next_month=month+1
                    future=ic.loc[ic.index.to_period('M')==next_month]
                    future_ends=ends.reindex(future.index)
                    complete=(len(future)>=10 and next_month.end_time.normalize()<=prices.index[-1]
                              and future_ends.notna().all() and future_ends.le(prices.index[-1]).all())
                    for rec in records:
                        rec.update(window=window,desired_capped_weight=desired[rec['factor']])
                        logs.append(rec)
                        lag.append({**rec,'evaluation_month':str(next_month),'future_month_complete':complete,
                            'next_month_ic':float(future[rec['factor']].mean()) if complete and future[rec['factor']].count()>=10 else np.nan})
            if date not in next_signal:continue
            fixed=family_score(x,families,prior).droplevel(0)
            common_pools[date]=fixed.index
            industry=rows.set_index('symbol').industry.fillna('UNKNOWN')
            scores={}
            for variant in variants:
                w=weights[126] if variant=='WINDOW126' else weights[252]
                if variant.startswith('DROP_'):w=remove_family(w,families[variant.removeprefix('DROP_')])
                score=family_score(x,families,w,require_all=False).droplevel(0).reindex(fixed.index).dropna()
                if len(score)!=len(fixed):raise ValueError('variant changed common score coverage')
                scores[variant]=ordered_scores(score)
            groups[date]=dict(scores=scores,industry=industry,fixed=ordered_scores(fixed))
            panel_day=x[factors].droplevel(0).reindex(fixed.index)
            analysis_scores=panel_day.assign(COMPOSITE=scores['BASE_RISK'])
            y,end_dates=horizon_returns(prices,date,next_signal[date],q['horizons'])
            common_y=y.notna().all(axis=1)
            for factor in analysis_scores:
                s=analysis_scores[factor]
                for horizon in y:
                    for sample in ['native','common_all_horizons']:
                        if sample=='common_all_horizons' and any(pd.isna(e) for e in end_dates.values()):continue
                        if pd.isna(end_dates[horizon]):continue
                        mask=common_y if sample=='common_all_horizons' else pd.Series(True,index=y.index)
                        row=cross_section_diagnostic(s,y[horizon],mask)
                        if row is not None:
                            diagnostics.append(dict(date=date,factor=factor,horizon=horizon,sample=sample,outcome_end=end_dates[horizon],**row))
            corr=panel_day.corr(method='spearman',min_periods=30)
            ranked={c:ordered_scores(panel_day[c].dropna()) for c in factors}
            top={c:set(ranked[c].head(15).index) for c in factors}
            top10={c:set(ranked[c].head(max(1,int(np.ceil(len(ranked[c])*.1)))).index) for c in factors}
            for a,b in itertools.combinations(factors,2):
                redundancy.append(dict(date=date,factor_a=a,factor_b=b,correlation=corr.loc[a,b],
                    top15_overlap=len(top[a]&top[b])/15,
                    top_decile_jaccard=len(top10[a]&top10[b])/max(1,len(top10[a]|top10[b]))))
            for c in factors:
                names=list(top[c]);sector=industry.reindex(names).value_counts(normalize=True)
                exposure=dict(date=date,factor=c,top_industry=sector.index[0] if len(sector) else 'UNKNOWN',
                              largest_industry_weight=sector.iloc[0] if len(sector) else np.nan,
                              theme_weight=industry.reindex(names).isin(p['selection']['theme_industries']).mean())
                exposure.update({f'exposure_{f}':panel_day.loc[names,f].mean() for f in factors})
                exposures.append(exposure)
        print(f"Diagnostics {folder.name}: {len(groups)} weeks, {len(diagnostics)} horizon rows",flush=True)
    for name,records in [('weights',logs),('lag_prediction',lag),('horizon_weekly',diagnostics),('redundancy_weekly',redundancy),('factor_exposures',exposures)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv.gz',index=False)
    # Preserve original membership separately from common complete-family score eligibility.
    members=[]
    for chunk in pd.read_csv(source/'benchmark_members.csv.gz',chunksize=500000,parse_dates=['date']):
        members.append(chunk[chunk.strategy.eq('BASE_V2_MULTI_FACTOR')&chunk.date.isin(groups)])
    original_pools={d:g.symbol.tolist() for d,g in pd.concat(members).groupby('date')}
    assert set(original_pools)==set(groups)
    pd.DataFrame([dict(date=d,original_pool=len(original_pools[d]),common_pool=len(common_pools[d])) for d in groups]).to_csv(out/'pool_coverage.csv',index=False)
    return groups,common_pools,original_pools,variants,prices.pct_change(fill_method=None)


class LazyRisk:
    def __init__(self,groups,variant,returns,p):self.groups,self.variant,self.returns,self.p=groups,variant,returns,p
    def __getitem__(self,date):
        return history_risk(self.returns,date,self.groups[date]['scores'][self.variant].head(100).index,self.p['selection'])


def factory_for(arm,groups,common,original,returns,p,risk_log=None):
    account=None;selections=[]
    variant=arm if arm in ['BASE_RISK','WINDOW126'] or arm.startswith('DROP_') else 'BASE_RISK'
    if arm in ['BASE_RISK','WINDOW126','CONSTRAINED15'] or arm.startswith('DROP_'):
        adapted={d:dict(ic=g['scores'][variant],industry=g['industry'],fixed=g['fixed']) for d,g in groups.items()}
        account=Account('IC_CONSTRAINED15' if arm=='CONSTRAINED15' else 'IC_CONSTRAINED15_RISK',
                        adapted,LazyRisk(groups,variant,returns,p),returns,p)
    state={'names':None}
    def factory(date,held):
        if account is not None:
            target=account(date,held)
        elif arm=='COMMON_EQ_MATCHED':
            if date in groups:state['names']=common[date]
            r=risk_log.loc[date]
            target=pd.Series(r.target_equity/len(state['names']),index=state['names']) if r.action else None
        elif date not in groups:
            target=None
        else:
            score=groups[date]['scores']['BASE_RISK']
            if arm=='RAW_TOP15':names=score.head(15).index
            elif arm=='TOP_DECILE':names=score.head(int(np.ceil(len(score)/10))).index
            elif arm=='COMMON_EQ':names=common[date]
            elif arm=='ORIGINAL_EQ':names=original[date]
            else:raise ValueError(arm)
            target=pd.Series(1/len(names),index=names)
        if target is not None and len(target)<=20:
            selections.extend(dict(date=date,symbol=s,weight=w,weekly=date in groups) for s,w in target.items())
        return target
    return factory,account,selections


def replay_all(p,q,parent,out,prices,groups,common,original,variants,returns):
    prices=prices.loc[min(groups):];dates=prices.index
    schedule=pd.DataFrame({'execution_date':dates[1:]},index=dates[:-1])
    arms=['BASE_RISK','RAW_TOP15','TOP_DECILE','CONSTRAINED15','ORIGINAL_EQ','COMMON_EQ','COMMON_EQ_MATCHED','WINDOW126']+[v for v in variants if v.startswith('DROP_')]
    summary=[];subperiods=[];navs={};parities=[]
    for fee in q['fees']:
        baseline_log=None
        for arm in arms:
            factory,account,selections=factory_for(arm,groups,common,original,returns,p,baseline_log)
            result=compact_replay(prices,schedule,factory,fee)
            key=f'{arm}_{int(round(fee*10000))}bp';folder=out/key;folder.mkdir(exist_ok=True)
            for name,frame in result.items():frame.to_csv(folder/f'{name}.csv.gz',index=name=='daily')
            if account is not None:
                logs=pd.DataFrame(account.logs).set_index('date');logs.to_csv(folder/'decisions.csv')
                if arm=='BASE_RISK':baseline_log=logs
            if selections:pd.DataFrame(selections).to_csv(folder/'selections.csv.gz',index=False)
            daily=result['daily'];ex=result['executions'];nav=daily.post_nav;navs[key]=nav
            if arm=='BASE_RISK' and fee>0:
                prior=parent/f'IC_CONSTRAINED15_RISK_{int(round(fee*10000))}bp'
                old=pd.read_csv(prior/'daily.csv.gz',parse_dates=['date']).set_index('date')
                pd.testing.assert_frame_equal(daily,old,rtol=1e-11,atol=1e-11,check_freq=False)
                old_ex=pd.read_csv(prior/'executions.csv.gz',parse_dates=['date','signal_date'])
                pd.testing.assert_frame_equal(ex[old_ex.columns],old_ex,rtol=1e-11,atol=1e-11,check_dtype=False)
                parities.append(dict(fee=fee,max_nav_error=float((nav-old.post_nav).abs().max()),execution_match=True))
            row=dict(arm=arm,fee=fee,**{k:float(v[0]) for k,v in metrics(nav).items()},
                annual_two_way_turnover=ex.two_way_turnover.sum()/((len(nav)-1)/252),
                average_equity=(1-daily.cash/daily.post_nav).mean(),
                blocked_exit_events=int(ex.blocked_exit_value.gt(1e-10).sum()))
            summary.append(row)
            for period,(start,stop) in PERIODS.items():
                s=nav.loc[start:stop];pos=nav.index.get_loc(s.index[0])
                if pos:s=pd.concat([nav.iloc[pos-1:pos],s])
                subperiods.append(dict(arm=arm,fee=fee,period=period,**{k:float(v[0]) for k,v in metrics(s).items()}))
            print(f"{key}: CAGR {row['cagr']:.2%}, DD {row['max_drawdown']:.2%}",flush=True)
    pd.DataFrame(summary).to_csv(out/'performance.csv',index=False)
    pd.DataFrame(subperiods).to_csv(out/'subperiods.csv',index=False)
    pd.DataFrame(navs).to_csv(out/'nav.csv')
    pd.DataFrame(parities).to_csv(out/'parent_parity.csv',index=False)
    # Fixed ex-ante random plans, not a search for a lucky seed or best K.
    spec=q['random_top15'];rng=np.random.default_rng(spec['seed']);signals=sorted(groups)
    plans=np.stack([np.stack([rng.choice(prices.columns.get_indexer(common[d]),15,replace=False)
                             for _ in range(spec['replicates'])]) for d in signals])
    np.savez_compressed(out/'random_plans.npz',plans=plans,signals=np.array(signals,dtype='datetime64[ns]'))
    random_rows=[]
    for fee in spec['fees']:
        ex=[dates.get_loc(d)+1 for d in signals]
        nav,_,_=replay_batch(prices.to_numpy(),ex,plans,fee)
        pd.DataFrame(nav,index=dates).to_csv(out/f'random_nav_{int(round(fee*10000))}bp.csv.gz')
        for period,(start,stop) in PERIODS.items():
            loc=np.flatnonzero((dates.year>=int(start))&(dates.year<=int(stop)))
            vals=nav[max(0,loc[0]-1):loc[-1]+1]
            m=metrics(vals)
            for i in range(spec['replicates']):random_rows.append(dict(fee=fee,period=period,replicate=i,**{k:float(v[i]) for k,v in m.items()}))
    pd.DataFrame(random_rows).to_csv(out/'random_performance.csv',index=False)


def summarize_diagnostics(out):
    h=pd.read_csv(out/'horizon_weekly.csv.gz',parse_dates=['date'])
    rows=[]
    for period,(start,stop) in PERIODS.items():
        sub=h[h.date.dt.year.between(int(start),int(stop))]
        for key,g in sub.groupby(['factor','horizon','sample']):
            row=dict(period=period,factor=key[0],horizon=key[1],sample=key[2],weeks=len(g),
                     coverage=g.coverage.mean(),top15_coverage=g.top15_coverage.mean())
            for metric in ['ic','high_minus_pool','top15_minus_pool','top15_minus_decile','high_minus_low']:
                row.update({f'{metric}_{k}':v for k,v in hac_mean(g[metric]).items()})
            row.update({f'q{i}':g[f'q{i}'].mean() for i in range(1,11)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'horizon_summary.csv',index=False)
    # Paired horizon differences use exactly the same signal dates and common evaluable stocks.
    diffs=[]
    c=h[h['sample'].eq('common_all_horizons')]
    for period,(start,stop) in PERIODS.items():
        sub=c[c.date.dt.year.between(int(start),int(stop))]
        for factor,g in sub.groupby('factor'):
            x=g.pivot(index='date',columns='horizon',values='ic')
            for horizon in ['20','60']:
                diffs.append(dict(period=period,factor=factor,long_horizon=horizon,**hac_mean(x[horizon]-x['5'])))
    pd.DataFrame(diffs).to_csv(out/'horizon_paired_difference.csv',index=False)
    red=pd.read_csv(out/'redundancy_weekly.csv.gz',parse_dates=['date']);rrows=[]
    for period,(start,stop) in PERIODS.items():
        g=red[red.date.dt.year.between(int(start),int(stop))]
        s=g.groupby(['factor_a','factor_b'])[['correlation','top15_overlap','top_decile_jaccard']].mean().reset_index()
        s['period']=period;rrows.append(s)
    pd.concat(rrows).to_csv(out/'redundancy_summary.csv',index=False)
    lag=pd.read_csv(out/'lag_prediction.csv.gz',parse_dates=['date']);lrows=[];monthly=[]
    for (date,window),g in lag.groupby(['date','window']):
        g=g.dropna(subset=['next_month_ic','mean_rank_ic'])
        if len(g)<5:continue
        weight=g.weight/g.weight.sum()
        monthly.append(dict(date=date,window=window,factors=len(g),
            past_next_cross_factor_rank_corr=g.mean_rank_ic.corr(g.next_month_ic,method='spearman'),
            weighted_next_ic=(weight*g.next_month_ic).sum(),equal_next_ic=g.next_month_ic.mean(),
            weighting_advantage=(weight*g.next_month_ic).sum()-g.next_month_ic.mean(),
            delta_weight_next_ic=((g.weight-g.previous_weight)*g.next_month_ic).sum(),
            smoothing_gap=(g.desired_capped_weight-g.weight).abs().sum(min_count=1),
            fallback=g.reason.iloc[0]!='positive_rank_ic'))
    m=pd.DataFrame(monthly);m.to_csv(out/'lag_monthly.csv',index=False)
    for period,(start,stop) in PERIODS.items():
        s=lag[lag.date.dt.year.between(int(start),int(stop))]
        for (window,factor),g in s.groupby(['window','factor']):
            g=g.dropna(subset=['next_month_ic','mean_rank_ic'])
            lrows.append(dict(period=period,window=window,factor=factor,months=len(g),
                past_next_time_rank_corr=g.mean_rank_ic.corr(g.next_month_ic,method='spearman') if len(g)>5 else np.nan,
                past_mean_ic=g.mean_rank_ic.mean(),next_mean_ic=g.next_month_ic.mean()))
    pd.DataFrame(lrows).to_csv(out/'lag_factor_summary.csv',index=False)
    nav=pd.read_csv(out/'nav.csv',index_col='date',parse_dates=True);attr=[]
    chain=['ORIGINAL_EQ','COMMON_EQ','TOP_DECILE','RAW_TOP15','CONSTRAINED15','BASE_RISK']
    for period,(start,stop) in PERIODS.items():
        s=nav.loc[start:stop];pos=nav.index.get_loc(s.index[0])
        if pos:s=pd.concat([nav.iloc[pos-1:pos],s])
        lograte=np.log(s.iloc[-1]/s.iloc[0])*252/(len(s)-1)
        for fee in [0,3,13]:
            for a,b in zip(chain,chain[1:]):
                attr.append(dict(period=period,fee_bp=fee,from_arm=a,to_arm=b,annual_log_return_difference=lograte[f'{b}_{fee}bp']-lograte[f'{a}_{fee}bp']))
            a,b='COMMON_EQ_MATCHED','BASE_RISK'
            attr.append(dict(period=period,fee_bp=fee,from_arm=a,to_arm=b,annual_log_return_difference=lograte[f'{b}_{fee}bp']-lograte[f'{a}_{fee}bp']))
        for arm in chain+['COMMON_EQ_MATCHED']:
            for fee in [3,13]:
                attr.append(dict(period=period,fee_bp=fee,from_arm=arm+'_0bp',to_arm=arm,
                    annual_log_return_difference=lograte[f'{arm}_{fee}bp']-lograte[f'{arm}_0bp']))
    pd.DataFrame(attr).to_csv(out/'attribution_log_returns.csv',index=False)


def run(out=OUT):
    p,q,parent=freeze(out)
    prices=load_prices(p).loc[:q['evaluation_end']]
    groups,common,original,variants,returns=prepare(p,q,out,prices)
    replay_all(p,q,parent,out,prices,groups,common,original,variants,returns)
    summarize_diagnostics(out)
    assert account_hashes()==json.loads((out/'forward_before.json').read_text())
    w=pd.read_csv(out/'weights.csv.gz',parse_dates=['date','latest_outcome_end'])
    assert w.latest_outcome_end.dropna().lt(w.loc[w.latest_outcome_end.notna(),'date']).all()
    assert w.weight.between(-1e-12,.2+1e-12).all()
    assert np.allclose(w.groupby(['date','window']).weight.sum(),1)
    validation=dict(status='PASS',weekly_signals=len(groups),mature_label_weights=True,
                    parent_base_risk_reproduced=True,forward_unchanged=True,
                    no_automatic_admission_or_live_change=True)
    write_json(out/'validation.json',validation)
    write_json(out/'completed.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='completed.json'})
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    run(parser.parse_args().output)
