#!/usr/bin/env python3
"""Checkpointed fixed15 factor extensions, isolated from forward accounts."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.candidate_factors import CANDIDATES,FIELDS,normalize_annual,annual_candidates,dividend_asof,momentum_12_1,standardize_candidates
from analysis.ic_weight_comparison import execution_labels,daily_rank_ic
from analysis.single_account_adaptive import prior_weights,update_ic_weights,family_score,history_risk
from analysis.rebalance_frequency import schedules,asof_weights,classify_executions
from analysis.factor_diagnosis import compact_replay
from analysis.holdings_count import eligible_base,metrics
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import sha,write_json,load_matrix
from scripts.run_rebalance_frequency import period_metrics,verify_sealed
from universe.risk import build_hard_risk_universe_report
from universe.dynamic import eligibility_from_report
from universe.strategy import point_in_time_industry
from strategies.holding_buffer import ordered_scores

PROTOCOL=ROOT/'research_protocols/20261022_candidate_factors_v1.json'
OUT=ROOT/'output/candidate_factors_20261022'

def freeze():
    q=json.loads(PROTOCOL.read_text());p=json.loads((ROOT/q['parent_protocol']).read_text())
    src=json.loads((ROOT/q['source_protocol']).read_text());parent=ROOT/q['parent_output']
    files=dict(json.loads((parent/'frozen.json').read_text())['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed inherited input '+name)
    for name,digest in json.loads((parent/'completed.json').read_text()).items():
        if sha(parent/name)!=digest:raise ValueError('Changed parent output '+name)
    rawseal=json.loads((ROOT/p['prior_replay']/'started.json').read_text())['raw']
    for name in ['income_vip.csv.gz','cashflow_vip.csv.gz','daily_basic.csv.gz']:
        path=ROOT/src['data_source']/name;digest=sha(path)
        if digest!=rawseal[name]:raise ValueError('Changed raw candidate source '+name)
        files[str(path.relative_to(ROOT))]=digest
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        seal=json.loads((folder/'complete.json').read_text())
        for name in ['BASE_V2_MULTI_FACTOR_weekly.csv.gz','BASE_V2_MULTI_FACTOR_ic.csv']:
            digest=sha(folder/name)
            if digest!=seal[name]:raise ValueError('Changed factor cache')
            files[str((folder/name).relative_to(ROOT))]=digest
    for path in [PROTOCOL,Path(__file__),ROOT/'analysis/candidate_factors.py',parent/'completed.json',
                 ROOT/'analysis/ic_weight_comparison.py',ROOT/'analysis/financial_asof.py']:
        files[str(path.relative_to(ROOT))]=sha(path)
    stamp={'files':files,'protocol':sha(PROTOCOL)};OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text())!=stamp:raise ValueError('Frozen inputs changed')
    if (OUT/'completed.json').exists():raise ValueError('Completed experiment is immutable')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():write_json(OUT/'forward_before.json',account_hashes())
    return p,q,src

def prepare(p,q,src,prices):
    source=ROOT/src['data_source'];dates=prices.index[prices.index>=pd.Timestamp('2011-01-01')]
    quarters=dates.to_period('Q').unique()
    if all((OUT/f'inputs_{x}'/'complete.json').exists() for x in quarters):
        for x in quarters:verify_sealed(OUT/f'inputs_{x}')
        return
    print('Loading PIT candidate inputs and rebuilding daily eligibility',flush=True)
    matrices=load_matrix(source/'daily.csv.gz',['close','amount'],prices.index,prices.columns,{'amount':1000.})
    basic=pd.read_csv(source/'stock_basic.csv.gz',dtype=str);basic=basic[basic.ts_code.isin(prices.columns)]
    names=pd.read_csv(source/'namechange.csv.gz')
    base=eligible_base(matrices['close'],matrices['amount'],basic,names,dates)
    del matrices
    base['quarter']=base.date.dt.to_period('Q').astype(str)
    tables={}
    for kind,file in [('income','income_vip'),('balance','balancesheet_vip'),('cashflow','cashflow_vip')]:
        cols=['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type']+FIELDS[kind]
        tables[kind]=normalize_annual(pd.read_csv(source/(file+'.csv.gz'),usecols=cols),kind)
    balance=pd.read_csv(source/'balancesheet_vip.csv.gz',usecols=['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit=pd.read_csv(source/'fina_audit.csv.gz');industry=pd.read_csv(source/'index_member_all.csv.gz')
    dividend=pd.read_csv(source/'daily_basic.csv.gz',usecols=['ts_code','trade_date','dv_ttm'])
    momentum=momentum_12_1(prices);labels,ends=execution_labels(prices,5)
    calendar=json.loads((ROOT/p['factor_source']/'calendar.json').read_text())
    weeks=pd.DatetimeIndex(calendar['signals'])
    plans=schedules(prices.index,weeks,q['evaluation_start'],q['evaluation_end'])
    saved_dates=weeks.union(plans['MONTHLY'])
    write_json(OUT/'schedules.json',{k:[str(d.date()) for d in v] for k,v in plans.items() if k in q['frequencies']})
    for quarter,b in base.groupby('quarter',sort=True):
        folder=OUT/f'inputs_{quarter}'
        if (folder/'complete.json').exists():verify_sealed(folder);continue
        folder.mkdir(exist_ok=True)
        risk=build_hard_risk_universe_report(b.drop(columns='quarter'),balance,audit,names)
        mask=eligibility_from_report(risk);idx=mask.index[mask]
        chunks=[]
        for date,g in idx.to_frame(index=False).groupby('date',sort=True):
            a=annual_candidates(tables,date).reindex(g.symbol)
            a.index=pd.MultiIndex.from_arrays([[date]*len(a),g.symbol],names=['date','symbol'])
            chunks.append(a)
        raw=pd.concat(chunks).sort_index()
        raw=raw.join(dividend_asof(raw.index,dividend))
        raw['MOMENTUM_12_1']=momentum.loc[raw.index.get_level_values(0).unique()].stack().reindex(raw.index)
        ind=point_in_time_industry(raw.index,industry);standard=standardize_candidates(raw,ind)
        target=labels.loc[standard.index.get_level_values(0).unique()].stack()
        target.index=target.index.set_names(['date','symbol'])
        ic,count=daily_rank_ic(standard,target)
        ic['outcome_end']=ends.reindex(ic.index)
        ic.to_csv(folder/'ic.csv');count.to_csv(folder/'ic_names.csv')
        take=raw.index.get_level_values(0).isin(saved_dates)
        standard.loc[take].to_csv(folder/'standard.csv.gz');raw.loc[take].to_csv(folder/'raw.csv.gz')
        cov=raw[CANDIDATES].notna().groupby(level=0).mean();cov['pool']=raw.groupby(level=0).size();cov.to_csv(folder/'coverage.csv')
        check=raw.reset_index()
        assert (check.latest_disclosure.isna()|check.latest_disclosure.lt(check.date)).all()
        assert (check.dividend_snapshot.isna()|check.dividend_snapshot.le(check.date)).all()
        write_json(folder/'audit.json',{'rows':len(raw),'days':raw.index.get_level_values(0).nunique(),
            'financial_disclosures_strictly_prior':True,'dividend_max_age_days':7,'min_raw_coverage':cov[CANDIDATES].min().to_dict()})
        write_json(folder/'complete.json',{f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name!='complete.json'})
        print(f'Prepared {quarter}: {len(raw)} candidate rows',flush=True)

def make_weights(p,q):
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families'];prior=prior_weights(families)
    old=pd.concat([pd.read_csv(f/'BASE_V2_MULTI_FACTOR_ic.csv',parse_dates=['date','outcome_end']) for f in sorted((ROOT/p['factor_source']).glob('inputs_*'))]).set_index('date').sort_index()
    new=pd.concat([pd.read_csv(f/'ic.csv',parse_dates=['date','outcome_end']) for f in sorted(OUT.glob('inputs_*'))]).set_index('date').sort_index()
    assert not old.index.has_duplicates and not new.index.has_duplicates
    shared=old.index.intersection(new.index)
    assert (old.loc[shared,'outcome_end'].fillna(pd.Timestamp('1900'))==new.loc[shared,'outcome_end'].fillna(pd.Timestamp('1900'))).all()
    all_ic=old[prior.index].join(new[CANDIDATES]);end=old.outcome_end
    # Parent had no eligible panel on2011-01-07: use its actual update calendar,
    # not a theoretical weekly date before the first available cross section.
    expected=pd.read_csv(ROOT/'output/single_account_fixed15_risk_20261021/factor_weights.csv',parse_dates=['date']).pivot(index='date',columns='factor',values='weight')
    update_dates=expected.index
    recipes={};weight_frames={};logs=[]
    for arm in q['arms']:
        if arm.startswith('COMMON_'):continue
        added=[] if arm=='BASE' else CANDIDATES if arm=='ALL5' else [arm]
        recipe={k:list(v) for k,v in families.items()}
        for c in added:recipe[q['candidate_families'][c]].append(c)
        cols=list(prior.index)+added;w=prior.reindex(cols,fill_value=0)
        records=[]
        for d in update_dates:
            w,log=update_ic_weights(all_ic[cols],end,d,w,p['ic']);records.extend(log)
        f=pd.DataFrame(records);f['arm']=arm;logs.append(f)
        weight_frames[arm]=f.pivot(index='date',columns='factor',values='weight');recipes[arm]=recipe
    all_logs=pd.concat(logs,ignore_index=True);all_logs.to_csv(OUT/'factor_weights.csv',index=False)
    all_ic.assign(outcome_end=end).to_csv(OUT/'daily_ic.csv')
    pd.testing.assert_frame_equal(weight_frames['BASE'],expected,check_dtype=False,atol=1e-12,rtol=1e-12)
    return families,recipes,weight_frames

def prepare_scores(p,q,prices):
    families,recipes,weights=make_weights(p,q);prior=prior_weights(families)
    parent=ROOT/q['parent_output'];plans={k:pd.DatetimeIndex(v) for k,v in json.loads((OUT/'schedules.json').read_text()).items()}
    needed=plans['WEEKLY'].union(plans['MONTHLY']);groups={k:{} for k in q['arms']};coverage=[];ranks=[];redundancy=[]
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        quarter=folder.name.removeprefix('inputs_')
        x=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        mf=parent/f'month_inputs_{quarter}'/'panel.csv.gz'
        if mf.exists():
            m=pd.read_csv(mf,parse_dates=['date']).set_index(['date','symbol']);m=m.loc[~m.index.get_level_values(0).isin(x.index.get_level_values(0).unique())];x=pd.concat([x,m])
        x=x.loc[x.index.get_level_values(0).isin(needed)]
        if x.empty:continue
        new=pd.read_csv(OUT/f'inputs_{quarter}'/'standard.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        raw=pd.read_csv(OUT/f'inputs_{quarter}'/'raw.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        for d,g in x.groupby(level=0):
            fixed=family_score(g,families,prior).droplevel(0);pool=fixed.index
            newg=new.reindex(g.index)[CANDIDATES]
            rawg=raw.reindex(g.index)[CANDIDATES]
            valid=rawg.notna().all(axis=1).droplevel(0).reindex(pool).fillna(False)
            merged=g.join(newg.fillna(0));ind=g.industry.droplevel(0)
            coverage.append(dict(date=d,pool=len(pool),complete=int(valid.sum()),**{c:float(rawg[c].droplevel(0).reindex(pool).notna().mean()) for c in CANDIDATES}))
            corr=g[list(prior.index)].join(newg).corr(method='spearman')
            for c in CANDIDATES:
                for old in prior.index:redundancy.append(dict(date=d,candidate=c,old_factor=old,correlation=corr.loc[c,old]))
            for arm in q['arms']:
                source=arm.removeprefix('COMMON_')
                w,updated=asof_weights(weights[source],d)
                score=family_score(merged,recipes[source],w,require_all=False).droplevel(0).reindex(pool)
                if score.isna().any():raise ValueError('Missing adaptive score in original comparison pool')
                if arm.startswith('COMMON_'):score=score.loc[valid]
                score=ordered_scores(score)
                if len(score)<100 and not arm.startswith('COMMON_'):raise ValueError('Fewer than100 primary candidates')
                groups[arm][d]=dict(ic=score.head(100),fixed=score.head(100),industry=ind)
                for rank,(symbol,value) in enumerate(score.head(100).items(),1):ranks.append(dict(arm=arm,date=d,symbol=symbol,score=value,rank=rank,weight_update=updated))
        print('Scored',quarter,flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(ranks).to_csv(OUT/'rankings.csv.gz',index=False)
    pd.DataFrame(redundancy).to_csv(OUT/'redundancy.csv.gz',index=False)
    return groups,plans

def run(stage):
    p,q,src=freeze();prices_all=load_prices(p).loc[:q['evaluation_end']]
    if stage in ['all','prepare']:prepare(p,q,src,prices_all)
    if stage=='prepare':return
    groups,plans=prepare_scores(p,q,prices_all)
    returns=prices_all.pct_change(fill_method=None);prices=prices_all.loc[q['evaluation_start']:];dates=prices.index
    schedule=pd.DataFrame({'execution_date':dates[1:]},index=dates[:-1]);rows=[];periods=[];navs={};annual=[];parity=[]
    for arm in q['arms']:
        cache={d:(history_risk(returns,d,g['ic'].index,p['selection']) if len(g['ic']) else (pd.DataFrame(),pd.DataFrame())) for d,g in groups[arm].items()}
        for frequency in q['frequencies']:
            selected_groups={d:groups[arm][d] for d in plans[frequency]}
            for fee in q['fees']:
                key=f'{arm}_{frequency}_{int(round(fee*10000))}bp';dest=OUT/key;dest.mkdir(exist_ok=True)
                if (dest/'complete.json').exists():verify_sealed(dest)
                else:
                    account=Account('IC_CONSTRAINED15_RISK',selected_groups,cache,returns,p);targets=[]
                    def factory(date,held):
                        target=account(date,held)
                        if target is not None:targets.extend(dict(date=date,symbol=s,weight=float(w)) for s,w in target.items())
                        return target
                    result=compact_replay(prices,schedule,factory,fee);log=pd.DataFrame(account.logs)
                    result['daily'].to_csv(dest/'daily.csv.gz')
                    classify_executions(result['executions'],log,dates[0]).to_csv(dest/'executions.csv.gz',index=False)
                    log.rename(columns={'weekly':'scheduled_reselection'}).to_csv(dest/'decisions.csv.gz',index=False)
                    pd.DataFrame(targets).to_csv(dest/'targets.csv.gz',index=False)
                    write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='complete.json'})
                daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True);ex=pd.read_csv(dest/'executions.csv.gz',parse_dates=['date','signal_date']);nav=daily.post_nav
                if arm=='BASE':
                    old=ROOT/q['parent_output']/f'{frequency}_{int(round(fee*10000))}bp'
                    olddaily=pd.read_csv(old/'daily.csv.gz',index_col='date',parse_dates=True);oldex=pd.read_csv(old/'executions.csv.gz',parse_dates=['date','signal_date'])
                    pd.testing.assert_frame_equal(daily,olddaily,check_dtype=False,atol=1e-10,rtol=1e-10)
                    pd.testing.assert_frame_equal(ex,oldex,check_dtype=False,atol=1e-10,rtol=1e-10)
                    parity.append(dict(frequency=frequency,fee=fee,max_nav_error=float((nav-olddaily.post_nav).abs().max()),daily_execution_parity=True))
                years=(len(nav)-1)/252
                row=dict(arm=arm,frequency=frequency,fee=fee,**{k:float(v[0]) for k,v in metrics(nav).items()},
                    annual_turnover=ex.two_way_turnover.sum()/years,average_equity=(1-daily.cash/nav).mean(),
                    actual_trade_days=int(ex.trade_legs.gt(0).sum()),risk_only_days=int((ex.trigger.eq('RISK_ONLY')&ex.trade_legs.gt(0)).sum()),
                    terminal_unquoted_fraction=daily.unquoted_value.iloc[-1]/nav.iloc[-1],
                    empty_signal_pools=sum(len(g['ic'])==0 for g in selected_groups.values()))
                rows.append(row);navs[key]=nav
                for r in period_metrics(nav,arm,fee):r['frequency']=frequency;periods.append(r)
                for year,g in nav.groupby(nav.index.year):
                    loc=nav.index.get_loc(g.index[0]);annual.append(dict(arm=arm,frequency=frequency,fee=fee,year=year,return_=g.iloc[-1]/nav.iloc[max(0,loc-1)]-1))
                print(f'{key}: CAGR{row["cagr"]:.2%}, DD{row["max_drawdown"]:.2%}',flush=True)
    oldnav=pd.read_csv(ROOT/q['parent_output']/'nav.csv',index_col='date',parse_dates=True)
    for c in ['H00300.CSI','H00016.CSI','POOL_EQ_3bp']:navs[c]=oldnav[c]
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(periods).to_csv(OUT/'subperiods.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False);pd.DataFrame(navs).to_csv(OUT/'nav.csv');pd.DataFrame(parity).to_csv(OUT/'parent_parity.csv',index=False)
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    write_json(OUT/'validation.json',{'status':'PASS','parent_replay_parity':len(parity),'forward_unchanged':True,'arms':len(rows)})
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['completed.json','independent_validation.json']})
    print('CANDIDATE FACTOR EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','backtest'],default='all');run(parser.parse_args().stage)
