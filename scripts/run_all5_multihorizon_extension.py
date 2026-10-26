#!/usr/bin/env python3
"""Compare original13 versus all18 factors under unified and economic horizons."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.candidate_factors import CANDIDATES,FIELDS,normalize_annual,annual_candidates,dividend_asof,momentum_12_1,standardize_candidates
from analysis.holdings_count import eligible_base,metrics
from analysis.ic_weight_comparison import execution_labels,daily_rank_ic
from analysis.multihorizon_selection import StudyAccount,build_mixed_monthly_weights
from analysis.single_account_adaptive import family_score,history_risk,prior_weights
from backtest.delisting_replay_v2 import replay
from scripts.run_mvp_holdings_count import load_matrix,sha,write_json
from scripts.run_multihorizon_selection_ablation import asset_inputs
from scripts.run_rebalance_frequency import classify_executions,verify_sealed
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from universe.dynamic import eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import point_in_time_industry
from strategies.holding_buffer import ordered_scores

PROTOCOL=ROOT/'research_protocols/20261025_all5_multihorizon_extension_v1.json'
OUT=ROOT/'output/all5_multihorizon_extension_20261025'
IC20_ORIGINAL=ROOT/'output/multihorizon_selection_ablation_v2_20261025/ic20'
ARMS=['ORIGINAL13_5D','ALL18_5D','ORIGINAL13_MULTI','ALL18_MULTI']


def freeze():
    q=json.loads(PROTOCOL.read_text());parent_path=ROOT/q['parent_protocol'];parent=json.loads(parent_path.read_text())
    base_path=ROOT/parent['parent_protocol'];p=json.loads(base_path.read_text());candidate=json.loads((ROOT/q['candidate_protocol']).read_text())
    files={str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL),str(parent_path.relative_to(ROOT)):sha(parent_path),str(base_path.relative_to(ROOT)):sha(base_path)}
    for path in [Path(__file__),ROOT/'analysis/multihorizon_selection.py',ROOT/'analysis/candidate_factors.py',ROOT/q['candidate_output']/'factor_weights.csv',ROOT/q['candidate_output']/'daily_ic.csv',ROOT/q['candidate_output']/'rankings.csv.gz',ROOT/q['multihorizon_output']/'monthly_weights.csv',ROOT/q['multihorizon_output']/'rankings.csv.gz']:
        files[str(path.relative_to(ROOT))]=sha(path)
    for manifest in sorted(IC20_ORIGINAL.glob('inputs_*/complete.json')):files[str(manifest.relative_to(ROOT))]=sha(manifest)
    OUT.mkdir(exist_ok=True);stamp={'files':files,'protocol':sha(PROTOCOL)}
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text())!=stamp:raise ValueError('frozen study changed')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():write_json(OUT/'forward_before.json',account_hashes())
    return p,parent,q,candidate


def prepare_candidate_ic20(p,parent,q,candidate,prices):
    destroot=OUT/'candidate_ic20';source=ROOT/json.loads((ROOT/candidate['source_protocol']).read_text())['data_source'];destroot.mkdir(exist_ok=True)
    frozen=sorted((ROOT/q['candidate_output']).glob('inputs_*'))
    if frozen and all((destroot/f.name/'complete.json').exists() for f in frozen):
        for f in frozen:verify_sealed(destroot/f.name)
        return
    print('Preparing five candidate factors with20-session labels',flush=True)
    matrices=load_matrix(source/'daily.csv.gz',['close','amount'],prices.index,prices.columns,{'amount':1000.})
    basic=pd.read_csv(source/'stock_basic.csv.gz',dtype=str);basic=basic[basic.ts_code.isin(prices.columns)]
    names=pd.read_csv(source/'namechange.csv.gz');base=eligible_base(matrices['close'],matrices['amount'],basic,names,prices.index);base['quarter']=base.date.dt.to_period('Q').astype(str)
    tables={}
    for kind,file in [('income','income_vip'),('balance','balancesheet_vip'),('cashflow','cashflow_vip')]:
        cols=['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type']+FIELDS[kind]
        tables[kind]=normalize_annual(pd.read_csv(source/(file+'.csv.gz'),usecols=cols),kind)
    balance=pd.read_csv(source/'balancesheet_vip.csv.gz',usecols=['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit=pd.read_csv(source/'fina_audit.csv.gz');industry=pd.read_csv(source/'index_member_all.csv.gz')
    dividend=pd.read_csv(source/'daily_basic.csv.gz',usecols=['ts_code','trade_date','dv_ttm'])
    mom=momentum_12_1(prices);labels,ends=execution_labels(prices,20)
    for quarter,b in base.groupby('quarter',sort=True):
        frozenfolder=ROOT/q['candidate_output']/f'inputs_{quarter}'
        if not frozenfolder.exists():continue
        folder=destroot/f'inputs_{quarter}';folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():verify_sealed(folder);continue
        risk=build_hard_risk_universe_report(b.drop(columns='quarter'),balance,audit,names);mask=eligibility_from_report(risk);idx=mask.index[mask]
        chunks=[]
        for date,g in idx.to_frame(index=False).groupby('date',sort=True):
            a=annual_candidates(tables,date).reindex(g.symbol);a.index=pd.MultiIndex.from_arrays([[date]*len(a),g.symbol],names=['date','symbol']);chunks.append(a)
        raw=pd.concat(chunks).sort_index();raw=raw.join(dividend_asof(raw.index,dividend));raw['MOMENTUM_12_1']=mom.loc[raw.index.get_level_values(0).unique()].stack().reindex(raw.index)
        ind=point_in_time_industry(raw.index,industry);standard=standardize_candidates(raw,ind)
        target=labels.loc[standard.index.get_level_values(0).unique()].stack();target.index=target.index.set_names(['date','symbol'])
        ic,count=daily_rank_ic(standard,target);ic['outcome_end']=ends.reindex(ic.index)
        ic.to_csv(folder/'ic.csv');count.to_csv(folder/'ic_names.csv')
        expected=pd.read_csv(frozenfolder/'standard.csv.gz',parse_dates=['date']).set_index(['date','symbol'])[CANDIDATES]
        observed=standard.reindex(expected.index)[CANDIDATES]
        if not np.allclose(expected,observed,equal_nan=True,atol=1e-12,rtol=1e-12):raise AssertionError('candidate weekly parity '+quarter)
        write_json(folder/'audit.json',{'quarter':quarter,'rows':len(standard),'weekly_factor_parity':True,'label_sessions':20})
        write_json(folder/'complete.json',{f.name:sha(f) for f in folder.iterdir() if f.name!='complete.json'})
        print('CANDIDATE_IC20',quarter,len(standard),flush=True);del raw,standard,risk,chunks;gc.collect()


def load_all_ic(q):
    five=pd.read_csv(ROOT/q['candidate_output']/'daily_ic.csv',parse_dates=['date','outcome_end']).set_index('date').sort_index()
    original20=[];candidate20=[]
    for folder in sorted((ROOT/q['candidate_output']).glob('inputs_*')):
        x=pd.read_csv(IC20_ORIGINAL/folder.name/'ic.csv',index_col=0,parse_dates=[0,'outcome_end']);x.index.name='date';original20.append(x)
        y=pd.read_csv(OUT/'candidate_ic20'/folder.name/'ic.csv',index_col=0,parse_dates=[0,'outcome_end']);y.index.name='date';candidate20.append(y)
    original20=pd.concat(original20).sort_index();candidate20=pd.concat(candidate20).sort_index()
    if not five.index.equals(original20.index) or not five.index.equals(candidate20.index):raise ValueError('IC axes differ')
    return five,original20,candidate20


def build_weights(p,parent,q,candidate):
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families'];recipe={k:list(v) for k,v in families.items()}
    for factor,family in candidate['candidate_families'].items():recipe[family].append(factor)
    current=pd.read_csv(ROOT/q['multihorizon_output']/'monthly_weights.csv');weights={}
    for rule,arm in [('ORIGINAL13_5D','BASE_5D'),('ORIGINAL13_MULTI','MULTI_HORIZON')]:
        x=current[current.rule==arm].drop(columns='rule');x['date']=pd.to_datetime(x.date);weights[rule]=x.set_index('date')
    old=pd.read_csv(ROOT/q['candidate_output']/'factor_weights.csv',parse_dates=['date']);old=old[old.arm=='ALL5'];weights['ALL18_5D']=old.pivot(index='date',columns='factor',values='weight')
    five,original20,candidate20=load_all_ic(q);factors=list(prior_weights(families).index)+CANDIDATES
    initial=prior_weights(families).reindex(factors,fill_value=0);updates=weights['ORIGINAL13_5D'].index
    mapping={str(k):v for k,v in parent['horizons'].items()};ic=pd.DataFrame(index=five.index);ends=pd.DataFrame(index=five.index)
    for factor in factors:
        if factor in CANDIDATES:source=candidate20
        else:
            horizon=next(h for h,names in mapping.items() if factor in names);source=five if horizon=='5' else original20
        ic[factor]=source[factor];ends[factor]=source.outcome_end
    multi,log=build_mixed_monthly_weights(ic,ends,updates,initial,p['ic']);weights['ALL18_MULTI']=multi
    log.assign(arm='ALL18_MULTI').to_csv(OUT/'weight_evidence.csv',index=False)
    pd.concat(weights,names=['arm','date']).to_csv(OUT/'monthly_weights.csv')
    return families,recipe,weights


def build_groups(p,parent,q,candidate,prices):
    families,recipe,weights=build_weights(p,parent,q,candidate);prior=prior_weights(families)
    weeks=pd.DatetimeIndex(json.loads((ROOT/q['candidate_output']/'schedules.json').read_text())['WEEKLY'])
    expected_candidate=pd.read_csv(ROOT/q['candidate_output']/'rankings.csv.gz',parse_dates=['date']);expected_candidate=expected_candidate[expected_candidate.arm=='ALL5']
    expected_current=pd.read_csv(ROOT/q['multihorizon_output']/'rankings.csv.gz',parse_dates=['date'])
    groups={a:{} for a in ARMS};risk={a:{} for a in ARMS};rankrows=[];returns=prices.pct_change(fill_method=None)
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        quarter=folder.name.removeprefix('inputs_');base=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        new=pd.read_csv(ROOT/q['candidate_output']/f'inputs_{quarter}'/'standard.csv.gz',parse_dates=['date']).set_index(['date','symbol'])[CANDIDATES]
        for date,panel in base.groupby(level=0):
            if date not in weeks:continue
            fixed=family_score(panel,families,prior).droplevel(0);merged=panel.join(new.reindex(panel.index).fillna(0));industry=panel.droplevel(0).industry
            for arm in ARMS:
                use_recipe=recipe if arm.startswith('ALL18') else families;use_panel=merged if arm.startswith('ALL18') else panel
                w=weights[arm].loc[:date].iloc[-1];score=family_score(use_panel,use_recipe,w,require_all=False).droplevel(0).reindex(fixed.index).dropna();score=ordered_scores(score)
                groups[arm][date]=dict(ic=score.head(100),fixed=score.head(100),industry=industry);risk[arm][date]=history_risk(returns,date,score.head(100).index,p['selection'])
                for rank,(symbol,value) in enumerate(score.head(100).items(),1):rankrows.append(dict(arm=arm,date=date,symbol=symbol,score=value,rank=rank))
            checks={'ALL18_5D':expected_candidate[expected_candidate.date==date],
                    'ORIGINAL13_5D':expected_current[(expected_current.date==date)&(expected_current.rule=='BASE_5D')],
                    'ORIGINAL13_MULTI':expected_current[(expected_current.date==date)&(expected_current.rule=='MULTI_HORIZON')]}
            for arm,frame in checks.items():
                if groups[arm][date]['ic'].index.tolist()!=frame.sort_values('rank').symbol.tolist():raise AssertionError('ranking parity '+arm+' '+str(date))
    pd.DataFrame(rankrows).to_csv(OUT/'rankings.csv.gz',index=False);return groups,risk,returns


def period_metrics(nav,start,stop=None):
    x=nav.loc[start:stop] if stop else nav.loc[start:];pos=nav.index.get_loc(x.index[0]);x=pd.concat([nav.iloc[pos-1:pos],x]) if pos else x
    return {k:float(v[0]) for k,v in metrics(x).items()}


def run_accounts(p,parent,q,full,groups,risk,returns):
    prices=full.loc[q['evaluation_start']:q['evaluation_end']];schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    ledger,events,transfers,adj,dead,bars=asset_inputs(full,parent);summary=[];periods=[];annual=[];navs={};paths={}
    for arm in ARMS:
        for fee in q['fees']:
            key=f'{arm}_{round(fee*10000)}bp';dest=OUT/key
            account=StudyAccount(groups[arm],risk[arm],returns,p,'FULL')
            result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,transfers=transfers,otc_bars=bars,confirmations=ledger['confirmations'],fee=fee)
            result['decisions']=pd.DataFrame(account.logs);result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0]);dest.mkdir(exist_ok=True)
            for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
            write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
            daily=result['daily'];nav=daily.post_nav;ex=result['executions'];years=(len(nav)-1)/252
            summary.append(dict(key=key,arm=arm,fee=fee,**period_metrics(nav,q['evaluation_start'],q['evaluation_end']),annual_two_way_turnover=ex.two_way_turnover.sum()/years,average_equity=float((1-daily.cash/daily.post_nav).mean())))
            for label,start,stop in q['periods']:periods.append(dict(key=key,arm=arm,fee=fee,period=label,**period_metrics(nav,start,stop)))
            for year,x in nav.groupby(nav.index.year):
                pos=nav.index.get_loc(x.index[0]);basev=nav.iloc[pos-1] if pos else 1.;annual.append(dict(key=key,arm=arm,fee=fee,year=year,return_value=float(x.iloc[-1]/basev-1)))
            navs[key]=nav;paths[key]=str(dest.relative_to(ROOT));print('ACCOUNT',key,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(periods).to_csv(OUT/'periods.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False);pd.DataFrame(navs).to_csv(OUT/'nav.csv');write_json(OUT/'accounts.json',paths)
    # Both original13 variants must reproduce article146 exactly after the same asset ledger.
    for arm,old in [('ORIGINAL13_5D','BASE_5D_FULL'),('ORIGINAL13_MULTI','MULTI_HORIZON_FULL')]:
        for fee in q['fees']:
            got=pd.read_csv(OUT/f'{arm}_{round(fee*10000)}bp'/'daily.csv.gz',index_col='date',parse_dates=True)
            ref=pd.read_csv(ROOT/q['multihorizon_output']/f'{old}_{round(fee*10000)}bp'/'daily.csv.gz',index_col='date',parse_dates=True)
            pd.testing.assert_frame_equal(got[ref.columns],ref,check_dtype=False,atol=1e-12,rtol=1e-12)
    # Quantify whether adding five factors actually changes the weekly15 names.
    rows=[]
    for base,added in [('ORIGINAL13_5D','ALL18_5D'),('ORIGINAL13_MULTI','ALL18_MULTI')]:
        left=pd.read_csv(OUT/f'{base}_3bp'/'targets.csv.gz',parse_dates=['date']);right=pd.read_csv(OUT/f'{added}_3bp'/'targets.csv.gz',parse_dates=['date'])
        weeks=sorted(set(left.date)&set(right.date))
        for date in weeks:
            a=set(left.loc[(left.date==date)&(left.weight>0),'symbol']);b=set(right.loc[(right.date==date)&(right.weight>0),'symbol'])
            rows.append(dict(base=base,added=added,date=date,common=len(a&b),replaced_each_side=len(a^b)/2,same=a==b,jaccard=len(a&b)/len(a|b)))
    pd.DataFrame(rows).to_csv(OUT/'holding_overlap.csv',index=False)
    write_json(OUT/'validation.json',{'original13_account_parity':True,'three_frozen_ranking_parities':True,'forward_accounts_unchanged':account_hashes()==json.loads((OUT/'forward_before.json').read_text()),'accounts':8})


def run(stage):
    p,parent,q,candidate=freeze();full=load_prices(p).loc[:q['evaluation_end']]
    if stage in ['all','prepare']:prepare_candidate_ic20(p,parent,q,candidate,full)
    if stage=='prepare':return
    groups,risk,returns=build_groups(p,parent,q,candidate,full);run_accounts(p,parent,q,full,groups,risk,returns)
    if account_hashes()!=json.loads((OUT/'forward_before.json').read_text()):raise AssertionError('forward account changed')
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completed.json'})
    print('COMPLETE all-five extension',flush=True)


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['all','prepare','run'],default='all');run(a.parse_args().stage)
