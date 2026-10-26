#!/usr/bin/env python3
"""Run the frozen multi-horizon and fixed15 leave-one-rule-out study."""
from pathlib import Path
import copy
import gc
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from analysis.holdings_count import eligible_base, metrics
from analysis.ic_weight_comparison import execution_labels, daily_rank_ic
from analysis.multihorizon_selection import (
    StudyAccount,
    build_mixed_monthly_weights,
    build_monthly_weights,
    impose_trend_floor,
    score_panel,
)
from analysis.single_account_adaptive import family_score, history_risk, prior_weights
from backtest.delisting_replay_v2 import DelistingBridge, replay
from factors.preprocess import preprocess_factor_panel
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel
from scripts.run_mvp_holdings_count import load_matrix, sha, write_json
from scripts.run_rebalance_frequency import classify_executions, verify_sealed
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from universe.dynamic import eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import point_in_time_industry

PROTOCOL = ROOT/'research_protocols/20261025_multihorizon_selection_ablation_v1.json'
OUT = ROOT/'output/multihorizon_selection_ablation_v3_20261025'
IC20 = ROOT/'output/multihorizon_selection_ablation_v2_20261025/ic20'
FACTORS = ['PE','FREE_CASH_FLOW_YIELD','ROE','GROSS_MARGIN','NET_MARGIN',
           'LOW_DEBT_TO_ASSETS','CASH_PROFIT_QUALITY','REVENUE_GROWTH',
           'PROFIT_GROWTH','MOMENTUM','MOMENTUM_60D','REVERSAL_5D','VOLATILITY']
RULES = ['BASE_5D','SHORT_REVERSAL','TREND_FLOOR','MULTI_HORIZON']
ABLATIONS = ['FULL','NO_BUFFER','NO_CORRELATION','NO_INDUSTRY_CAP','NO_TECH_CAP']


def freeze():
    q = json.loads(PROTOCOL.read_text())
    ppath = ROOT/q['parent_protocol']; p = json.loads(ppath.read_text())
    source_protocol = json.loads((ROOT/'research_protocols/20261018_six_strategy_ic_comparison_v1.json').read_text())
    source = ROOT/source_protocol['data_source']
    baseline = ROOT/q['baseline_source']; factor_source = ROOT/q['factor_source']
    files = {str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL), str(ppath.relative_to(ROOT)):sha(ppath)}
    for path in [ROOT/'analysis/multihorizon_selection.py', Path(__file__),
                 ROOT/'backtest/delisting_replay_v2.py', ROOT/'data/research_corporate_actions_v1.json',
                 ROOT/json.loads((ROOT/q['asset_ledger_protocol']).read_text())['registry']]:
        files[str(path.relative_to(ROOT))] = sha(path)
    for path in [baseline/'factor_weights.csv', baseline/'rankings.csv.gz', baseline/'schedules.json',
                 factor_source/'calendar.json']:
        files[str(path.relative_to(ROOT))] = sha(path)
    for manifest in sorted(IC20.glob('inputs_*/complete.json')):
        files[str(manifest.relative_to(ROOT))] = sha(manifest)
    prior = ROOT/p['prior_replay']; rawseal = json.loads((prior/'started.json').read_text())['raw']
    for name, digest in rawseal.items():
        path = source/name
        if sha(path) != digest: raise ValueError('raw input changed '+name)
        files[str(path.relative_to(ROOT))] = digest
    OUT.mkdir(exist_ok=True)
    stamp = {'files':files, 'protocol':sha(PROTOCOL)}
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text()) != stamp:
        raise ValueError('frozen inputs changed; version the study')
    write_json(OUT/'frozen.json',stamp); write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists(): write_json(OUT/'forward_before.json',account_hashes())
    return p,q,source_protocol


def prepare_twenty_day_ic(p,q,source_protocol,prices):
    source=ROOT/source_protocol['data_source']; destroot=IC20;destroot.mkdir(exist_ok=True)
    existing=sorted((ROOT/q['factor_source']).glob('inputs_*'))
    if existing and all((destroot/f.name/'complete.json').exists() for f in existing):
        for f in existing: verify_sealed(destroot/f.name)
        return
    print('Preparing point-in-time daily20-session IC',flush=True)
    matrices=load_matrix(source/'daily.csv.gz',['close','amount'],prices.index,prices.columns,{'amount':1000.})
    close,amount=matrices['close'],matrices['amount']
    labels,end=execution_labels(prices,20)
    basic=pd.read_csv(source/'stock_basic.csv.gz',dtype=str);basic=basic[basic.ts_code.isin(prices.columns)]
    names=pd.read_csv(source/'namechange.csv.gz')
    base=eligible_base(close,amount,basic,names,prices.index);base['quarter']=base.date.dt.to_period('Q').astype(str)
    fincols=['ts_code','ann_date','end_date','eps','roe','grossprofit_margin','netprofit_margin','debt_to_assets','or_yoy','netprofit_yoy','ocfps','ocf_to_profit']
    finance=pd.read_csv(ROOT/source_protocol['financial_supplement'],usecols=fincols)
    balance=pd.read_csv(source/'balancesheet_vip.csv.gz',usecols=lambda x:x in ['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit=pd.read_csv(source/'fina_audit.csv.gz');industry=pd.read_csv(source/'index_member_all.csv.gz')
    weeks=pd.DatetimeIndex(json.loads((ROOT/q['factor_source']/'calendar.json').read_text())['signals'])
    for quarter,b in base.groupby('quarter',sort=True):
        frozen=ROOT/q['factor_source']/f'inputs_{quarter}'
        if not frozen.exists(): continue
        folder=destroot/f'inputs_{quarter}';folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():verify_sealed(folder);continue
        dates=pd.DatetimeIndex(sorted(b.date.unique()));start=max(0,prices.index.get_loc(dates[0])-90);stop=prices.index.get_loc(dates[-1])+1
        long=close.iloc[start:stop].stack().rename('close').reset_index();long.columns=['trade_date','ts_code','close']
        idx=pd.MultiIndex.from_frame(long[['trade_date','ts_code']]);long['adj_close']=prices.iloc[start:stop].stack().reindex(idx).to_numpy()
        raw=build_weekly_raw_factor_panel(long,finance,dates)
        risk=build_hard_risk_universe_report(b.drop(columns='quarter'),balance,audit,names)
        mask=eligibility_from_report(risk).reindex(raw.index).fillna(False).astype(bool)
        ind=point_in_time_industry(raw.index,industry);selected=raw.loc[mask,FACTORS]
        standard=preprocess_factor_panel(selected,industry=ind.reindex(selected.index),by_industry=True,min_industry_count=3)
        label=labels.loc[dates].stack().rename('label');label.index=label.index.set_names(['date','symbol'])
        ic,n=daily_rank_ic(standard.where(selected.notna()),label,30);ic=ic.reindex(dates);n=n.reindex(dates)
        ic['outcome_end']=end.reindex(dates);ic.to_csv(folder/'ic.csv');n.to_csv(folder/'ic_names.csv')
        # Exact weekly panel parity proves the20-day IC uses the same PIT universe and factor preprocessing.
        expected=pd.read_csv(frozen/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        observed=standard.loc[standard.index.get_level_values(0).isin(weeks.intersection(dates))]
        common=expected.index.intersection(observed.index)
        if not expected.index.equals(observed.index) or not np.allclose(expected[FACTORS],observed[FACTORS],equal_nan=True,atol=1e-12,rtol=1e-12):
            raise AssertionError('weekly PIT factor parity failed '+quarter)
        write_json(folder/'audit.json',{'quarter':quarter,'daily_rows':len(standard),'weekly_rows':len(common),
            'weekly_factor_parity':True,'label_sessions':20})
        write_json(folder/'complete.json',{f.name:sha(f) for f in folder.iterdir() if f.name!='complete.json'})
        print('IC20',quarter,len(standard),flush=True);del raw,risk,standard,selected,long;gc.collect()


def load_ic(q):
    five=[];twenty=[]
    for folder in sorted((ROOT/q['factor_source']).glob('inputs_*')):
        five.append(pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_ic.csv',parse_dates=['date','outcome_end']).set_index('date'))
        x=pd.read_csv(IC20/folder.name/'ic.csv',index_col=0,parse_dates=[0,'outcome_end'])
        x.index.name='date';twenty.append(x)
    five=pd.concat(five).sort_index();twenty=pd.concat(twenty).sort_index()
    if five.index.has_duplicates or twenty.index.has_duplicates or not five.index.equals(twenty.index):raise ValueError('IC date mismatch')
    return five,twenty


def build_weights(p,q):
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families'];initial=prior_weights(families)
    five,twenty=load_ic(q)
    base_log=pd.read_csv(ROOT/q['baseline_source']/'factor_weights.csv',parse_dates=['date'])
    base_log=base_log[base_log.arm=='BASE'];base=base_log.pivot(index='date',columns='factor',values='weight')[FACTORS]
    updates=base.index
    shortfamilies={k:list(v) for k,v in families.items()};shortfamilies['MOMENTUM_REVERSAL']=['REVERSAL_5D']
    short_initial=prior_weights(shortfamilies)
    short,shortlog=build_monthly_weights(five[short_initial.index],five.outcome_end,updates,short_initial,p['ic'])
    trend=base.apply(impose_trend_floor,axis=1)
    mapping={str(k):v for k,v in q['horizons'].items()};icmix=pd.DataFrame(index=five.index);ends=pd.DataFrame(index=five.index)
    for factor in FACTORS:
        horizon=next(h for h,names in mapping.items() if factor in names);source=five if horizon=='5' else twenty
        icmix[factor]=source[factor];ends[factor]=source.outcome_end
    multi,multilog=build_mixed_monthly_weights(icmix,ends,updates,initial,p['ic'])
    weights={'BASE_5D':base,'SHORT_REVERSAL':short,'TREND_FLOOR':trend,'MULTI_HORIZON':multi}
    pd.concat([base_log.assign(rule='BASE_5D'),shortlog.assign(rule='SHORT_REVERSAL'),
               multilog.assign(rule='MULTI_HORIZON')],ignore_index=True).to_csv(OUT/'weight_evidence.csv',index=False)
    pd.concat({k:v for k,v in weights.items()},names=['rule','date']).to_csv(OUT/'monthly_weights.csv')
    return families,shortfamilies,weights


def build_groups(p,q,prices):
    families,shortfamilies,weights=build_weights(p,q);prior=prior_weights(families)
    weeks=pd.DatetimeIndex(json.loads((ROOT/q['baseline_source']/'schedules.json').read_text())['WEEKLY'])
    expected=pd.read_csv(ROOT/q['baseline_source']/'rankings.csv.gz',parse_dates=['date']);expected=expected[expected.arm=='BASE']
    groups={r:{} for r in RULES};rankrows=[];returns=prices.pct_change(fill_method=None);risk={r:{} for r in RULES}
    for folder in sorted((ROOT/q['factor_source']).glob('inputs_*')):
        x=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        for date,panel in x.groupby(level=0):
            if date not in weeks:continue
            fixed=family_score(panel,families,prior).droplevel(0);industry=panel.droplevel(0).industry
            for rule in RULES:
                recipe=shortfamilies if rule=='SHORT_REVERSAL' else families
                w=weights[rule].loc[:date].iloc[-1]
                score=score_panel(panel,recipe,w,fixed.index)
                groups[rule][date]=dict(ic=score.head(100),fixed=score.head(100),industry=industry)
                risk[rule][date]=history_risk(returns,date,score.head(100).index,p['selection'])
                for rank,(symbol,value) in enumerate(score.head(100).items(),1):
                    rankrows.append(dict(rule=rule,date=date,symbol=symbol,score=value,rank=rank))
            e=expected[expected.date==date].sort_values('rank').symbol.tolist()
            if groups['BASE_5D'][date]['ic'].index.tolist()!=e:raise AssertionError('BASE ranking parity '+str(date))
    pd.DataFrame(rankrows).to_csv(OUT/'rankings.csv.gz',index=False)
    return groups,risk,returns


def asset_inputs(full,q):
    ledger=json.loads((ROOT/q['asset_ledger_protocol']).read_text())
    events=json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']
    transfers=json.loads((ROOT/ledger['registry']).read_text())['events']
    adj=pd.read_csv(ROOT/'output/fixed15_ml_delisting_v4_20261024/adjustments.csv.gz',index_col='date',parse_dates=True)
    dead=pd.to_datetime(pd.read_csv(ROOT/'output/asset_transition_fix_20261024/delist_dates.csv',index_col=0).iloc[:,0])
    bars={}
    for event in transfers:
        path=ROOT/ledger['market_data'][event['destination']]/(event['destination'].replace('.','_')+'_daily_raw.csv')
        b=pd.read_csv(path,index_col='date',parse_dates=True)
        bars[event['destination']]=b.loc[event['listing_date']:event.get('coverage_end',ledger['end'])]
    DelistingBridge(full,adj,dead,transfers,bars,ledger['confirmations'])
    return ledger,events,transfers,adj,dead,bars


def account_tasks(q):
    tasks=[]
    for rule in RULES:
        for fee in q['fees']:tasks.append((rule,'FULL',fee))
    for rule in ['BASE_5D','MULTI_HORIZON']:
        for ablation in ABLATIONS[1:]:tasks.append((rule,ablation,q['fees'][0]))
    return tasks


def period_metrics(nav,start,stop=None):
    x=nav.loc[start:stop] if stop else nav.loc[start:]
    pos=nav.index.get_loc(x.index[0]);x=pd.concat([nav.iloc[pos-1:pos],x]) if pos else x
    return {k:float(v[0]) for k,v in metrics(x).items()}


def run_accounts(p,q,full,groups,risk,returns):
    prices=full.loc[q['evaluation_start']:q['evaluation_end']];schedule=pd.DataFrame({'execution_date':prices.index[1:]},index=prices.index[:-1])
    ledger,events,transfers,adj,dead,bars=asset_inputs(full,q)
    summary=[];periods=[];annual=[];navs={};account_paths={}
    for rule,ablation,fee in account_tasks(q):
        key=f'{rule}_{ablation}_{round(fee*10000)}bp';dest=OUT/key
        if (dest/'complete.json').exists():verify_sealed(dest)
        else:
            account=StudyAccount(groups[rule],risk[rule],returns,p,ablation)
            result=replay(prices,schedule,account,actions=events,adjustment_factors=adj,delist_dates=dead,
                transfers=transfers,otc_bars=bars,confirmations=ledger['confirmations'],fee=fee)
            result['decisions']=pd.DataFrame(account.logs)
            result['executions']=classify_executions(result['executions'],result['decisions'],prices.index[0])
            dest.mkdir(exist_ok=True)
            for name,frame in result.items():frame.to_csv(dest/(name+'.csv.gz'),index=name=='daily')
            write_json(dest/'complete.json',{f.name:sha(f) for f in dest.iterdir() if f.name!='complete.json'})
        daily=pd.read_csv(dest/'daily.csv.gz',index_col='date',parse_dates=True);nav=daily.post_nav
        ex=pd.read_csv(dest/'executions.csv.gz',parse_dates=['date']);years=(len(nav)-1)/252
        m=period_metrics(nav,q['evaluation_start'],q['evaluation_end'])
        summary.append(dict(key=key,rule=rule,ablation=ablation,fee=fee,**m,
            annual_two_way_turnover=ex.two_way_turnover.sum()/years,average_equity=float((1-daily.cash/daily.post_nav).mean()),
            terminal_unquoted_fraction=float(daily.unquoted_value.iloc[-1]/daily.post_nav.iloc[-1])))
        for label,start,stop in q['periods']:
            periods.append(dict(key=key,rule=rule,ablation=ablation,fee=fee,period=label,**period_metrics(nav,start,stop)))
        for year,x in nav.groupby(nav.index.year):
            pos=nav.index.get_loc(x.index[0]);base=nav.iloc[pos-1] if pos else 1.
            annual.append(dict(key=key,rule=rule,ablation=ablation,fee=fee,year=year,return_value=float(x.iloc[-1]/base-1)))
        navs[key]=nav;account_paths[key]=str(dest.relative_to(ROOT));print('ACCOUNT',key,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(periods).to_csv(OUT/'periods.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False);pd.DataFrame(navs).to_csv(OUT/'nav.csv');write_json(OUT/'accounts.json',account_paths)
    # Exact unchanged baseline parity against the repaired delisting account.
    for fee in q['fees']:
        key=f'BASE_5D_FULL_{round(fee*10000)}bp'
        got=pd.read_csv(ROOT/account_paths[key]/'daily.csv.gz',index_col='date',parse_dates=True)
        ref=pd.read_csv(ROOT/f"output/fixed15_ml_delisting_v4_20261024/BASE_BROAD_{round(fee*10000)}bp/daily.csv.gz",index_col='date',parse_dates=True)
        pd.testing.assert_frame_equal(got[ref.columns],ref,check_dtype=False,atol=1e-12,rtol=1e-12)
    write_json(OUT/'validation.json',{'baseline_full_account_parity':True,'weekly_score_parity':True,
        'accounts':len(account_tasks(q)),'forward_accounts_unchanged':account_hashes()==json.loads((OUT/'forward_before.json').read_text())})


def run(stage):
    p,q,source=freeze();full=load_prices(p).loc[:q['evaluation_end']]
    if stage in {'all','prepare'}:prepare_twenty_day_ic(p,q,source,full)
    if stage=='prepare':return
    groups,risk,returns=build_groups(p,q,full);run_accounts(p,q,full,groups,risk,returns)
    if account_hashes()!=json.loads((OUT/'forward_before.json').read_text()):raise AssertionError('forward accounts changed')
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='completed.json'})
    print('COMPLETE multi-horizon selection study',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','run'],default='all')
    run(parser.parse_args().stage)
