#!/usr/bin/env python3
"""Frozen weekly/fortnightly/monthly replay. Never changes paper accounts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis.rebalance_frequency import schedules,asof_weights,classify_executions,index_nav
from analysis.single_account_adaptive import family_score,prior_weights,history_risk
from analysis.factor_diagnosis import compact_replay
from analysis.holdings_count import eligible_base,metrics
from scripts.run_single_account_adaptive import Account,account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices
from scripts.run_mvp_holdings_count import sha,write_json,load_matrix
from scripts.build_dynamic_universe_comparison import build_weekly_raw_factor_panel
from universe.dynamic import eligibility_from_report
from universe.risk import build_hard_risk_universe_report
from universe.strategy import point_in_time_industry
from factors.preprocess import preprocess_factor_panel
from strategies.holding_buffer import ordered_scores

PROTOCOL=ROOT/'research_protocols/20261022_rebalance_frequency_v1.json'
OUT=ROOT/'output/rebalance_frequency_20261022'
PERIODS={'FULL':('2012','2026'),'EARLY':('2012','2015'),'MID':('2016','2020'),'RECENT':('2021','2026')}

def freeze():
    q=json.loads(PROTOCOL.read_text());parent=ROOT/q['parent_output']
    old=json.loads((parent/'frozen.json').read_text())
    files=dict(old['files'])
    for name,digest in files.items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Changed parent source: {name}')
    for name,digest in json.loads((parent/'completed.json').read_text()).items():
        if sha(parent/name)!=digest:raise ValueError(f'Changed parent result: {name}')
    p=json.loads((ROOT/q['parent_protocol']).read_text())
    src=json.loads((ROOT/q['source_protocol']).read_text())
    raw_seal=json.loads((ROOT/p['prior_replay']/'started.json').read_text())
    for name in ['daily.csv.gz','stock_basic.csv.gz','namechange.csv.gz','balancesheet_vip.csv.gz','fina_audit.csv.gz','index_member_all.csv.gz']:
        path=ROOT/src['data_source']/name
        if sha(path)!=raw_seal['raw'][name]:raise ValueError(f'Raw input changed: {name}')
        files[str(path.relative_to(ROOT))]=sha(path)
    fin=ROOT/src['financial_supplement']
    if sha(fin)!=raw_seal['financial_supplement']:raise ValueError('Financial supplement changed')
    files[str(fin.relative_to(ROOT))]=sha(fin)
    for path in [PROTOCOL,Path(__file__),ROOT/'analysis/rebalance_frequency.py',parent/'completed.json',
                 ROOT/'output/single_account_fixed15_risk_20261021/factor_weights.csv',
                 ROOT/'scripts/build_dynamic_universe_comparison.py',ROOT/'universe/risk.py',ROOT/'universe/dynamic.py',ROOT/'universe/strategy.py']:
        files[str(path.relative_to(ROOT))]=sha(path)
    stamp=dict(protocol=sha(PROTOCOL),files=files)
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'frozen.json').exists() and json.loads((OUT/'frozen.json').read_text())!=stamp:raise ValueError('Frozen run changed; use new directory')
    if (OUT/'completed.json').exists():raise ValueError('Completed run: do not overwrite')
    write_json(OUT/'frozen.json',stamp);write_json(OUT/'protocol.json',q)
    if not (OUT/'forward_before.json').exists():write_json(OUT/'forward_before.json',account_hashes())
    return p,q,src

def verify_sealed(folder):
    for name,digest in json.loads((folder/'complete.json').read_text()).items():
        if sha(folder/name)!=digest:raise ValueError(f'Changed checkpoint {folder/name}')

def prepare_months(p,q,src,prices,plans):
    source=ROOT/src['data_source'];factor_source=ROOT/p['factor_source']
    months=plans['MONTHLY'];weekly=plans['WEEKLY']
    # One overlapping weekly signal per quarter validates identical reconstruction.
    checks=pd.DatetimeIndex(pd.Series(weekly,index=weekly).groupby(weekly.to_period('Q')).first())
    needed=months.union(checks)
    folders={str(period):OUT/f'month_inputs_{period}' for period in needed.to_period('Q').unique()}
    pending=[quarter for quarter,folder in folders.items() if not (folder/'complete.json').exists()]
    if not pending:
        for folder in folders.values():verify_sealed(folder)
        return folders
    print('Loading raw close/amount matrices for true calendar month ends',flush=True)
    matrices=load_matrix(source/'daily.csv.gz',['close','amount'],prices.index,prices.columns,{'amount':1000.})
    close,amount=matrices['close'],matrices['amount']
    basic=pd.read_csv(source/'stock_basic.csv.gz',dtype=str);basic=basic[basic.ts_code.isin(prices.columns)]
    names=pd.read_csv(source/'namechange.csv.gz')
    base=eligible_base(close,amount,basic,names,needed)
    del amount,matrices
    base['quarter']=base.date.dt.to_period('Q').astype(str)
    finance=pd.read_csv(ROOT/src['financial_supplement'],usecols=['ts_code','ann_date','end_date','eps','roe','grossprofit_margin','netprofit_margin','debt_to_assets','or_yoy','netprofit_yoy','ocfps','ocf_to_profit'])
    balance=pd.read_csv(source/'balancesheet_vip.csv.gz',usecols=lambda c:c in ['ts_code','ann_date','f_ann_date','end_date','report_type','update_flag','total_hldr_eqy_exc_min_int'])
    audit=pd.read_csv(source/'fina_audit.csv.gz');industry=pd.read_csv(source/'index_member_all.csv.gz')
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families'];cols=[c for fs in families.values() for c in fs]
    for quarter,b in base.groupby('quarter',sort=True):
        folder=folders[quarter]
        if (folder/'complete.json').exists():verify_sealed(folder);continue
        folder.mkdir(exist_ok=True)
        days=pd.DatetimeIndex(sorted(b.date.unique()))
        quarter_days=prices.index[prices.index.to_period('Q').astype(str)==quarter]
        left=max(0,prices.index.get_loc(quarter_days[0])-90);right=prices.index.get_loc(days[-1])+1
        long=close.iloc[left:right].stack().rename('close').reset_index()
        long.columns=['trade_date','ts_code','close']
        idx=pd.MultiIndex.from_frame(long[['trade_date','ts_code']])
        long['adj_close']=prices.iloc[left:right].stack().reindex(idx).to_numpy()
        raw=build_weekly_raw_factor_panel(long,finance,days)
        risk=build_hard_risk_universe_report(b.drop(columns='quarter'),balance,audit,names)
        mask=eligibility_from_report(risk).reindex(raw.index).fillna(False).astype(bool)
        selected=raw.loc[mask,cols];ind=point_in_time_industry(selected.index,industry)
        panel=preprocess_factor_panel(selected,industry=ind,by_industry=True,min_industry_count=3)
        panel['industry']=ind.fillna('');panel['raw_count']=selected.notna().sum(axis=1)
        original=pd.read_csv(factor_source/f'inputs_{quarter}'/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        overlap=days.intersection(weekly)
        a=panel.loc[panel.index.get_level_values(0).isin(overlap),cols].sort_index()
        z=original.loc[original.index.get_level_values(0).isin(overlap),cols].sort_index()
        for frame in [a,z]:
            frame.index=pd.MultiIndex.from_arrays([frame.index.get_level_values(0).as_unit('ns'),frame.index.get_level_values(1)],names=['date','symbol'])
        pd.testing.assert_frame_equal(a,z,check_dtype=False,atol=1e-11,rtol=1e-11)
        panel.to_csv(folder/'panel.csv.gz');risk.to_csv(folder/'universe.csv.gz',index=False)
        write_json(folder/'parity.json',dict(overlapping_weekly_days=len(overlap),rows=len(a),status='PASS'))
        write_json(folder/'complete.json',{f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file() and f.name!='complete.json'})
        print(f'Rebuilt {quarter}: {len(days)} dates; weekly score parity PASS',flush=True)
    return folders

def prepare_groups(p,prices,plans,folders):
    families=json.loads((ROOT/p['family_protocol']).read_text())['candidate_families']
    prior=prior_weights(families)
    weights=pd.read_csv(ROOT/'output/single_account_fixed15_risk_20261021/factor_weights.csv',parse_dates=['date'])
    weights=weights.pivot(index='date',columns='factor',values='weight')
    groups={};weight_usage=[];rankings=[];returns=prices.pct_change(fill_method=None)
    all_dates=plans['WEEKLY'].union(plans['MONTHLY'])
    for folder in sorted((ROOT/p['factor_source']).glob('inputs_*')):
        panel=pd.read_csv(folder/'BASE_V2_MULTI_FACTOR_weekly.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
        quarter=folder.name.removeprefix('inputs_')
        if quarter in folders:
            month=pd.read_csv(folders[quarter]/'panel.csv.gz',parse_dates=['date']).set_index(['date','symbol'])
            month=month.loc[~month.index.get_level_values(0).isin(panel.index.get_level_values(0).unique())]
            panel=pd.concat([panel,month]).sort_index()
        panel=panel.loc[panel.index.get_level_values(0).isin(all_dates)]
        for date,g in panel.groupby(level='date',sort=True):
            w,update=asof_weights(weights,date)
            fixed=family_score(g,families,prior).droplevel(0)
            score=family_score(g,families,w,require_all=False).droplevel(0).reindex(fixed.index).dropna()
            if len(score)!=len(fixed):raise ValueError('Adaptive score changed common pool')
            groups[date]=dict(ic=ordered_scores(score),fixed=ordered_scores(fixed),industry=g.industry.droplevel(0))
            weight_usage.append(dict(date=date,weight_update=update,candidates=len(score)))
            for rank,(symbol,value) in enumerate(groups[date]['ic'].head(100).items(),1):
                rankings.append(dict(date=date,symbol=symbol,score=value,rank=rank))
    assert set(groups)==set(all_dates)
    pd.DataFrame(weight_usage).to_csv(OUT/'weight_usage.csv',index=False)
    pd.DataFrame(rankings).to_csv(OUT/'rankings.csv.gz',index=False)
    print(f'Scores prepared: {len(groups)} dates; building identical risk inputs',flush=True)
    cache={d:history_risk(returns,d,g['ic'].head(100).index,p['selection']) for d,g in groups.items()}
    return groups,cache,returns

def period_metrics(nav,arm,fee):
    result=[]
    for period,(start,stop) in PERIODS.items():
        x=nav.loc[start:stop];loc=nav.index.get_loc(x.index[0])
        if loc:x=pd.concat([nav.iloc[loc-1:loc],x])
        result.append(dict(arm=arm,fee=fee,period=period,**{k:float(v[0]) for k,v in metrics(x).items()}))
    return result

def run():
    p,q,src=freeze();prices_all=load_prices(p).loc[:q['evaluation_end']]
    weekly=pd.DatetimeIndex(json.loads((ROOT/p['factor_source']/'calendar.json').read_text())['signals'])
    plans=schedules(prices_all.index,weekly,q['evaluation_start'],q['evaluation_end'])
    write_json(OUT/'schedules.json',{k:[str(d.date()) for d in dates] for k,dates in plans.items()})
    folders=prepare_months(p,q,src,prices_all,plans)
    groups,cache,returns=prepare_groups(p,prices_all,plans,folders)
    prices=prices_all.loc[q['evaluation_start']:];dates=prices.index
    schedule=pd.DataFrame({'execution_date':dates[1:]},index=dates[:-1])
    navs={};rows=[];periods=[];annual=[];cost_rows=[];parities=[]
    parent=ROOT/q['parent_output']
    for frequency in q['frequencies']:
        subgroup={d:groups[d] for d in plans[frequency]}
        for fee in q['fees']:
            account=Account('IC_CONSTRAINED15_RISK',subgroup,cache,returns,p)
            selected=[]
            def factory(date,held):
                target=account(date,held)
                if target is not None:selected.extend(dict(date=date,symbol=s,weight=float(w)) for s,w in target.items())
                return target
            result=compact_replay(prices,schedule,factory,fee)
            daily=result['daily'];nav=daily.post_nav
            log=pd.DataFrame(account.logs)
            ex=classify_executions(result['executions'],log,dates[0])
            key=f'{frequency}_{int(round(fee*10000))}bp';folder=OUT/key;folder.mkdir(exist_ok=True)
            daily.to_csv(folder/'daily.csv.gz');ex.to_csv(folder/'executions.csv.gz',index=False)
            log.rename(columns={'weekly':'scheduled_reselection'}).to_csv(folder/'decisions.csv.gz',index=False)
            pd.DataFrame(selected).to_csv(folder/'targets.csv.gz',index=False)
            if frequency=='WEEKLY':
                old_folder=parent/f'BASE_RISK_{int(round(fee*10000))}bp'
                old=pd.read_csv(old_folder/'daily.csv.gz',index_col='date',parse_dates=True)
                pd.testing.assert_frame_equal(daily,old,check_dtype=False,check_freq=False,rtol=1e-11,atol=1e-11)
                old_ex=pd.read_csv(old_folder/'executions.csv.gz',parse_dates=['date','signal_date'])
                pd.testing.assert_frame_equal(ex[old_ex.columns],old_ex,check_dtype=False,rtol=1e-11,atol=1e-11)
                parities.append(dict(fee=fee,nav_error=float((nav-old.post_nav).abs().max()),daily_and_execution_match=True))
            years=(len(nav)-1)/252
            row=dict(arm=frequency,fee=fee,**{k:float(v[0]) for k,v in metrics(nav).items()},
                annual_two_way_turnover=ex.two_way_turnover.sum()/years,
                average_equity=(1-daily.cash/daily.post_nav).mean(),actual_trade_dates=int(ex.trade_legs.gt(0).sum()),
                scheduled_signals=len(subgroup),risk_only_trade_dates=int((ex.trigger.eq('RISK_ONLY')&ex.trade_legs.gt(0)).sum()),
                blocked_exit_events=int(ex.blocked_exit_value.gt(1e-10).sum()),
                terminal_unquoted_fraction=float(daily.unquoted_value.iloc[-1]/nav.iloc[-1]),
                terminal_writeoff_total_return=float((nav.iloc[-1]-daily.unquoted_value.iloc[-1])/nav.iloc[0]-1))
            rows.append(row);periods.extend(period_metrics(nav,frequency,fee));navs[key]=nav
            for trigger,g in ex.groupby('trigger'):
                cost_rows.append(dict(arm=frequency,fee=fee,trigger=trigger,action_dates=len(g),actual_trade_dates=int(g.trade_legs.gt(0).sum()),
                    buy_notional=float(g.buy_notional.sum()),sell_notional=float(g.sell_notional.sum()),paid_fees_initial_nav_units=float(g.fees.sum()),
                    annual_two_way_turnover=g.two_way_turnover.sum()/years,annual_fee_fraction=float((g.fees/g.pre_nav).sum()/years)))
            for year,g in nav.groupby(nav.index.year):
                loc=nav.index.get_loc(g.index[0]);annual.append(dict(arm=frequency,fee=fee,year=year,return_=g.iloc[-1]/nav.iloc[max(0,loc-1)]-1))
            print(f'{key}: CAGR {row["cagr"]:.2%}, DD {row["max_drawdown"]:.2%}, turnover {row["annual_two_way_turnover"]:.2f}x',flush=True)
    # Preserve costed own-pool reference, avoiding a new benchmark rebalanced to suit a winner.
    old_nav=pd.read_csv(parent/'nav.csv',index_col='date',parse_dates=True)
    for fee in q['fees']:
        nav=old_nav[f'COMMON_EQ_{int(round(fee*10000))}bp'].reindex(dates)
        assert nav.notna().all()
        navs[f'POOL_EQ_{int(round(fee*10000))}bp']=nav
        periods.extend(period_metrics(nav,'POOL_EQ',fee))
        for year,g in nav.groupby(nav.index.year):
            loc=nav.index.get_loc(g.index[0]);annual.append(dict(arm='POOL_EQ',fee=fee,year=year,return_=g.iloc[-1]/nav.iloc[max(0,loc-1)]-1))
    index_manifest=[]
    for code in q['index_benchmarks']:
        path=OUT/'market'/f'{code}.csv'
        if not path.exists():index_manifest.append(dict(code=code,status='UNAVAILABLE'));continue
        frame=pd.read_csv(path,dtype={'trade_date':str})
        frame.index=pd.to_datetime(frame.trade_date,format='%Y%m%d')
        if frame.index.has_duplicates:raise ValueError('Duplicate index dates')
        try:nav=index_nav(frame.close,dates)
        except ValueError:
            index_manifest.append(dict(code=code,status='INCOMPLETE',missing=int(frame.close.reindex(dates).isna().sum())));continue
        navs[code]=nav;periods.extend(period_metrics(nav,code,0))
        for year,g in nav.groupby(nav.index.year):
            loc=nav.index.get_loc(g.index[0]);annual.append(dict(arm=code,fee=0,year=year,return_=g.iloc[-1]/nav.iloc[max(0,loc-1)]-1))
        index_manifest.append(dict(code=code,status='COMPLETE',rows=len(frame),sha256=sha(path)))
    write_json(OUT/'index_validation.json',index_manifest)
    if not all(any(x['code']==c and x['status']=='COMPLETE' for x in index_manifest) for c in ['000300.SH','000016.SH']):
        raise ValueError('Required price benchmarks unavailable; portfolio checkpoints preserved')
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame(periods).to_csv(OUT/'subperiods.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(navs).to_csv(OUT/'nav.csv')
    pd.DataFrame(cost_rows).to_csv(OUT/'cost_attribution.csv',index=False)
    pd.DataFrame(parities).to_csv(OUT/'parent_parity.csv',index=False)
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    write_json(OUT/'validation.json',dict(status='PASS',weekly_reproduced=True,shared_bootstrap=True,weights_same=True,forward_unchanged=True,monthly_rebuilt=True))
    write_json(OUT/'completed.json',{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['completed.json','independent_validation.json']})
    print('FREQUENCY EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':run()
