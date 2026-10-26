#!/usr/bin/env python3
"""Isolated long-history beta research; never writes production configuration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from analysis.beta_research import rolling_market_sensitivity, execution_return, market_regression, controlled_beta
from analysis.holdings_count import metrics
from backtest.research_replay import replay
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.run_six_strategy_ic_comparison import load_prices, verify

PROTOCOL = ROOT/'research_protocols/20261019_beta_research_v1.json'
OUTPUT = ROOT/'output/beta_research_20261019'


def forward_hashes():
    base = ROOT/'output/six_strategy_forward_20260928'
    return {str(f.relative_to(base)): sha(f) for folder in ['initial','plans','days']
            for f in sorted((base/folder).rglob('*')) if f.is_file()}


def fetch(out):
    import tushare as ts
    p = json.loads(PROTOCOL.read_text()); out.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(os.environ.get('TUSHARE_TOKEN'))
    if pro is None:
        raise RuntimeError('Tushare credentials unavailable')
    directory = out/'market'; directory.mkdir(exist_ok=True)
    for code in [p['market'], p['market_control']]:
        frames = []
        for year in [2010, 2014, 2018, 2022, 2026]:
            dest = directory/f'{code}_{year}.csv'
            lo, hi = f'{year}0101', min(f'{year+3}1231', '20261015')
            if not dest.exists():
                f = pro.index_daily(ts_code=code, start_date=lo, end_date=hi)
                if f.empty or len(f) >= 3000 or f.ts_code.ne(code).any():
                    raise ValueError('invalid/truncated index response')
                f.to_csv(dest, index=False)
                write_json(dest.with_suffix('.json'), {'endpoint':'index_daily','ts_code':code,
                    'start_date':lo,'end_date':hi,'rows':len(f),'sha256':sha(dest),
                    'retrieved_at':datetime.now(timezone.utc).isoformat()})
            if sha(dest) != json.loads(dest.with_suffix('.json').read_text())['sha256']:
                raise ValueError('index cache corrupted')
            frames.append(pd.read_csv(dest, dtype={'trade_date':str}))
            print(f'index {code} {year}: {len(frames[-1])}', flush=True)
        f = pd.concat(frames).sort_values('trade_date')
        if f.trade_date.duplicated().any() or f.close.le(0).any():
            raise ValueError('invalid index dates/prices')
        f.to_csv(directory/f'{code}.csv', index=False)


def calculate(out):
    p = json.loads(PROTOCOL.read_text()); prior = ROOT/p['prior_replay']
    if (out/'completed.json').exists():
        raise ValueError('completed output exists; do not overwrite research')
    out.mkdir(parents=True, exist_ok=True)
    before = forward_hashes(); write_json(out/'forward_before.json', before)
    write_json(out/'protocol.json', p)
    prices = load_prices(p)
    markets = {}
    for code in [p['market'],p['market_control']]:
        f = pd.read_csv(out/'market'/f'{code}.csv', dtype={'trade_date':str})
        f.index = pd.to_datetime(f.trade_date, format='%Y%m%d')
        close = f.close.reindex(prices.index)
        if close.isna().any():
            raise ValueError('market does not cover full stock calendar')
        markets[code] = close.pct_change(fill_method=None)
    inputs, ranks, hashes = [], [], {}
    cols = ['date','symbol','eligible','industry_l1','total_mv_yuan']
    for directory in sorted(prior.glob('inputs_*')):
        verify(directory)
        inputs.append(pd.read_csv(directory/'universe.csv.gz', usecols=cols, parse_dates=['date']))
        ranks.append(pd.read_csv(directory/'ranked_top100.csv.gz', parse_dates=['date']))
        hashes[str(directory.relative_to(ROOT))] = sha(directory/'complete.json')
    universe = pd.concat(inputs, ignore_index=True)
    universe = universe[universe.eligible].copy()
    ranks = pd.concat(ranks, ignore_index=True)
    dates = pd.DatetimeIndex(sorted(universe.date.unique()))
    stamp = {'protocol':sha(PROTOCOL), 'prices':sha(prior/'prices.npz'), 'prior_nav':sha(prior/'nav.csv'),
             'inputs':hashes,'market':{c:sha(out/'market'/f'{c}.csv') for c in markets},
             'code':{n:sha(ROOT/n) for n in ['analysis/beta_research.py','scripts/run_beta_research.py',
                      'backtest/research_replay.py','analysis/holdings_count.py']}}
    write_json(out/'frozen.json', stamp)
    estimates = {}
    for key, code, window, minimum in [('main',p['market'],252,200),('short',p['market'],126,100),('control',p['market_control'],252,200)]:
        bparts, vparts, nparts = [], [], []
        for left in range(0, prices.shape[1], 256):
            b,v,n = rolling_market_sensitivity(prices.iloc[:,left:left+256], markets[code], window, minimum)
            bparts.append(b.reindex(dates)); vparts.append(v.reindex(dates)); nparts.append(n.reindex(dates))
        estimates[key] = (pd.concat(bparts,axis=1), pd.concat(vparts,axis=1), pd.concat(nparts,axis=1))
        print(f'estimated {key}: {len(dates)} signals', flush=True)
    vol20 = prices.pct_change(fill_method=None).rolling(20,min_periods=15).std().reindex(dates)*np.sqrt(252)
    base_groups = {d:g.set_index('symbol') for d,g in universe.groupby('date')}
    portfolios = {v:{} for v in p['variants']}
    coverage, signal_rows, ic_rows, exposure_rows, composition = [], [], [], [], []
    for d in dates:
        g = base_groups[d].copy()
        g['beta'] = estimates['main'][0].loc[d].reindex(g.index)
        g['vol252'] = estimates['main'][1].loc[d].reindex(g.index)
        g['paired_days'] = estimates['main'][2].loc[d].reindex(g.index)
        g['beta126'] = estimates['short'][0].loc[d].reindex(g.index)
        g['beta_csi300'] = estimates['control'][0].loc[d].reindex(g.index)
        g['vol20'] = vol20.loc[d].reindex(g.index)
        good = g.dropna(subset=['beta','vol252','beta126','beta_csi300','vol20']).sort_index()
        coverage.append(dict(date=d,base_names=len(g),common_names=len(good),
                             included=len(good)>=p['minimum_common_names']))
        if len(good)<p['minimum_common_names']:
            continue
        def sort_names(col):
            return good.sort_values(col,kind='stable').index
        ordered = sort_names('beta')
        portfolios['BASE_ELIGIBLE'][d] = list(g.index)
        portfolios['COMMON'][d] = list(good.index)
        for q, names in enumerate(np.array_split(ordered.to_numpy(),5),1):
            portfolios[f'BETA_Q{q}'][d] = list(names)
        for name,col in [('LOW_BETA50','beta'),('LOW_VOL252_50','vol252'),('LOW_VOL20_50','vol20'),
                         ('LOW_BETA126_50','beta126'),('LOW_BETA_CSI300_50','beta_csi300')]:
            portfolios[name][d] = list(sort_names(col)[:p['k']])
        good['controlled_beta'] = controlled_beta(good)
        low = set(portfolios['LOW_BETA50'][d])
        signal_rows.append(dict(date=d,beta_vol_rank_corr=good.beta.corr(good.vol252,method='spearman'),
             overlap_vol252=len(low & set(portfolios['LOW_VOL252_50'][d]))/p['k'],
             overlap_vol20=len(low & set(portfolios['LOW_VOL20_50'][d]))/p['k'],
             overlap_short=len(low & set(portfolios['LOW_BETA126_50'][d]))/p['k'],
             overlap_market=len(low & set(portfolios['LOW_BETA_CSI300_50'][d]))/p['k'],
             controlled_coverage=good.controlled_beta.notna().mean()))
        for horizon in [5,20]:
            future = execution_return(prices,d,horizon).reindex(good.index)
            for factor in ['beta','vol252','vol20','controlled_beta']:
                score = -good[factor]
                valid = score.notna() & future.notna()
                value = score[valid].corr(future[valid],method='spearman') if valid.sum()>=30 and future[valid].nunique()>1 else np.nan
                ic_rows.append(dict(date=d,horizon=horizon,factor=factor,rank_ic=value,names=int(valid.sum())))
        for name in p['variants']:
            selected = portfolios[name][d]
            x = g.reindex(selected)
            counts = x.industry_l1.fillna('UNKNOWN').replace('','UNKNOWN').value_counts()/len(x)
            exposure_rows.append(dict(date=d,variant=name,names=len(x),target_beta=x.beta.mean(),
                beta_weight_coverage=x.beta.notna().mean(),target_vol252=x.vol252.mean(),
                median_market_cap=x.total_mv_yuan.median(),max_industry_weight=counts.max(),
                unknown_industry_weight=counts.get('UNKNOWN',0)))
            for industry, weight in counts.items():
                composition.append(dict(date=d,variant=name,industry=industry,weight=weight))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(signal_rows).to_csv(out/'signal_diagnostics.csv',index=False)
    pd.DataFrame(ic_rows).to_csv(out/'ic.csv',index=False)
    pd.DataFrame(exposure_rows).to_csv(out/'target_exposure.csv',index=False)
    pd.DataFrame(composition).to_csv(out/'industry_weights.csv',index=False)
    # Save point-in-time beta estimates for reproducible six-strategy risk checks.
    estimates['main'][0].to_csv(out/'beta252_weekly.csv.gz',index_label='date')
    signals = pd.DatetimeIndex(sorted(portfolios['COMMON']))
    if len(signals)<100:
        raise ValueError('insufficient signals')
    prices_eval = prices.loc[signals[0]:p['end']]
    execution = prices.index[prices.index.get_indexer(signals)+1]
    schedule = pd.DataFrame({'execution_date':execution},index=signals)
    navs, performance, subperiods, audits = {}, [], [], []
    for variant, targets in portfolios.items():
        def factory(d, held):
            names = targets[d]
            return pd.Series(1/len(names),index=names)
        main = None
        for fee in p['fees']:
            result = replay(prices_eval,schedule,factory,fee=fee)
            daily, log = result['daily'], result['executions']
            nav = daily.post_nav
            tag = variant if fee==p['fees'][0] else variant+'__STRESS'
            navs[tag] = nav
            m = {k:float(v[0]) for k,v in metrics(nav).items()}
            m.update(market_regression(nav.pct_change(fill_method=None),markets[p['market']]))
            performance.append(dict(variant=variant,fee=fee,**m,
                annual_two_way_turnover=log.two_way_turnover.sum()*252/(len(nav)-1),
                mean_cash_fraction=log.cash_fraction.mean(),mean_actual_holdings=log.holdings.mean(),
                terminal_unquoted_fraction=daily.unquoted_value.iloc[-1]/nav.iloc[-1],
                terminal_zero_unquoted_total_return=(nav.iloc[-1]-daily.unquoted_value.iloc[-1])/nav.iloc[0]-1))
            for lo,hi in p['subperiods']:
                ix=np.flatnonzero((nav.index>=lo)&(nav.index<=hi)); a=max(0,ix[0]-1); b=ix[-1]+1
                subperiods.append(dict(variant=variant,fee=fee,period=lo[:4]+'-'+hi[:4],
                    **{k:float(v[0]) for k,v in metrics(nav.iloc[a:b]).items()},
                    **market_regression(nav.iloc[a:b].pct_change(fill_method=None),markets[p['market']])))
            if fee==p['fees'][0]:
                main=result
                log.assign(variant=variant).to_csv(out/f'executions_{variant}.csv',index=False)
                result['targets'].to_csv(out/f'targets_{variant}.csv.gz',index=False)
                audits.append(dict(variant=variant,executions=len(log),trades=len(result['trades']),
                    missing_targets=int(log.missing_targets.sum()),blocked_exit_events=int(log.blocked_exit_value.gt(0).sum()),
                    all_missing_hold_events=int(log.all_targets_missing_hold.sum())))
            del result
        print(f'replayed {variant}',flush=True)
        del main
    nav = pd.DataFrame(navs);nav.to_csv(out/'nav.csv',index_label='date')
    pd.DataFrame(performance).to_csv(out/'performance.csv',index=False)
    pd.DataFrame(subperiods).to_csv(out/'subperiods.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'execution_checks.csv',index=False)
    # Descriptive IC summaries; overlapping labels do not get naive t-statistics.
    ic=pd.DataFrame(ic_rows);summary=[]
    for (factor,horizon),g in ic.groupby(['factor','horizon']):
        v=g.rank_ic.dropna()
        summary.append(dict(factor=factor,horizon=horizon,mean_ic=v.mean(),median_ic=v.median(),
                            positive_fraction=v.gt(0).mean(),observations=len(v),mean_names=g.names.mean()))
    pd.DataFrame(summary).to_csv(out/'ic_summary.csv',index=False)
    # Six unchanged FIXED K50 histories: target snapshots and realized NAV beta.
    oldnav=pd.read_csv(prior/'nav.csv',index_col='date',parse_dates=True)
    six, weekly, rolling = [], [], {}
    b=estimates['main'][0]
    for strategy,g in ranks[ranks['rank'].le(50)].groupby('strategy'):
        s=oldnav[strategy+'_K50']
        s=s.reindex(nav.index)
        if s.isna().any():raise ValueError('missing six-strategy NAV')
        r=s.pct_change(fill_method=None)
        real,_,_=rolling_market_sensitivity(s.to_frame(),markets[p['market']].reindex(s.index),252,200)
        rolling[strategy]=real.iloc[:,0]
        w=[]
        for d,gs in g.groupby('date'):
            if d not in signals:continue
            bs=b.loc[d].reindex(gs.symbol)
            row=dict(strategy=strategy,date=d,target_beta_known_contribution=float(bs.sum()/len(bs)),
                     target_beta_known_normalized=float(bs.mean()),known_weight=float(bs.notna().mean()))
            w.append(row);weekly.append(row)
        reg=market_regression(r,markets[p['market']])
        six.append(dict(strategy=strategy,**reg,
            **{k:float(v[0]) for k,v in metrics(s).items()},
            mean_target_beta=pd.DataFrame(w).target_beta_known_normalized.mean(),
            mean_beta_known_weight=pd.DataFrame(w).known_weight.mean()))
    pd.DataFrame(six).to_csv(out/'six_strategy_risk.csv',index=False)
    pd.DataFrame(weekly).to_csv(out/'six_strategy_target_beta.csv',index=False)
    pd.DataFrame(rolling).to_csv(out/'six_strategy_rolling_beta.csv',index_label='date')
    # Unit-cash control: long common-universe equity scaled to low-beta target;
    # no leverage, to help distinguish stock selection from simply less equity.
    exposures=pd.DataFrame(exposure_rows).pivot(index='date',columns='variant',values='target_beta')
    scales=(exposures.LOW_BETA50/exposures.COMMON).clip(lower=0,upper=1)
    if not np.isfinite(scales).all():raise ValueError('invalid cash control')
    def cash_factory(d,held):
        names=portfolios['COMMON'][d];scale=float(scales.loc[d])
        return pd.Series(scale/len(names),index=names) if scale>0 else pd.Series(dtype=float)
    cash_result=replay(prices_eval,schedule,cash_factory,fee=p['fees'][0])
    cash_result['daily'].to_csv(out/'cash_control_nav.csv',index_label='date')
    scales.to_csv(out/'cash_control_stock_budget.csv',header=['stock_budget'])
    # Above is a descriptive risk-matching control, not exact daily beta neutrality.
    after=forward_hashes()
    if before!=after:raise AssertionError('forward accounts changed')
    write_json(out/'validation.json',dict(status='PASS',forward_unchanged=True,
        market_calendar_complete=True,price_checksum_verified=True,year_inputs_verified=True,
        signal_count=len(signals),first_signal=str(signals[0].date()),first_execution=str(execution[0].date()),
        last_date=str(nav.index[-1].date()),stock_count=prices.shape[1],
        cash_reconciliations='asserted at each execution by research_replay',
        cash_control='supplemental pre-result risk-matching control; stock budget clipped to [0,1]'))
    files={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='completed.json'}
    write_json(out/'completed.json',dict(status='COMPLETE',files=files,finished=datetime.now(timezone.utc).isoformat()))
    print('BETA RESEARCH COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['fetch','run'],required=True)
    ap.add_argument('--output',type=Path,default=OUTPUT);a=ap.parse_args()
    (fetch if a.stage=='fetch' else calculate)(a.output)
