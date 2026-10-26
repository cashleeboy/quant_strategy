#!/usr/bin/env python3
"""Independent cash/units/NAV reconciliation plus model timing and universe audit."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes
from scripts.run_fixed15_ml import OUT as TRAIN


def run(audit=False):
    OUT=ROOT/'output/fixed15_ml_20261024_audit' if audit else TRAIN
    for path,digest in json.loads((OUT/'frozen.json').read_text())['files'].items():
        assert sha(ROOT/path)==digest,path
    for path,digest in json.loads((OUT/'completed.json').read_text()).items():
        assert sha(OUT/path)==digest,path
    # Audit outputs retain rejected prefixes but never certify their full horizon.
    if audit:
        for path,digest in json.loads((TRAIN/'frozen.json').read_text())['files'].items():
            assert sha(ROOT/path)==digest,path
    states=json.loads((OUT/'status.json').read_text()) if audit else {}
    q=json.loads((TRAIN/'protocol.json').read_text())
    parent=ROOT/q['repaired_parent']
    z=np.load(ROOT/'output/mvp_holdings_count_20261017_complete_factors/prices.npz')
    full=pd.DataFrame(z['values'],index=pd.DatetimeIndex(z['dates']),columns=z['symbols'])
    prices=full.loc[q['start']:q['end']];marks=prices.ffill().fillna(0).to_numpy()
    names=prices.columns;pos={s:i for i,s in enumerate(names)}
    dead=pd.to_datetime(pd.read_csv(parent/'delist_dates.csv',index_col=0).iloc[:,0]).reindex(names)
    adj=pd.read_csv(parent/'event_adjustments.csv.gz',index_col='date',parse_dates=True)
    registry={e['event_id']:e for e in json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']}
    training=pd.read_csv(TRAIN/'training.csv',parse_dates=['refit_date','window_floor','train_start','train_end','latest_outcome_end'])
    ranking=pd.read_csv(TRAIN/'rankings.csv.gz',parse_dates=['date','refit_date'])
    diag=pd.read_csv(TRAIN/'diagnostics.csv',parse_dates=['date','refit_date'])
    coverage=pd.read_csv(TRAIN/'coverage.csv',parse_dates=['date']).set_index('date')
    plans=pd.DatetimeIndex(json.loads((ROOT/'output/candidate_factors_20261022/schedules.json').read_text())['WEEKLY'])
    expected_refits=pd.Series(plans,index=plans).groupby(plans.to_period('M')).min().tolist()
    for backend in ['lightgbm','catboost']:
        t=training[training.backend.eq(backend)]
        assert t.refit_date.tolist()==expected_refits
        assert t.latest_outcome_end.lt(t.refit_date).all()
        assert t.train_end.lt(t.refit_date).all() and t.train_start.ge(t.window_floor).all()
        assert t.train_dates.ge(q['minimum_train_dates']).all() and t.train_rows.ge(q['minimum_train_rows']).all()
        for row in t.itertuples():
            assert row.window_floor==full.index[full.index<=row.refit_date][-q['train_lookback_sessions']:].min()
        for area in q['areas']:
            d=diag[diag.backend.eq(backend)&diag.area.eq(area)].set_index('date')
            assert d.index.tolist()==plans.tolist()
            assert (d.refit_date.dt.to_period('M')==d.index.to_period('M')).all()
            if area=='BROAD':np.testing.assert_array_equal(d.candidates,coverage.loc[d.index,'eligible'])
            r=ranking[ranking.backend.eq(backend)&ranking.area.eq(area)]
            assert not r.duplicated(['date','symbol']).any()
            if area=='TECH15':
                assert r.industry.isin(['电子','计算机','通信','传媒','电力设备']).all()
        for folder in sorted((TRAIN/'training'/backend).iterdir()):
            params=json.loads((folder/'parameters.json').read_text())
            assert params['learning_rate']==.05
            if backend=='lightgbm':assert params['n_estimators']==80 and params['n_jobs']==2
            else:assert params['iterations']==80 and params['thread_count']==2 and params['allow_writing_files'] is False
    records=[];trade_count=0;action_count=0
    references=json.loads((OUT/'accounts.json').read_text());assert len(references)==12
    for key,path in references.items():
        folder=ROOT/path
        daily=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        complete=states.get(key,{'status':'COMPLETE'})['status']=='COMPLETE'
        if complete:assert daily.index.equals(prices.index)
        else:
            s=states[key]
            assert daily.index[-1]==pd.Timestamp(s['last_valid_date'])
            assert pd.Timestamp(s['failure_date'])>daily.index[-1]
            assert daily.index.equals(prices.index[:len(daily)])
        trades=pd.read_csv(folder/'trades.csv.gz',parse_dates=['date'])
        actions=pd.read_csv(folder/'asset_actions.csv.gz',parse_dates=['date','last_source_quote'])
        positions=pd.read_csv(folder/'positions.csv.gz',parse_dates=['date'])
        targets=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date','execution_date'])
        executions=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
        decisions=pd.read_csv(folder/'decisions.csv.gz',parse_dates=['date'])
        assert decisions.requested_k.isin([0,15]).all()
        assert decisions.selected.le(15).all()
        by_trade={d:g for d,g in trades.groupby('date')};by_action={d:g for d,g in actions.groupby('date')}
        by_position={d:g for d,g in positions.groupby('date')}
        fee=int(key.split('_')[-1].removesuffix('bp'))/10000
        units=np.zeros(len(names));cash=1.;lastnav=1.;maxerr=0.;seen=set()
        for i,date in enumerate(daily.index):
            mark=marks[i];gross=float(units@(mark-marks[i-1])) if i else 0.;conversion=0.
            for row in ([] if date not in by_action else by_action[date].itertuples()):
                assert row.event_id not in seen;seen.add(row.event_id)
                e=registry[row.event_id];assert date==pd.Timestamp(e['effective_date'])
                s,t=pos[row.source],pos[row.destination]
                assert (row.source,row.destination)==(e['source'],e['destination'])
                received=units[s]*adj.loc[row.last_source_quote,row.source]*e['ratio_numerator']/e['ratio_denominator']/adj.loc[date,row.destination]
                np.testing.assert_allclose(received,row.received_units,atol=1e-12)
                amount=received*mark[t]-units[s]*mark[s]
                np.testing.assert_allclose(amount,row.action_pnl,atol=1e-12)
                conversion+=amount;units[t]+=received;units[s]=0
            np.testing.assert_allclose(cash+units@mark,daily.loc[date,'pre_nav'],atol=1e-10)
            fees=0.
            for row in ([] if date not in by_trade else by_trade[date].itertuples()):
                j=pos[row.symbol];assert np.isfinite(prices.loc[date,row.symbol])
                np.testing.assert_allclose(row.price,prices.loc[date,row.symbol],atol=1e-12)
                np.testing.assert_allclose(units[j],row.shares_before,atol=1e-12)
                np.testing.assert_allclose(row.notional*fee,row.fee,atol=1e-12)
                units[j]+=row.notional/row.price*(1 if row.side=='BUY' else -1)
                np.testing.assert_allclose(units[j],row.shares_after,atol=1e-12)
                cash+=(-row.notional if row.side=='BUY' else row.notional)-row.fee;fees+=row.fee
            nav=float(cash+units@mark);maxerr=max(maxerr,abs(nav-daily.loc[date,'post_nav']))
            np.testing.assert_allclose(nav,daily.loc[date,'post_nav'],atol=1e-9)
            np.testing.assert_allclose(cash,daily.loc[date,'cash'],atol=1e-9)
            np.testing.assert_allclose(nav-lastnav,gross+conversion-fees,atol=1e-9)
            np.testing.assert_allclose(conversion,daily.loc[date,'action_pnl'],atol=1e-12)
            held=units*mark>1e-10
            assert not (held&dead.le(date).to_numpy()).any()
            recorded=by_position.get(date,pd.DataFrame(columns=['symbol','units']))
            assert set(recorded.symbol)==set(names[held])
            if len(recorded):np.testing.assert_allclose(recorded.units,units[[pos[s] for s in recorded.symbol]],atol=1e-12)
            lastnav=nav
        nextday=pd.Series(prices.index[1:],index=prices.index[:-1])
        assert (targets.date.map(nextday).to_numpy()==targets.execution_date.to_numpy()).all()
        assert (executions.signal_date.map(nextday).to_numpy()==executions.date.to_numpy()).all()
        assert cash>=-1e-10
        trade_count+=len(trades);action_count+=len(actions)
        records.append(dict(key=key,max_nav_error=maxerr,events=len(actions),trade_rows=len(trades),
                            full_horizon_complete=complete,last_verified_date=str(daily.index[-1].date())))
        print('Verified',key,flush=True)
    assert account_hashes()==json.loads((TRAIN/'forward_before.json').read_text())
    if audit:
        full_keys={k for k,s in states.items() if s['status']=='COMPLETE'}
        assert set(pd.read_csv(OUT/'comparison.csv').key)==full_keys
    write_json(OUT/'independent_validation.json',dict(status='PASS',scope='Full accounts plus explicitly rejected prefixes; not a complete comparison when blocked',accounts=len(records),
        complete_accounts=sum(r['full_horizon_complete'] for r in records),blocked_accounts=sum(not r['full_horizon_complete'] for r in records),
        actual_backend_refits=len(training),weekly_predictions=len(diag),trade_rows=trade_count,
        asset_actions=action_count,paper_accounts_unchanged=True,records=records))
    print('PASS',len(records),'accounts',len(training),'real model refits',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');run(parser.parse_args().audit)
