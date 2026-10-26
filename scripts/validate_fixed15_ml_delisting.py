#!/usr/bin/env python3
"""Reconstruct shares and cash independently, including both equity classifications."""
from pathlib import Path
import json,sys
from decimal import Decimal,ROUND_HALF_UP
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from scripts.run_fixed15_ml_delisting import OUT
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes


def run():
    for path,h in json.loads((OUT/'frozen.json').read_text())['files'].items():assert sha(ROOT/path)==h,path
    for path,h in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/path)==h,path
    q=json.loads((OUT/'protocol.json').read_text());e=json.loads((ROOT/q['registry']).read_text())['events'][0]
    z=np.load(ROOT/'output/mvp_holdings_count_20261017_complete_factors/prices.npz')
    prices=pd.DataFrame(z['values'],index=pd.DatetimeIndex(z['dates']),columns=z['symbols']).loc[q['start']:q['end']]
    prices[e['claim']]=np.nan
    bars=pd.read_csv(ROOT/q['market_data']/'400236_OC_daily_raw.csv',index_col='date',parse_dates=True)
    prices[e['destination']]=bars.close.reindex(prices.index)
    marks=prices.ffill().fillna(0);marks.loc[e['delist_date']:,e['claim']]=e['last_raw_close']
    adj=pd.read_csv(OUT/'adjustments.csv.gz',index_col='date',parse_dates=True)
    old_registry={a['event_id']:a for a in json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']}
    names=prices.columns;lookup={s:i for i,s in enumerate(names)};px=prices.to_numpy();mv=marks.to_numpy()
    registry_dead=pd.to_datetime(pd.read_csv(ROOT/'output/asset_transition_fix_20261024/delist_dates.csv',index_col=0).iloc[:,0])
    dead=registry_dead.reindex(names);dead.loc[e['claim']]=pd.Timestamp(e['listing_date'])
    results=[];exitrows=[];alltrades=0;allactions=0
    for key,path in json.loads((OUT/'accounts.json').read_text()).items():
        folder=ROOT/path;fee=int(key.split('_')[-1].removesuffix('bp'))/10000
        daily=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        trades=pd.read_csv(folder/'trades.csv.gz',parse_dates=['date'])
        acts=pd.read_csv(folder/'asset_actions.csv.gz',parse_dates=['date','last_source_quote'])
        positions=pd.read_csv(folder/'positions.csv.gz',parse_dates=['date'])
        targets=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date','execution_date'])
        ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
        assert daily.index.equals(prices.index)
        tgroup={d:g for d,g in trades.groupby('date')};agroup={d:g for d,g in acts.groupby('date')};pgroup={d:g for d,g in positions.groupby('date')}
        units=np.zeros(len(names));cash=1.;lastnav=1.;seen=set();error=0.
        for i,date in enumerate(prices.index):
            mark=mv[i];gross=float(units@(mark-mv[i-1])) if i else 0.;action_pnl=0.
            for a in ([] if date not in agroup else agroup[date].itertuples()):
                assert a.event_id not in seen;seen.add(a.event_id)
                source,destination=a.source,a.destination;s,t=lookup[source],lookup[destination]
                if a.kind=='stock_swap':
                    r=old_registry[a.event_id];assert date==pd.Timestamp(r['effective_date'])
                    assert (source,destination)==(r['source'],r['destination'])
                    f=adj.loc[a.last_source_quote,source];g=adj.loc[date,destination]
                    ratio=r['ratio_numerator']/r['ratio_denominator'];received=units[s]*f*ratio/g
                elif a.kind=='delist_to_claim':
                    assert date==pd.Timestamp(e['delist_date']) and (source,destination)==(e['source'],e['claim'])
                    f=adj.loc[e['last_source_quote'],source];received=units[s]*f
                    np.testing.assert_allclose(mark[s]/f,mark[t],atol=1e-10)
                elif a.kind=='claim_to_otc':
                    assert date==pd.Timestamp(e['listing_date']) and (source,destination)==(e['claim'],e['destination'])
                    assert pd.Timestamp(q['confirmations'][e['event_id']])<=pd.Timestamp(e['confirmation_deadline'])
                    received=units[s]
                else:raise AssertionError('unexpected action')
                np.testing.assert_allclose(units[s],a.source_units,atol=1e-12)
                np.testing.assert_allclose(received,a.received_units,atol=1e-12)
                change=received*mark[t]-units[s]*mark[s]
                np.testing.assert_allclose(change,a.action_pnl,atol=1e-12)
                assert a.fee==0 and a.cash_change==0
                units[s]=0.;units[t]+=received;action_pnl+=change
            np.testing.assert_allclose(cash+units@mark,daily.loc[date,'pre_nav'],atol=1e-9)
            fees=0.
            for a in ([] if date not in tgroup else tgroup[date].itertuples()):
                j=lookup[a.symbol];assert np.isfinite(px[i,j]) and a.symbol!=e['claim']
                np.testing.assert_allclose(units[j],a.shares_before,atol=1e-12)
                np.testing.assert_allclose(px[i,j],a.price,atol=1e-12)
                np.testing.assert_allclose(a.notional*fee,a.fee,atol=1e-12)
                if a.symbol==e['destination']:
                    assert a.side=='SELL' and date>=pd.Timestamp(e['listing_date'])
                    bar=bars.loc[date];assert bar.volume_shares>0 and bar.amount_yuan>0
                    previous=e['last_raw_close'] if date==bars.index[0] else bars.close.iloc[bars.index.get_loc(date)-1]
                    limit=float((Decimal(str(previous))*.95 if False else Decimal(str(previous))*Decimal('.95')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
                    limit=max(.01,min(limit,previous-.01))
                    if date==bars.index[0]:limit=e['first_day_low']
                    assert not (np.isclose(bar.high,bar.low) and bar.close<=limit+1e-10)
                    exitrows.append(dict(key=key,date=str(date.date()),price=a.price,raw_shares=a.notional/a.price,notional=a.notional,fee=a.fee,
                                         market_volume=bar.volume_shares,market_amount=bar.amount_yuan))
                units[j]+=a.notional/a.price*(1 if a.side=='BUY' else -1)
                np.testing.assert_allclose(units[j],a.shares_after,atol=1e-12)
                cash+=(-a.notional if a.side=='BUY' else a.notional)-a.fee;fees+=a.fee
            nav=float(cash+units@mark);error=max(error,abs(nav-daily.loc[date,'post_nav']))
            np.testing.assert_allclose(nav,daily.loc[date,'post_nav'],atol=1e-9)
            np.testing.assert_allclose(cash,daily.loc[date,'cash'],atol=1e-9)
            np.testing.assert_allclose(nav-lastnav,gross+action_pnl-fees,atol=1e-9)
            np.testing.assert_allclose(action_pnl,daily.loc[date,'action_pnl'],atol=1e-12)
            np.testing.assert_allclose(units[lookup[e['claim']]]*mark[lookup[e['claim']]],daily.loc[date,'locked_equity_value'],atol=1e-12)
            held=units*mark>1e-10;assert not (held&dead.le(date).to_numpy()).any()
            p=pgroup.get(date,pd.DataFrame(columns=['symbol','units']));assert set(p.symbol)==set(names[held])
            if len(p):np.testing.assert_allclose(p.units,units[[lookup[s] for s in p.symbol]],atol=1e-12)
            lastnav=nav
        assert not targets.symbol.isin([e['claim'],e['destination']]).any()
        nextday=pd.Series(prices.index[1:],index=prices.index[:-1])
        assert (targets.date.map(nextday).to_numpy()==targets.execution_date.to_numpy()).all()
        assert (ex.signal_date.map(nextday).to_numpy()==ex.date.to_numpy()).all()
        assert abs(units[lookup[e['source']]])<1e-10 and abs(units[lookup[e['claim']]])<1e-10 and abs(units[lookup[e['destination']]])<1e-10
        results.append(dict(key=key,max_nav_error=error,actions=len(acts),trades=len(trades)))
        alltrades+=len(trades);allactions+=len(acts);print('Verified',key,flush=True)
    assert len(results)==12 and len(exitrows)==8
    parity=pd.read_csv(OUT/'parity.csv');assert parity.max_nav_error.max()<1e-10
    assert account_hashes()==json.loads((ROOT/q['score_parent']/'forward_before.json').read_text())
    pd.DataFrame(exitrows).to_csv(OUT/'verified_otc_exits.csv',index=False)
    write_json(OUT/'independent_validation.json',dict(status='PASS',accounts=len(results),trade_rows=alltrades,asset_actions=allactions,
        otc_exits=len(exitrows),baseline_full_parity=4,ml_prefix_parity=8,paper_accounts_unchanged=True,records=results))
    print('PASS complete12 accounts',flush=True)


if __name__=='__main__':run()
