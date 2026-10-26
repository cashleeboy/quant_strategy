#!/usr/bin/env python3
"""Independently replay fills AND corporate actions into shares and cash."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes

OUT=ROOT/'output/asset_transition_fix_20261024'


def run():
    for path,h in json.loads((OUT/'frozen.json').read_text())['files'].items():assert sha(ROOT/path)==h,path
    for path,h in json.loads((OUT/'completed.json').read_text()).items():assert sha(OUT/path)==h,path
    registry={e['event_id']:e for e in json.loads((ROOT/'data/research_corporate_actions_v1.json').read_text())['events']}
    z=np.load(ROOT/'output/mvp_holdings_count_20261017_complete_factors/prices.npz')
    prices=pd.DataFrame(z['values'],index=pd.DatetimeIndex(z['dates']),columns=z['symbols']).loc['2012-01-06':'2026-10-15']
    marks=prices.ffill().fillna(0)
    adj=pd.read_csv(OUT/'event_adjustments.csv.gz',index_col='date',parse_dates=True)
    dead=pd.read_csv(OUT/'delist_dates.csv',index_col=0).iloc[:,0];dead=pd.to_datetime(dead)
    names=prices.columns;pos={s:i for i,s in enumerate(names)};markvalues=marks.to_numpy()
    records=[];action_count=0;trade_count=0
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'daily.csv.gz').exists():continue
        daily=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        trade=pd.read_csv(folder/'trades.csv.gz',parse_dates=['date']);acts=pd.read_csv(folder/'asset_actions.csv.gz',parse_dates=['date','last_source_quote'])
        positions=pd.read_csv(folder/'positions.csv.gz',parse_dates=['date']);targets=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date','execution_date'])
        ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
        by_trade={d:g for d,g in trade.groupby('date')};by_action={d:g for d,g in acts.groupby('date')}
        by_position={d:g for d,g in positions.groupby('date')}
        fee=int(folder.name.split('_')[-1].removesuffix('bp'))/10000
        shares=np.zeros(len(names));cash=1.;maxerr=0.;priornav=1.;seen=set()
        for i,d in enumerate(prices.index):
            mark=markvalues[i]
            gross=float(shares@(mark-markvalues[i-1])) if i else 0.
            conversion=0.
            for row in ([] if d not in by_action else by_action[d].itertuples()):
                assert row.event_id not in seen;seen.add(row.event_id)
                event=registry[row.event_id];assert d==pd.Timestamp(event['effective_date'])
                s,t=pos[row.source],pos[row.destination]
                assert row.source==event['source'] and row.destination==event['destination']
                np.testing.assert_allclose(shares[s],row.source_units,atol=1e-12)
                ratio=float(event['ratio_numerator'])/float(event['ratio_denominator'])
                oldf=adj.loc[row.last_source_quote,row.source];newf=adj.loc[d,row.destination]
                newunits=shares[s]*oldf*ratio/newf
                np.testing.assert_allclose(newunits,row.received_units,atol=1e-12)
                np.testing.assert_allclose(row.received_raw_shares,shares[s]*oldf*ratio,atol=1e-12)
                np.testing.assert_allclose(row.destination_units_before,shares[t],atol=1e-12)
                amount=newunits*mark[t]-shares[s]*mark[s]
                np.testing.assert_allclose(amount,row.action_pnl,atol=1e-12)
                conversion+=amount;shares[t]+=newunits;shares[s]=0
                assert row.fee==0 and row.cash_change==0
            np.testing.assert_allclose(cash+shares@mark,daily.loc[d,'pre_nav'],atol=1e-10)
            paid=0.
            for row in ([] if d not in by_trade else by_trade[d].itertuples()):
                j=pos[row.symbol];assert np.isfinite(prices.loc[d,row.symbol])
                np.testing.assert_allclose(shares[j],row.shares_before,atol=1e-12)
                np.testing.assert_allclose(row.fee,row.notional*fee,atol=1e-12)
                delta=row.notional/row.price*(1 if row.side=='BUY' else -1)
                shares[j]+=delta
                np.testing.assert_allclose(shares[j],row.shares_after,atol=1e-12)
                cash+=(-row.notional if row.side=='BUY' else row.notional)-row.fee;paid+=row.fee
            nav=float(cash+shares@mark);maxerr=max(maxerr,abs(nav-daily.loc[d,'post_nav']))
            np.testing.assert_allclose(nav,daily.loc[d,'post_nav'],atol=1e-9)
            np.testing.assert_allclose(cash,daily.loc[d,'cash'],atol=1e-9)
            np.testing.assert_allclose(nav-priornav,gross+conversion-paid,atol=1e-10)
            np.testing.assert_allclose(conversion,daily.loc[d,'action_pnl'],atol=1e-12)
            held=shares*mark>1e-10
            assert not (held & dead.reindex(names).le(d).to_numpy()).any()
            g=by_position.get(d,pd.DataFrame(columns=['symbol','units','value']))
            assert set(g.symbol)==set(names[held])
            if len(g):np.testing.assert_allclose(g.units,shares[[pos[s] for s in g.symbol]],atol=1e-12)
            priornav=nav
        nextday=pd.Series(prices.index[1:],index=prices.index[:-1])
        assert (targets.date.map(nextday).to_numpy()==targets.execution_date.to_numpy()).all()
        assert (ex.signal_date.map(nextday).to_numpy()==ex.date.to_numpy()).all()
        assert not acts.event_id.duplicated().any()
        action_count+=len(acts);trade_count+=len(trade)
        records.append(dict(key=folder.name,max_nav_error=maxerr,events=len(acts),terminal_unquoted_fraction=float(daily.unquoted_value.iloc[-1]/daily.post_nav.iloc[-1])))
        print('Verified shares/cash/actions',folder.name,flush=True)
    assert len(records)==20
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    parity=pd.read_csv(OUT/'legacy_parity.csv');assert parity.pre_action_max_nav_error.max()<1e-10
    result=dict(status='PASS',accounts=len(records),actions=action_count,trade_rows=trade_count,
                no_event_full_parity=int(parity.no_event_full_parity.sum()),paper_accounts_unchanged=True,
                accounts_reconciliation=records)
    write_json(OUT/'independent_validation.json',result)
    print('PASS',len(records),'accounts',action_count,'corporate actions',trade_count,'trades')

if __name__=='__main__':run()
