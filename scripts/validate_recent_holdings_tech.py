#!/usr/bin/env python3
"""Read-only independent reconciliation of the new diagnostic outputs."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_mvp_holdings_count import sha,write_json
from scripts.run_single_account_adaptive import account_hashes
from universe.strategy import point_in_time_industry

OUT=ROOT/'output/recent_holdings_tech_20261023'

def run():
    for name,digest in json.loads((OUT/'frozen.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in json.loads((OUT/'completed.json').read_text()).items():
        assert sha(OUT/name)==digest,name
    q=json.loads((OUT/'protocol.json').read_text())
    ranks=pd.read_csv(OUT/'rankings.csv.gz',parse_dates=['date','weight_update'])
    assert ranks.weight_update.le(ranks.date).all()
    assert ranks.industry.isin(q['technology_definition']).all()
    membership=pd.read_csv(ROOT/'data/all_a_history_20100101_20261015/index_member_all.csv.gz')
    unique=ranks.drop_duplicates(['date','symbol']).set_index(['date','symbol'])
    independent=point_in_time_industry(unique.index,membership)
    assert (independent.to_numpy()==unique.industry.to_numpy()).all()
    lookups={(a,d):g.set_index('symbol') for (a,d),g in ranks.groupby(['arm','date'])}
    selection_checks=0;trade_checks=0;accounts=0
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'daily.csv.gz').exists():continue
        d=pd.read_csv(folder/'daily.csv.gz',index_col='date',parse_dates=True)
        ex=pd.read_csv(folder/'executions.csv.gz',parse_dates=['date','signal_date'])
        t=pd.read_csv(folder/'trades.csv.gz',parse_dates=['date'])
        target=pd.read_csv(folder/'targets.csv.gz',parse_dates=['date','execution_date'])
        fee=int(folder.name.split('_')[-1].removesuffix('bp'))/10000
        np.testing.assert_allclose(t.fee,t.notional*fee,atol=1e-12)
        fills=t.groupby('date').agg(fees=('fee','sum'),notional=('notional','sum')).reindex(ex.date,fill_value=0)
        np.testing.assert_allclose(ex.fees,fills.fees,atol=1e-10)
        np.testing.assert_allclose(ex.buy_notional+ex.sell_notional,fills.notional,atol=1e-10)
        np.testing.assert_allclose(ex.pre_nav-ex.post_nav,ex.fees,atol=1e-10)
        assert d.cash.ge(-1e-9).all()
        next_day=pd.Series(d.index[1:],index=d.index[:-1])
        assert (ex.signal_date.map(next_day).to_numpy()==ex.date.to_numpy()).all()
        if len(target):
            assert target.weight.gt(0).all()
            assert target.groupby('date').weight.sum().le(1+1e-10).all()
            assert (target.date.map(next_day).to_numpy()==target.execution_date.to_numpy()).all()
        if 'TECH15' in folder.name or 'TECH_LITERAL' in folder.name:
            a=folder.name.split('_')[0]
            log=pd.read_csv(folder/'decisions.csv.gz',parse_dates=['date']).set_index('date')
            assert log.budget.between(.2-1e-10,1+1e-10).all()
            cap=5 if 'LITERAL' in folder.name else 15
            assert log.selected.le(cap).all()
            sums=target.groupby('date').weight.sum().reindex(log.index,fill_value=0)
            np.testing.assert_allclose(sums.loc[log.action],log.loc[log.action,'target_equity'],atol=1e-10)
            for date,g in target.groupby('date'):
                np.testing.assert_allclose(g.weight,log.loc[date,'budget']/15,atol=1e-10)
                if not log.loc[date,'weekly']:continue
                eligible=lookups[a,date]
                assert g.symbol.isin(eligible.index).all()
                counts=eligible.loc[g.symbol,'industry'].value_counts()
                assert counts.max()<=3
                selection_checks+=1
        trade_checks+=len(t);accounts+=1
    assert accounts==22,accounts
    checks=pd.read_csv(OUT/'attribution_checks.csv')
    assert len(checks)==56 and checks.attribution_error.max()<1e-9
    for folder in sorted(OUT.glob('holdings_*')):
        key=folder.name.removeprefix('holdings_')
        ledger=pd.read_csv(folder/'daily_holdings.csv.gz',parse_dates=['date'])
        daily=pd.read_csv(OUT/key/'daily.csv.gz',index_col='date',parse_dates=True)
        sums=ledger.groupby('date').agg(net=('net_pnl','sum'),value=('value','sum'),prior=('prior_weight','sum'))
        np.testing.assert_allclose(sums.net,daily.post_nav.diff().reindex(sums.index),atol=1e-10)
        np.testing.assert_allclose(sums.value+daily.cash.reindex(sums.index),daily.post_nav.reindex(sums.index),atol=1e-10)
        assert sums.prior.between(-1e-10,1+1e-8).all()
        for f in folder.glob('*_stocks.csv'):
            x=pd.read_csv(f)
            assert np.isfinite(x.contribution).all()
    assert account_hashes()==json.loads((OUT/'forward_before.json').read_text())
    result=dict(status='PASS',accounts=accounts,trade_rows=trade_checks,technology_selection_checks=selection_checks,
                historical_industry_rows=len(unique),attribution_period_checks=len(checks),
                maximum_attribution_error=float(checks.attribution_error.max()),paper_accounts_unchanged=True)
    write_json(OUT/'independent_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
