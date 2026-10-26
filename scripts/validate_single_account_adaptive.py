#!/usr/bin/env python3
"""Independent saved-result checks: source seal, maturity, targets and accounting."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_mvp_holdings_count import sha, write_json
from scripts.run_single_account_adaptive import OUT, PROTOCOL, account_hashes
from scripts.run_six_strategy_ic_comparison import load_prices


def validate(out=OUT):
    p=json.loads((out/"protocol.json").read_text())
    complete=json.loads((out/"completed.json").read_text())
    for name,digest in complete.items():
        assert sha(out/name)==digest, f"changed result: {name}"
    frozen=json.loads((out/"frozen.json").read_text())
    assert sha(PROTOCOL)==frozen["protocol"]
    for name,digest in frozen["files"].items():
        assert sha(ROOT/name)==digest, f"changed source: {name}"
    w=pd.read_csv(out/"factor_weights.csv",parse_dates=["date","latest_outcome_end"])
    assert w.latest_outcome_end.dropna().lt(w.loc[w.latest_outcome_end.notna(),"date"]).all()
    assert w.weight.between(-1e-12,.2+1e-12).all()
    assert np.allclose(w.groupby("date").weight.sum(),1)
    assert (w.weight-w.previous_weight).abs().max()<=.05+1e-12
    assert w.date.drop_duplicates().dt.to_period("M").is_unique
    parity=pd.read_csv(out/"fixed_score_parity.csv")
    assert parity.fixed_top100_matches.all()
    # Read original PIT industry; validate saved targets, not just the selector function.
    source=ROOT/p["factor_source"]
    inds=pd.concat([pd.read_csv(folder/"BASE_V2_MULTI_FACTOR_weekly.csv.gz",
                  usecols=["date","symbol","industry"],parse_dates=["date"])
                  for folder in sorted(source.glob("inputs_*"))])
    industry={d:g.set_index("symbol").industry.fillna("UNKNOWN").replace("","UNKNOWN")
              for d,g in inds.groupby("date")}
    returns=load_prices(p).pct_change(fill_method=None)
    correlation_checks=0
    audits=[]
    for fee in p["fees"]:
        for arm in p["arms"]:
            folder=out/f"{arm}_{int(round(fee*10000))}bp"
            daily=pd.read_csv(folder/"daily.csv.gz",parse_dates=["date"]).set_index("date")
            ex=pd.read_csv(folder/"executions.csv.gz",parse_dates=["date","signal_date"])
            targets=pd.read_csv(folder/"targets.csv.gz",parse_dates=["date","execution_date"])
            dec=pd.read_csv(folder/"decisions.csv",parse_dates=["date"]).set_index("date")
            trades=pd.read_csv(folder/"trades.csv.gz",parse_dates=["date"])
            assert (daily.cash>=-1e-9).all()
            assert np.allclose(ex.pre_nav-ex.fees,ex.post_nav)
            assert np.allclose(ex.fees,(ex.buy_notional+ex.sell_notional)*fee)
            assert np.allclose(trades.fee,trades.notional*fee)
            # Engine outputs next-session execution; date mapping is checked independently.
            dates=daily.index
            assert all(dates.get_loc(r.date)==dates.get_loc(r.signal_date)+1 for r in ex.itertuples())
            total_weights=targets.groupby("date").weight.sum()
            assert total_weights.le(1+1e-12).all()
            assert np.allclose(total_weights,dec.loc[total_weights.index].target_equity)
            changed=dec[dec.risk_changed]
            change_positions=dates.get_indexer(changed.index)
            assert (np.diff(change_positions)>=p["risk"]["minimum_sessions_between_budget_changes"]).all()
            previous_budget=1.
            initialized=False
            for r in dec.itertuples():
                if r.risk_changed:
                    if initialized:
                        assert abs(r.budget-previous_budget)>=p["risk"]["adjustment_threshold"]-1e-12
                    previous_budget=r.budget;initialized=True
                else:
                    assert np.isclose(r.budget,previous_budget)
            if arm.startswith("IC_"):
                for date,g in targets.groupby("date"):
                    if not dec.loc[date,"weekly"]:
                        continue
                    k=int(dec.loc[date,"requested_k"])
                    base_weights=g.set_index("symbol").weight/dec.loc[date,"budget"]
                    assert np.allclose(base_weights,1/k)
                    sectors=industry[date].reindex(base_weights.index)
                    assert sectors.notna().all()
                    assert base_weights.groupby(sectors).sum().max()<=p["selection"]["industry_cap"]+1e-12
                    assert base_weights[sectors.isin(p["selection"]["theme_industries"])].sum()<=p["selection"]["theme_cap"]+1e-12
                    history=returns.loc[:date,base_weights.index].tail(p["selection"]["history_sessions"])
                    assert history.count().ge(p["selection"]["minimum_name_observations"]).all()
                    corr=history.corr(min_periods=p["selection"]["minimum_pair_observations"]).to_numpy()
                    off_diagonal=corr[~np.eye(len(corr),dtype=bool)]
                    assert np.isfinite(corr).all()
                    assert (off_diagonal<=p["selection"]["correlation_cap"]+1e-12).all()
                    correlation_checks+=1
            weekly=dec[dec.weekly]
            risk=arm.endswith("_RISK")
            audits.append(dict(arm=arm,fee=fee,weekly_decisions=len(weekly),
                underfilled_weeks=int(weekly.selected.lt(weekly.requested_k).sum()),
                k10_weeks=int(weekly.requested_k.eq(10).sum()),
                k15_weeks=int(weekly.requested_k.eq(15).sum()),
                k20_weeks=int(weekly.requested_k.eq(20).sum()),
                extra_risk_actions=int((~dec.weekly & dec.action).sum()),
                risk_review_days=len(dec) if risk else 0,
                insufficient_risk_history_days=int(dec.insufficient_risk_history.sum()) if risk else 0,
                max_actual_holdings=int(ex.holdings.max()),
                max_unquoted_fraction=float((daily.unquoted_value/daily.post_nav).max()),
                final_unquoted_fraction=float((daily.unquoted_value/daily.post_nav).iloc[-1])))
    assert account_hashes()==json.loads((out/"forward_before.json").read_text())
    pd.DataFrame(audits).to_csv(out/"audit_summary.csv",index=False)
    result=dict(status="PASS",sealed_results_checked=len(complete),frozen_sources_checked=len(frozen["files"]),
        monthly_updates=w.date.nunique(),evaluated_weekly_rankings=len(parity),
        adaptive_family_renormalized_stock_weeks=int(parity.adaptive_available_family_renormalized.sum()),
        candidate_stock_weeks=int(parity.candidates.sum()),forward_unchanged=True,
        independently_recomputed_correlation_target_dates=correlation_checks,
        limits="Accounting, causality boundaries and target limits only; not real-market executability certification.")
    write_json(out/"independent_validation.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    validate()
