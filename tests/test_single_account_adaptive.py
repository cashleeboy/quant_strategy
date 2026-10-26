import numpy as np
import pandas as pd
import unittest
from analysis.single_account_adaptive import (
    prior_weights, capped_positive_weights, update_ic_weights, family_score,
    history_risk, choose_portfolio, constrained_selection, estimated_budget,
    budget_change_allowed,
)
from analysis.ic_weight_comparison import combine_scores


IC = dict(lookback=252, minimum=2, factor_cap=.2, maximum_monthly_weight_change=.05)
SELECT = dict(candidate_limit=100, buffer_exit_multiple=2, industry_cap=.2, theme_cap=.35,
              theme_industries=["tech"], correlation_cap=.8, history_sessions=126,
              minimum_name_observations=100, minimum_pair_observations=80,
              k_candidates=[10,15,20], diversification_ratio_squared_target=3.)
RISK = dict(lookback=60, minimum_complete_observations=40, target_annual_volatility=.15,
            minimum_budget=.2, maximum_budget=1., minimum_sessions_between_budget_changes=5,
            adjustment_threshold=.1)


def test_water_fill_and_no_negative_allocation():
    x = pd.Series([100, 5, 4, 3, 2, 1, -1], index=list("abcdefg"))
    w = capped_positive_weights(x, .2)
    assert np.isclose(w.sum(), 1) and w.max() <= .2 + 1e-12 and w["g"] == 0
    with unittest.TestCase().assertRaises(ValueError):
        capped_positive_weights(pd.Series([1, 2]), .2)


def test_label_maturity_caps_and_future_invariance():
    dates = pd.bdate_range("2020-01-01", periods=12)
    ic = pd.DataFrame(np.arange(72).reshape(12,6) / 100, index=dates, columns=list("abcdef"))
    end = pd.Series(dates, index=dates).shift(-3)
    prior = pd.Series(1/6, index=ic.columns)
    signal = dates[7]
    a, log = update_ic_weights(ic, end, signal, prior, IC)
    changed = ic.copy()
    changed.loc[end.ge(signal) | end.isna()] = -100
    b, _ = update_ic_weights(changed, end, signal, prior, IC)
    pd.testing.assert_series_equal(a,b)
    assert max(r["latest_outcome_end"] for r in log) < signal
    assert a.max() <= .2+1e-12 and (a-prior).abs().max() <= .05+1e-12
    c,_ = update_ic_weights(ic * -1, end, signal, prior, IC)
    pd.testing.assert_series_equal(c,prior)


def test_fixed_score_exact_parity_with_production_recipe():
    rng=np.random.default_rng(7)
    index=pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")],range(50)],names=["date","symbol"])
    panel=pd.DataFrame(rng.normal(size=(50,6)),index=index,columns=list("abcdef"))
    panel.iloc[0,0]=np.nan
    families={"one":list("ab"),"two":list("cde"),"three":["f"]}
    weights=prior_weights(families)
    pd.testing.assert_series_equal(family_score(panel,families,weights),
                                  combine_scores(panel,families,weights),atol=1e-14,rtol=1e-14)


def test_zero_weight_missing_factor_does_not_drop_entire_stock():
    index=pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")],range(4)],names=["date","symbol"])
    x=pd.DataFrame({"a":[np.nan,1,2,3],"b":[1,2,3,4],"c":[4,2,3,1]},index=index)
    families={"one":["a","b"],"two":["c"]}
    w=pd.Series({"a":.5,"b":0.,"c":.5})
    strict=family_score(x,families,w)
    available=family_score(x,families,w,require_all=False)
    assert index[0] not in strict.index and index[0] in available.index


def fixture_selection(rho=0):
    names=[f"s{i:02d}" for i in range(30)]
    scores=pd.Series(np.arange(30,0,-1),index=names,dtype=float)
    industry=pd.Series([f"sector{i%10}" for i in range(30)],index=names)
    corr=pd.DataFrame(np.eye(30)*(1-rho)+rho,index=names,columns=names)
    return scores,industry,corr


def test_dynamic_k_uses_diversification_not_return_and_underfill_cash():
    scores,industry,corr=fixture_selection()
    w, log=choose_portfolio(scores,industry,[],corr,corr,SELECT,True)
    assert len(w)==10 and log[-1]["requested_k"]==10
    scores,industry,corr=fixture_selection(.6)
    w,log=choose_portfolio(scores,industry,[],corr,corr,SELECT,True)
    assert len(w)==20 and log[-1]["requested_k"]==20 and not log[-1]["meets_target"]
    industry[:]="one"
    w,_=choose_portfolio(scores,industry,[],corr,corr,SELECT,False)
    assert len(w)==3 and np.isclose(w.sum(),.2)


def test_industry_theme_correlation_buffer_and_unknown_pair():
    scores,industry,corr=fixture_selection()
    industry.iloc[:10]="tech"
    corr.iloc[0,1]=corr.iloc[1,0]=.9
    chosen=constrained_selection(scores,industry,["s29"],15,corr,SELECT)
    assert "s29" in chosen
    assert not {"s00","s01"}.issubset(chosen)
    assert sum(industry[s]=="tech" for s in chosen)<=3
    assert industry[chosen].value_counts().max()<=3
    corr.loc["s29",:]=np.nan
    corr.loc[:,"s29"]=np.nan
    assert "s29" not in constrained_selection(scores,industry,["s29"],15,corr,SELECT)


def test_return_prefix_invariance_and_risk_cooldown():
    rng=np.random.default_rng(2)
    dates=pd.bdate_range("2020-01-01",periods=150)
    ret=pd.DataFrame(rng.normal(0,.03,(150,2)),index=dates,columns=["a","b"])
    signal=dates[125]
    a,b=history_risk(ret,signal,["a","b"],SELECT)
    budget,vol,n=estimated_budget(ret,signal,pd.Series(.5,index=["a","b"]),RISK)
    changed=ret.copy();changed.loc[dates[126]:]=100
    c,d=history_risk(changed,signal,["a","b"],SELECT)
    pd.testing.assert_frame_equal(a,c);pd.testing.assert_frame_equal(b,d)
    assert (budget,vol,n)==estimated_budget(changed,signal,pd.Series(.5,index=["a","b"]),RISK)
    assert .2<=budget<=1 and n==60
    assert not budget_change_allowed(.5,.7,4,RISK,True)
    assert not budget_change_allowed(.5,.55,20,RISK,True)
    assert budget_change_allowed(.5,.7,5,RISK,True)
    assert budget_change_allowed(1,.95,0,RISK,False)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in globals().items() if name.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
