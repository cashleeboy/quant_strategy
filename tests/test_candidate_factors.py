import unittest
import numpy as np
import pandas as pd
from analysis.candidate_factors import *

class CandidateTest(unittest.TestCase):
    def tables(self):
        base=dict(ts_code='A',ann_date='20220401',f_ann_date='20220402',end_date='20211231',report_type=1,comp_type=1)
        old={**base,'end_date':'20201231','ann_date':'20210401','f_ann_date':'20210401','total_assets':80.}
        rows={'balance':[old,{**base,'total_assets':100.}],
              'income':[{**base,'revenue':90.,'oper_cost':60.,'n_income':15.}],
              'cashflow':[{**base,'n_cashflow_act':10.}]}
        return {k:normalize_annual(pd.DataFrame(v),k) for k,v in rows.items()}
    def test_formula(self):
        r=annual_candidates(self.tables(),'2022-04-03').loc['A']
        self.assertAlmostEqual(r.LOW_ACCRUALS,-5/90);self.assertAlmostEqual(r.LOW_ASSET_GROWTH,-.25);self.assertAlmostEqual(r.GROSS_PROFIT_ASSETS,.3)
    def test_disclosure_lag(self):
        r=annual_candidates(self.tables(),'2022-04-02').loc['A']
        self.assertTrue(pd.isna(r.GROSS_PROFIT_ASSETS));self.assertEqual(r.report_end,pd.Timestamp('2020-12-31'))
    def test_no_fallback_missing_latest_income(self):
        t=self.tables();t['income'].loc[:,'available_date']=pd.Timestamp('2022-06-01')
        r=annual_candidates(t,'2022-04-03').loc['A'];self.assertTrue(pd.isna(r.GROSS_PROFIT_ASSETS));self.assertAlmostEqual(r.LOW_ASSET_GROWTH,-.25)
    def test_financial_company(self):
        t=self.tables();t['balance']['comp_type']=2
        self.assertTrue(annual_candidates(t,'2022-04-03')[['LOW_ACCRUALS','LOW_ASSET_GROWTH','GROSS_PROFIT_ASSETS']].isna().all().all())
    def test_exact_prior_year(self):
        t=self.tables();t['balance']=t['balance'].loc[t['balance'].end_date.dt.year.eq(2021)]
        r=annual_candidates(t,'2022-04-03').loc['A'];self.assertTrue(pd.isna(r.LOW_ACCRUALS));self.assertAlmostEqual(r.GROSS_PROFIT_ASSETS,.3)
    def test_conflicting_versions(self):
        b=dict(ts_code='A',ann_date='20220401',f_ann_date='20220402',end_date='20211231',report_type=1,comp_type=1)
        f=normalize_annual(pd.DataFrame([{**b,'total_assets':100},{**b,'total_assets':101}]),'balance')
        self.assertTrue(f.ambiguous.all());self.assertTrue(f.total_assets.isna().all())
    def test_dividend_no_future_or_fill(self):
        idx=pd.MultiIndex.from_product([pd.to_datetime(['2022-01-06','2022-01-07','2022-01-14','2022-01-15','2022-01-21']),['A']],names=['date','symbol'])
        raw=pd.DataFrame({'ts_code':['A','A'],'trade_date':[20220107,20220121],'dv_ttm':[3.,np.nan]})
        r=dividend_asof(idx,raw).DIVIDEND_TTM
        self.assertTrue(pd.isna(r.iloc[0]));self.assertAlmostEqual(r.iloc[1],.03);self.assertAlmostEqual(r.iloc[2],.03)
        self.assertTrue(pd.isna(r.iloc[3]));self.assertTrue(pd.isna(r.iloc[4]))
    def test_skip_month_endpoints(self):
        x=pd.DataFrame({'A':np.arange(1,301,dtype=float)})
        r=momentum_12_1(x);self.assertAlmostEqual(r.iloc[252,0],232/1-1);self.assertTrue(pd.isna(r.iloc[251,0]))
        x.iloc[252]=1e9;self.assertAlmostEqual(momentum_12_1(x).iloc[252,0],232/1-1)
    def test_preserve_missing_standardization(self):
        idx=pd.MultiIndex.from_product([[pd.Timestamp('2020-01-01')],['A','B','C']],names=['date','symbol'])
        raw=pd.DataFrame(np.nan,index=idx,columns=CANDIDATES);raw['DIVIDEND_TTM']=[.01,.02,np.nan]
        z=standardize_candidates(raw,pd.Series('X',index=idx));self.assertTrue(z.LOW_ACCRUALS.isna().all());self.assertTrue(pd.isna(z.DIVIDEND_TTM.iloc[2]))

if __name__=='__main__':unittest.main()
