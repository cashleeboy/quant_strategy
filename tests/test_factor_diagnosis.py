import unittest
import numpy as np
import pandas as pd
from analysis.factor_diagnosis import horizon_returns,cross_section_diagnostic,hac_mean,remove_family,compact_replay
from backtest.research_replay import replay


class FactorDiagnosisTests(unittest.TestCase):
    def test_execution_aligned_horizons_and_tail(self):
        dates=pd.bdate_range('2020-01-01',periods=10)
        prices=pd.DataFrame({'a':np.arange(10,20,dtype=float)},index=dates)
        y,end=horizon_returns(prices,dates[0],dates[4],[1,5,20])
        self.assertAlmostEqual(y.loc['a','1'],12/11-1)
        self.assertAlmostEqual(y.loc['a','5'],16/11-1)
        self.assertAlmostEqual(y.loc['a','NEXT'],15/11-1)
        self.assertTrue(pd.isna(end['20']))
        prices.loc[dates[1],'a']=np.nan
        self.assertTrue(horizon_returns(prices,dates[0],dates[4],[1])[0].isna().all().all())

    def test_missing_future_does_not_replace_top15(self):
        scores=pd.Series(np.arange(40,0,-1),index=range(40))
        y=pd.Series(0.,index=range(40));y.iloc[:15]=1;y.iloc[15]=100;y.iloc[0]=np.nan
        row=cross_section_diagnostic(scores,y,pd.Series(True,index=scores.index))
        self.assertEqual(row['top15_return'],1)
        self.assertAlmostEqual(row['top15_coverage'],14/15)

    def test_family_deletion_preserves_relative_remaining_weights(self):
        w=pd.Series({'a':.2,'b':.3,'c':.5})
        d=remove_family(w,['a']);self.assertEqual(w.a,.2)
        self.assertEqual(d.a,0);self.assertAlmostEqual(d.b/d.c,.6);self.assertAlmostEqual(d.sum(),1)

    def test_hac_constant_and_missing(self):
        r=hac_mean([.1,.1,np.nan,.1,.1]);self.assertEqual(r['n'],4)
        self.assertAlmostEqual(r['low'],.1);self.assertAlmostEqual(r['high'],.1)

    def test_compact_engine_matches_original_missing_cash_hold_and_buffer(self):
        dates=pd.bdate_range('2020-01-01',periods=9)
        prices=pd.DataFrame({'a':[10,11,np.nan,12,13,12,14,15,16],
                             'b':[20,21,np.nan,22,21,23,24,np.nan,25]},index=dates)
        schedule=pd.DataFrame({'execution_date':dates[1:]},index=dates[:-1])
        def factory(date,held):
            i=dates.get_loc(date)
            if i==0:return pd.Series({'a':.4,'b':.4})
            if i==1:return pd.Series({'b':.6})
            if i==2:return None
            if i==3:return pd.Series(dtype=float)
            if i==4:return pd.Series({'a':.5})
            return pd.Series(.3,index=held)
        for fee in [0,.0003,.0013]:
            a=replay(prices,schedule,factory,fee=fee);b=compact_replay(prices,schedule,factory,fee)
            pd.testing.assert_frame_equal(a['daily'],b['daily'])
            pd.testing.assert_frame_equal(a['executions'],b['executions'][a['executions'].columns],check_dtype=False)


if __name__=='__main__':unittest.main()
