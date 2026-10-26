import numpy as np
import pandas as pd
import unittest

from analysis.multihorizon_selection import (
    build_mixed_monthly_weights,
    impose_trend_floor,
    select_with_ablation,
)


class MultiHorizonSelectionTests(unittest.TestCase):
    def test_trend_floor_preserves_mass_and_other_proportions(self):
        original=pd.Series({'MOMENTUM':0.,'MOMENTUM_60D':.05,'A':.57,'B':.38})
        result=impose_trend_floor(original)
        self.assertTrue(np.isclose(result.sum(),1))
        self.assertEqual(result.MOMENTUM,.10);self.assertEqual(result.MOMENTUM_60D,.10)
        self.assertTrue(np.isclose(result.A/result.B,original.A/original.B))

    def test_mixed_horizon_maturity_is_checked_per_factor(self):
        dates=pd.bdate_range('2024-01-01',periods=5)
        ic=pd.DataFrame({'A':[1,1,1,1,1],'B':[2,2,2,2,2]},index=dates)
        ends=pd.DataFrame({'A':dates+pd.offsets.BDay(1),'B':dates+pd.offsets.BDay(4)},index=dates)
        initial=pd.Series({'A':.5,'B':.5})
        spec={'lookback':5,'minimum':1,'factor_cap':.8,'maximum_monthly_weight_change':.5}
        weights,log=build_mixed_monthly_weights(ic,ends,[pd.Timestamp('2024-01-05')],initial,spec)
        a=log[log.factor.eq('A')].iloc[0];b=log[log.factor.eq('B')].iloc[0]
        self.assertEqual(a.valid_ic_days,3);self.assertEqual(b.valid_ic_days,0)
        self.assertLess(pd.Timestamp(a.latest_outcome_end),pd.Timestamp('2024-01-05'))
        self.assertEqual(weights.shape,(1,2))

    def test_no_buffer_does_not_priority_holdings_below_top15(self):
        names=[f'S{i:02}' for i in range(20)]
        scores=pd.Series(np.arange(20,0,-1),index=names,dtype=float)
        industry=pd.Series([f'I{i}' for i in range(20)],index=names)
        corr=pd.DataFrame(np.eye(20),index=names,columns=names)
        spec={'candidate_limit':100,'buffer_exit_multiple':2,'industry_cap':1.,'theme_cap':1.,'theme_industries':[],'correlation_cap':.8}
        full=select_with_ablation(scores,industry,['S19'],corr,spec,'FULL')
        no_buffer=select_with_ablation(scores,industry,['S19'],corr,spec,'NO_BUFFER')
        self.assertIn('S19',full.index);self.assertNotIn('S19',no_buffer.index)
        self.assertEqual(no_buffer.index.tolist(),names[:15])

    def test_correlation_ablation_removes_only_pair_filter(self):
        names=['A','B','C'];scores=pd.Series([3.,2.,1.],index=names)
        industry=pd.Series(['I1','I2','I3'],index=names)
        corr=pd.DataFrame([[1,.9,0],[.9,1,0],[0,0,1]],index=names,columns=names)
        spec={'candidate_limit':100,'buffer_exit_multiple':2,'industry_cap':1.,'theme_cap':1.,'theme_industries':[],'correlation_cap':.8}
        full=select_with_ablation(scores,industry,[],corr,spec,'FULL')
        removed=select_with_ablation(scores,industry,[],corr,spec,'NO_CORRELATION')
        self.assertEqual(full.index.tolist(),['A','C']);self.assertEqual(removed.index.tolist(),['A','B','C'])


if __name__=='__main__':unittest.main()
