import unittest
import numpy as np
import pandas as pd
from analysis.beta_research import rolling_market_sensitivity, execution_return, market_regression, controlled_beta


class BetaResearchTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2020-01-01', periods=50)
        self.market = pd.Series(np.sin(np.arange(50))*.01, index=self.dates)
        self.prices = pd.DataFrame({'a': (1+.002+1.7*self.market).cumprod(),
                                    'b': (1-.0001+.6*self.market).cumprod()})

    def test_known_slope_and_intercept(self):
        b, v, n = rolling_market_sensitivity(self.prices, self.market, 20, 15)
        np.testing.assert_allclose(b.iloc[-1], [1.7, .6], atol=1e-10)
        self.assertEqual(n.iloc[-1, 0], 20)
        self.assertGreater(v.iloc[-1, 0], v.iloc[-1, 1])
        self.assertAlmostEqual(market_regression(self.prices.a.pct_change(), self.market)['annual_intercept'], .002*252)

    def test_prefix_invariant(self):
        b, _, _ = rolling_market_sensitivity(self.prices, self.market, 20, 15)
        p = self.prices.copy(); p.iloc[35:] *= 10
        after, _, _ = rolling_market_sensitivity(p, self.market, 20, 15)
        pd.testing.assert_frame_equal(b.iloc[:35], after.iloc[:35])
        prefix, _, _ = rolling_market_sensitivity(self.prices.iloc[:35], self.market.iloc[:35], 20, 15)
        pd.testing.assert_frame_equal(b.iloc[:35], prefix)

    def test_missing_does_not_bridge(self):
        p = self.prices.copy(); p.iloc[30, 0] = np.nan
        b, _, n = rolling_market_sensitivity(p, self.market, 20, 15)
        self.assertEqual(n.iloc[32, 0], 18)
        self.assertAlmostEqual(b.iloc[32, 0], 1.7)
        label = execution_return(p, p.index[28], 5)
        self.assertTrue(np.isnan(label.a))
        self.assertAlmostEqual(label.b, p.b.iloc[34]/p.b.iloc[29]-1)

    def test_zero_market_variance_and_minimum(self):
        b, _, _ = rolling_market_sensitivity(self.prices, self.market*0, 20, 15)
        self.assertTrue(b.isna().all().all())
        b, _, _ = rolling_market_sensitivity(self.prices, self.market, 20, 15)
        self.assertTrue(b.iloc[:15].isna().all().all())
        self.assertTrue(execution_return(self.prices, self.dates[-3], 5).isna().all())

    def test_residual_orthogonal_controls(self):
        rng = np.random.default_rng(8); n = 120
        f = pd.DataFrame({'beta': rng.normal(size=n), 'vol252': rng.random(n),
                          'total_mv_yuan': np.exp(rng.normal(size=n)), 'industry_l1': ['A','B']*(n//2)})
        r = controlled_beta(f)
        self.assertAlmostEqual(r.mean(), 0)
        self.assertAlmostEqual(r.cov(f.vol252), 0)
        self.assertAlmostEqual(r.cov(np.log(f.total_mv_yuan)), 0)


if __name__ == '__main__':
    unittest.main()
