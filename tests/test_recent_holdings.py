import unittest
import numpy as np
import pandas as pd
from backtest.research_replay import replay
from analysis.recent_holdings import attribute, contribution


class HoldingsTest(unittest.TestCase):
    def run_case(self,missing=False):
        dates=pd.date_range('2020-12-29',periods=6,freq='B')
        prices=pd.DataFrame({'A':[10,10,np.nan if missing else 12,13,11,14], 'B':[10,11,10,11,12,12]},index=dates)
        schedule=pd.DataFrame({'execution_date':dates[1:]},index=dates[:-1])
        targets={dates[0]:pd.Series({'A':.5}),dates[2]:pd.Series({'B':.7}),dates[4]:pd.Series(dtype=float)}
        result=replay(prices,schedule,lambda d,h:targets.get(d),fee=.0003)
        ledger=attribute(prices,result['daily'],result['trades'])
        self.assertTrue(pd.api.types.is_numeric_dtype(ledger.prior_weight))
        return prices,result,ledger

    def test_actual_not_target_weights(self):
        prices,r,x=self.run_case()
        # The first buy is after the second close, no profit on that day's jump.
        self.assertEqual(x.iloc[0].gross_pnl,0)
        self.assertLess(x.iloc[0].net_pnl,0)
        self.assertEqual(x.loc[x.date.eq(prices.index[2])].iloc[0].gross_pnl,.1)

    def test_carry_across_year_and_additivity(self):
        prices,r,x=self.run_case()
        for start in ['2020','2021']:
            c=contribution(x,r['daily'].post_nav,start,'2021')
            self.assertTrue(np.isfinite(c.contribution).all())
        self.assertGreater(x.loc[x.date.eq(prices.index[3]),'gross_pnl'].sum(),0)

    def test_missing_quote_stale_mark(self):
        prices,r,x=self.run_case(True)
        self.assertEqual(x.loc[x.date.eq(prices.index[2]),'gross_pnl'].sum(),0)
        contribution(x,r['daily'].post_nav,'2020','2021')


if __name__=='__main__':unittest.main()
