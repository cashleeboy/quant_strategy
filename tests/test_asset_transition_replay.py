import unittest,copy
import numpy as np
import pandas as pd
from backtest.action_replay import replay
from backtest.research_replay import replay as legacy
from backtest.corporate_actions import AssetTransitionError,adjusted_swap_quantity


class TransitionTest(unittest.TestCase):
    def setUp(self):
        self.d=pd.bdate_range('2020-01-01',periods=7)
        self.p=pd.DataFrame({'OLD':[20,20,np.nan,np.nan,np.nan,np.nan,np.nan], 'NEW':[30,30,30,30,33,36,36]},index=self.d)
        self.a=pd.DataFrame({'OLD':2.,'NEW':3.},index=self.d)
        self.dead=pd.Series({'OLD':self.d[4],'NEW':pd.NaT})
        self.s=pd.DataFrame({'execution_date':self.d[1:]},index=self.d[:-1])
        self.e=dict(event_id='swap',kind='stock_swap',source='OLD',destination='NEW',announcement_date=str(self.d[1].date()),
            record_date=str(self.d[3].date()),effective_date=str(self.d[4].date()),ratio_numerator=2.,ratio_denominator=1.,evidence=['verified'])
    def run_case(self,factory=None,events=None):
        return replay(self.p,self.s,factory or (lambda d,h:pd.Series({'OLD':.6}) if d==self.d[0] else None),
            actions=[self.e] if events is None else events,adjustment_factors=self.a,delist_dates=self.dead,fee=0.)
    def test_raw_adjusted_units(self):
        self.assertAlmostEqual(adjusted_swap_quantity(10,2,4,5),16)
        r=self.run_case();e=r['asset_actions'].iloc[0]
        self.assertAlmostEqual(e.source_units,.03);self.assertAlmostEqual(e.source_raw_shares,.06)
        self.assertAlmostEqual(e.received_units,.04);self.assertAlmostEqual(e.received_raw_shares,.12)
        self.assertAlmostEqual(e.action_pnl,.72);self.assertAlmostEqual(r['daily'].post_nav.iloc[4],1.72)
    def test_no_fee_and_no_premature_trade(self):
        r=self.run_case();self.assertEqual(len(r['trades']),1)
        self.assertTrue(r['asset_actions'].fee.eq(0).all());self.assertTrue(r['asset_actions'].cash_change.eq(0).all())
        self.assertTrue(r['positions'].query('symbol=="NEW"').date.ge(self.d[4]).all())
    def test_existing_destination_is_added(self):
        r=self.run_case(lambda d,h:pd.Series({'OLD':.6,'NEW':.3}) if d==self.d[0] else None)
        self.assertAlmostEqual(r['asset_actions'].iloc[0].destination_units_after,.05)
    def test_sale_only_on_later_strategy_instruction(self):
        r=self.run_case(lambda d,h:pd.Series({'OLD':.6}) if d==self.d[0] else pd.Series(dtype=float) if d==self.d[4] else None)
        sale=r['trades'].query('side=="SELL"').iloc[0]
        self.assertEqual(sale.symbol,'NEW');self.assertEqual(sale.date,self.d[5])
        self.assertAlmostEqual(sale.notional,1.44)
    def test_same_day_pending_sell_after_swap(self):
        r=self.run_case(lambda d,h:pd.Series({'OLD':.6}) if d==self.d[0] else pd.Series(dtype=float) if d==self.d[3] else None)
        self.assertEqual(r['trades'].query('side=="SELL"').iloc[0].date,self.d[4])
    def test_unknown_delisting_blocks(self):
        with self.assertRaises(AssetTransitionError):self.run_case(events=[])
    def test_missing_destination_quote_blocks(self):
        self.p.loc[self.d[4],'NEW']=np.nan
        with self.assertRaises(AssetTransitionError):self.run_case()
    def test_changed_source_adjustment_blocks(self):
        self.a.loc[self.d[3],'OLD']=2.1
        with self.assertRaises(AssetTransitionError):self.run_case()
    def test_duplicate_or_future_event_blocks(self):
        with self.assertRaises(ValueError):self.run_case(events=[self.e,self.e])
        e=copy.deepcopy(self.e);e['announcement_date']=str(self.d[5].date())
        with self.assertRaises(ValueError):self.run_case(events=[e])
    def test_no_action_exact_legacy_parity(self):
        self.dead[:]=pd.NaT
        def f(d,h):return pd.Series({'NEW':.7}) if d in [self.d[0],self.d[3]] else None
        old=legacy(self.p,self.s,f,fee=.0013)
        new=replay(self.p,self.s,f,actions=[],adjustment_factors=self.a,delist_dates=self.dead,fee=.0013)
        for key in old:pd.testing.assert_frame_equal(new[key][old[key].columns],old[key])
    def test_stale_target_blocks_rebuy(self):
        with self.assertRaises(AssetTransitionError):
            self.run_case(lambda d,h:pd.Series({'OLD':.6}) if d in [self.d[0],self.d[4]] else None)
    def test_inverted_ratio(self):
        self.assertAlmostEqual(adjusted_swap_quantity(.1,1/.5424,2,3),.1*2/(.5424*3))

if __name__=='__main__':unittest.main()
