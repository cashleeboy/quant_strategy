import unittest
import copy
import numpy as np
import pandas as pd
from backtest.delisting_replay_v2 import replay,DelistingBridge,lower_limit,conservative_lower_limit
from backtest.action_replay import replay as old_replay
from backtest.corporate_actions import AssetTransitionError


class DelistingReplayV2Tests(unittest.TestCase):
    def setUp(self):
        self.d=pd.bdate_range('2024-01-01',periods=9)
        self.p=pd.DataFrame({'old':[1.8,1.8]+[np.nan]*7,'other':[1.]*9},index=self.d)
        self.adj=pd.DataFrame({'old':[10.,10.]+[np.nan]*7},index=self.d)
        self.dead=pd.Series({'old':self.d[3],'other':pd.NaT})
        self.e=dict(event_id='x',source='old',claim='right',destination='otc',delist_announcement=self.d[3],delist_date=self.d[3],
            listing_announcement=self.d[5],listing_date=self.d[6],confirmation_deadline=self.d[4],last_source_quote=self.d[1],
            raw_share_ratio=1.,last_raw_close=.18,first_day_low=.17,first_day_high=.19,tick_size=.01,limit_rate=.05,evidence=['test'])
        self.b=pd.DataFrame({'open':[.19,.20,.20],'high':[.19,.20,.20],'low':[.19,.20,.20],'close':[.19,.20,.20],
            'volume_shares':[10000.]*3,'amount_yuan':[1900.,2000.,2000.]},index=self.d[6:])
        self.schedule=pd.DataFrame({'execution_date':self.d[1:]},index=self.d[:-1])
        self.kw=dict(actions=[],adjustment_factors=self.adj,delist_dates=self.dead,transfers=[self.e],otc_bars={'otc':self.b},confirmations={'x':self.d[4]},fee=0.)

    def factory(self,date,held):
        return pd.Series({'old':.5}) if date==self.d[0] else pd.Series(dtype=float)

    def test_locked_claim_preserves_raw_units_and_nav(self):
        r=replay(self.p,self.schedule,self.factory,**self.kw)
        a=r['asset_actions'];self.assertEqual(a.kind.tolist(),['delist_to_claim','claim_to_otc'])
        self.assertAlmostEqual(a.iloc[0].received_units,.5/1.8*10)
        self.assertAlmostEqual(a.iloc[0].action_pnl,0.)
        self.assertAlmostEqual(r['daily'].loc[self.d[4],'locked_equity_value'],.5)
        self.assertEqual(r['trades'][r['trades'].symbol.eq('otc')].date.iloc[0],self.d[6])
        self.assertFalse(r['trades'].symbol.eq('right').any())
        self.assertAlmostEqual(r['daily'].post_nav.iloc[-1],.5+.5/1.8*10*.19)

    def test_no_premature_sale_even_cash_orders(self):
        r=replay(self.p,self.schedule,self.factory,**self.kw)
        self.assertEqual(len(r['trades']),2)
        locked=r['positions'][r['positions'].symbol.eq('right')]
        self.assertFalse(locked.quoted.any());self.assertFalse(locked.sellable.any())

    def test_confirmation_is_required_and_deadline_enforced(self):
        for conf in [{},{'x':self.d[5]}]:
            with self.assertRaises(AssetTransitionError):replay(self.p,self.schedule,self.factory,**{**self.kw,'confirmations':conf})

    def test_unexplained_missing_otc_day_fails(self):
        with self.assertRaisesRegex(ValueError,'missing OTC'):
            replay(self.p,self.schedule,self.factory,**{**self.kw,'otc_bars':{'otc':self.b.drop(self.d[7])}})

    def test_zero_volume_marks_but_does_not_execute(self):
        b=self.b.copy();b.loc[self.d[6],['volume_shares','amount_yuan']]=0
        r=replay(self.p,self.schedule,self.factory,**{**self.kw,'otc_bars':{'otc':b}})
        self.assertEqual(r['trades'].iloc[-1].date,self.d[7])
        self.assertGreater(r['daily'].loc[self.d[6],'post_nav'],1.)

    def test_limit_down_does_not_fake_exit(self):
        b=self.b.copy();b.loc[self.d[6],['open','high','low','close']]=.17
        b.loc[self.d[6],'amount_yuan']=1700
        r=replay(self.p,self.schedule,self.factory,**{**self.kw,'otc_bars':{'otc':b}})
        self.assertEqual(r['trades'].iloc[-1].date,self.d[7])

    def test_hold_does_not_force_liquidation_on_listing(self):
        def f(date,held):return self.factory(date,held) if date==self.d[0] else None
        r=replay(self.p,self.schedule,f,**self.kw)
        self.assertEqual(len(r['trades']),1)
        self.assertIn('otc',r['positions'][r['positions'].date.eq(self.d[-1])].symbol.tolist())

    def test_exit_only_assets_cannot_be_new_targets(self):
        for symbol in ['right','otc']:
            with self.assertRaises(AssetTransitionError):
                replay(self.p,self.schedule,lambda d,h:pd.Series({symbol:.5}),**self.kw)

    def test_unknown_terminated_asset_still_fails(self):
        kw={**self.kw,'transfers':[],'otc_bars':{},'confirmations':{}}
        with self.assertRaises(AssetTransitionError):replay(self.p,self.schedule,self.factory,**kw)

    def test_no_action_portfolio_has_exact_legacy_parity(self):
        f=lambda d,h:pd.Series({'other':.6})
        a=old_replay(self.p,self.schedule,f,actions=[],adjustment_factors=self.adj,delist_dates=self.dead,fee=.0013)
        b=replay(self.p,self.schedule,f,**{**self.kw,'fee':.0013})
        pd.testing.assert_frame_equal(a['daily'],b['daily'][a['daily'].columns])
        pd.testing.assert_frame_equal(a['trades'],b['trades'])

    def test_future_otc_prices_cannot_change_prelisting_results(self):
        a=replay(self.p,self.schedule,self.factory,**self.kw)
        b=self.b.copy();b.loc[self.d[7]:,['open','close','high','low']]*=2
        r=replay(self.p,self.schedule,self.factory,**{**self.kw,'otc_bars':{'otc':b}})
        pd.testing.assert_frame_equal(a['daily'].loc[:self.d[5]],r['daily'].loc[:self.d[5]])

    def test_bounded_coverage_rejects_held_tail(self):
        e={**self.e,'coverage_end':self.d[7]}
        kw={**self.kw,'transfers':[e],'otc_bars':{'otc':self.b.iloc[:2]}}
        f=lambda d,h:self.factory(d,h) if d==self.d[0] else None
        with self.assertRaisesRegex(AssetTransitionError,'beyond verified coverage'):replay(self.p,self.schedule,f,**kw)
        r=replay(self.p,self.schedule,self.factory,**kw)
        self.assertEqual(r['trades'].iloc[-1].date,self.d[6])

    def test_conservative_guard_catches_rounding_boundary(self):
        self.assertEqual(conservative_lower_limit(.78),.75)
        self.assertEqual(conservative_lower_limit(.75),.72)
        self.assertEqual(conservative_lower_limit(.57),.55)
        self.assertEqual(conservative_lower_limit(.18),.17)
        e={**self.e,'last_raw_close':.78,'first_day_low':.74,'first_day_high':.82,'limit_rule':'conservative_envelope'}
        p=self.p.copy();p.loc[self.d[:2],'old']=7.8
        b=self.b.copy();b.loc[:,['open','high','low','close']]=np.array([.75,.72,.73])[:,None]*np.ones((1,4))
        b['amount_yuan']=b.close*b.volume_shares
        r=replay(p,self.schedule,self.factory,**{**self.kw,'transfers':[e],'otc_bars':{'otc':b}})
        self.assertEqual(r['trades'].iloc[-1].date,self.d[8])

    def test_two_rights_are_not_merged(self):
        p=self.p.copy();p['old2']=p.old
        adj=self.adj.copy();adj['old2']=adj.old
        dead=pd.concat([self.dead,pd.Series({'old2':self.d[3]})])
        e={**self.e,'event_id':'y','source':'old2','claim':'right2','destination':'otc2'}
        f=lambda d,h:pd.Series({'old':.25,'old2':.25}) if d==self.d[0] else pd.Series(dtype=float)
        r=replay(p,self.schedule,f,**{**self.kw,'adjustment_factors':adj,'delist_dates':dead,
            'transfers':[self.e,e],'otc_bars':{'otc':self.b,'otc2':self.b},'confirmations':{'x':self.d[4],'y':self.d[4]}})
        self.assertEqual(len(r['asset_actions']),4)
        self.assertEqual(set(r['trades'].symbol),{'old','old2','otc','otc2'})

    def test_tick_limit(self):
        self.assertEqual(lower_limit(.18),.17)
        self.assertEqual(lower_limit(.20),.19)


if __name__=='__main__':unittest.main()
