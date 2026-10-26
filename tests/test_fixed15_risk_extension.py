import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.run_fixed15_risk_extension import resolved_protocol
from scripts.run_single_account_adaptive import Account

ROOT = Path(__file__).resolve().parents[1]


class Fixed15RiskExtensionTests(unittest.TestCase):
    def setUp(self):
        self.parent=json.loads((ROOT/'research_protocols/20261020_single_account_adaptive_v1.json').read_text())
        self.ext=json.loads((ROOT/'research_protocols/20261021_fixed15_risk_extension_v1.json').read_text())
        self.p=resolved_protocol(self.ext,self.parent)
        self.dates=pd.bdate_range('2020-01-01',periods=140)
        self.date=self.dates[125]
        self.names=[f's{i:02}' for i in range(30)]
        scores=pd.Series(np.arange(30,0,-1),index=self.names,dtype=float)
        industry=pd.Series([f'sector{i%10}' for i in range(30)],index=self.names)
        corr=pd.DataFrame(np.eye(30),index=self.names,columns=self.names)
        self.groups={self.date:dict(fixed=scores.iloc[::-1],ic=scores,industry=industry)}
        self.cache={self.date:(corr,corr)}
        # Shared large fluctuations cause a material budget reduction, without relying on real data.
        values=np.tile(np.where(np.arange(140)%2,.06,-.06)[:,None],(1,30))
        self.returns=pd.DataFrame(values,index=self.dates,columns=self.names)

    def account(self,arm,returns=None):
        return Account(arm,self.groups,self.cache,self.returns if returns is None else returns,self.p)

    def test_only_arm_list_changes(self):
        self.assertEqual({k:v for k,v in self.p.items() if k!='arms'},
                         {k:v for k,v in self.parent.items() if k!='arms'})

    def test_new_arm_is_fixed15_constrained_buffered_with_risk(self):
        held=['s29']
        baseline=self.account('IC_CONSTRAINED15')
        new=self.account('IC_CONSTRAINED15_RISK')
        a,b=baseline(self.date,held),new(self.date,held)
        self.assertEqual(list(a.index),list(b.index))
        self.assertEqual(len(b),15)
        self.assertIn('s29',b.index)
        self.assertLess(b.sum(),1)
        np.testing.assert_allclose(b/b.sum(),a)
        self.assertEqual(new.k_trials[-1]['requested_k'],15)
        # Actual holdings, rather than old targets, drive buffer retention.
        without=self.account('IC_CONSTRAINED15_RISK')(self.date,[])
        self.assertNotIn('s29',without.index)

    def test_daily_budget_change_keeps_fixed15_stock_list(self):
        new=self.account('IC_CONSTRAINED15_RISK')
        a=new(self.date,[])
        date=self.dates[130]
        with patch('scripts.run_single_account_adaptive.estimated_budget',return_value=(.8,.18,60)):
            b=new(date,list(a.index))
        self.assertEqual(list(a.index),list(b.index))
        self.assertEqual(len(b),15)
        self.assertAlmostEqual(b.sum(),.8)
        self.assertFalse(new.logs[-1]['weekly'])
        self.assertTrue(new.logs[-1]['risk_changed'])

    def test_future_returns_do_not_change_initial_target(self):
        modified=self.returns.copy()
        modified.loc[self.dates[126]:]=100
        a=self.account('IC_CONSTRAINED15_RISK')(self.date,[])
        b=self.account('IC_CONSTRAINED15_RISK',modified)(self.date,[])
        pd.testing.assert_series_equal(a,b)


if __name__=='__main__':
    unittest.main()
