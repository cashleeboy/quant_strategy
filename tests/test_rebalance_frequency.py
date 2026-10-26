import unittest
import pandas as pd
import numpy as np
from analysis.rebalance_frequency import schedules,asof_weights,classify_executions,index_nav

class FrequencyTests(unittest.TestCase):
    def test_true_month_end_partial_month_and_common_bootstrap(self):
        days=pd.bdate_range('2020-01-01','2020-03-12')
        weeks=pd.to_datetime(['2020-01-03','2020-01-10','2020-01-17','2020-01-24','2020-01-31','2020-02-07','2020-02-14','2020-02-21','2020-02-28','2020-03-06'])
        plan=schedules(days,weeks,'2020-01-03','2020-03-12')
        self.assertEqual(list(plan['MONTHLY']),list(pd.to_datetime(['2020-01-03','2020-01-31','2020-02-28'])))
        self.assertTrue(all(x[0]==days[2] for x in plan.values()))

    def test_empty_holiday_week_does_not_reset_fortnight_phase(self):
        days=pd.bdate_range('2020-01-01','2020-02-15')
        weeks=pd.to_datetime(['2020-01-03','2020-01-10','2020-01-17','2020-01-31','2020-02-07','2020-02-14'])
        p=schedules(days,weeks,'2020-01-03','2020-02-15')
        self.assertEqual(list(p['FORTNIGHTLY']),list(pd.to_datetime(['2020-01-03','2020-01-17','2020-01-31','2020-02-14'])))

    def test_month_end_is_not_last_friday(self):
        days=pd.bdate_range('2020-03-01','2020-04-03')
        weeks=pd.to_datetime(['2020-03-06','2020-03-13','2020-03-20','2020-03-27'])
        p=schedules(days,weeks,'2020-03-06','2020-04-03')
        self.assertIn(pd.Timestamp('2020-03-31'),p['MONTHLY'])
        self.assertNotIn(pd.Timestamp('2020-03-27'),p['MONTHLY'])

    def test_weights_cannot_read_future_update(self):
        weights=pd.DataFrame({'A':[.2,.4]},index=pd.to_datetime(['2020-01-03','2020-02-07']))
        w,d=asof_weights(weights,pd.Timestamp('2020-01-31'))
        self.assertEqual(w.A,.2);self.assertEqual(d,pd.Timestamp('2020-01-03'))

    def test_index_entry_next_close_and_no_missing_fill(self):
        days=pd.bdate_range('2020-01-01',periods=4)
        levels=pd.Series([90.,100.,110.,105.],index=days)
        np.testing.assert_allclose(index_nav(levels,days),[1,1,1.1,1.05])
        with self.assertRaises(ValueError):index_nav(levels.drop(days[2]),days)

    def test_joint_risk_day_is_not_double_counted(self):
        days=pd.bdate_range('2020-01-01',periods=5)
        decisions=pd.DataFrame({'date':days,'weekly':[True,False,True,True,False],'risk_changed':[True,True,False,True,False]})
        executions=pd.DataFrame({'signal_date':days,'action':['TARGET']*4+['HOLD']})
        out=classify_executions(executions,decisions,days[0])
        self.assertEqual(out.trigger.tolist(),['BOOTSTRAP','RISK_ONLY','SCHEDULED_ONLY','SCHEDULED_AND_RISK','HOLD'])

if __name__=='__main__':unittest.main()
