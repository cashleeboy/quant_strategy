import unittest
import importlib.util
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.fixed15_ml import training_sample, strict_regressor, fit_model, predict
from analysis.ic_weight_comparison import execution_labels


class Fixed15MLTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2020-01-01', periods=80)
        self.panel = pd.DataFrame({'a': np.arange(32, dtype=float)}, index=pd.MultiIndex.from_product(
            [self.dates[::5], ['a', 'b']], names=['date', 'symbol']))
        px = pd.DataFrame({'a': np.arange(80)+100., 'b': np.arange(80)+200.}, index=self.dates)
        label, self.end = execution_labels(px)
        label.index.name='date';label.columns.name='symbol'
        self.labels = label.stack().reindex(self.panel.index)
        self.spec = dict(train_lookback_sessions=30, minimum_train_dates=3, minimum_train_rows=6)

    def test_calendar_horizon_is_sessions_not_weekly_rows(self):
        self.assertEqual(self.end.loc[self.dates[5]], self.dates[11])

    def test_strict_label_maturity_and_calendar_window(self):
        x, y, log = training_sample(self.panel,self.labels,self.end,self.dates,self.dates[51],self.spec)
        self.assertLess(log['latest_outcome_end'],self.dates[51])
        self.assertGreaterEqual(log['train_start'],self.dates[22])
        self.assertNotIn(self.dates[45],x.index.get_level_values('date'))  # label ends exactly at refit
        self.assertTrue(y.notna().all())

    def test_future_features_and_unmatured_labels_do_not_change_training(self):
        date=self.dates[51]
        a,b,_=training_sample(self.panel,self.labels,self.end,self.dates,date,self.spec)
        changed=self.panel.copy();changed.loc[changed.index.get_level_values('date')>=self.dates[45]]=1e9
        labels=self.labels.copy();labels.loc[labels.index.get_level_values('date')>=self.dates[45]]=-1e9
        c,d,_=training_sample(changed,labels,self.end,self.dates,date,self.spec)
        pd.testing.assert_frame_equal(a,c);pd.testing.assert_series_equal(b,d)

    def test_insufficient_history_fails(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            training_sample(self.panel,self.labels,self.end,self.dates,self.dates[10],self.spec)

    def test_backend_fallback_is_forbidden(self):
        with patch('analysis.fixed15_ml._make_regressor',return_value=(object(),'hist_gradient_boosting')):
            with self.assertRaises(RuntimeError):strict_regressor('catboost')
        with self.assertRaises(ValueError):strict_regressor('auto')

    def test_real_models_and_training_only_medians(self):
        if not all(importlib.util.find_spec(name) for name in ['lightgbm','catboost']):
            self.skipTest('Install requirements-ml-research.txt for real-backend smoke test')
        rng=np.random.default_rng(42)
        x=pd.DataFrame(rng.normal(size=(100,2)),columns=['a','b']);x.loc[:4,'a']=np.nan
        y=x.b**2
        for backend in ['lightgbm','catboost']:
            model,med=fit_model(backend,x,y)
            self.assertAlmostEqual(med.a,x.a.median())
            test=pd.DataFrame({'a':[np.nan,1e9], 'b':[1.,2.]})
            result=predict(model,test,med)
            self.assertEqual(len(result),2);self.assertTrue(np.isfinite(result).all())


if __name__=='__main__':unittest.main()
