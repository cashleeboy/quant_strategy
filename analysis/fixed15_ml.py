"""Strict causal training utilities; deliberately separate from legacy ML defaults."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.factor_ml import MLScoreConfig, _make_regressor, _prepare_matrix


def strict_regressor(backend):
    if backend not in {"lightgbm", "catboost"}:
        raise ValueError("Explicit LightGBM or CatBoost backend required")
    cfg = MLScoreConfig(backend, 5, 252, 26, 1000, 20, 42)
    model, actual = _make_regressor(cfg)
    if actual != backend:
        raise RuntimeError(f"Requested {backend}, obtained {actual}; install optional dependencies")
    if backend == "lightgbm":
        model.set_params(n_jobs=2, deterministic=True, force_col_wise=True)
    else:
        model.set_params(thread_count=2, allow_writing_files=False)
    return model


def training_sample(panel, labels, outcome_end, sessions, signal, spec):
    """Cut the exchange calendar before filtering weekly samples or label validity."""
    signal = pd.Timestamp(signal)
    if signal not in sessions or not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("Invalid exchange session calendar")
    if panel.index.has_duplicates:
        raise ValueError("Duplicate factor rows")
    history = sessions[sessions <= signal][-spec["train_lookback_sessions"]:]
    dates = panel.index.get_level_values("date")
    ends = pd.to_datetime(outcome_end.reindex(dates))
    mask = dates.isin(history) & (dates < signal) & ends.lt(signal).to_numpy()
    candidates = panel.loc[mask]
    y = labels.reindex(candidates.index).replace([np.inf, -np.inf], np.nan)
    x = candidates.loc[y.notna()]
    y = y.loc[x.index]
    ndays = x.index.get_level_values("date").nunique()
    if ndays < spec["minimum_train_dates"] or len(x) < spec["minimum_train_rows"]:
        raise ValueError(f"Insufficient matured training sample on {signal}: {ndays} dates, {len(x)} rows")
    used_dates = x.index.get_level_values("date")
    log = dict(refit_date=signal, window_floor=history.min(), train_start=used_dates.min(),
               train_end=used_dates.max(), latest_outcome_end=outcome_end.reindex(used_dates).max(),
               train_dates=ndays, train_rows=len(x), candidate_rows=len(candidates),
               excluded_missing_labels=len(candidates)-len(x))
    return x, y, log


def fit_model(backend, x, y):
    """No test-set early stopping, future-aware feature dropping or silent recovery."""
    matrix, medians = _prepare_matrix(x, list(x.columns))
    model = strict_regressor(backend)
    model.fit(pd.DataFrame(matrix, columns=x.columns), y.to_numpy())
    return model, medians


def predict(model, x, medians):
    matrix, _ = _prepare_matrix(x, list(medians.index), medians)
    values = np.asarray(model.predict(pd.DataFrame(matrix, columns=medians.index)))
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("Invalid model prediction")
    return pd.Series(values, index=x.index, name="score")
