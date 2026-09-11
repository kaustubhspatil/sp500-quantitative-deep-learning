"""Metrics and walk-forward validation utilities."""

import numpy as np
import pandas as pd

from . import config


def regression_metrics(y_true, y_pred):
    """MSE, MAE, directional accuracy, and the gross long/flat Sharpe."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "signal_sharpe": signal_sharpe(y_pred, y_true),
    }


def signal_sharpe(predictions, actual_returns):
    """
    Annualized Sharpe of the naive long/flat strategy implied by the signal
    (long when prediction > 0, flat otherwise), gross of costs.
    """
    strat = np.where(np.asarray(predictions) > 0, np.asarray(actual_returns), 0.0)
    if strat.std() == 0:
        return 0.0
    return float(strat.mean() / strat.std() * np.sqrt(config.TRADING_DAYS))


def annualized_sharpe(daily_returns):
    r = np.asarray(daily_returns)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(config.TRADING_DAYS))


def max_drawdown(equity):
    """Worst peak-to-trough decline of an equity curve (as a negative fraction)."""
    equity = np.asarray(equity)
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def walk_forward_folds(dates, n_folds=4, min_train_years=8):
    """
    Expanding-window walk-forward splits over a sorted DatetimeIndex of
    unique trading dates. Each fold trains on everything before the fold's
    test window and tests on the next contiguous slice. Returns a list of
    (train_end_date, test_start_date, test_end_date).
    """
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    first_test = dates[0] + pd.DateOffset(years=min_train_years)
    test_dates = dates[dates >= first_test]
    fold_size = len(test_dates) // n_folds
    folds = []
    for k in range(n_folds):
        chunk = test_dates[k * fold_size: (k + 1) * fold_size if k < n_folds - 1 else None]
        train_end = dates[dates < chunk[0]][-1]
        folds.append((train_end, chunk[0], chunk[-1]))
    return folds


def metrics_table(results, index_name="Model"):
    """Assemble a comparison DataFrame from {name: metrics_dict}."""
    df = pd.DataFrame(results).T
    df.index.name = index_name
    return df
