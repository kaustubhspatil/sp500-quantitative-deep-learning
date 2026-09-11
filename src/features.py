"""
Hand-built technical indicators and feature engineering.

Every indicator uses only TRAILING windows (pandas `rolling`/`ewm` defaults),
so the feature at date t never sees data after t. The prediction target is
the NEXT-day return, aligned with `shift(-1)` - the one deliberate look-ahead,
because it is the thing we are trying to predict.
"""

import numpy as np
import pandas as pd

from . import config


# ── Indicators (each takes per-ticker series, returns an aligned series) ──

def rsi(close, window=14):
    """Wilder's Relative Strength Index, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd_histogram(close, fast=12, slow=26, signal=9):
    """MACD histogram (MACD line minus signal line), scaled by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - signal_line) / close


def bollinger(close, window=20, num_std=2.0):
    """Bollinger %B (position inside the bands) and bandwidth."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / mid
    return pct_b, bandwidth


def atr(high, low, close, window=14):
    """Average True Range, normalized by close (a volatility measure)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean() / close


# ── Feature matrix construction ──────────────────────────────────────────

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "vol_21d", "atr_14",
    "rsi_14", "macd_hist", "boll_pct_b", "boll_bw",
    "px_vs_sma10", "px_vs_sma50",
    "volume_z",
    "vix", "vix_chg_5d", "treasury_10y", "treasury_chg_21d",
]


def build_stock_features(group):
    """
    Per-ticker feature block. `group` is one ticker's OHLCV rows sorted by date.
    Returns a DataFrame indexed like `group` with the stock-level features
    plus the forward-return target.
    """
    close, high, low = group["Close"], group["High"], group["Low"]
    volume = group["Volume"].astype(float)

    f = pd.DataFrame(index=group.index)
    ret = close.pct_change()

    # momentum at several horizons
    f["ret_1d"] = ret
    f["ret_5d"] = close.pct_change(5)
    f["ret_21d"] = close.pct_change(21)
    f["ret_63d"] = close.pct_change(63)

    # volatility
    f["vol_21d"] = ret.rolling(21).std() * np.sqrt(config.TRADING_DAYS)
    f["atr_14"] = atr(high, low, close)

    # oscillators / mean-reversion
    f["rsi_14"] = rsi(close)
    f["macd_hist"] = macd_histogram(close)
    f["boll_pct_b"], f["boll_bw"] = bollinger(close)

    # trend location
    f["px_vs_sma10"] = close / close.rolling(10).mean() - 1
    f["px_vs_sma50"] = close / close.rolling(50).mean() - 1

    # volume anomaly (z-score against trailing 21 days)
    vol_mean = volume.rolling(21).mean()
    vol_std = volume.rolling(21).std()
    f["volume_z"] = ((volume - vol_mean) / vol_std.replace(0, np.nan)).clip(-5, 5)

    # target: next-day simple return (the one deliberate forward-shift)
    f["target"] = ret.shift(-config.TARGET_HORIZON)
    return f


def build_feature_panel(prices, market, tickers=None):
    """
    Full long-format feature panel across the universe.

    Returns a DataFrame with columns [Date, Ticker, <features...>, target],
    with warm-up NaNs and the final unlabeled day dropped.
    """
    tickers = tickers or sorted(
        t for t in prices["Ticker"].unique() if t not in config.BENCHMARKS
    )
    blocks = []
    for ticker in tickers:
        group = prices[prices["Ticker"] == ticker].sort_values("Date")
        f = build_stock_features(group)
        f.insert(0, "Ticker", ticker)
        f.insert(0, "Date", group["Date"].values)
        blocks.append(f)
    panel = pd.concat(blocks, ignore_index=True)

    # market context, shared across tickers on each date
    mkt = market.copy()
    mkt["vix_chg_5d"] = mkt["vix"].pct_change(5)
    mkt["treasury_chg_21d"] = mkt["treasury_10y"].diff(21)
    panel = panel.merge(mkt.reset_index(), on="Date", how="left")

    panel = panel.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return panel[["Date", "Ticker"] + FEATURE_COLUMNS + ["target"]]
