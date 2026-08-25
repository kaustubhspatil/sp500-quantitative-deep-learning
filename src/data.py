"""Raw data loading and reshaping."""

import pandas as pd

from . import config


def load_prices():
    """
    Load the long-format OHLCV panel (one row per date x ticker).
    Returns a DataFrame with columns [Date, Close, High, Low, Open, Volume, Ticker].
    """
    df = pd.read_csv(config.DATA_DIR / "sp500_stocks.csv", parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_market():
    """Load VIX and 10Y Treasury closes, joined on date."""
    vix = pd.read_csv(config.DATA_DIR / "vix.csv", parse_dates=["Date"],
                      index_col="Date")["Close"].rename("vix")
    tsy = pd.read_csv(config.DATA_DIR / "treasury_10y.csv", parse_dates=["Date"],
                      index_col="Date")["Close"].rename("treasury_10y")
    return pd.concat([vix, tsy], axis=1).sort_index()


def load_summary_stats():
    """Per-stock summary statistics computed by the download script."""
    return pd.read_csv(config.DATA_DIR / "stock_summary_stats.csv")


def pivot_field(prices, field="Close", tickers=None):
    """Wide DataFrame: dates x tickers for a single OHLCV field."""
    wide = prices.pivot(index="Date", columns="Ticker", values=field).sort_index()
    if tickers is not None:
        wide = wide[[t for t in tickers if t in wide.columns]]
    return wide


def stock_universe(prices):
    """All tickers except the reserved benchmarks."""
    return sorted(t for t in prices["Ticker"].unique() if t not in config.BENCHMARKS)


def temporal_split(df, date_col=None):
    """
    Split any date-indexed (or date-column) frame into train/val/test
    using the project's fixed boundaries. Returns three frames.
    """
    dates = df[date_col] if date_col else df.index
    train = df[dates <= config.TRAIN_END]
    val = df[(dates > config.TRAIN_END) & (dates <= config.VAL_END)]
    test = df[dates > config.VAL_END]
    return train, val, test
