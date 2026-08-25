"""Project-wide configuration: paths, universe, temporal splits, constants."""

from pathlib import Path

# ── Paths (everything is relative to the capstone root) ──────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHART_IMG_DIR = DATA_DIR / "chart_images"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RUNS_DIR = ROOT / "runs"  # TensorBoard logs

# ── Universe ─────────────────────────────────────────────────────────────
# SPY and QQQ are reserved as benchmarks - they are never model inputs.
BENCHMARKS = ["SPY", "QQQ"]

# The 10 stocks used for the sequence models and chart-image models
# (2 per major sector bucket, chosen for liquidity and long history).
FOCUS_TICKERS = [
    "AAPL", "MSFT",   # technology
    "JPM", "GS",      # financials
    "UNH", "JNJ",     # healthcare
    "WMT", "PG",      # consumer
    "XOM", "CAT",     # energy / industrials
]

# ── Temporal splits (strictly ordered, no shuffling across time) ─────────
TRAIN_END = "2018-12-31"   # 2005-2018  (14 years) - fit models and scalers
VAL_END = "2021-12-31"     # 2019-2021  (3 years)  - early stopping, tuning
                           # 2022-2025  (4 years)  - untouched test period

# ── Modeling constants ───────────────────────────────────────────────────
TARGET_HORIZON = 1      # predict next-day return
CHART_HORIZON = 5       # chart images are labeled with the 5-day forward direction
SEQ_LEN = 60            # sliding-window length for LSTM / 1D CNN
CHART_WINDOW = 60       # trading days rendered per candlestick image
SEED = 42

# ── Backtest assumptions ─────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000.0
TRANSACTION_COST_BPS = 10.0   # round-trip, per unit of turnover
TRADING_DAYS = 252


def get_device():
    """Return the best available torch device (cuda > mps > cpu)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED):
    """Seed python, numpy and torch for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
