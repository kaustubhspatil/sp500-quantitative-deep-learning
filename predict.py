"""
Live inference demo - Day 10 presentation.

Loads the saved production artifacts (scaler + trained feedforward network),
scores the most recent trading day in the dataset, and prints the signal the
desk would act on tomorrow morning.

This is the "model makes a prediction on recent data" walkthrough: it takes
RAW price data as input and runs the complete pipeline - features, scaling,
inference, position sizing - with no manual preprocessing steps.

Usage:
    python predict.py                 # score the latest date in the dataset
    python predict.py --date 2025-06-30
    python predict.py --top 10        # show the 10 strongest signals
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch

from src import config, data, features
from src.models import StockPredictor


def load_pipeline():
    """Load every artifact needed for inference, exactly as saved by the notebooks."""
    feature_columns = json.load(open(config.PROCESSED_DIR / "feature_columns.json"))
    scaler = joblib.load(config.MODELS_DIR / "feature_scaler.pkl")

    model = StockPredictor(len(feature_columns), (128, 64, 32), dropout=0.3)
    model.load_state_dict(torch.load(config.MODELS_DIR / "ffnn_best.pt",
                                     map_location="cpu"))
    model.eval()   # critical: disables dropout and uses BatchNorm running stats
    return model, scaler, feature_columns


def predict_for_date(as_of=None, top_n=None):
    """
    Build features from raw prices, score one date, and return the ranked signal.

    `as_of` is the date whose CLOSE the decision is made on; the prediction is
    the expected return for the NEXT trading day.
    """
    model, scaler, feature_columns = load_pipeline()

    prices = data.load_prices()
    market = data.load_market()
    panel = features.build_feature_panel(prices, market)

    as_of = pd.Timestamp(as_of) if as_of else panel["Date"].max()
    day = panel[panel["Date"] == as_of].copy()
    if day.empty:
        raise SystemExit(f"No feature row for {as_of.date()} "
                         f"(latest available: {panel['Date'].max().date()})")

    X = scaler.transform(day[feature_columns])
    with torch.no_grad():
        day["predicted_return"] = model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    day["signal"] = np.where(day["predicted_return"] > 0, "LONG", "FLAT")
    longs = day["signal"].eq("LONG")
    # equal-weight across the names the model wants to hold
    day["weight"] = np.where(longs, 1 / max(longs.sum(), 1), 0.0)
    day["dollar_position"] = day["weight"] * config.INITIAL_CAPITAL

    ranked = day.sort_values("predicted_return", ascending=False)
    return as_of, ranked[["Ticker", "predicted_return", "signal", "weight",
                          "dollar_position"]].reset_index(drop=True), top_n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="decision date (YYYY-MM-DD); default = latest")
    ap.add_argument("--top", type=int, default=15, help="how many names to display")
    args = ap.parse_args()

    as_of, ranked, _ = predict_for_date(args.date, args.top)
    n_long = (ranked["signal"] == "LONG").sum()

    print()
    print("=" * 68)
    print(f"  JARVIS CAPITAL - DAILY EQUITY SIGNAL")
    print(f"  Model: feedforward NN (128-64-32) | decision date: {as_of.date()}")
    print(f"  Prediction horizon: next trading day's return")
    print("=" * 68)
    print(f"\n  Universe scored:   {len(ranked)} stocks")
    print(f"  Going LONG:        {n_long}  ({n_long / len(ranked):.0%} of the book)")
    print(f"  Holding CASH:      {len(ranked) - n_long}")
    print(f"  Capital deployed:  ${ranked['dollar_position'].sum():,.0f} "
          f"of ${config.INITIAL_CAPITAL:,.0f}")

    print(f"\n  Top {args.top} by predicted return:\n")
    head = ranked.head(args.top).copy()
    head["predicted_return"] = (head["predicted_return"] * 10_000).map("{:+.1f} bps".format)
    head["dollar_position"] = head["dollar_position"].map("${:,.0f}".format)
    head["weight"] = head["weight"].map("{:.2%}".format)
    print(head.to_string(index=False))

    worst = ranked.tail(3).copy()
    worst["predicted_return"] = (worst["predicted_return"] * 10_000).map("{:+.1f} bps".format)
    print(f"\n  Weakest 3 signals:\n")
    print(worst[["Ticker", "predicted_return", "signal"]].to_string(index=False))

    print("\n" + "-" * 68)
    print("  Reminder for the committee: predicted magnitudes are small by")
    print("  construction (daily returns are near-noise). The signal's value")
    print("  is in the sign and the ranking, realized over many days - see")
    print("  reports/backtest_report.md for the net-of-cost record.")
    if n_long == len(ranked):
        print()
        print("  NOTE: the model is long the entire universe today. This is the")
        print("  documented long-bias limitation - on most days it holds ~47.7 of")
        print("  48 names, which is why the ITERATE verdict stands.")
    print("-" * 68 + "\n")


if __name__ == "__main__":
    main()
