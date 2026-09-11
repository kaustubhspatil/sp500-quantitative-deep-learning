"""
Candlestick chart image generation for the transfer-learning model.

Renders 60-day OHLCV windows as clean 224x224 candlestick images (green up /
red down, volume strip along the bottom) with no axes, labels, or gridlines -
the CNN should learn from price geometry, not chart chrome. Images are saved
in an ImageFolder-compatible layout:

    chart_images/<split>/up/    - subsequent 5-day return > 0
    chart_images/<split>/down/  - subsequent 5-day return <= 0
"""

import matplotlib
import matplotlib.pyplot as plt

# Rendering uses fig.savefig + plt.close only (never plt.show), so any backend
# works — including headless Agg. Do NOT force a backend here: matplotlib.use()
# at import time would break inline display for every notebook that imports us.
import numpy as np
import pandas as pd

from . import config

UP_COLOR = "#008300"
DOWN_COLOR = "#e34948"


def render_candlestick_image(window, out_path, size_px=224, volume=True):
    """
    Render one OHLCV window (DataFrame with Open/High/Low/Close/Volume rows
    in date order) to a square PNG at `out_path`.
    """
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    if volume:
        ax = fig.add_axes([0, 0.22, 1, 0.78])
        axv = fig.add_axes([0, 0, 1, 0.18])
    else:
        ax = fig.add_axes([0, 0, 1, 1])
        axv = None

    o = window["Open"].to_numpy()
    h = window["High"].to_numpy()
    l = window["Low"].to_numpy()
    c = window["Close"].to_numpy()
    x = np.arange(len(window))
    up = c >= o
    colors = np.where(up, UP_COLOR, DOWN_COLOR)

    ax.vlines(x, l, h, colors=colors, linewidth=0.7)                    # wicks
    ax.bar(x, np.abs(c - o), bottom=np.minimum(o, c), width=0.75,       # bodies
           color=colors, linewidth=0)
    ax.set_xlim(-1, len(window))
    ax.axis("off")

    if axv is not None:
        v = window["Volume"].to_numpy(dtype=float)
        vmax = v.max() or 1.0
        axv.bar(x, v / vmax, width=0.75, color=colors, linewidth=0)
        axv.set_xlim(-1, len(window))
        axv.set_ylim(0, 1.05)
        axv.axis("off")

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_chart_dataset(prices, tickers, out_dir, *, window=config.CHART_WINDOW,
                           horizon=config.CHART_HORIZON, stride=5,
                           start=None, end=None, verbose=True):
    """
    Slide a `window`-day window (every `stride` days) over each ticker between
    `start` and `end`, render each window to up/ or down/ by the sign of the
    subsequent `horizon`-day return, and write labels.csv with exact values.

    The label uses returns from AFTER the window ends - the image never
    contains the days being predicted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "up").mkdir(exist_ok=True)
    (out_dir / "down").mkdir(exist_ok=True)

    records = []
    for ticker in tickers:
        g = prices[prices["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if start is not None:
            g = g[g["Date"] >= pd.Timestamp(start)].reset_index(drop=True)
        if end is not None:
            g = g[g["Date"] <= pd.Timestamp(end)].reset_index(drop=True)

        for i in range(0, len(g) - window - horizon, stride):
            win = g.iloc[i:i + window]
            close_end = g["Close"].iloc[i + window - 1]
            close_fwd = g["Close"].iloc[i + window - 1 + horizon]
            fwd_ret = close_fwd / close_end - 1
            label = "up" if fwd_ret > 0 else "down"

            end_date = win["Date"].iloc[-1].strftime("%Y%m%d")
            fname = f"{ticker}_{end_date}.png"
            render_candlestick_image(win, out_dir / label / fname)
            records.append({"filename": f"{label}/{fname}", "ticker": ticker,
                            "window_end": win["Date"].iloc[-1],
                            "forward_return": fwd_ret, "label": label})
        if verbose:
            n = sum(1 for r in records if r["ticker"] == ticker)
            print(f"  {ticker}: {n} images")

    labels = pd.DataFrame(records)
    labels.to_csv(out_dir / "labels.csv", index=False)
    return labels
