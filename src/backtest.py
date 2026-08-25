"""
Portfolio backtesting engine with transaction costs.

Strategy simulated: each trading day, go equal-weight LONG every stock whose
predicted next-day return is positive; hold cash for the rest. Costs are
charged on turnover (the change in weights day over day), so a position held
for weeks pays once, while a signal that flips daily pays constantly.

Deliberately simple - trades at the close, long/flat only, no leverage,
no slippage beyond the cost parameter. Not a production engine.
"""

import numpy as np
import pandas as pd

from . import config
from .evaluation import annualized_sharpe, max_drawdown


class PortfolioBacktester:
    """
    Args:
        predictions:    DataFrame (dates x tickers) of predicted next-day returns.
        actual_returns: DataFrame (dates x tickers) of realized next-day returns,
                        aligned with `predictions` (same index/columns).
        cost_bps:       round-trip transaction cost in basis points of turnover.
        capital:        starting capital in dollars.
    """

    def __init__(self, predictions, actual_returns,
                 cost_bps=config.TRANSACTION_COST_BPS,
                 capital=config.INITIAL_CAPITAL):
        predictions, actual_returns = predictions.align(actual_returns, join="inner")
        self.predictions = predictions
        self.actual = actual_returns
        self.cost = cost_bps / 10_000
        self.capital = capital
        self._run()

    def _run(self):
        # weights held ON each day: equal weight across positive-signal names
        signal = (self.predictions > 0) & self.actual.notna()
        n_long = signal.sum(axis=1)
        weights = signal.div(n_long.replace(0, np.nan), axis=0).fillna(0.0)

        gross = (weights * self.actual.fillna(0.0)).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = weights.iloc[0].abs().sum()  # entering day one
        costs = turnover * self.cost

        self.weights = weights
        self.turnover = turnover
        self.daily_returns = gross - costs
        self.daily_costs_paid = costs
        self.equity = self.capital * (1 + self.daily_returns).cumprod()
        self.drawdown = self.equity / self.equity.cummax() - 1.0

    # ── Reporting ────────────────────────────────────────────────────────

    def stats(self):
        r = self.daily_returns
        years = len(r) / config.TRADING_DAYS
        total_return = self.equity.iloc[-1] / self.capital - 1
        return {
            "total_return": float(total_return),
            "annualized_return": float((1 + total_return) ** (1 / years) - 1),
            "annualized_vol": float(r.std() * np.sqrt(config.TRADING_DAYS)),
            "sharpe": annualized_sharpe(r),
            "max_drawdown": max_drawdown(self.equity),
            "win_rate": float((r[r != 0] > 0).mean()),
            "avg_positions": float((self.weights > 0).sum(axis=1).mean()),
            "annual_turnover": float(self.turnover.sum() / years),
            "total_costs": float((self.daily_costs_paid * self.equity.shift(1)
                                  .fillna(self.capital)).sum()),
            "final_value": float(self.equity.iloc[-1]),
            "dollar_pnl": float(self.equity.iloc[-1] - self.capital),
        }

    def monthly_returns(self):
        return (1 + self.daily_returns).resample("ME").prod() - 1

    def drawdown_events(self, top_n=3):
        """The `top_n` deepest peak-to-trough episodes with recovery times."""
        dd = self.drawdown
        in_dd = dd < 0
        episodes, start = [], None
        for date, flag in in_dd.items():
            if flag and start is None:
                start = date
            elif not flag and start is not None:
                seg = dd.loc[start:date]
                episodes.append({"start": start, "trough": seg.idxmin(),
                                 "end": date, "depth": float(seg.min()),
                                 "days": int(len(seg))})
                start = None
        if start is not None:  # still underwater at the end
            seg = dd.loc[start:]
            episodes.append({"start": start, "trough": seg.idxmin(), "end": pd.NaT,
                             "depth": float(seg.min()), "days": int(len(seg))})
        return (pd.DataFrame(episodes).sort_values("depth").head(top_n)
                .reset_index(drop=True))

    def regime_table(self, vix):
        """Performance sliced by VIX regime on each trading day."""
        vix = vix.reindex(self.daily_returns.index).ffill()
        bins = [-np.inf, 15, 25, 35, np.inf]
        labels = ["Low vol (VIX<15)", "Normal (15-25)", "High vol (25-35)", "Crisis (VIX>35)"]
        regime = pd.cut(vix, bins=bins, labels=labels)
        rows = []
        for label in labels:
            r = self.daily_returns[regime == label]
            if len(r) == 0:
                continue
            invested = self.weights[regime == label].sum(axis=1) > 0
            rows.append({
                "regime": label, "days": len(r),
                # mean daily return in bps: comparable across regimes without the
                # absurd compounding of annualizing a handful of crisis days
                "mean_daily_bps": float(r.mean() * 10_000),
                "total_return": float((1 + r).prod() - 1),
                "sharpe": annualized_sharpe(r),
                "pct_days_invested": float(invested.mean()),
            })
        return pd.DataFrame(rows).set_index("regime")


def buy_and_hold(benchmark_returns, capital=config.INITIAL_CAPITAL):
    """Equity curve and stats for a passive benchmark (e.g. SPY)."""
    r = benchmark_returns.dropna()
    equity = capital * (1 + r).cumprod()
    years = len(r) / config.TRADING_DAYS
    total = equity.iloc[-1] / capital - 1
    return equity, {
        "total_return": float(total),
        "annualized_return": float((1 + total) ** (1 / years) - 1),
        "annualized_vol": float(r.std() * np.sqrt(config.TRADING_DAYS)),
        "sharpe": annualized_sharpe(r),
        "max_drawdown": max_drawdown(equity),
        "final_value": float(equity.iloc[-1]),
        "dollar_pnl": float(equity.iloc[-1] - capital),
    }
