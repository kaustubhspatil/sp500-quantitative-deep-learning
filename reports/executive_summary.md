# Executive Summary: Deep Learning Equity Signal Evaluation

**Prepared by:** Kaustubh Patil
**Date:** 20 August 2026
**Audience:** CIO, Head of Risk, Technical Committee

---

## Strategy Overview

Jarvis Capital's equity desk generates daily trading signals from a rules-based linear
model whose trailing Sharpe ratio (~0.4) sits below the firm's 0.6 target. We evaluated
whether deep learning improves signal quality: **six models across four architectures**
(feedforward networks, LSTM, 1D CNN, and an ImageNet-pretrained ResNet-18 reading
candlestick charts) trained on 2005–2018 data for 48 S&P 500 stocks, tuned on 2019–2021,
and evaluated on a **sealed 2022–2025 test period** through an identical $1,000,000
long/flat backtest, net of 10 basis points transaction costs.

## Key Results (net of costs, 2022–2025)

| Metric | Best DL model (FFNN) | Linear (incumbent) | Buy & hold SPY |
|---|---|---|---|
| Annualized return (net) | **+13.1%** | −7.7% | +11.3% |
| Annualized volatility | 16.0% | 19.9% | 18.0% |
| Sharpe ratio | **0.85** | **−0.30** | 0.69 |
| Max drawdown | −21.3% | −41.2% | −24.5% |
| Directional accuracy | 51.8% | 50.4% | n/a |
| Dollar P&L ($1M capital) | **+$627,692** | **−$271,380** | +$531,026 |

*Figure: `figures/bt_equity_curve.png` — strategy vs SPY with drawdown panel.*

**Finding 1 — the incumbent destroys value after costs.** The linear signal trades ~159×
its capital per year; at 10 bps that is $473,796 in transaction costs over four years,
turning a +0.50 gross Sharpe into **−0.30 net**. Retiring it as a standalone signal is
justified on this evidence alone.

**Finding 2 — the deep learning signal clears the firm's target.** The feedforward
network's 0.85 net Sharpe exceeds the 0.6 target and beats SPY, with a shallower worst
drawdown. Its edge is *cost-robust*: turnover of only 4.3×/yr means its Sharpe declines
just 0.87 → 0.81 as assumed costs rise from 0 to 25 bps. Walk-forward validation of the
sequence sibling (four regime folds, 2019–2025) shows the signal family is stable:
gross Sharpe +0.59 ± 0.07 across the COVID crash, the 2022 bear, and two melt-ups.

**Finding 3 — the honest caveat.** An equal-weight buy-and-hold of the same 48 stocks
scored **0.88** — marginally above the FFNN. The network holds 47.7 of 48 stocks on an
average day: most of its performance is a learned long bias that the bullish test period
rewarded, and its residual stock-skipping has not yet demonstrated standalone alpha.

## Risk Analysis

- **Worst episode:** −21.3% peak-to-trough (Jan → Sep 2022, the rate-hike bear market),
  recovering by Jun 2023; SPY drew down −24.5% in the same episode. Worst single month
  **Apr 2022, −8.9%**; best **Nov 2023, +8.3%**; 30 of 48 months positive.
- **Regime behavior:** net performance degrades as volatility rises — +6.3 bps/day when
  VIX < 15, +3.7 in normal markets, +3.3 when VIX is 25–35. The signal is *calm-market
  seeking*; a bear-market repeat is its dominant risk because it is nearly always long.
- **Failure mode:** in a sustained downturn the strategy would track the market down;
  the proposed VIX > 35 risk override (cut gross exposure 50%) is untested at scale —
  the test period contained only 8 such days.

## Recommendation — ITERATE

1. **Retire the incumbent linear signal now.** It is value-destroying net of costs under
   every assumption tested.
2. **Do not yet allocate live capital to the DL signal as standalone alpha.** It has not
   beaten the passive equal-weight alternative (0.85 vs 0.88), and its long-bias
   dependence is untested in a bear regime.
3. **Four-week iteration plan:** (a) train against a cost-aware, sign-oriented objective
   instead of MSE; (b) add a volatility-targeted position-sizing overlay instead of
   binary long/flat; (c) ensemble the FFNN with the walk-forward-stable LSTM;
   (d) extend the test to international universes and to the 2008 crisis via the
   held-back 2005–2018 walk-forward folds.

## Next Steps (If Given 4 More Weeks)

1. Cost-aware objective and turnover regularization at training time
2. Volatility-scaled position sizing and the VIX crisis override, backtested properly
3. FFNN + LSTM ensemble with champion/challenger monthly retraining
4. Production inference pipeline: nightly feature build → prediction → dashboard, with
   the monitoring thresholds in `model_documentation.md`
