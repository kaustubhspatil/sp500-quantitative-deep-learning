# Backtest Report: Deep Learning Equity Signal

**Strategy:** Daily long/flat equal-weight signal on 48 S&P 500 stocks
**Test period:** 3 Jan 2022 → 29 Dec 2025 (1,001 trading days), sealed until evaluation
**Capital:** $1,000,000 · **Costs:** 10 bps per unit of turnover (sensitivity: 0–25 bps)
**Rule (identical for every model):** long every stock with a positive predicted next-day
return, equal weight; cash otherwise. No leverage, no shorting.

---

## 1. Model Comparison Summary

Signal-level metrics on the sealed test period (gross Sharpe = long/flat signal before
costs; portfolio Sharpe = the backtest, net of 10 bps):

| Model | Params | Dir. acc | Signal Sharpe (gross) | Portfolio Sharpe (net) | Max DD | Train time |
|---|---|---|---|---|---|---|
| Linear regression (incumbent) | 18 | 50.4% | 0.37 | **−0.30** | −41.2% | <1 s |
| Feedforward NN 128-64-32 ★ | 13,121 | 51.8% | 0.46 | **0.85** | −21.3% | 156 s |
| Feedforward NN 256-128-64 | 46,721 | 51.5% | 0.45 | — | — | 105 s |
| Feedforward NN 128-64 (2 layers) | 11,009 | 50.5% | 0.34 | — | — | — |
| Feedforward NN 256-…-16 (5 layers) | 49,377 | 51.8% | 0.47 | — | — | — |
| LSTM (64 hidden, 60-day) | 56,641 | 51.3% | 0.42 | 0.35 | −20.5% | ~6 min |
| 1D CNN (32-64-128, 60-day) | 47,617 | 49.1% | 0.32 | −0.37 | −38.9% | ~4 min |
| ResNet-18 feature extraction † | 1,026 trainable | see §7 | n/a | n/a | n/a | see §7 |
| ResNet-18 fine-tuned † | 8.4M trainable | see §7 | n/a | n/a | n/a | see §7 |
| *Always long (reference)* | — | 51.8% | 0.47 | — | — | — |

★ Selected model — highest net Sharpe with the lowest turnover (4.3×/yr) and the
flattest cost-sensitivity slope.
† The chart-image models classify 5-day direction on a strided sample and are evaluated
as classifiers (§7), not in the daily backtest.

## 2. Equity Curve

`figures/bt_equity_curve.png` (selected model vs SPY, with drawdown panel) and
`figures/bt_all_equity_curves.png` (all signals, one engine).

## 3. Strategy Performance (Net of Costs)

| Metric | Strategy (FFNN) | Benchmark (SPY) | Excess |
|---|---|---|---|
| Total return | +62.8% | +53.1% | +9.7 pp |
| Annualized return | +13.1% | +11.3% | +1.8 pp |
| Annualized volatility | 16.0% | 18.0% | −2.0 pp |
| Sharpe ratio | 0.85 | 0.69 | +0.16 |
| Max drawdown | −21.3% | −24.5% | +3.2 pp |
| Daily win rate | 54.1% | — | — |
| Avg positions held | 47.7 of 48 | — | — |
| Annual turnover | 4.3× | ~0 | — |
| Total transaction costs | $17,787 | $0 | — |
| Final portfolio value | $1,627,692 | $1,531,026 | +$96,666 |
| Dollar P&L | +$627,692 | +$531,026 | +$96,666 |

**Passive control:** equal-weight buy & hold of the same 48 stocks returned +66.1%
(Sharpe 0.88, max DD −20.9%). The strategy did **not** beat this control — the decisive
input to the ITERATE recommendation.

## 4. Monthly Returns

Full table in `monthly_returns.csv`; heatmap in `figures/bt_monthly_heatmap.png`.

- **Best month:** Nov 2023, +8.3% · **Worst month:** Apr 2022, −8.9%
- **Positive months:** 30 of 48 (63%) · **Months beating SPY:** 26 of 48 (54%)
- 2022 was the stress year (−1.9, −6.0, +5.9, −8.9 … net −13% for the year vs SPY −18%);
  2023–2025 delivered steady gains with only single-month interruptions.

## 5. Drawdown Analysis

| Event | Start | Trough | Recovered | Depth | Length |
|---|---|---|---|---|---|
| Largest | 4 Jan 2022 | 29 Sep 2022 | 6 Jun 2023 | −21.3% | 357 trading days |
| 2nd | 19 Feb 2025 | 7 Apr 2025 | 26 Jun 2025 | −16.6% | 89 days |
| 3rd | 31 Jul 2023 | 26 Oct 2023 | 11 Dec 2023 | −10.6% | 94 days |

## 6. Regime Analysis (VIX buckets, net of costs)

| Regime | Days | Mean daily return | Sharpe | Notes |
|---|---|---|---|---|
| Low vol (VIX < 15) | 256 | +6.3 bps | 1.71 | Best conditions |
| Normal (15–25) | 588 | +3.7 bps | 0.62 | Consistent |
| High vol (25–35) | 149 | +3.3 bps | 0.24 | Edge thins |
| Crisis (VIX > 35) | 8 | +138 bps | — | Rebound days; sample far too small to trust |

The signal is calm-market seeking. The crisis bucket's outsized number is 8 days of
post-selloff rebounds (April 2025) — reported for completeness, not evidence.

## 7. Chart-Image Models (ResNet-18)

Evaluated as 5-day direction classifiers on 1,880 test images (10 focus stocks,
60-day windows, 5-day stride), against both an accuracy bar and a ranking bar:

| Model | Test accuracy | Test AUC | Trainable params |
|---|---|---|---|
| Majority class ("always up") | 55.6% | 0.500 | — |
| ResNet-18, feature extraction | 55.1% | 0.516 | 1,026 |
| ResNet-18, fine-tuned | 49.2% | 0.491 | 8.4M |

**Neither mode beat always-up.** Feature extraction lands a hair below the majority
bar on accuracy with an AUC of 0.516 — barely-above-chance ranking, not a tradeable
edge. Fine-tuning is *worse than chance on both* (AUC 0.491): it overfit
catastrophically, driving training loss to 0.03 while validation loss tripled, within
four epochs on ~6,900 images. Grad-CAM heatmaps (`figures/tl_gradcam.png`) show the
network attends to candle clusters and the volume strip — economically sensible
regions — but the 5-day drift dominates anything it can read from chart geometry.
The AUC column is what rules the approach out: even ignoring the classification
cutoff, the model's confidence does not order charts by subsequent direction. The
vision approach is exploratory only. Details in notebook 05.

## 7b. How Much History Does the Sequence Model Use?

Two probes, both in notebook 04 (`figures/seq_window_and_attribution.png`):

| Input window | Dir. accuracy | Signal Sharpe (gross) |
|---|---|---|
| 20 days | 51.4% | +0.47 |
| 40 days | 51.1% | +0.37 |
| 60 days (selected) | 51.3% | +0.42 |
| 120 days | 51.4% | +0.42 |

Quadrupling the input window moves accuracy by 0.3 points — inside run-to-run
noise. **Gradient attribution explains why:** tracing the trained 60-day model's
prediction back to its inputs shows the **last 10 days carry 99% of total
gradient mass**. The network's effective memory is roughly two weeks regardless
of how much history it is handed. This is the documented answer to "why not
longer sequences?" — and it corroborates the EDA's autocorrelation finding that
older prices hold almost no incremental information about tomorrow.

## 8. Look-Ahead Bias Checks

- [x] Features computed from trailing windows only — *proven* by recomputation on
      truncated data (notebook 02)
- [x] Train/val/test strictly temporal; test years sealed until notebook 06
- [x] Scaler fit on training years only, persisted and reused
- [x] Predictions at close of *t* target the *t*→*t+1* return
- [x] Sequence warmup windows draw only on earlier periods
- [x] Walk-forward validation performed (4 expanding folds, notebook 04)
- [x] Costs charged on every position change, including day one

## 9. Sensitivity Analysis

Net Sharpe across the cost grid (`figures/bt_cost_sensitivity.png`):

| Costs | Linear | FFNN ★ | LSTM | 1D CNN |
|---|---|---|---|---|
| 0 bps | +0.50 | +0.87 | +0.62 | +0.81 |
| 5 bps | +0.10 | +0.86 | +0.49 | +0.22 |
| **10 bps (assumed)** | **−0.30** | **+0.85** | **+0.35** | **−0.37** |
| 15 bps | −0.70 | +0.83 | +0.21 | −0.95 |
| 20 bps | −1.10 | +0.82 | +0.08 | −1.54 |
| 25 bps | −1.50 | +0.81 | −0.06 | −2.13 |

The conclusion is cost-robust for the selected model and catastrophic for the
high-turnover signals: doubling assumed costs to 20 bps changes the FFNN's Sharpe by
−0.03 and the incumbent's by −0.80.
