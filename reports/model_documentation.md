# Model Documentation — Deep Learning Equity Signal

**Model:** Feedforward Neural Network "FFNN 128-64-32" (daily long/flat equity signal)
**Prepared for:** Model Validation / Technical Review Committee
**Prepared by:** Kaustubh Patil
**Date:** 20 August 2026
**Status:** Candidate — recommended for iteration, not yet for production deployment

---

## 1. Model Purpose

The model supports the equity desk's daily signal generation: at each market close it
produces a predicted next-day return for each of 48 large-cap US stocks. The trading rule
consumes the *sign* of the prediction — an equal-weight long position in every stock with a
positive prediction, cash otherwise (long/flat, no leverage, no shorting).

It is a candidate replacement for the desk's incumbent linear-regression signal, whose
backtested performance over the 2022–2025 test period is **negative net of transaction
costs** (Sharpe −0.30 after 10 bps per unit turnover; see §5).

## 2. Methodology

| Item | Specification |
|---|---|
| Algorithm | Feedforward network: 3 hidden blocks (Linear → BatchNorm → ReLU → Dropout 0.3), widths 128-64-32, linear output head; 13,121 parameters |
| Loss / optimizer | MSE; Adam (lr 1e-3, weight decay 1e-5), batch 1024 |
| Regularization | Dropout, batch normalization, early stopping (patience 10 epochs on validation loss), best-weight checkpointing |
| Features | 17 per (date, stock): momentum (1/5/21/63-day returns), volatility (21-day realized, ATR-14), oscillators (RSI-14, MACD histogram, Bollinger %B and bandwidth), trend (price vs SMA-10/50), volume z-score, market context (VIX level and 5-day change, 10Y Treasury level and 21-day change) |
| Feature scaling | StandardScaler **fit on training years only**, persisted (`models/feature_scaler.pkl`) |
| Training data | 160,595 (date, stock) rows: 48 S&P 500 stocks, 2005-04-05 → 2018-12-31 |
| Validation data | 36,336 rows, 2019 → 2021 (early stopping and model selection) |
| Test data | 48,048 rows, 2022 → 2025 — sealed until final evaluation |
| Target | Next-day simple return |
| Implementation | PyTorch 2.13 (CUDA); all code in `capstone/src/`; training reproducible with fixed seed 42 |

## 3. Performance (Sealed Test Period, 2022-01 → 2025-12)

**Signal quality** (prediction-level):

| Metric | FFNN | Linear incumbent | Always-long bar |
|---|---|---|---|
| Directional accuracy | 51.8% | 50.4% | 51.8% |
| MSE | 4.0e-4 | 4.0e-4 | 4.0e-4 (predict-zero floor) |
| Signal Sharpe (gross) | 0.46 | 0.37 | 0.47 |

**Portfolio backtest** ($1,000,000, net of 10 bps costs on turnover):

| Metric | FFNN | Linear incumbent | SPY B&H | Universe B&H |
|---|---|---|---|---|
| Annualized return (net) | **+13.1%** | −7.7% | +11.3% | +13.6% |
| Annualized volatility | 16.0% | 19.9% | 18.0% | 16.0% |
| Sharpe (net) | **0.85** | −0.30 | 0.69 | 0.88 |
| Max drawdown | −21.3% | −41.2% | −24.5% | −20.9% |
| Annual turnover | 4.3× | 159× | ~0 | ~0 |
| Total costs paid | $17,787 | $473,796 | $0 | ≈$0 |
| Dollar P&L | **+$627,692** | −$271,380 | +$531,026 | +$660,768 |

## 4. Baseline Comparison and Honest Caveat

The model decisively beats the incumbent. However, it does **not** beat the equal-weight
universe buy-and-hold (0.85 vs 0.88 Sharpe), and it holds 47.7 of 48 stocks on an average
day: most of its performance is a *learned long bias*, which the strongly bullish
2022–2025 test period rewarded. Its residual timing decisions (skipping ~0.3 stocks/day)
have not demonstrated standalone alpha. This is the central reason the recommendation is
**iterate**, not deploy — see the Executive Summary.

Supporting robustness evidence:

- **Walk-forward validation (sequence-model sibling, notebook 04):** four expanding-window
  folds spanning 2019–2025 produced directional accuracy 52.3% ± 0.5% and gross Sharpe
  +0.59 ± 0.07 — stable across the COVID crash, the 2022 bear, and two melt-ups.
- **Cost sensitivity:** the FFNN's net Sharpe moves only 0.87 → 0.81 as costs rise 0 → 25
  bps (turnover 4.3×/yr). The incumbent collapses +0.50 → −1.50 over the same grid.

## 5. Limitations and Known Failure Modes

1. **Long-bias dependence.** In a sustained bear market, a signal that is long 99% of the
   time will track the market down; the −21.3% drawdown in 2022 demonstrates this.
2. **Regime shift.** Training data (2005–2018) contains near-zero interest rates for a
   decade; the model met 4–5% rates in the test period only through the Treasury feature.
   Structural breaks (new market microstructure, changed correlations) degrade it silently.
3. **No fundamental or news information.** Features are purely price/volume/market-derived;
   earnings surprises, M&A, and macro shocks are invisible until they appear in prices.
4. **Single-asset-class scope.** Evaluated only on 48 large-cap US equities; no evidence
   of generalization to other universes.
5. **MSE-trained, sign-consumed.** The loss optimizes squared error, but the trading rule
   uses only the sign — a known objective mismatch (candidate fix in the iteration plan).
6. **Short effective memory.** Gradient attribution on the sequence sibling shows the last
   10 days carry 99% of the model's input influence, and window lengths from 20 to 120 days
   score within 0.3 accuracy points of each other. The model cannot exploit long-horizon
   structure — anything slower than about two weeks is invisible to it.

## 6. Monitoring Plan

| Monitor | Threshold | Action |
|---|---|---|
| Rolling 63-day directional accuracy | < 50% for 21 consecutive days | Alert; review feature drift |
| Rolling 126-day net Sharpe | < 0 | Reduce allocation 50% |
| Feature PSI (train vs live, monthly) | PSI > 0.2 on any feature | Trigger revalidation |
| VIX regime | VIX > 35 | Risk override: cut gross exposure 50% (crisis behavior is the least-tested regime) |
| Drawdown from peak | −15% | Escalate to risk committee |
| Retraining cadence | Monthly, expanding window | Champion/challenger vs live model |

## 7. Data Quality Assessment

- Source: Yahoo Finance daily OHLCV (adjusted closes), 2005–2025; VIX and 10Y Treasury
  from the same source. 259,073 rows, no missing trading days in the core universe.
- Three tickers (AVGO, META, ABBV) begin later due to IPO/spinoff; per-ticker rolling
  features start when the ticker starts (no imputation of pre-listing history).
- **Leakage controls verified programmatically:** features recomputed on truncated data
  are identical (notebook 02); splits are strictly temporal; scaler fit on train only;
  costs charged on every position change. Survivorship caveat: the universe was selected
  from *current* S&P 500 constituents, which flatters absolute returns for **all**
  strategies and benchmarks equally; relative comparisons remain valid.
