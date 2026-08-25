# Deep Learning Equity Signal — Module 2 Capstone

**Jarvis Capital — can deep learning beat the desk's linear signal?**

The equity desk's rules-based strategy has delivered a Sharpe ratio of ~0.4 against a firm
target of 0.6. This project tests whether deep learning improves signal quality: three
neural architectures (feedforward, LSTM, 1D CNN) plus an ImageNet-pretrained ResNet-18
reading candlestick charts, all evaluated on 20 years of S&P 500 data with a sealed
2022–2025 test period, and settled by a $1M backtest net of transaction costs.

## Repository layout

```
capstone/
├── data/
│   ├── sp500_stocks.csv          ← raw daily OHLCV, 50 tickers, 2005–2025
│   ├── vix.csv, treasury_10y.csv ← market context series
│   ├── processed/                ← model-ready parquet + predictions (built by nb 02–04)
│   └── chart_images/             ← candlestick images for transfer learning (built by nb 05)
├── notebooks/
│   ├── 01_eda.ipynb              ← two decades of market structure, and what it implies
│   ├── 02_preprocessing.ipynb    ← 17 leak-free features, temporal splits, scaling
│   ├── 03_feedforward_nn.ipynb   ← linear incumbent vs feedforward networks
│   ├── 04_sequence_models.ipynb  ← LSTM + 1D CNN, walk-forward validation
│   ├── 05_transfer_learning.ipynb← ResNet-18 on chart images + Grad-CAM
│   └── 06_backtest.ipynb         ← $1M, 10 bps costs, verdict
├── src/                          ← all reusable code (features, models, training, backtest…)
├── models/                       ← trained weights (.pt) and the fitted scaler (.pkl)
├── reports/                      ← executive summary, backtest report, model documentation
│   └── figures/                  ← every chart, publication-ready PNG
├── runs/                         ← TensorBoard logs (tensorboard --logdir=runs)
└── README.md
```

## Reproduce end-to-end

**1. Environment** (Python 3.11, NVIDIA GPU optional but recommended):

```bash
python -m venv venv && venv\Scripts\activate          # Windows
pip install numpy pandas matplotlib seaborn scikit-learn joblib pyarrow scipy \
            tensorboard pillow jupyter nbformat nbconvert ipykernel
# GPU (CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# or CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Windows note:** install into a venv at a *short path* (e.g. `C:\venvs\dl`) — PyTorch's
> package paths can exceed the 260-character Windows path limit inside deeply nested
> site-packages directories.

**2. Run the notebooks in order** (each persists what the next one loads):

```bash
cd capstone
jupyter notebook   # run 01 → 06 top to bottom
```

Or headless:

```bash
cd capstone/notebooks
for nb in 01_eda 02_preprocessing 03_feedforward_nn 04_sequence_models 05_transfer_learning 06_backtest; do
    python -m nbconvert --to notebook --execute --inplace $nb.ipynb
done
```

Approximate runtimes on an RTX 4060 laptop: 01–02 ≈ 2 min each, 03 ≈ 5 min,
04 ≈ 30 min (walk-forward retrains the LSTM four times), 05 ≈ 30 min
(first run also renders ~10k chart images), 06 ≈ 2 min.

**3. Watch training live (optional):**

```bash
tensorboard --logdir=runs
```

**4. Live inference demo** (the Day-10 presentation walkthrough) — takes raw
prices, runs the full pipeline, prints tomorrow's signal:

```bash
python predict.py --top 10
```

## Design decisions that matter

- **SPY and QQQ are benchmarks only** — never model inputs.
- **Strict temporal discipline everywhere:** train 2005–2018, validate 2019–2021,
  test 2022–2025; the scaler is fit on train only; sequence windows never cross the
  boundaries; notebook 02 *proves* the features are leak-free by recomputing them on
  truncated data.
- **Pooled panel:** all 48 stocks train one model per architecture (the EDA shows one
  dominant market factor, making pooling statistically reasonable and giving 160k+
  training rows).
- **Honest yardsticks:** directional accuracy is judged against always-long (~52%), not
  50%; Sharpe is judged net of 10 bps costs against SPY buy-and-hold; every claim is
  stress-tested across VIX regimes, four walk-forward folds, and a 0–25 bps cost grid.

## Deliberate deviations from the brief

Every one of these was a decision, not an omission. They are listed here so a
reviewer can challenge the reasoning rather than hunt for the gap.

| Brief says | What this project does | Why |
|---|---|---|
| Target = next-day **log** return | Simple returns end-to-end | At daily horizons the two correlate 0.9997 (median gap 0.0013 bp — measured in notebook 01 §3). Simple returns make the portfolio arithmetic *exact*: an equal-weight portfolio's return is the weighted mean of simple returns, and equity compounds as ∏(1+r). Log-return additivity would only matter for aggregating one asset across time, which nothing downstream does. |
| Indicators via the `ta` library, incl. **OBV** | 17 indicators hand-written in `src/features.py`; no OBV | Hand-built indicators are auditable, unit-testable, and provably trailing-only (notebook 02 proves it by recomputation) rather than trusting a third-party implementation. The set covers all six required families — momentum, volatility, oscillators, bands, trend, volume — with `volume_z` in place of OBV; OBV's cumulative construction adds a path-dependent level that the z-score captures more stably. |
| 70 / 15 / 15 split by row count | Calendar splits: 2005–18 / 2019–21 / 2022–25 (≈67/16/17) | Aligns split boundaries with market regimes and whole years, so the test period is a coherent economic era (bear → recovery → melt-up) rather than an arbitrary row index. Same discipline, more defensible boundaries. |
| Chart images from **30-day** windows | 60-day windows | Matches the sequence models' 60-day input, so the vision and LSTM experiments see the same amount of history and their results are directly comparable. |
| Walk-forward: expand by one **month** at a time | Four expanding folds of ~21 months each | Monthly refits over 2019–2025 would mean ~80 LSTM trainings for a model already shown to be near the noise floor. Four folds spanning distinct regimes (COVID crash, 2022 bear, AI recovery, melt-up) answer the actual question — *is the edge stable across market conditions?* — at 5% of the compute. Per-fold spread is reported, as the brief requires. |
| Feature extraction = "extract 512-d vectors, fit a logistic regression on them" | Frozen backbone + a trainable `nn.Linear(512, 2)` head, cross-entropy loss | These are the same model. A linear layer trained under cross-entropy *is* multinomial logistic regression on the 512-d features; doing it in-graph avoids materializing the feature matrix and keeps one inference path for both modes. |
| Optional: GRU comparison, Huber loss, embeddings, multi-seed runs | Not run | Out of scope once the core finding (signal ≈ noise floor; costs decide) was established. Each is listed in the executive summary's four-week iteration plan. |

## Deliverables

- [`reports/executive_summary.md`](reports/executive_summary.md) — one page, for the CIO
- [`reports/backtest_report.md`](reports/backtest_report.md) — full backtest, for the desk
- [`reports/model_documentation.md`](reports/model_documentation.md) — for model validation (OSFI E-23 style)
- `reports/figures/` — all charts at 150 dpi
