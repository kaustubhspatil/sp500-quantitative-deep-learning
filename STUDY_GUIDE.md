# Study Guide — Capstone 2: The Deep Learning Trading Project
### A complete plain-language walkthrough: what it is, how it was built, what we found, and what went wrong along the way

---

## 1. The big picture — what is this project?

Imagine a trading desk at a fund ("Jarvis Capital"). Every day it decides which stocks to hold and which to sit out of. It currently uses a simple **linear model** — basically a weighted sum of technical indicators — and its results are mediocre: a Sharpe ratio around 0.4 when the firm wants 0.6. (Sharpe ratio = return earned per unit of risk taken; higher is better; 1.0 is very good for stocks.)

The question we were hired to answer: **can modern deep learning read the market better than that simple model?**

So we took 20 years of daily prices for 48 big US stocks, built four kinds of neural networks — a plain feedforward network, an LSTM (a network with memory), a 1D convolutional network (a pattern scanner), and even a **ResNet that looks at pictures of candlestick charts** — and then settled the question the only way that counts: a simulated **$1,000,000 portfolio traded over 2022–2025, with realistic trading costs subtracted**.

**The end result (the honest version):**
- The best deep learning model made **+$627,692** and hit a Sharpe of **0.85** — beating the firm's target and beating the S&P 500.
- The desk's existing linear signal turned out to be **losing money after costs** (−$271,380) — it trades so frantically that fees eat it alive.
- But simply buying and holding all 48 stocks equally scored **0.88** — slightly better than our best model.
- So the honest recommendation was **"iterate"**: kill the old signal today, but don't hand the new one real money yet.

That nuanced answer — instead of an oversold "deep learning wins!" — is the whole point of the project.

---

## 2. The data — what did we work with?

- **Daily prices (open, high, low, close, volume) for 48 S&P 500 stocks**, 2005–2025 — about 259,000 rows. Two extra tickers, SPY and QQQ (index funds), were kept strictly as *benchmarks*, never as model inputs.
- **The VIX** — the market's "fear gauge" (expected volatility). High VIX = scared market.
- **The 10-year Treasury yield** — the interest-rate backdrop.

The prediction target: **tomorrow's return for each stock** — a tiny, noisy number. This matters enormously (see the next section).

---

## 3. Notebook by notebook — what we did and why

### Notebook 01 — Exploration: learning what markets are like

Before modeling, we established four facts that shaped everything:

1. **Returns have "fat tails."** If daily stock moves followed the textbook bell curve, a ±4-standard-deviation day should happen about once every 63 years. The S&P had *dozens* — clustered in 2008 and March 2020. Meaning: averages hide the days that actually matter; every result must be checked in stressed periods too.
2. **Volatility comes in storms.** Calm days follow calm days; violent days follow violent days. The market has *regimes* (we used VIX bands: calm / normal / stressed / crisis), and a model must be judged per-regime.
3. **Stocks move together.** Average correlation between our 48 stocks is ~0.4, and it *spikes toward 0.7+ in every crisis* — exactly when you'd want diversification, it disappears. So "48 stocks" is not 48 independent bets.
4. **The most important chart of the whole project:** the autocorrelation of returns. Tomorrow's return is almost **pure noise** relative to today's (correlation ≈ −0.1, mostly indistinguishable from zero). What *is* predictable is the *size* of moves (volatility), not their direction. Consequence: any model claiming 70% accuracy on next-day direction has a bug or a leak. A genuinely good model gets **52–55%**. We set that expectation upfront — and it protected us from fooling ourselves later.

### Notebook 02 — Preprocessing: features without cheating

We hand-built **17 features per stock per day**, each with a reason:
- **Momentum**: returns over the last 1, 5, 21, 63 days (does the trend continue or reverse?)
- **Volatility state**: 21-day realized volatility, ATR (how "hot" is this stock right now?)
- **Mean-reversion gauges**: RSI, MACD, Bollinger band position (is it overstretched?)
- **Trend location**: price vs its 10- and 50-day averages
- **Volume anomaly**: is today's volume weirdly high?
- **Market context**: VIX level and change, Treasury yield and change (same for every stock on a given day)

The golden rule: **at any date, a feature may only use information from that date or earlier.** All rolling windows look backward. The one deliberate forward-looking number is the target itself (tomorrow's return — the thing being predicted).

We *proved* this instead of promising it: we deleted all data after mid-2015, recomputed the features, and checked that the surviving rows were **byte-identical**. If any feature secretly peeked forward, deleting the future would have changed the past. (Fun fact: this test initially "failed" — not because of a leak, but because our comparison misaligned rows. Even the test needed debugging.)

**Time splits:** train on 2005–2018, tune on 2019–2021, and **seal 2022–2025 as the test period** — never shuffled, never touched until the end. The scaler (which standardizes features) learned its statistics from training years only. We deliberately *kept* the resulting weirdness — e.g., interest rates in the test years look nothing like the near-zero training era — because that's exactly the unfamiliarity a real deployed model would face.

We also measured how much signal each feature carries (the "information coefficient" — rank correlation with tomorrow's return). Best features: ±0.03. That sounds tiny, and it is — in liquid stock markets, 0.02–0.05 is a real edge and anything much bigger means a mistake. Reality-check passed.

### Notebook 03 — Feedforward networks vs. the incumbent

The setup: three bars to clear.
- **Predict zero** (the "markets are efficient" answer) — the MSE floor.
- **Always predict up** — since stocks rise slightly more often than not, this scores **51.8%** direction accuracy for free. THIS is the bar, not 50%.
- **Linear regression** — a stand-in for the desk's incumbent model.

Then feedforward neural networks — two widths (128→64→32 and 256→128→64 neurons) and three **depths** (2, 3, and 5 hidden layers) — trained with the full professional pattern: mini-batches, the 5-step loop (forward → loss → backward → update → reset), **early stopping** (quit when validation error stops improving, keep the best checkpoint), and TensorBoard logs.

The depth experiment answers "would a bigger network help?" and the answer is *asymmetric*: going 2 → 3 layers gains 1.2 points of accuracy, but 3 → 5 layers gains nothing (+0.04 points) for 3.8× the parameters. Too little capacity hurts; past three layers the limit is signal, not model size. Note also that all three had **identical validation loss** — three networks that trade quite differently look the same on the metric they were trained to minimize. That's the whole argument for judging on directional accuracy and dollars instead.

Results on the sealed test years:
- Neural net: **51.8% direction accuracy, 0.46 gross Sharpe** — beats the linear model (50.4%, 0.37)…
- …but sits *exactly at* the always-long bar (51.8%, 0.47). Deep learning matched the free strategy, gross of costs.
- A fascinating nuance: the *linear* model was better at ranking extreme days (top-vs-bottom decile spread of +13 bps/day vs the network's +1.2), while the *network* was better at day-by-day direction. Two models can be "good" in entirely different ways — only the money test can arbitrate.

### Notebook 04 — Networks with memory (LSTM and 1D CNN)

Instead of one day's snapshot, these models see the **entire trailing 60 days** (a 60×17 window):
- The **LSTM** has gates that learn what to remember and forget across the window — trained with gradient clipping (prevents exploding updates) and learning-rate scheduling.
- The **1D CNN** slides small pattern-detector filters across the window. (Classic trap handled: convolution layers expect dimensions in a different order, so the model permutes them internally — the most common shape bug in this area.)

Two experiments answer "how much history does the model actually use?":
- **Sequence length**: retrain with 20-, 40-, 60-, and 120-day windows. If longer memory helped, accuracy would climb with window length.
- **Temporal attribution**: push the prediction backwards through the network and measure how strongly each of the 60 input days moves the output. This is an X-ray of the model's *effective* memory, which turns out to be far shorter than its 60-day input — gradient mass piles up in the most recent days.

Both agree with notebook 01: old prices carry almost no extra information about tomorrow. That's the documented answer to "why not use longer windows?"

The star of this notebook is **walk-forward validation** — the gold standard for time series. Instead of one train/test split, we trained **four separate LSTMs**, each on all data before its own test window, then tested on the next ~21 months: 2019→COVID, stimulus→2022 bear, bear→AI recovery, 2024→2025 melt-up. Result: direction accuracy **52.3% ± 0.5%** and gross Sharpe **0.59 ± 0.07 across all four regimes** — remarkably *stable*. A single lucky split can flatter a model; four different eras cannot.

Test-period scores: LSTM 51.3% / 0.42 gross; CNN 49.1% / 0.32 (below the coin flip on direction — honest and reported as such). Memory did not unlock a new level of performance — consistent with notebook 01's "returns are noise" finding.

### Notebook 05 — Can a vision model read charts? (ResNet + Grad-CAM)

The fun experiment: traders claim they can "read" candlestick charts. Can a computer-vision model?

- We rendered **10,190 chart images** — 60 days of candlesticks + volume bars, clean 224×224 pictures with no axes or text — each labeled by whether the price rose over the *following* 5 days (the picture never contains the days being predicted).
- We took **ResNet-18**, a network pre-trained on 1.2 million ordinary photos, and adapted it two ways: **feature extraction** (freeze everything, train only a tiny new final layer) and **fine-tuning** (also unfreeze the last two stages). Rules followed: ImageNet normalization (mandatory for pretrained models), and *never* flip chart images horizontally — time flows left to right; a flip reverses cause and effect.

Results, judged against **two** honest bars — accuracy against "always answer up" (55.6%, because markets drift upward), and **AUC** against 0.5 (AUC asks a different question: ignoring the yes/no cutoff, does the model's confidence *rank* charts correctly? A model can fail on accuracy but still have useful ranking — this checks for that):
- Feature extraction: **55.1% accuracy, AUC 0.516** — just below the free answer, with barely-above-chance ranking.
- Fine-tuning: **49.2% accuracy, AUC 0.491** — *worse than chance on both*. It memorized the training images catastrophically within 4 epochs (training error collapsed to near zero while validation error tripled).

The AUC column is what actually rules the approach out. Accuracy alone could be dismissed as a bad cutoff choice; AUC says that even ignoring the cutoff entirely, the model's confidence doesn't order charts by what happens next.
- **Grad-CAM** — a technique that highlights *where in the image* the network looked — showed it attending to sensible regions (recent candles, volume spikes), not corners or whitespace. So the model looks at the right things; there just isn't enough signal in chart *shapes* to beat the market's upward drift.

Verdict: the most expensive signal per unit of accuracy in the project — exploratory only. A negative result, stated plainly, is still a result.

### Notebook 06 — The backtest: every signal becomes dollars

The great equalizer. Every model's predictions went through **one identical trading simulation**:
- Each day, hold an equal-weight long position in every stock the model expects to rise; hold cash otherwise. No leverage, no shorting.
- **Charge 10 basis points (0.10%) of cost on every position change.** A model that flips its mind daily pays constantly; a patient model pays almost nothing.
- Start with $1,000,000 on 3 Jan 2022; run through 29 Dec 2025.
- Benchmarks: buy-and-hold SPY, and buy-and-hold all 48 stocks equally.

The scoreboard (all *net of costs*):

| Strategy | Sharpe | Money made |
|---|---|---|
| **Feedforward NN** | **0.85** | **+$627,692** |
| Universe buy & hold | 0.88 | +$660,768 |
| SPY buy & hold | 0.69 | +$531,026 |
| LSTM | 0.35 | +$185,475 |
| Linear (incumbent) | **−0.30** | **−$271,380** |
| 1D CNN | −0.37 | −$324,557 |

Three headline discoveries:

1. **Turnover is destiny.** The linear model trades 159× its capital per year → $474k in fees → its +0.50 *gross* Sharpe becomes **−0.30 net**. The CNN (246×/yr) is even worse. The feedforward net trades only 4.3×/yr → $17.8k total fees → its result barely moves whether costs are 0 or 25 bps. We swept the whole cost grid to prove the conclusion is robust.
2. **The winning model's secret is that it's almost always long.** It held 47.7 of 48 stocks on the average day. Its performance is mostly a *learned bullish bias*, which the strongly rising 2022–25 test period rewarded. That's why it narrowly loses to just holding everything (0.85 vs 0.88).
3. **Regime honesty:** the strategy earns best in calm markets (+6.3 bps/day) and thins out as volatility rises (+3.3). The "crisis" bucket showed a huge number — from **only 8 days**, statistically meaningless, and we labeled it as such.

**The three-test verdict** (computed, not asserted):
- Beats the incumbent? **PASS** (massively)
- Beats doing nothing (buy & hold)? **FAIL** (0.85 < 0.88)
- Survives doubled costs? **PASS**

→ **Recommendation: ITERATE.** Retire the money-losing incumbent *today* (that alone saves ~$118k/year in fees). Don't give the new model live capital until it proves an edge over passive holding. Four-week plan: train against a cost-aware objective, add volatility-based position sizing, ensemble the feedforward + LSTM, and stress-test on the 2008 crisis via walk-forward.

---

## 4. The challenges — what actually went wrong (and how we fixed it)

1. **PyTorch wouldn't install.** Windows has a 260-character limit on file paths, and the Microsoft Store Python's deeply nested folders plus PyTorch's absurdly long internal paths blew past it. Fix: create a virtual environment at a short path (`C:\Users\mailt\venvs\dl`). Mundane, but it blocks everything until solved.
2. **The leak test that "failed."** Our truncate-and-recompute leak check initially raised an alarm — caused not by a leak but by row misalignment in the comparison itself (deleting the future removes each stock's last row *mid-table*, shifting positions). Fixed by matching rows on (date, stock) keys. Lesson: verify your verifier.
3. **Charts that lied by layout.** Several first-draft figures had real defects: training curves squashed flat by the first epoch's huge loss (fix: log scale), titles overwriting subtitles (fix in the shared plotting module — and the fix itself had a bug, erasing titles, because matplotlib stores left-aligned titles in a different slot than centered ones), and a regime chart showing a nonsense **+2807% annualized return** because annualizing 8 crisis days is mathematically silly (fix: report average daily return instead).
4. **The narrative had to bend to the data, repeatedly.** The decile chart showed the linear model out-ranking the network in the tails — the write-up was changed to describe that tension rather than paper over it. The "iterate" verdict itself is this principle at project scale.
5. **ResNet fine-tuning self-destructed** — textbook overfitting on ~7,000 images with 8.4M trainable parameters. We reported it as the finding it is: with small data, freeze more, train less.
6. **A file-corrupting quote bug in the slide generator** — a patch script accidentally converted every double quote to a single quote, breaking on apostrophes. Fixed by regenerating the file cleanly. Lesson: prefer precise, targeted edits over sweeping find-and-replace on code.

---

## 5. What we found — the takeaways in one place

- **Next-day stock returns are almost pure noise** — 52–55% direction accuracy is the realistic ceiling; treat anything higher as a bug.
- **Costs, not accuracy, decide who survives.** Rankings *inverted* once fees were charged: the gross winner (linear) became the biggest loser.
- **The right benchmark is "do nothing."** Beating a bad model is easy; beating buy-and-hold is the actual bar — and our best model narrowly missed it.
- **Walk-forward stability is worth more than one great number.** Four fresh models, four market eras, nearly identical scores — that's evidence of a real (if modest) signal.
- **Vision models can't out-read the drift** — charts contain less predictive shape than traders believe, and the model's attention maps prove it was at least looking in the right places.
- **An honest "iterate" is a stronger deliverable than an oversold "deploy."** Every number supporting the verdict is reproducible from the sealed test.

---

## 6. Mini-glossary (the words, in plain terms)

| Term | Plain meaning |
|---|---|
| **Sharpe ratio** | Return per unit of risk. ~0.7 = decent, 1.0 = very good for stocks. Firm target: 0.6. |
| **Basis point (bp)** | 0.01%. Our trading cost: 10 bps = 0.10% per position change. |
| **Turnover** | How much you trade per year, as a multiple of your capital. 159×/yr = frantic; 4.3×/yr = patient. |
| **Drawdown** | The fall from a portfolio's peak to its lowest point after. Ours: −21.3% (SPY: −24.5%). |
| **Gross vs net** | Before vs after trading costs. Only net matters. |
| **Directional accuracy** | How often the model calls up/down correctly. The honest bar is ~52% (always-up), not 50%. |
| **Early stopping** | Stop training when validation error stops improving; keep the best checkpoint. |
| **Walk-forward validation** | Retrain repeatedly on expanding history, test on each next slice — a distribution of results instead of one number. |
| **LSTM** | A neural network with learned memory gates, for sequences. |
| **1D CNN** | A network that slides pattern-detecting filters along a time window. |
| **Transfer learning** | Reusing a network pre-trained on one task (photos) for another (charts). |
| **Feature extraction vs fine-tuning** | Freeze the pretrained network and train only a new head — vs also unfreezing deeper layers. Try freezing first. |
| **Grad-CAM** | A heatmap showing where in an image a vision network "looked" when deciding. |
| **Look-ahead bias** | Accidentally using future information in a backtest — the deadliest sin; we proved its absence rather than assumed it. |
| **AUC** | "If I pick one up-chart and one down-chart at random, how often does the model score the up-chart higher?" 0.5 = coin flip. Measures *ranking*, independent of the yes/no cutoff. |
| **Log vs simple return** | log = ln(P₁/P₀), simple = P₁/P₀ − 1. Nearly identical daily (corr. 0.9997); we use simple because portfolio math is exact with it. |
| **Temporal attribution** | Measuring which input days actually move a sequence model's output, by tracing the prediction backwards to the inputs. |
| **VIX regime** | Market state by fear-gauge level: calm (<15), normal, stressed, crisis (>35). |
