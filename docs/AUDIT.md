# StrategyFinder: adversarial quant audit

*Audit date 2026-11-15, against commit `7c56dcb`. Line numbers refer to that commit.*

## Verdict in one paragraph

The engine's mechanics are mostly clean. Indicators are causal, fills happen at the next open, stops are filled stop-first, and costs are charged on every fill. The selection-bias correction (a Deflated Sharpe Ratio inside the fitness) does its job on pure noise. **Research integrity is where it fails.** The "validation" segment is optimised on directly, so the app has no out-of-sample data during search. It is a single 80% window hill-climbed by up to a million iterations, and its only out-of-sample test is one 20% holdout that gets revealed 25 strategies at a time and can be sorted. Nothing in the selection rewards parameter plateaus, regime consistency or sample size beyond soft penalties. The multiple-testing ledger resets every run. On pure random walks the app's top-ranked strategy reports **+13% to +206% in-sample ROI** and **validation Sharpe of 0.9 to 3.9**, and that is the number the headline tile shows. The DSR column and the post-run verdict correctly call it luck, but the UI leads with the inflated figure.

---

## 0. Empirical evidence (null-hypothesis runs)

I ran the real optimiser worker on i.i.d. random walks: 1,826 daily bars (≈5y of crypto), σ = 3.5%/day, default settings (2% risk sizing, 0.1% fee, 0.05% slippage, stagnation 5,000). The strategy set contains no edge by construction. There were 6 seeds per null, and each run's whole 25-strategy board was inspected.

| | Driftless walk | Walk with +40%/yr drift |
|---|---|---|
| Iterations per run | 6,378 – 15,677 | 5,688 – 16,151 |
| #1 validation Sharpe (annualised) | 0.94 – 3.86 | 1.42 – 2.76 |
| #1 validation trades | 17 – 50 | 27 – 55 |
| #1 in-sample ROI (shown live in the headline tile) | +13% – +206% | +35% – +144% |
| #1 DSR | 0 – 15% | 0 – 13% |
| Board verdicts Robust / Fragile / Overfit | 0 / 0 / 150 | 0 / 0 / 150 |
| Board strategies with holdout ROI > 0 | 39 / 150 | **96 / 150** |
| Runs where the *best-of-25* holdout ROI is positive | 4 / 6 (up to +35%) | **6 / 6** (up to +27%) |
| #1 base fitness > median ±1-step neighbour fitness | **6 / 6** | **6 / 6** |
| #1 in-sample ROI: base vs. median neighbour | e.g. 206 → 120, 92 → 53 | e.g. 125 → 80, 144 → 111 |

On 3,000 random parents (one mutation each), **4.3%** of children were identical to the parent after repair and a further **6.4%** were different genomes with identical trades and P&L. About **11% of mutations are wasted**, and the 6.4% still count as independent trials.

Takeaways:
1. The DSR gate works, and on noise it is the only thing standing between the user and a false discovery.
2. The champion always sits on a peak of its neighbourhood. That is the expected outcome of max-selection on a noisy surface (regression to the mean), and the selection never corrects for it.
3. On a trending asset, the holdout test used by the verdict, `holdout.roi > 0`, is passed by 64% of noise strategies. A user who sorts the leaderboard by Holdout ROI finds a "winner" every time.

---

## 1. Backtesting methodology: gap analysis

| Institutional practice | Status | Evidence |
|---|---|---|
| Causal indicators, no full-sample normalisation of signals | **Pass** | Every indicator is recursive or a trailing window (`src/engine/indicators.ts:36-388`). Signals are evaluated on the close of bar *t* and filled at the open of *t+1* (`backtester.ts:340-399`). |
| Intrabar fill realism | **Pass** (with caveat) | Gaps fill at the open, the stop is assumed hit first when a bar touches both levels, and ATR is taken from the signal bar (`backtester.ts:401-418`). Caveat: basket high and low are the **sum of component highs and lows** (`basket.ts:149-151`). That envelope is not achievable, so basket take-profits fill on prices that never existed. |
| No full-sample statistics in features | **Fail (minor)** | Basket volume is divided by each leg's **whole-history mean volume, holdout included** (`basket.ts:143-153`). This changes the relative leg weighting inside the `VOLUME` indicator. |
| Hypothesis-driven design | **Fail** | Archetypes (`genome.ts:400-426`) constrain the grammar, but no hypothesis is declared before the data is seen. The search explores about 170 structural niches (archetype × indicator families × exit mode, `genome.ts:965-968`) and keeps whatever fits. |
| Strict train / validation / holdout separation | **Fail** | The split is chronological 60/20/20 (`backtester.ts:549-553`), but validation is an **optimisation target**. `assetBaseScore` weights validation excess return at 1.5, more than any other term (`backtester.ts:1084-1087`), and the DSR term is computed on validation (`evaluation.ts:200`). Validation is effectively in-sample. |
| Walk-forward / purged CV | **Fail** | Not implemented. There is one fixed boundary. |
| Untouched, one-shot holdout | **Partial** | Sealed during search (pass: `runBacktest` stops at `holdoutStart`, `backtester.ts:998`). But `finish()` reveals the holdout for **all 25** board entries (`optimizer.worker.ts:794-805`). The Holdout ROI column is sortable (`Leaderboard.tsx:200-214`), which turns the holdout into a selection set. The reveal ledger is keyed on the **end date string** (`holdoutLog.ts:14-17`), so it resets every calendar day. Switching the range between 1y, 2y and 5y also produces a "new" fingerprint, even though their holdouts (last 73, 146 and 365 days) are nested. |
| Commission / fees | **Pass** | Charged per side on every fill (`backtester.ts:352, 383-384, 424`). |
| Spread, volatility-dependent slippage, impact, capacity | **Fail** | Slippage is a constant % of price whatever the volatility, bar size, liquidity or order size (`backtester.ts:264, 351, 373`). A 0.3% round trip is trivial on daily BTC but large relative to a 15m bar's range. DOGE and ADA get BTC-grade costs. |
| Short financing / funding | **Fail** | Default `fundingPct8h: 0` (`config.ts:78`) even though `DEFAULT_FUNDING_PCT_8H = 0.01` exists (`config.ts:54`). There is no stock borrow fee, so long-short mode shorts for free. |
| Total-return data | **Fail** | Yahoo `quote.close` is split-adjusted but not dividend-adjusted (`market-data.ts:298-313`). Buy & Hold on SPY loses about 1.3%/yr of yield all the time, while a strategy loses it only when invested. That biases "alpha vs. B&H" in favour of strategies. |
| Survivorship bias | **Fail** | The default universes are hindsight winners (`trading.ts:420-421`: SOL, DOGE, NVDA, …). Nothing delisted (LUNA, FTT) can appear, and there is no point-in-time universe. |
| Silent data substitution | **Pass** (flagged) | The synthetic fallback is banner-flagged (`App.tsx:474-480`). It should still block holdout reveal and verdicts. |
| Minimum sample size | **Fail** | `MIN_TRADES = 100` and `MIN_VALIDATION_TRADES = 25` are **soft** penalties **pooled across the whole universe** (`backtester.ts:1050-1053, 1132-1133`). With 6 assets, 4 validation trades per asset gets no penalty. The noise champions had 17–55 validation trades, and over a 1-year window the annual-Sharpe standard error is about 1.0. |
| Regime diversity | **Fail** | No regime labelling or per-regime reporting. Each of the three segments is one contiguous period. On 5y daily data as of today, validation is roughly Nov-2024 → Nov-2025, a single regime. |
| Tail risk | **Partial** | The intrabar drawdown penalty is good (`backtester.ts:1088-1089`). There is no CVaR, no Monte Carlo of trade order, and the "remove top 5 trades" test is post-hoc only (`optimizer.worker.ts:384-391`), never part of selection. |
| "Robust universe" | **Partial** | The median across assets is a good idea (`backtester.ts:1131`). But the default crypto universe has pairwise correlations around 0.7–0.9 over **the same dates**, so 6 assets give roughly 1.5–2 independent samples and **zero** extra regime diversity. |

---

## 2. Overfitting and data-snooping audit

### 2.1 In-sample vs. out-of-sample
- **No OOS data exists during search.** Every candidate is scored on train *and* validation (`evaluation.ts:156-169`). The hill climber iterates against the same 80% window until stagnation, which defaults to 5,000 non-improving iterations and can go up to **1,000,000** (`trading.ts:405-408`).
- **Leakage across runs.** `RunningStats` is created fresh per run (`optimizer.worker.ts:600`). Ten runs of 15k iterations on the same data each report a DSR built on N ≈ 15k, when the true N is 150k. Seeds, tickers, intervals, ranges, fees and stagnation limits are all researcher degrees of freedom that are never counted.
- **Leakage through the holdout.** See §1. The holdout is not a one-shot test as implemented.

### 2.2 Parameter sensitivity: spikes or plateaus
- The fitness evaluates **only the point**. `tweak` mutations (45% of all mutations, `genome.ts:771-791`) move one parameter one grid step, which is hill-climbing on a noisy surface. In every null run the champion beat its neighbourhood median (§0).
- `parameterVariants` (`genome.ts:909-938`) is computed only for the post-hoc report (`optimizer.worker.ts:393-403`). The "fragile" flag (worst neighbour ROI < 50% of base) never feeds back into ranking.
- The "±20%" test is not ±20%. `LOOKBACK_GRID = [5,7,10,14,20,21,26,30,50,100,150,200]` (`genome.ts:74`), so a 50-period perturbation becomes 100 (+100%), while 20 ↔ 21 is a +5% near-duplicate.

### 2.3 Objective function (`backtester.ts:1081-1141`)
| Term | Problem |
|---|---|
| `1.5 × tanh(validation ROI − B&H ROI)` | This is the dominant term. Raw ROI is not risk-adjusted. Comparing a 2%-risk, low-exposure strategy to 100%-exposure Buy & Hold makes it a bet on the validation window's direction: in a bull window it rewards high-exposure beta, and in a bear window it rewards anything that sits in cash. The UI calls the difference "alpha" (`Leaderboard.tsx:117`). It is not alpha. |
| `0.75 × tanh(search ROI − B&H)` | Same issue, on a window that overlaps the first term. |
| `0.35 × train (0.6 Sortino + 0.4 Sharpe)` | Sortino is capped at 99 and clamped to 4.5. A strategy with almost no losing bars saturates it. |
| `3 × DSR + 0.25 × z` | Correct in spirit. N is the raw trial count, not the effective count, and it only covers this run. It ranks candidates *within* a round under a frozen hurdle, so it cannot tell the search to stop. |
| `−0.15 × (params − 4)` | Negligible next to the validation term. Four extra parameters cost 0.6, while one lucky validation year is worth up to 1.5. |
| Soft trade-count penalties | A candidate with 30 search trades loses 4.2 points, but can buy them back with DSR and validation luck. These should be gates. |

---

## 3. Iteration mechanics: what actually happens from 1 to N

**The loop** (`optimizer.worker.ts:807-860`): each round, the hurdle is recomputed, the board is re-scored, and a batch of 25 × 4 × pool-size candidates is proposed from the board as it stood. The batch is scored in parallel and the results are applied in order. Any improvement of the #1 fitness resets `stagnation` to 0.

**The proposal mix** (`optimizer.worker.ts:666-678`):
- 7 hand-written seeds, then pure random genomes until the board holds 10.
- After that: 10%→25% random immigrants (the share grows with stagnation), 12% entry/exit crossover, and the rest mutation. Mutation picks the champion 40% of the time and a rank-biased parent otherwise. Each mutation applies `1 + ⌊3·progress⌋` (+ random) steps.

**What mutates** (`genome.ts:771-858`):
| Kind | Weight | What changes | Assessment |
|---|---|---|---|
| `tweak` | 45% | One number, ±1–2 grid steps | Blind hyper-parameter search. This is the main overfitting engine. |
| `second` | 13% | Add, replace or drop the filter or regime condition | Structural but arbitrary: the filter is drawn uniformly from ATR, ADX, BB-width and volume. |
| `exit` | 12% | static ↔ dynamic, trailing on/off, target on/off, reversal on/off | Partly no-op: `normalizeGenome` undoes trailing when TP < SL+1 and undoes reversal on mean-reversion strategies (`genome.ts:692-694`). |
| `operator` | 10% | gt ↔ cross, EMA ↔ SMA, MACD variant, filter gt ↔ lt | EMA ↔ SMA is near-zero impact. On filters, gt ↔ lt is a hypothesis inversion ("trade high vol" becomes "trade low vol") presented as a small step. |
| `trigger` | 10% | A new random entry trigger | Structural. |
| `time` | 6% | Time stop | 20% no-op when already null (`genome.ts:851`). |
| `archetype` | 4% | Flip TREND ↔ MR and redraw the entry | Structural. |

**Signal vs. noise.** About 55–60% of evaluations tweak numbers on existing structures. The structural blocks are economically shallow and highly collinear. RSI, Stoch, CCI and %B are four parametrisations of the same "short-horizon oversold" idea. Supertrend, Price-vs-MA, MA-cross and MACD are four parametrisations of "trend". The niche cap (`NICHE_CAP = 3`, `optimizer.worker.ts:98`) treats them as different ideas, so the board's apparent diversity is overstated. No mutation tests a market-structure hypothesis such as volatility-scaled sizing, time-of-day or session effects, cross-asset lead/lag, or carry/funding.

**Search inefficiencies:**
- About 11% of mutations are no-ops (§0). The behaviourally identical ones are counted in N, and the identical ones still advance `stagnation` (`optimizer.worker.ts:830-831`).
- About 12% of random genomes never trade, and they are still pushed into `trials` with Sharpe 0. That distorts the trial-variance estimate the hurdle depends on (`optimizer.worker.ts:691`).
- Near-duplicate grid values: lookback 20 vs 21 and ATR period 20 vs 21 (`genome.ts:74, 88`).
- `seen.clear()` at 250k (`optimizer.worker.ts:770`) re-admits re-evaluations late in long runs.
- Filters that are always true on a given asset (for example ATR% > 0.5 on crypto) are behaviourally the unfiltered parent, but they cost a parameter and a trial.

---

## 4. Iteration limits and multiple-testing bias

### Current state
- **No iteration cap.** Stopping is stagnation only: default 5,000, range 100 to **1,000,000** (`trading.ts:405-408`, `ControlBar.tsx:464-472`), plus the user's Stop button. A single improvement resets the counter, so long runs keep going.
- **Complexity penalty:** a token one (§2.3). `MAX_PARAMETERS = 8` is a hard cap.
- **Multiple-testing correction:** DSR on the validation segment, within one run, with raw N. It has three gaps. It does not persist across runs, it ignores trial correlation (effective N), and it is used only for ranking, never to stop or to block.

### How much search does the data support?
I evaluated the app's own `expectedMaxSharpe` under the null (annual-Sharpe estimator variance ≈ 1/T years). It gives the annualised Sharpe the luckiest of *N* independent skill-less strategies is expected to show:

| Selection window | N = 10 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|
| 1 year (today's validation segment) | 1.57 | 2.53 | 3.26 | 3.86 | 4.39 |
| 4 years (today's train + validation) | 0.79 | 1.27 | 1.63 | 1.93 | 2.20 |

The largest **effective** trial count for which luck stays below a target annual Sharpe:

| Selection window | target SR 1.0 | target SR 1.5 |
|---|---|---|
| 1 y | 3 | 8 |
| 2 y | 7 | 33 |
| 4 y | 25 | 420 |
| 8 y | 242 | 51,134 |

Intraday bars do **not** raise these limits much. Annualised-Sharpe precision scales with calendar years, not bar count. One year of 15-minute candles is still one year of evidence.

### Recommendation: exact limits and stopping rules
1. **Budget effective trials, not iterations.** Estimate N_eff from the average pairwise correlation ρ̄ of trial return vectors, using López de Prado's approximation N_eff ≈ ρ̄ + (1 − ρ̄)·N, or better, ONC clustering of the trial returns. Store a 16-block return summary per trial to make this cheap.
2. **Per-dataset lifetime budget:** N_eff,max = the largest N with `expectedMaxSharpe(N, 1/T) ≤ 1.5`, where T is the years of OOS evidence used for selection. That gives **≈ 420 for the default 5y daily window** with 4y of walk-forward OOS. Hill-climb neighbours are highly correlated (ρ̄ ≈ 0.6–0.8), so this corresponds to roughly **2,000–5,000 raw evaluations**.
   - Default per run: **2,000 raw evaluations**. Hard ceiling: **10,000 per dataset window across all runs**. Delete the 1,000,000 option.
   - For windows under 2 years: refuse to issue a "Robust" verdict (the table shows < 35 effective trials are affordable).
3. **Stop on DSR peak, not stagnation.** The hurdle rises with every trial, so more search is only justified while the champion's DSR is still rising. Stop when the champion's DSR has not risen in the last **500** evaluations, or when the hurdle exceeds the champion's Sharpe minus one standard error. Keep `stagnation = 500` as a backstop.
4. **Persistent multiple-testing ledger:** key it by `(symbol set, interval, [start, end) date range)` with **overlap** matching, not an exact end date. Seed `RunningStats` with all prior trials whose window overlaps by ≥ 50%. Count each new seed, ticker or setting tried on the same window.
5. **Haircut everything shown:** report the Harvey–Liu haircut Sharpe (Holm or BHY at the cumulative N_eff) alongside the DSR. Require DSR ≥ 0.95 at the *lifetime* N_eff for "Robust".
6. **Complexity:** charge an information-criterion-style penalty of about `ln(T_trades)/T_trades` per parameter in Sharpe units, or require each extra condition to raise OOS DSR by ≥ 0.05 against its parent without the condition (a nested-model test).

---

## 5. Redesign roadmap (by impact, mapped to code)

### P0: stop the UI and the ledger from lying (small changes, biggest effect)
1. **Lead with deflated numbers.** Replace the "Best strategy ROI" tile (`TelemetryStrip.tsx:130-140`) with "Best DSR / luck hurdle Sharpe". In `Leaderboard.tsx:95-160`, label ROI, PnL, Sharpe and "Alpha" as **in-sample** and grey them out. Rename "Alpha" to "Excess ROI".
2. **One-shot holdout lock.** In `finish()` (`optimizer.worker.ts:794-805`), stop revealing all 25. The user picks **one** strategy (at most 3, with the haircut applied) and presses "Lock & reveal". After that the holdout is spent for that window. Remove the sortable Holdout ROI column (`Leaderboard.tsx:200-214`).
3. **Fix the ledger key.** In `holdoutLog.ts:14-17`, store `[holdoutStartTime, holdoutEndTime]` and count any later run whose holdout overlaps it. Persist the cumulative trial count there too (§4.4).
4. **Hard gates instead of soft penalties.** In `combineFitness` (`backtester.ts:1129-1141`), return `UNFIT` below **30 trades per asset per OOS fold** and below **100 OOS trades on the primary asset**. Count per asset, not pooled.
5. **Make the verdict stricter** (`optimizer.worker.ts:438-443`). Require the holdout to beat Buy & Hold on a risk-adjusted basis, *and* the holdout Sharpe CI lower bound > 0 (or the holdout DSR at N = number of locked strategies), *and* the lifetime-N DSR ≥ 0.95.
6. **Cost realism defaults.** Set funding to 0.01%/8h for crypto long-short and add a borrow-fee field for stocks (`config.ts:61-82`). Add slippage = max(fixed %, k × ATR%) with k ≈ 0.05, as a volatility-scaled option in `runBacktest` (`backtester.ts:263-265`). Run a **2× cost stress** in `analyze()`.

### P1: real validation pipeline
7. **Walk-forward OOS fitness.** Split the 80% search window into K = 5 contiguous folds. `runBacktest` already simulates causally over `[0, bars)`, so one pass yields per-fold metrics by generalising `searchMetrics` (`backtester.ts:817-976`) from 2 segments to K. That is no extra simulation cost. Then:
   - **Fitness = 25th percentile of per-fold DSR-adjusted Sharpe** (it rewards consistency, not one lucky year), gated on profitability in ≥ 4/5 folds.
   - **True WFO as a final check:** re-run the optimiser with the same budget on `[0, t_k)` and test on `[t_k, t_k+1)` for k = 2..5. This scores the *search process*, which is what the user is actually relying on. Run it once, at lock time, in the worker.
8. **Probability of Backtest Overfitting (CSCV, Bailey et al. 2014).** Keep a 16-block return matrix for the board plus a random sample of 500 trials. Compute PBO over the C(16,8) = 12,870 splits (cheap: 516 × 16 floats). Show it, and block "Robust" if PBO > 0.2.
9. **Plateau objective.** Rank board entrants by `min(base, median(±1-step neighbours))`, reusing `parameterVariants` (`genome.ts:909-938`). Only compute it for candidates that beat the board floor. That is about 2–5% of evaluations at 10–16 backtests each, cached through the `IndicatorCache`. Replace the irregular lookback grid with a geometric one (`[5,7,10,14,20,28,40,56,80,112,160,200]`, about √2 steps) so "one step" means the same thing everywhere.
10. **Regime decomposition.** Tag every bar causally with a 200-bar trend sign × realised-volatility tercile (6 buckets), and report trades, expectancy and Sharpe per bucket. Gate: non-negative expectancy in ≥ 4 of 6 buckets with ≥ 10 trades each, otherwise label the strategy "regime-specific". In `deriveLimits` (`evaluation.ts:129-138`), refuse windows in which fewer than 3 buckets each cover ≥ 10% of bars.
11. **Monte Carlo stress in `analyze()`** (`optimizer.worker.ts:379-430`): block-bootstrap the trade sequence 1,000×, then report the 5th-percentile ROI and 95th-percentile drawdown. Also run a +1-bar entry delay, a random 10% trade skip, and the 2× cost run. Make the verdict depend on the p5 ROI being > 0.
12. **Data fixes.** Use Yahoo `adjclose` and ratio-adjust OHLC for total return (`market-data.ts:298-313`). In `basket.ts:143-153`, use a causal expanding mean (or raw dollar volume) for volume normalisation. For basket stops, use per-leg intrabar paths or close-only checks instead of the summed high/low envelope. Ship a universe that includes dead assets (for example a dated top-20-by-market-cap list with delisted coins).

### P2: meaningful strategy discovery
13. **Hypothesis-first runs.** Before a run starts, the user picks one or two hypotheses from a short menu, for example "time-series momentum", "short-horizon reversal in uptrends", "volatility breakout", "volatility-compression mean reversion" or "trend with volatility-target sizing". Each hypothesis fixes the indicator family, the archetype and the direction of every filter. The search then tunes only within that hypothesis. Each hypothesis is one entry in the N_eff ledger, and a parameter neighbourhood shares its budget.
14. **Collapse collinear families.** Merge RSI, Stoch, CCI and %B into one "oscillator" block with a single lookback × threshold surface, and merge PRICE_MA, MA_CROSS, Supertrend and MACD into one "trend" block. Niches then become real ideas (`genomeNiche`, `genome.ts:965-968`).
15. **Add structural genes that encode economics, not curve shape.** These include volatility-target position sizing as a hypothesis (sizing is a run setting today), session and day-of-week filters for intraday, and cross-asset regime filters (for example BTC trend gating altcoins, or SPY trend gating single stocks). Remove the gt ↔ lt filter flip from `mutateOperator` (`genome.ts:809`). The filter direction belongs to the hypothesis.
16. **Fix the mutation waste.** Re-draw when repair returns the parent unchanged. Skip filters that are true on more than 95% of bars. Do not count behavioural duplicates (same trade list) as trials. Exclude non-trading genomes from the trial-variance estimate.

### P3: UI and UX guardrails
17. **Overfitting risk score (0–100)** on every row, blended from PBO, lifetime DSR, plateau ratio (neighbour median ÷ base), fold consistency, OOS trade count and regime coverage. Show it in red, amber or green before any return figure.
18. **Parameter stability heatmap** in `StrategyModal.tsx`: a 2-D grid over the strategy's two most influential parameters (period × level), computed on request in the worker. Plateaus show as broad warm regions and spikes as single hot cells.
19. **Budget meter** in `TelemetryStrip.tsx`: "Effective trials 312 / 420 · luck hurdle Sharpe 1.41". It turns amber at 75%. At 100% the Start button is disabled for that data window, with a message explaining that more search on this window would only produce noise.
20. **Holdout gating.** Hide the holdout until lock (done today during search, but extend it to after the run). Block holdout reveal and verdicts on synthetic data. Show "Holdout already spent on [date]" when windows overlap.
21. **Report regime and fold panels** in the modal: per-fold equity curves stacked on one chart, and a per-regime table. A strategy that made all its money in one fold or one regime becomes obvious at a glance.

### Suggested order of work
P0 items 1–6 (a few days) → item 7 (fold-based fitness, the single highest-leverage engine change) → items 8–9 → items 10–12 → P2 → the remaining P3 items. After each step, re-run the random-walk acceptance test in `tests/optimizer.test.ts:376-402`, extended to check the **whole board** rather than only #1. Add a drift variant, since that null is the one the current `holdout.roi > 0` criterion fails.

---

### References
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*. J. Portfolio Mgmt.
- Bailey, Borwein, López de Prado & Zhu (2014/2017), *The Probability of Backtest Overfitting* and *Pseudo-Mathematics and Financial Charlatanism* (MinBTL).
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 (purged / embargoed CV), ch. 11–12 (backtesting dangers, CPCV).
- Harvey & Liu (2015), *Backtesting*. J. Portfolio Mgmt. (haircut Sharpe). Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (t > 3).
- White (2000), *A Reality Check for Data Snooping*. Hansen (2005), *A Test for Superior Predictive Ability*.
- Pardo (2008), *The Evaluation and Optimization of Trading Strategies* (walk-forward analysis).
