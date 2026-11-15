# StrategyFinder

Discover, backtest and optimise algorithmic trading strategies for stocks, cryptocurrencies and weighted
multi-asset baskets. The whole engine (indicators, backtester, evolutionary optimiser) runs in a Web Worker
in your browser; a single Netlify Function proxies market data. Zero-config deploy on Netlify.

> Research tool only. Backtests are hypothetical and fitted to past data. Nothing here is financial advice.

## Quick start

Requires Node 22.13 or newer (`.nvmrc` pins 22).

```bash
npm install && npx netlify dev
```

Open <http://localhost:8888>. `netlify dev` serves the Vite app and the `market-data` function together.
The first run downloads Netlify's local edge runtime, so it needs internet access once.

Other commands:

| Command | What it does |
| --- | --- |
| `npm run dev` | Vite only, on <http://localhost:5173>. A small dev middleware serves the same `market-data` function, so live data works here too. |
| `npm test` | Unit and integration tests (indicators, backtester, genome, Pine export, function, optimiser worker). |
| `npm run build` | Type-check, then build to `dist/`. |
| `npm run preview` | Serve the production build locally. |

**Saved tickers.** A custom ticker is remembered once you run it, or press Enter in a ticker field.
Saved tickers appear as one-click chips next to the presets and as autocomplete suggestions; the × on a
chip forgets it. They are stored in the browser's `localStorage`.

**Works offline.** If the function is unreachable, or Yahoo / Binance fail or rate-limit, the app switches to
a built-in OHLCV generator and labels the run **Simulated** in amber. Simulated prices contain artificial
trend regimes, so results on them are illustrative only.

## Deploy: GitHub to Netlify

1. Push this repository to GitHub.
2. In Netlify choose **Add new site → Import an existing project → GitHub** and pick the repository.
3. Accept the detected settings. They come from `netlify.toml`:
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Functions directory: `netlify/functions`
4. Click **Deploy**. Every push to the production branch redeploys; pull requests get deploy previews.

No environment variables or API keys are required. Optional: set `VITE_MARKET_DATA_URL` to point a build at a
function hosted elsewhere.

CLI alternative: `npx netlify login`, `npx netlify init`, then `npx netlify deploy --build --prod`.

## How it works

```
Browser (main thread)                      Web Worker                         Netlify Function
┌──────────────────────────┐   start/stop  ┌───────────────────────────┐      ┌──────────────────────┐
│ React UI                 │ ────────────▶ │ indicators.ts             │      │ market-data.ts       │
│ control bar · telemetry  │               │ backtester.ts             │      │  Binance (crypto)    │
│ chart · leaderboard      │ ◀──────────── │ genome.ts                 │      │  Yahoo (stocks, …)   │
│ strategy breakdown       │  progress /   │ optimizer.worker.ts       │      │  normalise + CORS    │
└──────────────────────────┘  leaderboard  └───────────────────────────┘      └──────────────────────┘
        │  GET /.netlify/functions/market-data?symbols=…&interval=…&range=…              ▲
        └────────────────────────────────────────────────────────────────────────────────┘
```

Nothing heavy runs on the main thread or in the function, so the UI never blocks and the serverless
timeout is irrelevant to optimisation.

### Market data function

`GET /.netlify/functions/market-data?symbols=BTC-USD,NVDA&interval=1d&range=2y`

| Parameter | Values |
| --- | --- |
| `symbols` | Comma-separated tickers, up to 10. `BASE-USD` pairs go to Binance, falling back to Yahoo. Everything else goes to Yahoo. |
| `interval` | `15m`, `30m`, `1h`, `4h`, `1d` |
| `range` | `1y`, `2y`, `5y` |

The response is columnar JSON with CORS headers: `{ interval, range, generatedAt, series: { SYMBOL: { symbol,
source, venueSymbol, t, o, h, l, c, v } }, errors }`. `t` is the bar open time in UTC milliseconds, normalised so venues
line up: daily bars sit at 00:00 UTC of the exchange-local date, hourly bars at the top of the hour, and 4h
bars in UTC-aligned buckets. `15m` loads at most `1y` and `30m` at most `2y` (about 35,000 crypto
candles; longer requests get a 400). Yahoo serves 15m / 30m history for the last 60 days only and hourly
history for about 730 days, so stock requests longer than that are shortened (the response carries a
`note`). Crypto pairs come from Binance and get the full range.

### Backtester

- Signals are evaluated on the close of candle `t` and filled at the open of candle `t+1`.
- Every fill pays the fee (default 0.1% per side) and **volatility-scaled slippage**: the larger of a fixed
  floor (default 0.05%) and a multiple of ATR(14) as a % of price (default 0.03 ×), read on the last closed
  candle. Volatile candles and volatile assets cost more to trade.
- **Long/Short runs pay financing by default.** Crypto is shorted as a perpetual: every open position pays
  funding (default 0.01% per 8h). Stock and ETF shorts pay a borrow fee (default 2% a year). Long-Only runs
  are treated as spot positions and pay no financing.
- **Sizing is a run setting, not something the optimiser tunes.** Default: risk 2% of the starting capital at
  the stop, no compounding, exposure capped at 1× equity. Options: compounding, or 100% of equity.
- Stops and targets are simulated intrabar from the entry candle. A gap through a level fills at the open.
  If one candle touches both the stop and the target, the stop is assumed to fill first, and the trade is
  counted as "ambiguous" so you can see how often that assumption was needed.
- ATR levels use the ATR of the signal candle. Trailing stops update only after a candle has closed.
- **Max drawdown uses intrabar worst-case marks**: the low of every candle a long is open, the high for a short.
- Buy & Hold buys at the first open with the same costs and capital.

### Train / validation / holdout

Data is split chronologically: **Train 60% / Validation 20% / Holdout 20%**.

- The search sees train and validation only. While a run is live, every backtest stops at the holdout
  boundary; the holdout is not simulated, scored or sent to the UI.
- **The holdout stays sealed when the run ends.** You choose ONE strategy of the final leaderboard and press
  *Lock & reveal holdout*. Only that strategy's holdout is simulated and shown, once per run; every other
  strategy stays sealed, so the holdout can never be used to pick a winner (there is no holdout column to
  sort on).
- A holdout is only unseen once. Every reveal is recorded in `localStorage` with its symbols and holdout
  **date window**. A later run whose holdout overlaps an earlier reveal on any shared symbol is warned, and
  none of its strategies can be rated Robust. A new end date, or another range or timeframe over the same
  dates, does not reset the log.

### Market hypothesis and strategy genome

Every run tests **one market hypothesis**, picked in the control bar before the run starts. An entry is
built from up to three **slots**, joined by `AND`, and the hypothesis lists the primitives each slot may use
and fixes their direction:

1. **Entry trigger** (required) – a discrete **event**: a crossover, a channel breakout, an oscillator
   crossing back out of its extreme zone. An event fires once per setup, so a strategy cannot re-enter the
   same trend leg on every candle after a stop or target is hit (a `close > EMA` state trigger did exactly
   that). Triggers are never states.
2. **Regime filter** (optional) – a slow **state**: close above a slow EMA (100+ candles), or a rising slow
   EMA (56+ candles, its value vs. 10 candles earlier).
3. **Confirmation filter** (optional) – a fast **state**: trend strength (ADX) or volume.

| Hypothesis | Entry triggers (events) | Regime filter | Confirmation |
|---|---|---|---|
| **Trend following** – time-series momentum | EMA crossover · Donchian breakout (close above the prior N-candle high) · MACD signal cross | close > slow EMA · slow EMA rising | ADX above a level |
| **Volatility breakout** – a close beyond the range starts a move | Donchian breakout · close crosses above the upper Bollinger band | close > slow EMA · slow EMA rising | volume above average |
| **Buy the dip in an uptrend** – short-term overreaction | pullback rejection: RSI crosses back above an oversold level · close crosses back above the lower band | **required**: close > slow EMA or slow EMA rising | ADX above a level |
| **Range reversion** – trendless prices revert to a mean | RSI crosses back above oversold · close back above the lower band | – | ADX below a level |

- Mutations stay inside the hypothesis: they tune a number, swap or redraw the entry trigger, add / drop /
  swap the regime or confirmation filter, or change the exit. They never flip a filter's direction or
  change the archetype. A mutation is **guaranteed to differ** from its parent.
- In Long/Short mode the short rules are the exact mirror of the long rules. An actionable opposite-side
  entry **reverses** an open position at the next open – also on the open a time stop fires (the time stop
  only blocks re-entering the same side) – in the backtester and in the Pine export alike.
- **Round-number grids only.** Lookbacks from the geometric grid {5, 7, 10, 14, 20, 28, 40, 56, 80, 112,
  160, 200} (about √2 per step, no near-duplicates); oscillator levels in steps of 5; band multipliers in
  steps of 0.5; static stop / target from {1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30} %.
- **At most 10 tunable numbers** per strategy, with a fitness penalty for each one above 6.
- **Exits:** static % stop and target, or ATR stop (1–4×) with optional ATR target (1.5–8×), trailing stop,
  **opposite-signal exit** (the mirrored trigger closes the trade; any hypothesis) and an optional time
  stop (5, 10, 20 or 40 bars). A trailing stop is only kept when the target sits at least 1 ATR beyond the stop.

### Optimiser, trial budget and early stopping

Evolutionary hill-climbing with mutation: textbook seeds of the hypothesis and random genomes start the
population; each iteration mutates the champion or a rank-biased parent, recombines two leaderboard
members, or injects a fresh random genome.

- **Only new bets count.** A candidate whose genome was already proposed, that never trades, or whose
  trade log (entries, exits, direction and PnL on the primary asset) repeats one already evaluated is
  skipped: it is not a trial, uses no budget and cannot enter the leaderboard.
- **Evaluations per run:** 5,000 by default (settable from 100 to the window's remaining budget).
- **Cumulative trial budget per data window:** **25,000** evaluations across every run and session whose
  search period overlaps (a shared symbol, and at least half of the shorter window in common). The trials
  are kept in `localStorage`, so reloading the page, a new end date, another range or another timeframe over
  the same dates never resets them. They also seed the deflated-Sharpe hurdle and the haircut, so every new
  run on the same data is judged against everything tried on it before. Once the budget is spent, search on
  that window pauses: **Extend Budget (+10,000 trials)** lets you continue knowingly. Extensions are
  recorded too, every earlier trial still counts, and the meter warns that multiple-testing significance is
  degraded.
- **Early stopping:** a run ends when **1,500** evaluations in a row fail to improve the best **composite
  fitness** on the leaderboard (or at its evaluation limit, or the window's budget, or when you press
  **Stop**). Early stopping used to watch the deflated Sharpe, which sits at 0% through the first
  generations and stopped runs before good combinations had evolved.
- **Haircut Sharpe** (Harvey & Liu): the validation Sharpe's p-value is multiplied by the number of trials on
  the window (Bonferroni) and turned back into the Sharpe that survives. Shown next to the deflated Sharpe.

The telemetry strip shows a **trial-budget meter** for the active data window: trials used (all runs), its
level (fresh / significance degrading from 50% / nearly spent from 90% / spent), and the Sharpe the luckiest
of that many skill-less strategies is expected to reach. Every strategy carries an **overfitting risk score**
(0–100: selection bias 30, multiple testing 15, parameter spike 20, fold inconsistency 15, regime collapse 10,
thin sample 10; always "high" below a 50% deflated Sharpe), and its breakdown shows a **parameter-stability
heatmap**: the search-period Sharpe over the two parameters whose one-step neighbours move it most.

**Asset pool** (asset mode in the control bar, default BTC, ETH, SOL, BNB): every asset is traded
separately with the same genome, and the **minimum-trade gates apply to the trades pooled over the
assets** – one strategy must work across the pool rather than fit one asset's price path, and four assets
supply the sample a single daily chart cannot. Fitness is still the median per-asset score; the reported
deflated Sharpe is the median over the assets (pooling raises the sample for the gates, not the statistic),
and the holdout trade count of a locked strategy is pooled too.

**Robust universe** (on by default for crypto, single-asset mode): the same rules are backtested
separately on every asset of a universe (default BTC, ETH, SOL, XRP, ADA, DOGE; editable). Fitness is the
median per-asset score, trade counts are gated per asset (not pooled), and there is a penalty when fewer
than 2/3 of the assets are profitable on validation. The leaderboard and chart show the primary asset.

The search period (train + validation) is cut into **5 equal, sequential folds**, and every fold is scored
on its own (0.6 Sortino + 0.4 Sharpe, clamped). Fitness rests on the **median** fold score, plus a
**consistency bonus** when at least 3 of the 5 folds made money (and a penalty per profitable fold short of
that), and a **severe penalty** when any single fold suffers a drawdown beyond 35%. A fold in which the
strategy never held a position is **flat**: once the strategy was active in at least 3 folds, flat folds
are left out of the median and the consistency count (60% of the active folds must be profitable), so a
regime filter that sat out a bear year is not punished for doing its job. Grading the 25th-percentile
fold, as before, punished trend followers for every sideways year. Raw return over Buy & Hold does not
count: it is a bet on the market's direction, not evidence of an edge.

Every search-period bar is also labelled, causally, as **bull** or **bear** (sign of the trailing return
over about a quarter of a year) and, separately, **high volatility** (trailing one-month volatility in the
top third of the search period). A regime slice is judged when it covers at least 10% of the period and
holds at least 10 of the strategy's trades. A strategy that **collapses** in a slice (annualised Sharpe
below −0.5 there) loses 0.75 per slice; one that makes money in only one regime and collapses in all the
others is **disqualified**.

Fitness (higher is better), per asset:

```
+ 3.00 × Deflated Sharpe Ratio of the validation segment (plus a small slice of its z-score as a gradient)
+ 1.50 × median fold score (flat folds set aside, see above)
+ 0.50 when ≥ 3 of 5 folds are profitable, − 0.50 per profitable fold short of that
− 3 (and more) when any single fold draws down more than 35%
− penalty for intrabar drawdowns beyond 20% over the search period, steeper beyond 45%
− 0.75 for every regime slice the strategy collapses in
```

then, across assets, after the **hard sample-size gates**, which are **calibrated to the timeframe and the
length of the data**: 15 trades per year of data on daily candles, rising slowly with the number of
candles per day (≈ 26 on 4h, ≈ 39 on 1h, ≈ 59 on 15m), times the years of the search period and of the
validation segment, with floors of 30 and 10 trades. On 5 years of daily data that is 60 train + validation
and 15 validation trades; the holdout gate of a Robust verdict is computed the same way (16). A strategy
below any gate is disqualified outright. The counts are the primary asset's – or pooled over an asset pool.

```
  median of the per-asset scores
− 1.0 when fewer than 2/3 of the universe is profitable on validation
− 0.15 for each tunable parameter above 6
− 0.75 for reward:risk below 1:1, unless the validation win rate clears break-even by 10 points
```

**Plateaus, not spikes.** A strategy that could enter the leaderboard also has every one-grid-step
parameter neighbour backtested (each tunable number moved one step down and one step up, everything else
unchanged). When its neighbours are worse it is ranked by **70% of its own fitness + 30% of their median**
(a disqualified neighbour counts as at most 3 below the strategy); a plateau keeps its score, a spike loses
30% of its lead – enough to rank plateaus first without discarding a promising new rule combination the
moment one neighbour slips under a trade gate. The blend never raises a score, so it is re-applied whenever
the deflated-Sharpe hurdle moves and never lets a strategy jump the leaderboard.

The **Deflated Sharpe Ratio** (Bailey & López de Prado) is the probability that a Sharpe ratio is real given
the number of strategies tried so far, the spread of their Sharpe ratios, the sample length, and the
skewness and kurtosis of returns. The hurdle rises with every trial, so the leaderboard is re-scored as
the run goes on and a score can fall over time.

**Speed.** Candidate strategies are scored on several CPU cores: the app starts one evaluator
worker per core (minus one, up to 6) and the optimiser sends them rounds of 100 candidates each over
direct MessageChannels. Results are applied in proposal order, so a seeded run gives the same answer
whatever the pool size. With one or two cores, or if an evaluator fails to load, the optimiser scores
candidates itself. Within each evaluation, the expensive parts are kept small: only the bars inside
trades are visited when computing metrics, the second condition of an entry is only checked where the
trigger fired, and in a Robust universe or asset pool a candidate stops being backtested once it provably cannot
reach the leaderboard (an exact bound on the median, so it never changes the result).

The leaderboard keeps the 25 best behaviourally distinct strategies: two candidates whose equity returns
correlate above 0.9 count as the same idea and only the fitter one is listed.

### Robustness panel and verdict

When a run ends (or on demand from the strategy drawer) each strategy is tested on train + validation:

- **Parameter sensitivity:** every parameter is moved by ±20% (snapped to its grid, or to the next grid step
  when ±20% rounds back). Fragile when the worst variant loses more than half of the original ROI.
- **Top-trade removal:** net PnL without the 5 best trades.
- **Cross-asset:** the exact rules, untuned, on every universe asset.

Verdict (only the locked strategy gets one): **Overfit** when its deflated Sharpe is below 50% or it lost
money (or had a non-positive Sharpe) on the holdout. **Robust** only when ALL of these hold: deflated Sharpe
≥ 95%; holdout Sharpe 95% confidence interval above zero; holdout Sharpe above Buy & Hold's; at least 30
holdout trades; every test above passes; the holdout period was never revealed before; live (not simulated)
data. **Fragile** otherwise; the drawer lists every unmet condition.

## Read this before trusting a result

- **Expect small numbers and few "Robust" verdicts.** With 2% risk per trade and a deflated Sharpe hurdle,
  most strategies are labelled Overfit or Fragile. That is the tool being honest, not broken: a year or
  two of validation data cannot prove an edge against thousands of trials. On the audit's random walks
  (6 driftless, 6 with +40% a year of drift, one hypothesis per run, whole boards), no strategy reached a
  95% deflated Sharpe, none met the holdout conditions of a Robust verdict, and every locked #1 was rated
  Overfit. On five years of daily candles the breakout and dip-buy hypotheses rarely reach the minimum
  number of trades at all – use 4h or 1h candles for them.
- **The holdout is a one-shot test.** Re-running on the same dataset and picking the run whose holdout looks
  best turns the holdout into training data. The reveal log makes that visible and blocks a Robust verdict
  on a holdout that has been seen before.
- **The holdout confidence interval is wide.** The drawer shows the 95% interval for the holdout Sharpe; with
  a few dozen trades it usually includes zero.
- **Daily bars cannot resolve intrabar order.** The "ambiguous candles" count shows how many trades relied on
  the stop-first assumption. The drawer warns above 10%.
- **Baskets are synthetic.** Components are aligned on shared timestamps and combined into a buy-and-hold
  index. Basket highs and lows are sums of component highs and lows, which is a conservative envelope.
  The Robust universe is not available for baskets.
- **Crypto is tested on the USDT pair.** `XRP-USD` is routed to `XRPUSDT` on Binance; the telemetry tile and
  the drawer show the instrument actually used.
- **Pine export is a faithful translation, not a guarantee of identical numbers.** It sets
  `use_bar_magnifier=false` so it compiles on every TradingView plan; Premium users can switch it to `true`
  to settle candles that touch both stop and target with lower-timeframe data. The code is generated and
  syntax-tested here but was not run on TradingView.
- **Paper trade before going live.** The test window ends at the latest candle, so only new data is unseen.

## Project layout

```
netlify.toml                      Build, functions, dev and SPA redirect settings
netlify/functions/market-data.ts  Market-data proxy (Binance + Yahoo Finance)
src/types/trading.ts              Shared types and worker protocol
src/engine/indicators.ts          Indicator library and memoising cache
src/engine/backtester.ts          Execution model, metrics, split, benchmark, fitness
src/engine/genome.ts              Market hypotheses, grids, random generation, mutation, repair
src/engine/describe.ts            Plain-English rules and badges
src/engine/pine.ts                Pine Script v5 exporter
src/workers/optimizer.worker.ts   Optimisation loop (Web Worker)
src/workers/evaluator.worker.ts   Evaluator pool member: scores candidate strategies
src/engine/evaluation.ts          Shared scoring code and the evaluator protocol
src/lib/stats.ts                  Deflated Sharpe Ratio, haircut Sharpe, confidence intervals
src/lib/holdoutLog.ts             Holdout reveal log (symbols + date window, overlap matching)
src/lib/trialLedger.ts            Cumulative trial budget per data window (overlap matching)
src/lib/risk.ts                   Overfitting risk score
src/data/                         Data loader, offline generator, basket builder
src/components/                   ControlBar, TelemetryStrip, EquityChart, Leaderboard, StrategyModal
src/App.tsx                       Dashboard and worker orchestration
tests/                            Vitest suites
```

TypeScript is pinned to 5.9 because Netlify CLI's dev server loads the project's `typescript` package and
does not yet support TypeScript 7.
