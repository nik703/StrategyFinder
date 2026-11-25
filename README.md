# StrategyFinder

Discover, backtest and optimise algorithmic trading strategies for stocks, cryptocurrencies and weighted
multi-asset baskets. The whole engine (indicators, backtester, evolutionary optimiser) runs in a Web Worker
in your browser; a single Netlify Function proxies market data. Zero-config deploy on Netlify.

The app is one guided page in four steps:

1. **Top action bar & presets** – a sticky bar with the primary **1-Click Run** button (it scans with the
   active preset; no panel needs opening), **Stop**, the **trial-budget meter** of the selected data window
   and **Reset Runs & Clear Cache**. Below it, one-click preset cards (*Crypto Multi-Asset Swing (4h)* –
   the default – *Trend Breakout (1d)*, *Stock Momentum (1d)*, *Intraday Scalp (15m)*) set the assets,
   timeframe, realistic fees, direction mode and hypothesis; every granular knob sits in a collapsible
   **Advanced settings** drawer.
2. **Live discovery scanner** – the progress bar, robust strategies found against the target, the
   diversity-restart cycle and a pass / fail funnel of every robustness gate.
3. **Ranked leaderboard** – robust strategies first, then by composite fitness: equity sparklines, the
   deflated Sharpe, a Monte Carlo stress column, Sortino, max drawdown and trades, plus side-by-side
   comparison of up to three strategies.
4. **Strategy workbench** – a plain-English *Strategy verdict* with a status badge for every check, the
   equity curve, the Next Entry Forecast (Long vs Short readiness, exact trigger price, estimated fill,
   stop / target, trailing-stop and breakeven levels), the Monte Carlo stress curve, the parameter-stability
   heatmap, a cross-market check on unseen markets, the one-shot locked holdout reveal and the 1-click
   Pine Script v5 export.

**Reset Runs & Clear Cache** stops any scan, terminates every pooled worker, drops every pending evaluation
request and deletes every cached strategy, the leaderboard, the elite kept across restarts, the trial and
restart counters, the holdout reveal log, the saved runs and the candle cache (memory and IndexedDB),
restoring a clean 0-trial slate. It asks for confirmation first, because
clearing the history also resets multiple-testing tracking: earlier trials stop raising the luck hurdle and
holdouts already seen look untouched again. Settings and saved tickers are kept.

**Export / Import Snapshot** (top action bar) saves the multiple-testing ledger (trial counts, budget
extensions, holdout reveals) and the saved runs with their final leaderboards as one `.json` file, and merges
such a file into another browser. An import can only add history: a trial count is never lowered and a
recorded reveal is never removed, so a snapshot cannot be used to wipe the luck hurdle.

### Persistence

Everything the app remembers lives in one IndexedDB database (`strategyfinder`) with three stores:
`candles` (the incremental candle cache, one record per exchange, symbol and timeframe), `ledger` (trial
counts, budget extensions and holdout reveals) and `runs` (saved runs and leaderboard snapshots). The
legacy `localStorage` keys of earlier versions are migrated on first load and removed. Only closed candles
are cached; a later run downloads just the candles that closed since the last stored one (*Force Refresh
Data* in Advanced settings downloads the whole range again).

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
| `npm test` | Unit and integration tests (indicators, backtester, genome, direction modes, exits, Monte Carlo, diagnostics, Pine export and AST parity, function, optimiser worker, worker pool, IndexedDB storage and snapshots, candle cache, stateful Next Entry solver). |
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
│ leaderboard · compare    │ ◀──────────── │ genome.ts                 │      │  Yahoo (stocks, …)   │
│ strategy workbench       │  progress /   │ optimizer.worker.ts       │      │  normalise + CORS    │
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
`note`). Crypto pairs come from Binance and get the full range. Daily Yahoo candles are
**dividend-adjusted** (every price of a candle is scaled by `adjclose / close`), so a long position earns the
dividends Buy & Hold earns and ex-dividend gaps are not mistaken for price moves.

### Backtester

- Signals are evaluated on the close of candle `t` and filled at the open of candle `t+1`.
- Every fill pays the fee (default 0.1% per side) and **volatility-scaled slippage**: the larger of a fixed
  floor (default 0.05%) and a multiple of ATR(14) as a % of price (default 0.03 ×, never below 0.02 ×),
  read on the last closed candle. Volatile candles and volatile assets cost more to trade.
- **Long/Short runs always pay financing.** Crypto is shorted as a perpetual: every open position pays
  funding (default 0.01% per 8h, never below 0.005%). Stock and ETF shorts pay a borrow fee (default 2% a
  year, never below 0.5%). Long-Only runs are treated as spot positions and pay no financing. This applies
  to both Asymmetric and Mirrored L/S.
- **Execution realism cannot be switched off.** Next-bar fills, volatility slippage, financing and the
  holdout lock are hard-coded; the settings only let you make costs higher, and the engine enforces the
  floors above on every run (`enforceRealism`).
- **Sizing is a run setting, not something the optimiser tunes.** Default: risk 2% of the starting capital at
  the stop, no compounding, exposure capped at 1× equity. Options: compounding, or 100% of equity.
- Stops and targets are simulated intrabar from the entry candle. A gap through a level fills at the open.
  If one candle touches both the stop and the target, the stop is assumed to fill first, and the trade is
  counted as "ambiguous" so you can see how often that assumption was needed.
- ATR levels use the ATR of the signal candle. Trailing stops update only after a candle has closed.
- **Breakeven lock** (an exit option): once a candle has traded 1R in the trade's favour (1R = the initial
  stop distance), the stop moves to the entry price from the next candle on, so a winner can no longer turn
  into a full loss. It combines with fixed brackets and with the trailing ATR stop.
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
- A holdout is only unseen once. Every reveal is recorded in IndexedDB with its symbols and holdout
  **date window**. A later run whose holdout overlaps an earlier reveal on any shared symbol is warned, and
  none of its strategies can be rated Robust. A new end date, or another range or timeframe over the same
  dates, does not reset the log.

### Market hypothesis and strategy genome

Every run tests **one market hypothesis**, picked in the control bar before the run starts. Strategies
are deliberately **parsimonious** – simple rules trade more often and leave less room to fit noise:

1. **Entry trigger** (required) – a discrete **event**: a crossover, a channel breakout, an oscillator
   crossing back out of its extreme zone. An event fires once per setup, so a strategy cannot re-enter the
   same trend leg on every candle after a stop or target is hit. Triggers are never states.
2. **At most one filter** (optional) – a **state**: a slow regime (close above a slow EMA of 100+ candles,
   or a rising slow EMA of 56+ candles) or a confirmation (ADX trend strength, volume).

- **1–2 indicators per entry.** About **60%** of random strategies are the trigger alone and 40% a trigger
  plus one filter; mutation keeps that mix (a filter mutation drops an existing filter 75% of the time and
  adds one to a single-indicator strategy 50% of the time).
- **Orthogonal pairs only.** The filter must come from another indicator family than the trigger
  (moving averages, oscillators, bands, volatility, trend strength, volume): two moving-average rules or two
  momentum oscillators never pair, so an EMA-cross or MACD trigger can only be filtered by ADX.

| Hypothesis | Entry triggers (events) | The one optional filter |
|---|---|---|
| **Trend following** – time-series momentum | EMA or SMA crossover · close crosses above an EMA or SMA · Donchian breakout (close above the prior N-candle high) · MACD signal cross | ADX above a level, or (breakout trigger only) close > slow EMA/SMA / slow EMA/SMA rising |
| **Volatility breakout** – volatility clusters; a close beyond the range starts a move | Donchian breakout · close crosses above the upper Bollinger or **Keltner** band · **Bollinger-inside-Keltner squeeze release** · **ATR range expansion** (a candle with a true range of several ATRs closing higher) | close > slow EMA/SMA · slow EMA/SMA rising · volume above average |
| **Trend pullback** (buy the dip in an uptrend) – short-term overreaction | RSI crosses back above an oversold level · close back above the lower Bollinger or **Keltner** band · **pullback to dynamic support** (the candle dips to a short EMA/SMA and closes back above it) | **required**: close > slow EMA/SMA or slow EMA/SMA rising |
| **Range reversion** – trendless prices revert to a mean | RSI crosses back above oversold · close back above the lower Bollinger or Keltner band | ADX below a level |

- Mutations stay inside the hypothesis: they tune a number, swap or redraw the entry trigger, add / drop /
  swap the regime or confirmation filter, or change the exit. They never flip a filter's direction or
  change the archetype. A mutation is **guaranteed to differ** from its parent.
- **Direction mode** (a run setting):
  - **Long-Only** (the default) never sells short – equities and spot crypto drift upwards over time and
    charge for shorting, so a mirrored short side mostly bleeds in bull markets.
  - **Asymmetric L/S** shorts with the mirrored trigger, but only inside the short side's **own bear-regime
    filter** (close below a slow EMA/SMA, or a falling slow EMA/SMA). That filter is a separate gene, tuned
    independently of the long rules (it gets one parameter slot of its own), so the short side sits out
    bull markets instead of fading every pullback.
  - **Mirrored L/S** uses the exact mirror image of the long rules for the shorts.

  An actionable opposite-side entry **reverses** an open position at the next open – also on the open a
  time stop fires (the time stop only blocks re-entering the same side) – in the backtester, the Next Entry
  Forecast and the Pine export alike.
- **EMA / SMA lengths: every integer by default.** Each moving average can be an EMA or an SMA (a
  crossover uses the same type for both). With **EMA / SMA lengths → Every integer** (the default) the
  search fine-tunes each length one candle at a time, e.g. EMA(23) × EMA(87). That covers crossovers,
  close-vs-average rules, average slopes and the Bollinger period (its middle line is an SMA).
  - New lengths are drawn log-uniformly.
  - Mutations mix ±1–3 candle refinements with 10–35% jumps, and sometimes switch EMA ↔ SMA.
  - The plateau check and the stability map still look one **×√2 step** away (EMA 50 → 35 / 71), the
    same relative distance as the coarse grid. A ±1-candle neighbour (EMA 50 vs 51) would always look
    stable and let a fine-tuned spike through.
  - Every extra length tried is one more trial, so the deflated Sharpe hurdle rises accordingly.
  - **Coarse grid** restricts lengths to {5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200}: fewer
    combinations, less room to overfit.
- **Other numbers stay on round grids:** remaining lookbacks on the geometric grid above (about √2 per
  step), oscillator levels in steps of 5, band multipliers in steps of 0.5.
- **Static stops and targets scale with the timeframe:** 0.5–3.5% stop and 1–10% target on 15m–1h
  candles, 1–6% / 2–15% on 4h, 1.5–10% / 3–20% on daily. A 10% stop on a 15m chart is never hit, so capital
  sits in trades nothing closes. ATR-based exits, which scale on their own, are favoured (70% of random
  strategies).
- **At most 5 tunable numbers** per strategy, exits included (a hard cap – an Asymmetric short regime
  filter gets one more). About 60% of random strategies are a single trigger and 40% a trigger plus one
  non-redundant filter (the trend-pullback hypothesis always pairs its pullback with a slow trend filter);
  most carry 3–5 numbers. Settings that would add parameters without adding an idea are fixed: the exit ATR period (14, the same ATR that
  scales slippage), the MACD signal (10), the ADX filter period (14) and the volume average (20).
- **Exits:** static % stop and target, or ATR stop (1–4×) with optional ATR target (1.5–8×), trailing ATR
  stop, **breakeven lock** at +1R, **opposite-signal exit** (the mirrored trigger closes the trade; any
  hypothesis) and an optional time stop (5, 10, 20 or 40 bars). Half of the random ATR exits trail, and most
  trailing exits carry no fixed target, so winners can ride a trend. A trailing stop is only kept when the
  target sits at least 1 ATR beyond the stop; a breakeven lock only when the target lies beyond 1R.

### Optimiser, trial budget and scanning

Evolutionary hill-climbing with mutation: textbook seeds of the hypothesis and random genomes start the
population; each iteration mutates the champion or a rank-biased parent, recombines two leaderboard
members, or injects a fresh random genome.

- **Only new bets count.** A candidate whose genome was already proposed, that never trades, or whose
  trade log (entries, exits, direction and PnL on the primary asset) repeats one already evaluated is
  skipped: it is not a trial, uses no budget and cannot enter the leaderboard.
- **Evaluations per run:** 25,000 by default, at most 50,000 (the safety ceiling of one run).
- **Cumulative trial budget per data window:** **50,000** evaluations across every run and session whose
  search period overlaps (a shared symbol, and at least half of the shorter window in common). The trials
  are kept in IndexedDB, so reloading the page, a new end date, another range or another timeframe over
  the same dates never resets them. They also seed the deflated-Sharpe hurdle, so every new
  run on the same data is judged against everything tried on it before. Once the budget is spent, search on
  that window pauses: **Extend Budget (+10,000 trials)** lets you continue knowingly. Extensions are
  recorded too, every earlier trial still counts, and the meter warns that multiple-testing significance is
  degraded.
- **Scan until robust.** A run keeps scanning until it has found **N robust strategies** ("Stop after N
  robust": 1, 3 – the default – 5, or *Run Full Budget*), or its budget is spent. A strategy is robust when
  it passes **three orthogonal gates** on the search period (the holdout stays sealed). Each property is
  judged exactly once – nothing is also folded into the fitness as a penalty:
  1. **Sample and consistency:** the timeframe-scaled minimum-trade gates on the **effective sample size**
     (correlated assets of a pool count as fewer independent bets, see *Asset pool*), a positive
     **median-fold** risk-adjusted return (½ Sortino + ½ Calmar), and no fold beyond the 35% drawdown cap;
  2. **Significance:** a deflated Sharpe of at least 50% – the validation Sharpe beats the best a random walk
     is expected to reach after that many trials;
  3. **Stress and stability** (post-search, second evaluation tier): the Monte Carlo execution stress test
     passes and the one-grid-step parameter neighbours keep a positive median expectancy per trade.
  Robust strategies stay listed even when the fitness ranking would push them off the leaderboard, and are
  re-checked whenever the luck hurdle rises (a strategy can drop out). The target only ends a run after at
  least 1,000 evaluations. Robust here is a search-period verdict: the holdout of the one strategy you lock
  decides the final verdict.
- **Diversity restarts instead of early exit.** When **1,500** evaluations in a row fail to improve the best
  composite fitness, the run does not stop: it keeps the **top 5 strategies as the elite** and re-seeds the
  rest of its parent population with fresh random single- and dual-indicator genomes (a new **population
  cycle**). In the new cycle the fresh candidates form the population – the best of them by fitness,
  whether or not they reach the leaderboard – and are hill-climbed, mutated and crossed with the elite, so
  a long scan escapes the local optimum it stalled on instead of looping on minor tweaks. The leaderboard
  and the robust archive survive every restart.
- The live scan row shows robust strategies found / target, evaluations / budget with the population cycle,
  and how many scored candidates cleared each gate (gate 1, gate 2, and of those put through the second
  tier, how many survived gate 3), so you can see why candidates are being filtered.

The top action bar shows a **trial-budget meter** for the selected data window: trials used (all runs),
its level (fresh / significance degrading from 50% / nearly spent from 90% / spent) and, in its tooltip,
the Sharpe the luckiest of that many skill-less strategies is expected to reach. The workbench draws a
**parameter-stability heatmap**: the search-period Sharpe over the two parameters whose one-step neighbours
move it most.

**Asset pool** (the default preset's BTC, ETH, SOL, BNB; asset mode in Advanced settings): every asset is traded
separately with the same genome, and the **minimum-trade gates apply to the effective number of
independent bets pooled over the assets** – one strategy must work across the pool rather than fit one
asset's price path. Correlated coins trading at the same time are not independent evidence: each asset's
trades are divided by its design effect `1 + Σ overlap(a,b) · max(0, ρ(a,b))` over the other assets (ρ the
correlation of their bar returns, overlap the share of its in-market bars the other asset is also in the
market), so four highly correlated coins that are always in the market together count as little more than
one. Fitness is still the median per-asset score; the reported
deflated Sharpe is the median over the assets (pooling raises the sample for the gates, not the statistic),
and the holdout trade count of a locked strategy is pooled too.

The search period (train + validation) is cut into **5 equal, sequential folds**. Each fold's
**risk-adjusted return** is ½ Sortino (clamped to [−3, 4.5], scaled by 1/1.5) + ½ Calmar (CAGR ÷ max
drawdown, clamped to ±3). A fold in which the strategy never held a position is **flat**: once the strategy
was active in at least 3 folds, flat folds are left out, so a regime filter that sat out a bear year is not
punished for doing its job – and the fold *gate* of the robust scan uses exactly the same rule.

**One composite fitness** ranks every strategy (higher is better), per asset:

```
+ 1.5 × median-fold risk-adjusted return   (median over the judged folds of ½ Sortino/1.5 + ½ Calmar)
+ 1.0 × expectancy × trade frequency       (R-expectancy of the search period × trades a year, in units of
                                            10 R a year, clamped to [−1.5, 3])
```

Nothing else is stacked on top: drawdown is judged once (gate 1's per-fold cap), significance once (gate 2)
and parameter stability once (gate 3). There is no haircut Sharpe and no neighbourhood discount.

then the **median over the assets** (one asset, or every asset of a pool), after the **hard sample-size
gates**, which are **calibrated to the timeframe and the length of the data**: 15 trades per year of data
on daily candles, rising slowly with the number of candles per day (≈ 26 on 4h, ≈ 39 on 1h, ≈ 59 on 15m),
times the years of the search period and of the validation segment, with floors of 30 and 10 trades. On 5
years of daily data that is 60 train + validation and 15 validation trades; the holdout gate of a Robust
verdict is computed the same way (16). A strategy below any gate is disqualified outright. The counts are
the primary asset's, or the effective bets pooled over an asset pool.

The **deflated Sharpe ratio is a gate, not a score term**: it decides whether a strategy counts as robust
(≥ 50%) and whether a locked strategy can be rated Robust (≥ 95%), but it never moves the fitness. The
fitness therefore stays fixed for the whole run, and the search does not hill-climb the validation Sharpe
it is later judged on. Raw return over Buy & Hold does not count anywhere: it is a bet on the market's
direction, not evidence of an edge. Complexity is capped by the genome (5 numbers), not penalised.

**Plateaus, not spikes – judged once, after the search.** Evaluation runs in two tiers. Inside the search
loop every candidate gets the fast single-pass multi-fold score (tier 1). After every round, the top 20
contenders (gates 1 and 2 first, then by fitness) go through tier 2 once: every one-grid-step parameter
neighbour is backtested (each tunable number moved one step down and one step up, everything else
unchanged) and the trades are replayed 200+ times in the Monte Carlo stress test. Those two results are gate
3; they never change the fitness. The same one-step neighbourhood drives the workbench's stability heatmap:
one definition of "nearby settings" throughout.

The **Deflated Sharpe Ratio** (Bailey & López de Prado) is the probability that a Sharpe ratio is real given
the number of strategies tried so far, the spread of their Sharpe ratios, the sample length, and the
skewness and kurtosis of returns. The hurdle rises with every trial, so every listed strategy's deflated
Sharpe – and with it its robust status – is re-checked as the run goes on and can fall over time.

**Speed.** Candidate strategies are scored on several CPU cores: a `WorkerPool` starts
`max(1, min(cores − 1, 8))` evaluator workers and the optimiser's `BatchDispatcher` sends each round of
candidates to all of them at once, one contiguous chunk per worker, over direct MessageChannels. On the
default preset (4h, BTC/ETH/SOL/BNB, 25,000 evaluations) a 4-core machine finishes in about 13 s with three
evaluators versus about 22 s with one. Results are applied in proposal order, so a seeded run gives the same answer
whatever the pool size. With one or two cores, or if an evaluator fails to load, the optimiser scores
candidates itself. Within each evaluation, the expensive parts are kept small: only the bars inside
trades are visited when computing metrics, the second condition of an entry is only checked where the
trigger fired. Every candidate is backtested on every asset: the robustness gates are not fitness-based,
so a candidate that cannot make the leaderboard may still be one of the robust strategies a scan is after.

The leaderboard keeps the 25 best behaviourally distinct strategies: two candidates whose equity returns
correlate above 0.9 count as the same idea and only the fitter one is listed.

### Robustness panel and verdict

When a run ends (or when you pick a strategy in the workbench) it is tested on train + validation:

- **Top-trade removal:** net PnL without the 5 best trades (pooled over the assets of a pool).
- **Parameter stability:** the heatmap of the two most sensitive parameters (the gate itself is the one-step
  neighbourhood above).
- **Monte Carlo execution stress test** (`src/engine/monteCarlo.ts`, the StrategyQuant X / AmiBroker kind):
  500 replays of the strategy's search-period trades, each with 10% of the trades skipped at random, the
  order shuffled, and every fill charged an extra adverse slippage shock (exponential, mean = the run's
  whole per-side cost – fee + slippage floor, at least 5 bp – so costs double on average, with a 5% chance
  of a 5× spike). With a pooled asset pool the replayed trades are the pooled ones. Each trade's worst intrabar excursion is replayed
  too, so drawdowns are intrabar worst cases. It reports the **probability of profit** and the
  **95th-percentile worst-case drawdown**; it passes at ≥ 80% and < 50%. Every leaderboard entry carries the
  headline numbers; the workbench draws the percentile bands of the paths.
- **Cross-market generalization check** (workbench, on demand): the strategy's frozen rules on related
  markets the search never saw – for example SOL and BNB for a strategy found on BTC and ETH – using only
  their candles **before the run's holdout boundary**, so the holdout stays sealed. It passes when at least
  two unseen markets are tested, two thirds of them make money and the median Sharpe is positive. Assets that
  were part of the run are listed but not counted.

Verdict (only the locked strategy gets one): **Overfit** when its deflated Sharpe is below 50% or it lost
money (or had a non-positive Sharpe) on the holdout. **Robust** only when ALL of these hold: deflated Sharpe
≥ 95%; the search-period gates (validation, folds, neighbourhood); holdout Sharpe 95% confidence interval
above zero; holdout Sharpe above Buy & Hold's; enough holdout trades (the timeframe-scaled holdout gate);
profitable without its 5 best trades; the Monte Carlo stress test passed; generalisation shown – a pooled
multi-asset run, or a passed cross-market check (running the check after the reveal updates the verdict);
the holdout period was never revealed before; live (not simulated) data. **Fragile** otherwise; the
workbench lists every unmet condition.

The workbench translates all of this into a **Strategy verdict** box (why it looks strong, what to watch
out for) and six status badges with plain-English tooltips: *Beats luck* (deflated Sharpe), *Consistent
over time* (folds), *Survives stress* (Monte Carlo), *Stable parameters* (neighbourhood), *Works on other
markets* (cross-market check) and *Holdout*.

### Next Entry Forecast

The Strategy workbench shows where the strategy stands on the latest closed candle
(`src/engine/nextEntrySolver.ts`), for all three direction modes (an Asymmetric short side is forecast
with its bear-regime filter, and every new building block – Keltner bands, squeeze releases, range
expansions, pullbacks – is solved exactly like the other conditions):

- **Trigger price.** A synthetic next candle (open = last close, close = P) is appended to the last
  2,000 candles and run through the backtester's own indicator cache and rule evaluators. A 241-point
  grid scan across ±10 × ATR(14) (capped at ±30%) brackets the closes that fire each entry rule, and
  48 bisection steps find the exact close. The card shows that close, the move it needs, the estimated
  fill at the following open (plus slippage), and the stop loss and take profit at that fill – exactly the
  backtester's levels (the ATR of the signal candle) – with whether the stop will trail (and by how many
  ATRs) and the price at +1R where a breakeven lock moves it to the entry.
- **Pending entries.** If the rule already fired on the last candle, the entry is pending for the next
  open. If a position is open, the card shows its live stop, target, time stop and the close that
  would trigger a signal exit or reversal (the stop is the backtester's own trailing stop), and the
  price at +1R that will lock the stop at breakeven – or that the lock is already active.
- **Blockers.** A filter no single candle can satisfy (for example a low ADX, or a volume spike, with
  the next candle assumed to trade at average volume) is reported next to the price the rest of the
  rule needs, e.g. `$148.20 (Blocked: ADX(7) 12.1 < 15)`.
- **Readiness.** Each side scores 0–100% from the share of conditions met, how far each indicator is
  from its threshold (in ATRs for prices and averages, in points for oscillators), and the distance
  to the trigger. The scores give a verdict from Strong Long to Strong Short, or In Position.
- **Refresh Latest Price** re-fetches candles through the normal data pipeline without re-optimising.
  Baskets and asset pools get a table of every asset, closest to an entry first.

The forecast is a mechanical reading of the rules, not a prediction: it says what close would fire
the strategy, not whether price will get there.

## Read this before trusting a result

- **Expect small numbers and few "Robust" verdicts.** With 1–2% risk per trade and a deflated Sharpe hurdle,
  most strategies are labelled Overfit or Fragile. That is the tool being honest, not broken: a year or
  two of validation data cannot prove an edge against thousands of trials. On the audit's random walks
  (6 driftless, 6 with +40% a year of drift, one hypothesis per run, whole boards), no strategy reached a
  95% deflated Sharpe, none met the holdout conditions of a Robust verdict, and every locked #1 was rated
  Overfit. On five years of daily candles the breakout and dip-buy hypotheses rarely reach the minimum
  number of trades at all – use 4h or 1h candles for them. On live Binance pools (the default 4h preset
  and the daily Trend Breakout pool, every hypothesis, see `docs/AUDIT.md`) no strategy cleared the
  significance gate at the default budget either, and neither did any random-walk control. The search does
  find parsimonious strategies with a positive validation expectancy – but so it does on noise, and some
  live strategies beat Buy & Hold on their holdout and generalised to unseen coins: a year of validation
  data cannot separate a Sharpe-1.5 edge from the luckiest of 25,000 trials. Run the benchmark yourself
  with `LIVE_BENCHMARK=1 npx vitest run tests/live-benchmark.test.ts` (add `LIVE_BENCHMARK_POOLS=4h,1d`
  for both pools).
- **The holdout is a one-shot test.** Re-running on the same dataset and picking the run whose holdout looks
  best turns the holdout into training data. The reveal log makes that visible and blocks a Robust verdict
  on a holdout that has been seen before.
- **The holdout confidence interval is wide.** The drawer shows the 95% interval for the holdout Sharpe; with
  a few dozen trades it usually includes zero.
- **Daily bars cannot resolve intrabar order.** The "ambiguous candles" count shows how many trades relied on
  the stop-first assumption. The drawer warns above 10%.
- **Baskets are synthetic.** Components are aligned on shared timestamps and combined into a buy-and-hold
  index. A basket candle's high and low are its body widened by the root-sum-square of the legs' wicks (the
  sum of the legs' highs would be a price the basket almost never trades at).
- **Crypto is tested on the USDT pair.** `XRP-USD` is routed to `XRPUSDT` on Binance; the telemetry tile and
  the drawer show the instrument actually used.
- **Pine export is a faithful translation, not a guarantee of identical numbers.** It supports all three
  direction modes, every building block, the trailing ATR stop and the breakeven lock, and sets
  `use_bar_magnifier=false` so it compiles on every TradingView plan; Premium users can switch it to `true`
  to settle candles that touch both stop and target with lower-timeframe data. The code is generated from
  the same strategy AST the engine runs, and the parity tests execute it with a Pine-subset interpreter
  against the engine bar by bar (signals, stops, targets, trailing moves, breakeven locks); it was not run
  on TradingView itself.
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
src/engine/strategyAst.ts         Strategy AST: the one compiled form both the engine and the Pine exporter use
src/engine/pine.ts                Pine Script v5 exporter (compiles from the AST)
src/engine/monteCarlo.ts          Monte Carlo execution stress test
src/engine/nextEntrySolver.ts     Next Entry Forecast: trigger prices, readiness, carried state, non-price blockers
src/workers/workerPool.ts         WorkerPool (main thread) and BatchDispatcher (optimiser): sizing, dispatch, shutdown
src/workers/optimizer.worker.ts   Optimisation loop (Web Worker)
src/workers/evaluator.worker.ts   Evaluator pool member: scores candidate strategies
src/engine/evaluation.ts          Shared scoring code and the evaluator protocol
src/lib/stats.ts                  Deflated Sharpe Ratio, expected maximum Sharpe, confidence intervals
src/lib/holdoutLog.ts             Holdout reveal log (symbols + date window, overlap matching)
src/lib/trialLedger.ts            Cumulative trial budget per data window (overlap matching)
src/lib/resetRuns.ts              "Reset Runs & Clear Cache": wipes the ledger, holdout log and run caches
src/lib/storage.ts                IndexedDB stores (candles, ledger, runs), legacy migration, snapshots
src/lib/diagnostics.ts            Plain-English check badges and the Strategy verdict
src/lib/presets.ts                One-click quick-start presets
src/data/                         Data loader, incremental candle cache, offline generator, basket builder
src/components/                   TopActionBar, ControlBar + PresetCards (1), TelemetryStrip (2), Leaderboard + ComparePanel (3),
                                  StrategyWorkbench (4, tabbed) with NextEntryForecast, MonteCarloChart, StabilityHeatmap
src/components/ui/                shadcn/ui primitives (generated by the shadcn CLI – see CLAUDE.md)
src/components/common.tsx         Small app-level building blocks composed from those primitives
src/lib/theme.ts                  Resolves the theme tokens to colours for the canvas charts
src/index.css                     Tailwind v4 + shadcn/ui theme tokens (light and dark)
src/App.tsx                       Dashboard and worker orchestration
tests/                            Vitest suites
```

TypeScript is pinned to 5.9 because Netlify CLI's dev server loads the project's `typescript` package and
does not yet support TypeScript 7.
