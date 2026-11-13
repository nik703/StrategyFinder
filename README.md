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
| `interval` | `1h`, `4h`, `1d` |
| `range` | `1y`, `2y`, `5y` |

The response is columnar JSON with CORS headers: `{ interval, range, generatedAt, series: { SYMBOL: { symbol,
source, venueSymbol, t, o, h, l, c, v } }, errors }`. `t` is the bar open time in UTC milliseconds, normalised so venues
line up: daily bars sit at 00:00 UTC of the exchange-local date, hourly bars at the top of the hour, and 4h
bars in UTC-aligned buckets. Yahoo serves hourly history for about 730 days only, so `1h` / `4h` stock
requests longer than that are truncated (the response carries a `note`).

### Backtester

- Signals are evaluated on the close of candle `t` and filled at the open of candle `t+1`.
- Every fill pays the fee (default 0.1% per side) and slippage (default 0.05%). Perpetual funding can be
  charged on open positions (off by default, 0.01% per 8h when switched on).
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
- When the run ends (Stop or converged) the final leaderboard is evaluated on the holdout once, and the
  holdout part of the chart appears.
- A holdout is only unseen once. The app counts reveals per dataset (symbols, timeframe, range, end date) in
  `localStorage` and warns on later runs: "Holdout already viewed N times. It is no longer unseen data."

### Strategy genome

Every strategy follows one archetype and stays small:

- **Trend-following:** one trigger that fires when price or momentum turns up, plus an optional filter
  (ATR %, volume ratio, ADX strength or Bollinger bandwidth).
- **Mean reversion:** one trigger on an oversold reading or a short-term downswing (lookback ≤ 30), plus an
  optional slow up-trend regime filter (lookback ≥ 100) or a filter. "Fade Supertrend flips against the
  200-bar trend" is this archetype.
- At most 2 conditions per entry, always joined by `AND`. In Long/Short mode the short rules are the exact
  mirror of the long rules.
- **Round-number grids only.** Lookbacks from {5, 7, 10, 14, 20, 21, 26, 30, 50, 100, 150, 200}; oscillator
  levels in steps of 5; ATR, Bollinger and Supertrend multipliers in steps of 0.5; static stop / target from
  {1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30} %. Stochastic smoothing is fixed at 3 / 3.
- **At most 8 tunable numbers** per strategy, with a fitness penalty for each one above 4.
- **Exits:** static % stop and target, or ATR stop (1–4×) with optional ATR target (1.5–8×), trailing stop,
  trigger-reversal exit (trend only, the mirror of the entry trigger) and an optional time stop
  (5, 10, 20 or 40 bars). A trailing stop is only kept when the target sits at least 1 ATR beyond the stop.

### Optimiser

Evolutionary hill-climbing with mutation: coherent textbook seeds and random genomes start the population;
each iteration mutates the champion or a rank-biased parent, recombines two leaderboard members, or injects a
fresh random genome. Progress is posted every 25 iterations. The run ends when you press **Stop** or after
a configurable number of consecutive iterations without a better top score (default 5,000, set in the control bar).

**Robust universe** (on by default for crypto): the same rules are backtested separately on every asset of
a universe (default BTC, ETH, SOL, XRP, ADA, DOGE; editable). Fitness is the median per-asset score, trade
counts are summed across assets, and there is a penalty when fewer than 2/3 of the assets are profitable on
validation. The leaderboard and chart show the primary asset.

Fitness (higher is better), per asset:

```
+ 3.00 × Deflated Sharpe Ratio of the validation segment (plus a small slice of its z-score as a gradient)
+ 0.35 × train risk-adjusted return (0.6 Sortino + 0.4 Sharpe, clamped)
+ 1.50 × validation excess return over Buy & Hold
+ 0.75 × train + validation excess return over Buy & Hold
− penalty for intrabar drawdowns beyond 20%, steeper beyond 45%
− penalty for a loss on the training segment, and for train / validation divergence
```

then, across assets:

```
  median of the per-asset scores
− up to 6.0 for fewer than 100 trades, up to 2.0 for fewer than 25 validation trades
− 1.0 when fewer than 2/3 of the universe is profitable on validation
− 0.15 for each tunable parameter above 4
− 0.75 for reward:risk below 1:1, unless the validation win rate clears break-even by 10 points
```

The **Deflated Sharpe Ratio** (Bailey & López de Prado) is the probability that a Sharpe ratio is real given
the number of strategies tried so far, the spread of their Sharpe ratios, the sample length, and the
skewness and kurtosis of returns. The hurdle rises with every trial, so the leaderboard is re-scored as
the run goes on and a score can fall over time.

The leaderboard keeps the 25 best behaviourally distinct strategies: two candidates whose equity returns
correlate above 0.9 count as the same idea and only the fitter one is listed.

### Robustness panel and verdict

When a run ends (or on demand from the strategy drawer) each strategy is tested on train + validation:

- **Parameter sensitivity:** every parameter is moved by ±20% (snapped to its grid, or to the next grid step
  when ±20% rounds back). Fragile when the worst variant loses more than half of the original ROI.
- **Top-trade removal:** net PnL without the 5 best trades.
- **Cross-asset:** the exact rules, untuned, on every universe asset.

Verdict: **Overfit** when it lost money on the holdout or its deflated Sharpe is below 50%; **Fragile**
when it survives the holdout but fails a test above or its deflated Sharpe is below 95%; **Robust** otherwise.

## Read this before trusting a result

- **Expect small numbers and few "Robust" verdicts.** With 2% risk per trade and a deflated Sharpe hurdle,
  most strategies are labelled Overfit or Fragile. That is the tool being honest, not broken: a year or
  two of validation data cannot prove an edge against thousands of trials. On six driftless random walks
  the best strategy's deflated Sharpe stayed at or below 16%, every listed strategy was labelled Overfit,
  and the mean holdout Sharpe was −0.36.
- **The holdout is a one-shot test.** Re-running on the same dataset and picking the run whose holdout looks
  best turns the holdout into training data. The reveal counter exists to make that visible.
- **The holdout confidence interval is wide.** The drawer shows the 95% interval for the holdout Sharpe; with
  a few dozen trades it usually includes zero.
- **Daily bars cannot resolve intrabar order.** The "ambiguous candles" count shows how many trades relied on
  the stop-first assumption. The drawer warns above 10%.
- **Baskets are synthetic.** Components are aligned on shared timestamps and combined into a buy-and-hold
  index. Basket highs and lows are sums of component highs and lows, which is a conservative envelope.
  The Robust universe is not available for baskets.
- **Crypto is tested on the USDT pair.** `XRP-USD` is routed to `XRPUSDT` on Binance; the telemetry tile and
  the drawer show the instrument actually used.
- **Pine export is a faithful translation, not a guarantee of identical numbers.** It switches on Bar
  Magnifier (TradingView Premium) to settle candles that touch both stop and target. The code is generated
  and syntax-tested here but was not run on TradingView.
- **Paper trade before going live.** The test window ends at the latest candle, so only new data is unseen.

## Project layout

```
netlify.toml                      Build, functions, dev and SPA redirect settings
netlify/functions/market-data.ts  Market-data proxy (Binance + Yahoo Finance)
src/types/trading.ts              Shared types and worker protocol
src/engine/indicators.ts          Indicator library and memoising cache
src/engine/backtester.ts          Execution model, metrics, split, benchmark, fitness
src/engine/genome.ts              Archetypes, grids, random generation, mutation, repair
src/engine/describe.ts            Plain-English rules and badges
src/engine/pine.ts                Pine Script v5 exporter
src/workers/optimizer.worker.ts   Optimisation loop (Web Worker)
src/lib/stats.ts                  Deflated Sharpe Ratio, confidence intervals
src/lib/holdoutLog.ts             Holdout reveal counter
src/data/                         Data loader, offline generator, basket builder
src/components/                   ControlBar, TelemetryStrip, EquityChart, Leaderboard, StrategyModal
src/App.tsx                       Dashboard and worker orchestration
tests/                            Vitest suites
```

TypeScript is pinned to 5.9 because Netlify CLI's dev server loads the project's `typescript` package and
does not yet support TypeScript 7.
