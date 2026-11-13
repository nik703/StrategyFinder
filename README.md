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
source, t, o, h, l, c, v } }, errors }`. `t` is the bar open time in UTC milliseconds, normalised so venues
line up: daily bars sit at 00:00 UTC of the exchange-local date, hourly bars at the top of the hour, and 4h
bars in UTC-aligned buckets. Yahoo serves hourly history for about 730 days only, so `1h` / `4h` stock
requests longer than that are truncated (the response carries a `note`).

### Backtester

- Signals are evaluated on the close of candle `t` and filled at the open of candle `t+1`.
- Every fill pays the fee (default 0.1% per side) and slippage (default 0.05%).
- Stops and targets are simulated intrabar from the entry candle. A gap through a level fills at the open.
  If one candle touches both the stop and the target, the stop is assumed to fill first.
- ATR levels use the ATR of the signal candle. Trailing stops update only after a candle has closed.
- Data is split chronologically: the first 70% is in-sample, the last 30% out-of-sample.
- Buy & Hold buys at the first open with the same costs and capital.

### Strategy genome

- **Entries:** 1 to 4 conditions per side joined by `AND` or `OR`, from RSI, MACD (line, signal, histogram),
  EMA / SMA (price and average crossovers), Bollinger Bands (bands, %B, bandwidth), ATR %, Stochastic
  (%K, %D), ADX / DMI, volume spike ratio, Supertrend and CCI. Long-Only or Long/Short.
- **Static exits:** fixed stop loss (0.5% to 15%) and take profit (1% to 60%).
- **Dynamic exits:** ATR stop (1.0× to 4.0×), ATR target (1.5× to 8.0×), trailing ATR stop, and
  indicator-reversal exit rules.
- **Sizing:** 100% of equity, or volatility sizing that risks a fixed share of equity per stop distance.

### Optimiser

Evolutionary hill-climbing with mutation: textbook seed strategies and random genomes start the population;
each iteration mutates the champion or a rank-biased parent, recombines two leaderboard members, or injects a
fresh random genome. Progress is posted every 25 iterations. The run ends when you press **Stop** or after
2,000 consecutive iterations without a better top score.

Fitness (higher is better):

```
+ 1.00 × out-of-sample risk-adjusted return (0.6 Sortino + 0.4 Sharpe, clamped)
+ 0.35 × in-sample risk-adjusted return
+ 1.50 × out-of-sample excess return over Buy & Hold
+ 0.75 × full-period excess return over Buy & Hold
− penalty for fewer than 20 trades (and fewer than 6 out-of-sample trades)
− penalty for drawdowns beyond 20%, steeper beyond 45%
− penalty for an in-sample loss, and for in-sample / out-of-sample divergence
```

The leaderboard keeps the 25 best behaviourally distinct strategies: two candidates whose equity returns
correlate above 0.9 count as the same idea and only the fitter one is listed.

## Read this before trusting a result

- **Out-of-sample numbers are validation, not proof.** The ranking score uses the out-of-sample segment, so
  the search is fitted to it. On a pure random walk the optimiser still reports top strategies with Sharpe
  ratios near 2, and those same strategies score at or below zero on data it never saw. The strategy breakdown
  shows how good the best of N skill-less strategies would look by luck, next to the strategy's own figure.
  Forward-test on new data before risking capital.
- **Baskets are synthetic.** Components are aligned on shared timestamps and combined into a buy-and-hold
  index. Basket highs and lows are sums of component highs and lows, which is a conservative envelope.
- **Pine export is a faithful translation, not a guarantee of identical numbers.** TradingView uses its own
  data feed and intrabar fill model. The Pine code is generated and syntax-tested here but was not run on
  TradingView.

## Project layout

```
netlify.toml                      Build, functions, dev and SPA redirect settings
netlify/functions/market-data.ts  Market-data proxy (Binance + Yahoo Finance)
src/types/trading.ts              Shared types and worker protocol
src/engine/indicators.ts          Indicator library and memoising cache
src/engine/backtester.ts          Execution model, metrics, benchmark, fitness
src/engine/genome.ts              Random generation, mutation, crossover, repair
src/engine/describe.ts            Plain-English rules and badges
src/engine/pine.ts                Pine Script v5 exporter
src/workers/optimizer.worker.ts   Optimisation loop (Web Worker)
src/data/                         Data loader, offline generator, basket builder
src/components/                   ControlBar, TelemetryStrip, EquityChart, Leaderboard, StrategyModal
src/App.tsx                       Dashboard and worker orchestration
tests/                            Vitest suites
```

TypeScript is pinned to 5.9 because Netlify CLI's dev server loads the project's `typescript` package and
does not yet support TypeScript 7.
