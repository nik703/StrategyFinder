/**
 * Live benchmark sweep (opt-in, needs network access to Binance):
 *
 *   LIVE_BENCHMARK=1 npx vitest run tests/live-benchmark.test.ts
 *   (behind an HTTPS proxy also set NODE_USE_ENV_PROXY=1)
 *
 * By default it runs the default preset's live pool - Crypto Multi-Asset Swing (4h): BTC, ETH, SOL, BNB,
 * 2 years of 4h candles - for every hypothesis (Long-Only, plus Asymmetric L/S for the breakout and trend
 * ideas) at the default budget. The top strategy of each run gets the cross-market check on unseen majors
 * and is locked to reveal its holdout; every run is followed by its random-walk control (driftless walks
 * with the pool's length, timestamps and volatility, same settings). Options:
 *
 *   LIVE_BENCHMARK_POOLS=4h,1d           pools: 4h = the default preset's, 1d = Trend Breakout (1d)'s
 *   LIVE_BENCHMARK_HYPOTHESES=VOL_BREAKOUT,TREND_FOLLOWING   limit the hypotheses
 *   LIVE_BENCHMARK_EVALS=25000           evaluations per run
 *   LIVE_BENCHMARK_OUT=live-benchmark.md where the markdown table is written
 */

import { writeFileSync } from 'node:fs';
import { beforeAll, describe, it } from 'vitest';
import handler from '../netlify/functions/market-data';
import { buildDataset } from '../src/data/basket';
import { conditionFormula } from '../src/engine/describe';
import { gaussian, mulberry32 } from '../src/engine/genome';
import { DEFAULT_CONFIG, settingsFrom } from '../src/lib/config';
import { PRESETS } from '../src/lib/presets';
import type {
  BacktestSettings,
  CandleSeries,
  Hypothesis,
  MarketDataResponse,
  NamedDataset,
  OHLCV,
  TradeMode,
  WorkerRequest,
  WorkerResponse,
} from '../src/types/trading';

interface FakeScope {
  onmessage: ((event: { data: WorkerRequest }) => void) | null;
  postMessage: (message: WorkerResponse) => void;
}
const scope: FakeScope = { onmessage: null, postMessage: () => undefined };
type Of<T extends WorkerResponse['type']> = Extract<WorkerResponse, { type: T }>;

const OUT = process.env.LIVE_BENCHMARK_OUT ?? 'live-benchmark.md';
const EVALS = Number(process.env.LIVE_BENCHMARK_EVALS ?? 25_000);
const POOLS = (process.env.LIVE_BENCHMARK_POOLS ?? '4h').split(',').map((p) => p.trim());
const ONLY = process.env.LIVE_BENCHMARK_HYPOTHESES?.split(',').map((h) => h.trim()) ?? null;
const lines: string[] = [];
const say = (line: string): void => {
  lines.push(line);
  writeFileSync(OUT, `${lines.join('\n')}\n`);
};

function runOptimizer(datasets: NamedDataset[], settings: BacktestSettings, hypothesis: Hypothesis): Promise<Of<'done'>> {
  return new Promise((resolve, reject) => {
    scope.postMessage = (message) => {
      if (message.type === 'done') resolve(message);
      if (message.type === 'error') reject(new Error(message.message));
    };
    scope.onmessage?.({ data: { type: 'start', payload: { datasets, settings, hypothesis, seed: 2026, maxEvaluations: EVALS, robustTarget: 3 } } });
  });
}

function request(message: WorkerRequest): WorkerResponse {
  let reply: WorkerResponse | null = null;
  scope.postMessage = (response) => {
    reply = response;
  };
  scope.onmessage?.({ data: message });
  if (reply === null) throw new Error('The worker did not answer.');
  return reply;
}

async function live(symbols: string[], interval: string, range: string): Promise<Record<string, CandleSeries>> {
  const response = await handler(new Request(`http://localhost/.netlify/functions/market-data?symbols=${symbols.join(',')}&interval=${interval}&range=${range}`));
  const body = (await response.json()) as MarketDataResponse;
  if (!response.ok) throw new Error(JSON.stringify(body).slice(0, 400));
  for (const symbol of symbols) if (!body.series[symbol]) throw new Error(`${symbol}: ${body.errors?.[symbol] ?? 'missing'}`);
  return body.series;
}

const named = (series: Record<string, CandleSeries>, symbol: string): NamedDataset => ({ symbol, data: buildDataset(series, [{ symbol, weight: 1 }]).data });

/** Driftless random walk with the same timestamps and per-bar volatility as `like`. */
function walkLike(like: OHLCV, seed: number): OHLCV {
  const n = like.c.length;
  let s2 = 0;
  for (let i = 1; i < n; i++) s2 += Math.log(like.c[i] / like.c[i - 1]) ** 2;
  const sigma = Math.sqrt(s2 / (n - 1));
  const rng = mulberry32(seed);
  const data: OHLCV = { t: like.t.slice(), o: new Float64Array(n), h: new Float64Array(n), l: new Float64Array(n), c: new Float64Array(n), v: new Float64Array(n) };
  let close = 100;
  for (let i = 0; i < n; i++) {
    const open = close;
    close = open * Math.exp(sigma * gaussian(rng) - 0.5 * sigma * sigma);
    data.o[i] = open;
    data.c[i] = close;
    data.h[i] = Math.max(open, close) * (1 + Math.abs(gaussian(rng)) * sigma * 0.5);
    data.l[i] = Math.min(open, close) * (1 - Math.abs(gaussian(rng)) * sigma * 0.5);
    data.v[i] = 1000 * Math.exp(gaussian(rng) * 0.3);
  }
  return data;
}

beforeAll(async () => {
  if (!process.env.LIVE_BENCHMARK) return;
  (globalThis as unknown as { self: FakeScope }).self = scope;
  await import('../src/workers/optimizer.worker');
});

/** Board summary: how many strategies, how many with a positive validation expectancy, and their parameter counts. */
function boardSummary(done: Of<'done'>): { board: string; params: string } {
  const list = done.strategies;
  if (list.length === 0) return { board: '0', params: '–' };
  const positive = list.filter((s) => s.validation.trades > 0 && s.validation.netPnl / s.validation.trades > 0).length;
  const params = list.map((s) => s.params).sort((a, b) => a - b);
  const mid = params[params.length >> 1];
  return { board: `${positive}/${list.length}`, params: `${mid} / ${params[params.length - 1]}` };
}

/** Per-trade validation expectancy of a strategy, as text. */
const perTrade = (s: Of<'done'>['strategies'][number]): string => {
  const value = s.validation.netPnl / Math.max(1, s.validation.trades);
  return `${value >= 0 ? '+' : '-'}$${Math.abs(value).toFixed(0)}/trade`;
};

describe.skipIf(!process.env.LIVE_BENCHMARK)('live benchmark sweep', () => {
  it("runs every hypothesis on the default preset's live crypto pool, plus random-walk controls", async () => {
    say(`# Live sweep (Binance candles, ${EVALS.toLocaleString('en-US')} evaluations a run, stop after 3 robust)`);
    say(
      '| Pool | Hypothesis | Mode | Evals | Cycles | Robust | Best DSR | Board: val. expectancy > 0 | Params (median / max) | Top strategy | MC P(profit) / DD95 | Locked #1: holdout ROI · Sharpe vs B&H · verdict | Cross-market (unseen) |',
    );
    say('|---|---|---|---|---|---|---|---|---|---|---|---|---|');
    const pools = [
      { id: '4h', label: '4h 2y BTC·ETH·SOL·BNB', preset: 'crypto-swing-4h', unseen: ['XRP-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD'] },
      { id: '1d', label: '1d 5y BTC·ETH·SOL·XRP', preset: 'trend-breakout-1d', unseen: ['BNB-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD'] },
    ].filter((pool) => POOLS.includes(pool.id));
    const runs = (
      [
        ['VOL_BREAKOUT', 'long-only'],
        ['VOL_BREAKOUT', 'asymmetric'],
        ['TREND_FOLLOWING', 'long-only'],
        ['TREND_FOLLOWING', 'asymmetric'],
        ['DIP_IN_UPTREND', 'long-only'],
        ['RANGE_REVERSION', 'long-only'],
      ] as [Hypothesis, TradeMode][]
    ).filter(([h]) => ONLY === null || ONLY.includes(h));
    for (const pool of pools) {
      const preset = PRESETS.find((p) => p.id === pool.preset);
      if (!preset) throw new Error(pool.preset);
      const base = { ...DEFAULT_CONFIG, ...preset.patch };
      const symbols = base.pool.split(',').map((s) => s.trim());
      const series = await live(symbols, base.interval, base.range);
      const datasets = symbols.map((symbol) => named(series, symbol));
      const extra = await live(pool.unseen, base.interval, base.range);
      // Random-walk controls with the pool's own length, timestamps and volatility.
      const walks = datasets.map((d, k) => ({ symbol: `RW-${k}`, data: walkLike(d.data, 9100 + k) }));
      for (const [hypothesis, tradeMode] of runs) {
        const settings = settingsFrom({ ...base, tradeMode, hypothesis, fundingPct8h: 0.01 });
        const done = await runOptimizer(datasets, settings, hypothesis);
        const robust = done.strategies.filter((s) => s.robust?.passed);
        const pick = robust[0] ?? done.strategies[0];
        let lockText = '–';
        let crossText = '–';
        if (pick) {
          const cross = request({ type: 'cross-market', strategyId: pick.id, genome: pick.genome, datasets: pool.unseen.map((symbol) => named(extra, symbol)) }) as Of<'cross-market'>;
          crossText = `${Math.round(cross.report.share * cross.report.tested)}/${cross.report.tested} profitable, median Sharpe ${cross.report.medianSharpe.toFixed(2)} → ${cross.report.passed ? 'PASS' : 'fail'}`;
          const locked = request({ type: 'lock', strategyId: pick.id, priorReveals: 0, simulated: false }) as Of<'locked'>;
          const h = locked.strategy.holdout;
          lockText = `${h ? `${h.roi >= 0 ? '+' : ''}${h.roi.toFixed(1)}% · ${h.sharpe.toFixed(2)} vs ${locked.benchmark.holdout?.sharpe.toFixed(2)}` : '–'} · ${locked.strategy.verdict}`;
        }
        const top = pick
          ? `${pick.id}${pick.robust?.passed ? ' ✓' : ''} ${pick.genome.longEntry.conditions.map(conditionFormula).join(' AND ')}${
              pick.genome.shortRegime ? ` / short gate ${conditionFormula(pick.genome.shortRegime)}` : ''
            } (${pick.params}p, val. ${pick.validation.trades} trades, ${perTrade(pick)})`
          : '–';
        const mc = pick?.monteCarlo;
        const bestDsr = Math.max(0, ...done.strategies.map((s) => s.dsr));
        const summary = boardSummary(done);
        say(
          `| ${pool.label} | ${hypothesis} | ${tradeMode} | ${done.evaluations.toLocaleString('en-US')} | ${done.cycle} | **${done.robustFound}** | ${(bestDsr * 100).toFixed(0)}% | ${summary.board} | ${summary.params} | ` +
            `${top.split('|').join('/')} | ${mc ? `${(mc.probProfit * 100).toFixed(0)}% / ${mc.drawdownP95.toFixed(1)}%` : '–'} | ${lockText} | ${crossText} |`,
        );
        const control = await runOptimizer(walks, settings, hypothesis);
        const controlDsr = Math.max(0, ...control.strategies.map((s) => s.dsr));
        const controlSummary = boardSummary(control);
        say(
          `| random walks (${pool.label} vol) | ${hypothesis} | ${tradeMode} | ${control.evaluations.toLocaleString('en-US')} | ${control.cycle} | **${control.robustFound}** | ${(controlDsr * 100).toFixed(0)}% | ${controlSummary.board} | ${controlSummary.params} | – | – | – | – |`,
        );
      }
    }
  }, 14_400_000);
});
