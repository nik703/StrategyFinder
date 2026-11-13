/**
 * Optimisation engine – runs entirely inside a dedicated Web Worker.
 *
 * Indicator calculation, backtesting and the evolutionary search all happen
 * here, so the main thread stays responsive and nothing depends on (or is
 * limited by) a serverless function's execution timeout.
 *
 * Search: evolutionary hill-climbing with mutation.
 *   • seeding      – a handful of classic templates plus random genomes
 *   • exploitation – mutate the champion (hill-climb) or a rank-biased parent
 *   • recombination– block crossover between two leaderboard members
 *   • exploration  – fresh random immigrants, more often while stagnating
 *
 * The loop runs until the UI sends `stop` or the configured number of consecutive
 * iterations fail to improve the top fitness score.
 */

import {
  BacktestWorkspace,
  barsPerYear,
  buildBenchmark,
  computeFitness,
  evaluateGenome,
  UNFIT,
  type Benchmark,
} from '../engine/backtester';
import {
  crossoverGenomes,
  genomeNiche,
  genomeSignature,
  mulberry32,
  mutateGenome,
  randomGenome,
  repairGenome,
  type GenomeLimits,
  type Rng,
} from '../engine/genome';
import { IndicatorCache } from '../engine/indicators';
import {
  IN_SAMPLE_FRACTION,
  LEADERBOARD_SIZE,
  DEFAULT_MAX_STAGNATION,
  MAX_MAX_STAGNATION,
  MIN_MAX_STAGNATION,
  PROGRESS_EVERY,
  type ChartAxis,
  type OHLCV,
  type OptimizerStartPayload,
  type ProgressSnapshot,
  type StopReason,
  type StrategyGenome,
  type StrategyResult,
  type TradeMode,
  type WorkerRequest,
  type WorkerResponse,
} from '../types/trading';

/* -------------------------------------------------------------------------- */
/*  Worker plumbing                                                           */
/* -------------------------------------------------------------------------- */

interface WorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse): void;
}

const scope = self as unknown as WorkerScope;

/** Minimum number of candles required for a meaningful 70/30 split. */
const MIN_BARS = 150;
/** Maximum number of points per equity curve sent to the UI. */
const MAX_CHART_POINTS = 1200;
/** Leaderboard snapshots are throttled so the UI is not flooded on fast datasets. */
const LEADERBOARD_INTERVAL_MS = 150;
/** Maximum number of leaderboard slots a single structural niche may occupy. */
const NICHE_CAP = 3;
/**
 * Two strategies whose sampled equity returns correlate at least this much trade
 * the same idea. Only the fitter twin is listed, so the Top 25 holds 25
 * behaviourally distinct strategies instead of parameter tweaks of one.
 */
const CLONE_CORRELATION = 0.9;
/** Pure random search until the leaderboard holds this many strategies. */
const SEED_POPULATION = 10;

let activeRun = 0;
let stopRequested = false;

/**
 * Yield to the worker's event loop so a queued `stop` message can be handled.
 * A MessageChannel round-trip is not subject to the 4 ms clamp of nested timers.
 */
const yieldChannel = new MessageChannel();
let resumeLoop: (() => void) | null = null;
yieldChannel.port1.onmessage = () => {
  const resume = resumeLoop;
  resumeLoop = null;
  resume?.();
};
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    resumeLoop = resolve;
    yieldChannel.port2.postMessage(0);
  });
}

scope.onmessage = (event) => {
  const message = event.data;
  if (message.type === 'stop') {
    stopRequested = true;
    return;
  }
  if (message.type === 'start') {
    activeRun++;
    stopRequested = false;
    const token = activeRun;
    run(message.payload, token).catch((error: unknown) => {
      if (token !== activeRun) return;
      scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    });
  }
};

/* -------------------------------------------------------------------------- */
/*  Setup helpers                                                             */
/* -------------------------------------------------------------------------- */

function validate(data: OHLCV): void {
  const n = data.c.length;
  if (n < MIN_BARS) {
    throw new Error(`Not enough history: ${n} candles loaded, at least ${MIN_BARS} are required.`);
  }
  for (const key of ['t', 'o', 'h', 'l', 'v'] as const) {
    if (data[key].length !== n) throw new Error('Malformed market data: OHLCV columns have different lengths.');
  }
  for (let i = 0; i < n; i++) {
    if (!(data.o[i] > 0) || !(data.h[i] > 0) || !(data.l[i] > 0) || !(data.c[i] > 0)) {
      throw new Error(`Malformed market data: non-positive or missing price at candle ${i}.`);
    }
  }
}

/** Evenly spaced sample of bar indices that always contains bar 0, the split bar and the last bar. */
function buildAxis(t: Float64Array, splitIndex: number): ChartAxis {
  const n = t.length;
  const stride = Math.max(1, Math.ceil(n / MAX_CHART_POINTS));
  const picked = new Set<number>();
  for (let i = 0; i < n; i += stride) picked.add(i);
  picked.add(splitIndex);
  picked.add(n - 1);
  const barIndex = Int32Array.from([...picked].sort((a, b) => a - b));
  const time = new Float64Array(barIndex.length);
  let splitPos = 0;
  for (let k = 0; k < barIndex.length; k++) {
    time[k] = t[barIndex[k]];
    if (barIndex[k] === splitIndex) splitPos = k;
  }
  return { barIndex, time, splitPos, bars: n, splitIndex };
}

function sampleCurve(equity: Float64Array, axis: ChartAxis): Float32Array {
  const out = new Float32Array(axis.barIndex.length);
  for (let k = 0; k < out.length; k++) out[k] = equity[axis.barIndex[k]];
  return out;
}

/** Simple returns between consecutive samples of an equity curve. */
function curveReturns(curve: Float32Array): Float32Array {
  const out = new Float32Array(curve.length);
  for (let k = 1; k < curve.length; k++) out[k] = curve[k - 1] > 0 ? curve[k] / curve[k - 1] - 1 : 0;
  return out;
}

/** Pearson correlation of two equally long return series (0 when either is flat). */
function correlation(a: Float32Array, b: Float32Array): number {
  const n = a.length;
  let sa = 0;
  let sb = 0;
  for (let k = 0; k < n; k++) {
    sa += a[k];
    sb += b[k];
  }
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let k = 0; k < n; k++) {
    const da = a[k] - ma;
    const db = b[k] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

/** 5th–95th percentile of the valid in-sample values of a series. */
function typicalRange(series: Float64Array, to: number, fallback: readonly [number, number]): readonly [number, number] {
  const values: number[] = [];
  for (let i = 0; i < to; i++) {
    const v = series[i];
    if (v === v && v > 0) values.push(v);
  }
  if (values.length < 20) return fallback;
  values.sort((a, b) => a - b);
  const lo = values[Math.floor(values.length * 0.05)];
  const hi = values[Math.floor(values.length * 0.95)];
  return hi > lo ? [lo, hi] : fallback;
}

function deriveLimits(ind: IndicatorCache, splitIndex: number): GenomeLimits {
  const volume = ind.data.v;
  let withVolume = 0;
  for (let i = 0; i < volume.length; i++) if (volume[i] > 0) withVolume++;
  return {
    maxLookback: Math.max(30, Math.min(250, Math.floor(splitIndex / 4))),
    hasVolume: withVolume > volume.length * 0.5,
    // Thresholds for unbounded indicators are drawn from what the in-sample data actually shows.
    atrPercentRange: typicalRange(ind.atrPercent(14), splitIndex, [0.5, 6]),
    bandwidthRange: typicalRange(ind.bbBandwidth(20, 2), splitIndex, [2, 30]),
  };
}

/** Well-known textbook strategies that give the search a sensible starting population. */
function seedGenomes(mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome[] {
  const long = Math.min(200, limits.maxLookback);
  const mid = Math.min(50, limits.maxLookback);
  const seeds: StrategyGenome[] = [
    {
      // Supertrend trend-following with a trailing ATR stop.
      longEntry: { logic: 'AND', conditions: [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }] },
      shortEntry: null,
      exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 3, trailing: true, takeProfitAtrMult: null, longExit: null, shortExit: null },
      sizing: { mode: 'full' },
    },
    {
      // Moving-average crossover filtered by trend strength.
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'EMA', slowPeriod: mid, op: 'gt' },
          { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 },
        ],
      },
      shortEntry: null,
      exit: {
        mode: 'dynamic',
        atrPeriod: 14,
        stopAtrMult: 2.5,
        trailing: true,
        takeProfitAtrMult: null,
        longExit: {
          logic: 'AND',
          conditions: [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'EMA', slowPeriod: mid, op: 'lt' }],
        },
        shortExit: null,
      },
      sizing: { mode: 'full' },
    },
    {
      // Buy the dip inside an uptrend.
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'RSI', period: 14, op: 'lt', level: 34 },
          { ind: 'PRICE_MA', maType: 'EMA', period: long, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 5, takeProfitPct: 10 },
      sizing: { mode: 'full' },
    },
    {
      // MACD signal-line crossover above a medium-term average.
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'MACD', fast: 12, slow: 26, signal: 9, variant: 'signal', op: 'cross_above' },
          { ind: 'PRICE_MA', maType: 'EMA', period: mid, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 2, trailing: false, takeProfitAtrMult: 4, longExit: null, shortExit: null },
      sizing: { mode: 'full' },
    },
    {
      // Bollinger-band mean reversion.
      longEntry: {
        logic: 'AND',
        conditions: [{ ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'cross_above', level: 0 }],
      },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 4, takeProfitPct: 8 },
      sizing: { mode: 'full' },
    },
    {
      // Stochastic turn-up from oversold with a volume confirmation.
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'kd', op: 'cross_above', level: 50 },
          { ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'lt', level: 35 },
          { ind: 'VOLUME', period: 20, op: 'gt', level: 1.1 },
        ],
      },
      shortEntry: null,
      exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 2, trailing: false, takeProfitAtrMult: 3.5, longExit: null, shortExit: null },
      sizing: { mode: 'full' },
    },
    {
      // Momentum continuation: strong RSI and CCI in an established trend.
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'RSI', period: 14, op: 'gt', level: 55 },
          { ind: 'CCI', period: 20, op: 'gt', level: 50 },
          { ind: 'PRICE_MA', maType: 'SMA', period: mid, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 2.5, trailing: true, takeProfitAtrMult: 6, longExit: null, shortExit: null },
      sizing: { mode: 'full' },
    },
  ];
  // `repairGenome` mirrors the long rules into short rules in Long/Short mode and clamps all lookbacks.
  return seeds.map((seed) => repairGenome(seed, mode, limits, rng));
}

/* -------------------------------------------------------------------------- */
/*  Optimisation run                                                          */
/* -------------------------------------------------------------------------- */

interface Entry {
  result: StrategyResult;
  niche: string;
  /** Outcome fingerprint – strategies that produce identical results are treated as duplicates. */
  fingerprint: string;
  /** Sampled equity returns, used to detect behavioural clones. */
  returns: Float32Array;
}

async function run(payload: OptimizerStartPayload, token: number): Promise<void> {
  const { data, settings } = payload;
  validate(data);

  const n = data.c.length;
  const splitIndex = Math.floor(n * IN_SAMPLE_FRACTION);
  const periodsPerYear = barsPerYear(data.t);
  const ind = new IndicatorCache(data);
  const ws = new BacktestWorkspace(n);
  const benchmark: Benchmark = buildBenchmark(data, settings, splitIndex, periodsPerYear);
  const axis = buildAxis(data.t, splitIndex);
  const limits = deriveLimits(ind, splitIndex);
  const mode = settings.tradeMode;
  const requested = Math.round(payload.maxStagnation ?? DEFAULT_MAX_STAGNATION);
  const maxStagnation = Number.isFinite(requested)
    ? Math.max(MIN_MAX_STAGNATION, Math.min(MAX_MAX_STAGNATION, requested))
    : DEFAULT_MAX_STAGNATION;
  const rng = mulberry32(payload.seed ?? (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0);

  scope.postMessage({
    type: 'ready',
    benchmark: {
      full: benchmark.full,
      inSample: benchmark.inSample,
      outOfSample: benchmark.outOfSample,
      equity: sampleCurve(benchmark.equity, axis),
    },
    axis,
    barsPerYear: periodsPerYear,
  });

  const board: Entry[] = [];
  const seen = new Set<string>();
  const seeds = seedGenomes(mode, limits, rng);
  let iteration = 0;
  let stagnation = 0;
  let boardDirty = false;
  let lastBoardPost = 0;
  let itersPerSec = 0;
  const startedAt = performance.now();
  let lastBatchAt = startedAt;

  const snapshot = (): ProgressSnapshot => ({
    iteration,
    stagnation,
    bestFitness: board.length > 0 ? board[0].result.fitness : null,
    bestRoi: board.length > 0 ? board[0].result.full.roi : null,
    itersPerSec,
    elapsedMs: performance.now() - startedAt,
  });

  const postBoard = (): void => {
    scope.postMessage({ type: 'leaderboard', iteration, strategies: board.map((e) => e.result) });
    boardDirty = false;
    lastBoardPost = performance.now();
  };

  /** Rank-biased parent selection: quadratic bias towards the top of the leaderboard. */
  const pickParent = (): StrategyGenome => board[Math.floor(rng() * rng() * board.length)].result.genome;

  const propose = (): StrategyGenome => {
    if (seeds.length > 0) return seeds.shift() as StrategyGenome;
    if (board.length < SEED_POPULATION) return randomGenome(rng, mode, limits);
    const progress = stagnation / maxStagnation;
    const r = rng();
    // Exploration grows from 10% to 25% of proposals as the search stagnates.
    const explore = 0.1 + 0.15 * progress;
    if (r < explore) return randomGenome(rng, mode, limits);
    if (r < explore + 0.12) return crossoverGenomes(rng, pickParent(), pickParent(), mode, limits);
    // Hill-climb the champion 40% of the time, otherwise a rank-biased parent.
    const parent = rng() < 0.4 ? board[0].result.genome : pickParent();
    return mutateGenome(rng, parent, mode, limits, 1 + Math.floor(progress * 3));
  };

  /** Insert into the leaderboard if it qualifies. Returns true when the top fitness improved. */
  const consider = (genome: StrategyGenome): boolean => {
    const metrics = evaluateGenome(ind, genome, settings, ws, splitIndex, periodsPerYear);
    const fitness = computeFitness(metrics, benchmark);
    if (fitness <= UNFIT) return false;
    const best = board.length > 0 ? board[0].result.fitness : -Infinity;
    if (board.length >= LEADERBOARD_SIZE && fitness <= board[board.length - 1].result.fitness) return false;

    const { full } = metrics;
    const fingerprint = `${full.trades}|${full.netPnl.toFixed(2)}|${full.maxDrawdown.toFixed(4)}|${metrics.outOfSample.netPnl.toFixed(2)}`;
    if (board.some((e) => e.fingerprint === fingerprint)) return false;

    // Behavioural clones: a fitter twin already on the board wins; otherwise this one replaces its twins.
    const equity = sampleCurve(ws.equity, axis);
    const returns = curveReturns(equity);
    const twins: number[] = [];
    for (let k = 0; k < board.length; k++) {
      if (correlation(returns, board[k].returns) >= CLONE_CORRELATION) twins.push(k);
    }
    if (twins.length > 0) {
      if (fitness <= board[twins[0]].result.fitness) return false;
      for (let k = twins.length - 1; k >= 0; k--) board.splice(twins[k], 1);
    }

    const niche = genomeNiche(genome);
    let nicheCount = 0;
    let nicheWorst = -1;
    for (let k = 0; k < board.length; k++) {
      if (board[k].niche === niche) {
        nicheCount++;
        nicheWorst = k;
      }
    }
    if (nicheCount >= NICHE_CAP) {
      // The niche is full: only replace its weakest member, and only when strictly better.
      if (fitness <= board[nicheWorst].result.fitness) return false;
      board.splice(nicheWorst, 1);
    }

    const entry: Entry = {
      niche,
      fingerprint,
      returns,
      result: {
        id: `S-${iteration}`,
        genome,
        fitness,
        full: metrics.full,
        inSample: metrics.inSample,
        outOfSample: metrics.outOfSample,
        alpha: metrics.full.roi - benchmark.full.roi,
        alphaOos: metrics.outOfSample.roi - benchmark.outOfSample.roi,
        equity,
        foundAt: iteration,
      },
    };
    let at = board.length;
    while (at > 0 && board[at - 1].result.fitness < fitness) at--;
    board.splice(at, 0, entry);
    if (board.length > LEADERBOARD_SIZE) board.length = LEADERBOARD_SIZE;
    boardDirty = true;
    return fitness > best + 1e-9;
  };

  const step = (): void => {
    iteration++;
    let genome = propose();
    let signature = genomeSignature(genome);
    // Skip genomes that were already evaluated (a few retries, then accept the duplicate).
    for (let attempt = 0; attempt < 4 && seen.has(signature); attempt++) {
      genome = board.length >= SEED_POPULATION && rng() < 0.7
        ? mutateGenome(rng, genome, mode, limits, 1)
        : randomGenome(rng, mode, limits);
      signature = genomeSignature(genome);
    }
    const duplicate = seen.has(signature);
    if (seen.size > 250_000) seen.clear();
    seen.add(signature);
    const improved = duplicate ? false : consider(genome);
    stagnation = improved ? 0 : stagnation + 1;
  };

  const finish = (reason: StopReason): void => {
    postBoard();
    scope.postMessage({ type: 'done', reason, ...snapshot() });
  };

  for (;;) {
    for (let k = 0; k < PROGRESS_EVERY && stagnation < maxStagnation; k++) step();

    const now = performance.now();
    const batchMs = now - lastBatchAt;
    lastBatchAt = now;
    if (batchMs > 0) {
      const rate = (PROGRESS_EVERY * 1000) / batchMs;
      itersPerSec = itersPerSec === 0 ? rate : itersPerSec * 0.9 + rate * 0.1;
    }

    // Streamed every PROGRESS_EVERY (25) iterations.
    scope.postMessage({ type: 'progress', ...snapshot() });
    if (boardDirty && now - lastBoardPost >= LEADERBOARD_INTERVAL_MS) postBoard();

    if (stagnation >= maxStagnation) {
      finish('converged');
      return;
    }

    await yieldToEventLoop();
    if (token !== activeRun) return; // superseded by a newer run
    if (stopRequested) {
      finish('stopped');
      return;
    }
  }
}
