/**
 * Optimisation engine – runs entirely inside a dedicated Web Worker.
 *
 * Indicator calculation, backtesting and the evolutionary search all happen
 * here, so the main thread stays responsive and nothing depends on (or is
 * limited by) a serverless function's execution timeout.
 *
 * Honest by construction
 * ----------------------
 * • The data is split Train 60% / Validation 20% / Holdout 20%. While the search
 *   runs, every backtest stops at the holdout boundary – the holdout is never
 *   simulated, scored or sent anywhere. It is evaluated once, for the final
 *   leaderboard, when the run ends.
 * • Fitness is built on the Deflated Sharpe Ratio of the validation segment, so
 *   the bar rises with every strategy that is tried.
 * • With a "Robust universe" the same rules are backtested on every asset and the
 *   median score counts.
 *
 * Search: evolutionary hill-climbing with mutation (seeds, mutation of the
 * champion or a rank-biased parent, crossover, random immigrants). It runs until
 * the UI sends `stop` or the configured number of consecutive iterations fails
 * to improve the top fitness score.
 */

import {
  assetBaseScore,
  BacktestWorkspace,
  barsPerYear,
  benchmarkEquity,
  benchmarkSearch,
  combineFitness,
  deflatedTerm,
  evaluateSearch,
  holdoutMetrics,
  runBacktest,
  searchMetrics,
  segmentMetrics,
  splitFor,
  UNFIT,
  type SearchMetrics,
  type Split,
} from '../engine/backtester';
import {
  countParameters,
  crossoverGenomes,
  genomeNiche,
  genomeSignature,
  mulberry32,
  mutateGenome,
  parameterVariants,
  randomGenome,
  repairGenome,
  rewardToRisk,
  type GenomeLimits,
  type Rng,
} from '../engine/genome';
import { IndicatorCache } from '../engine/indicators';
import { deflatedSharpe, RunningStats } from '../lib/stats';
import {
  DEFAULT_MAX_STAGNATION,
  LEADERBOARD_SIZE,
  MAX_MAX_STAGNATION,
  MIN_MAX_STAGNATION,
  PROGRESS_EVERY,
  type AssetResult,
  type BacktestSettings,
  type BenchmarkResult,
  type ChartAxis,
  type NamedDataset,
  type OHLCV,
  type OptimizerStartPayload,
  type PerformanceMetrics,
  type ProgressSnapshot,
  type RobustnessReport,
  type StopReason,
  type StrategyGenome,
  type StrategyResult,
  type TradeMode,
  type Verdict,
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

/** Minimum number of candles required for a meaningful three-way split. */
const MIN_BARS = 150;
/** Maximum number of points per equity curve sent to the UI. */
const MAX_CHART_POINTS = 1200;
/** Leaderboard snapshots are throttled so the UI is not flooded on fast datasets. */
const LEADERBOARD_INTERVAL_MS = 150;
/** Maximum number of leaderboard slots a single structural niche may occupy. */
const NICHE_CAP = 3;
/**
 * Two strategies whose sampled equity returns correlate at least this much trade
 * the same idea. Only the fitter twin is listed.
 */
const CLONE_CORRELATION = 0.9;
/** Pure random search until the leaderboard holds this many strategies. */
const SEED_POPULATION = 10;
/** Trades removed by the "top trades" robustness test. */
const TOP_TRADES_REMOVED = 5;
/** Total memory budget for memoised indicator series, shared by all assets. */
const INDICATOR_BUDGET_BYTES = 256 * 1024 * 1024;

let activeRun = 0;
let stopRequested = false;
let session: Session | null = null;

/**
 * Yield to the worker's event loop so queued `stop` / `analyze` messages can be handled.
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

const fail = (error: unknown): void =>
  scope.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });

scope.onmessage = (event) => {
  const message = event.data;
  if (message.type === 'stop') {
    stopRequested = true;
    return;
  }
  if (message.type === 'analyze') {
    try {
      if (!session) throw new Error('No optimisation run is available to analyse.');
      if (message.universe) addUniverse(session, message.universe);
      const report = analyze(session, message.genome);
      // The verdict needs the holdout, which only exists once the run has ended.
      const verdict = session.finished ? verdictFor(report, finalResult(session, message.strategyId, message.genome)) : null;
      scope.postMessage({ type: 'analysis', strategyId: message.strategyId, report, verdict });
    } catch (error) {
      fail(error);
    }
    return;
  }
  if (message.type === 'start') {
    activeRun++;
    stopRequested = false;
    const token = activeRun;
    run(message.payload, token).catch((error: unknown) => {
      if (token === activeRun) fail(error);
    });
  }
};

/* -------------------------------------------------------------------------- */
/*  Session state                                                             */
/* -------------------------------------------------------------------------- */

/** Everything needed to backtest one asset. */
interface Asset {
  symbol: string;
  data: OHLCV;
  ind: IndicatorCache;
  ws: BacktestWorkspace;
  split: Split;
  periodsPerYear: number;
  /** Buy & Hold over the search period. */
  benchmark: SearchMetrics & { equity: Float64Array };
}

/** Per-asset ingredients of a fitness score that do not depend on the number of trials. */
interface AssetScore {
  base: number;
  metrics: SearchMetrics;
}

interface Entry {
  result: StrategyResult;
  niche: string;
  /** Outcome fingerprint – strategies that produce identical results are treated as duplicates. */
  fingerprint: string;
  /** Sampled equity returns, used to detect behavioural clones. */
  returns: Float32Array;
  scores: AssetScore[];
}

interface Session {
  /** The primary asset first. More than one asset = Robust-universe scoring. */
  assets: Asset[];
  /** Additional assets used only by the cross-asset robustness test. */
  extra: Asset[];
  settings: BacktestSettings;
  mode: TradeMode;
  limits: GenomeLimits;
  axis: ChartAxis;
  /** Validation Sharpe (per bar, primary asset) of every strategy evaluated so far. */
  trials: RunningStats;
  /** Frozen copy of (trials, variance) used for scoring; refreshed between batches. */
  hurdleTrials: number;
  hurdleVariance: number;
  board: Entry[];
  finished: boolean;
  /** Final results by id, including the holdout (only after the run has ended). */
  finals: Map<string, StrategyResult>;
}

function validate(dataset: NamedDataset): void {
  const { data, symbol } = dataset;
  const n = data.c.length;
  if (n < MIN_BARS) {
    throw new Error(`Not enough history for ${symbol}: ${n} candles loaded, at least ${MIN_BARS} are required.`);
  }
  for (const key of ['t', 'o', 'h', 'l', 'v'] as const) {
    if (data[key].length !== n) throw new Error(`Malformed market data for ${symbol}: OHLCV columns have different lengths.`);
  }
  for (let i = 0; i < n; i++) {
    if (!(data.o[i] > 0) || !(data.h[i] > 0) || !(data.l[i] > 0) || !(data.c[i] > 0)) {
      throw new Error(`Malformed market data for ${symbol}: non-positive or missing price at candle ${i}.`);
    }
  }
}

function buildAsset(dataset: NamedDataset, settings: BacktestSettings, budgetBytes: number): Asset {
  validate(dataset);
  const { data, symbol } = dataset;
  const split = splitFor(data.c.length);
  const periodsPerYear = barsPerYear(data.t);
  return {
    symbol,
    data,
    ind: new IndicatorCache(data, budgetBytes),
    ws: new BacktestWorkspace(data.c.length),
    split,
    periodsPerYear,
    benchmark: benchmarkSearch(data, settings, split, periodsPerYear),
  };
}

function addUniverse(s: Session, universe: NamedDataset[]): void {
  const known = new Set([...s.assets, ...s.extra].map((asset) => asset.symbol));
  const fresh = universe.filter((dataset) => !known.has(dataset.symbol));
  const budget = INDICATOR_BUDGET_BYTES / Math.max(1, s.assets.length + s.extra.length + fresh.length);
  for (const dataset of fresh) s.extra.push(buildAsset(dataset, s.settings, budget));
}

/** Evenly spaced sample of bar indices that always contains bar 0, both segment boundaries and the last bar. */
function buildAxis(t: Float64Array, split: Split): ChartAxis {
  const n = t.length;
  const stride = Math.max(1, Math.ceil(n / MAX_CHART_POINTS));
  const picked = new Set<number>();
  for (let i = 0; i < n; i += stride) picked.add(i);
  picked.add(split.validationStart);
  picked.add(split.holdoutStart - 1);
  picked.add(split.holdoutStart);
  picked.add(n - 1);
  const barIndex = Int32Array.from([...picked].sort((a, b) => a - b));
  const time = new Float64Array(barIndex.length);
  let validationPos = 0;
  let holdoutPos = 0;
  for (let k = 0; k < barIndex.length; k++) {
    time[k] = t[barIndex[k]];
    if (barIndex[k] === split.validationStart) validationPos = k;
    if (barIndex[k] === split.holdoutStart) holdoutPos = k;
  }
  return {
    barIndex,
    time,
    validationPos,
    holdoutPos,
    bars: n,
    validationIndex: split.validationStart,
    holdoutIndex: split.holdoutStart,
  };
}

/** Sample an equity curve on the chart axis. `points` limits it to the search period while a run is live. */
function sampleCurve(equity: Float64Array, axis: ChartAxis, points: number = axis.barIndex.length): Float32Array {
  const out = new Float32Array(points);
  for (let k = 0; k < points; k++) out[k] = equity[axis.barIndex[k]];
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
  const n = Math.min(a.length, b.length);
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

function deriveLimits(primary: Asset): GenomeLimits {
  const volume = primary.data.v;
  let withVolume = 0;
  for (let i = 0; i < volume.length; i++) if (volume[i] > 0) withVolume++;
  return {
    // Lookbacks are capped so indicators are warmed up for most of the training segment.
    maxLookback: Math.max(30, Math.min(200, Math.floor(primary.split.validationStart / 3))),
    hasVolume: withVolume > volume.length * 0.5,
  };
}

/** Coherent textbook strategies that give the search a sensible starting population. */
function seedGenomes(mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome[] {
  const dynamic = (stop: number, target: number | null, trailing: boolean, reversalExit = false): StrategyGenome['exit'] => ({
    mode: 'dynamic',
    atrPeriod: 14,
    stopAtrMult: stop,
    trailing,
    takeProfitAtrMult: target,
    reversalExit,
    longExit: null,
    shortExit: null,
  });
  const seeds: StrategyGenome[] = [
    {
      // Supertrend trend-following with a trailing ATR stop.
      archetype: 'TREND',
      longEntry: { logic: 'AND', conditions: [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }] },
      shortEntry: null,
      exit: dynamic(3, null, true),
      maxBarsInTrade: null,
    },
    {
      // Moving-average trend filtered by trend strength, closed when the averages cross back.
      archetype: 'TREND',
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'EMA', slowPeriod: 50, op: 'gt' },
          { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 },
        ],
      },
      shortEntry: null,
      exit: dynamic(2.5, null, true, true),
      maxBarsInTrade: null,
    },
    {
      // MACD signal-line crossover with a fixed ATR bracket.
      archetype: 'TREND',
      longEntry: { logic: 'AND', conditions: [{ ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'signal', op: 'cross_above' }] },
      shortEntry: null,
      exit: dynamic(2, 4, false),
      maxBarsInTrade: null,
    },
    {
      // Buy the dip inside a long-term uptrend.
      archetype: 'MEAN_REVERSION',
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'RSI', period: 14, op: 'lt', level: 30 },
          { ind: 'PRICE_MA', maType: 'SMA', period: 200, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 5, takeProfitPct: 10 },
      maxBarsInTrade: null,
    },
    {
      // Fade Supertrend flips that go against the 200-bar trend.
      archetype: 'MEAN_REVERSION',
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'SUPERTREND', period: 10, mult: 3, op: 'cross_below' },
          { ind: 'PRICE_MA', maType: 'SMA', period: 200, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: dynamic(2.5, 3, false),
      maxBarsInTrade: 20,
    },
    {
      // Bollinger-band mean reversion.
      archetype: 'MEAN_REVERSION',
      longEntry: { logic: 'AND', conditions: [{ ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'cross_above', level: 0 }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 4, takeProfitPct: 7.5 },
      maxBarsInTrade: null,
    },
    {
      // Oversold stochastic on a volume spike.
      archetype: 'MEAN_REVERSION',
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'lt', level: 20 },
          { ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 },
        ],
      },
      shortEntry: null,
      exit: dynamic(2, 3, false),
      maxBarsInTrade: null,
    },
  ];
  return seeds.map((seed) => repairGenome(seed, mode, limits, rng));
}

/* -------------------------------------------------------------------------- */
/*  Scoring                                                                   */
/* -------------------------------------------------------------------------- */

/** Backtest `genome` on the search period of every asset. The primary asset is run last, so its workspace stays loaded. */
function scoreAssets(s: Session, genome: StrategyGenome): AssetScore[] {
  const scores = new Array<AssetScore>(s.assets.length);
  for (let a = s.assets.length - 1; a >= 0; a--) {
    const asset = s.assets[a];
    const metrics = evaluateSearch(asset.ind, genome, s.settings, asset.ws, asset.split, asset.periodsPerYear);
    scores[a] = { base: assetBaseScore(metrics, asset.benchmark), metrics };
  }
  return scores;
}

/** Fitness of a scored genome against the current selection-bias hurdle. */
function fitnessOf(s: Session, genome: StrategyGenome, scores: AssetScore[]): { fitness: number; dsr: number } {
  let searchTrades = 0;
  let validationTrades = 0;
  let profitable = 0;
  const assetScores = scores.map((score) => {
    searchTrades += score.metrics.search.trades;
    validationTrades += score.metrics.validation.trades;
    if (score.metrics.validation.roi > 0) profitable++;
    return score.base + deflatedTerm(deflatedSharpe(score.metrics.validation, s.hurdleTrials, s.hurdleVariance));
  });
  const fitness = combineFitness({
    assetScores,
    searchTrades,
    validationTrades,
    profitableShare: profitable / scores.length,
    params: countParameters(genome, s.limits),
    rewardToRisk: rewardToRisk(genome),
    validationWinRate: scores[0].metrics.validation.winRate,
  });
  const dsr = deflatedSharpe(scores[0].metrics.validation, s.hurdleTrials, s.hurdleVariance).probability;
  return { fitness, dsr };
}

/** Re-score the leaderboard against the latest hurdle (it rises as more strategies are tried). */
function rescoreBoard(s: Session): void {
  s.hurdleTrials = s.trials.count;
  s.hurdleVariance = s.trials.variance;
  for (const entry of s.board) {
    const { fitness, dsr } = fitnessOf(s, entry.result.genome, entry.scores);
    entry.result.fitness = fitness;
    entry.result.dsr = dsr;
  }
  s.board.sort((a, b) => b.result.fitness - a.result.fitness);
}

function searchAssetResults(s: Session, scores: AssetScore[]): AssetResult[] {
  return scores.map((score, a) => ({ symbol: s.assets[a].symbol, ...score.metrics, holdout: null }));
}

/* -------------------------------------------------------------------------- */
/*  Robustness analysis                                                       */
/* -------------------------------------------------------------------------- */

const medianOf = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Robustness tests on the search period only (the holdout is not touched):
 * ±20% parameter sensitivity, removal of the best trades, and the same rules on other assets.
 */
function analyze(s: Session, genome: StrategyGenome): RobustnessReport {
  const baseScores = scoreAssets(s, genome);
  const base = fitnessOf(s, genome, baseScores);
  const baseRoi = baseScores[0].metrics.search.roi;

  // Top-trade removal – read straight from the primary workspace left by `scoreAssets`.
  const ws = s.assets[0].ws;
  const pnls = Array.from(ws.tradePnl.subarray(0, ws.tradeCount)).sort((x, y) => y - x);
  const removed = Math.min(TOP_TRADES_REMOVED, pnls.length);
  let netPnl = 0;
  for (const pnl of pnls) netPnl += pnl;
  let top = 0;
  for (let k = 0; k < removed; k++) top += pnls[k];

  // Parameter sensitivity.
  const rois: number[] = [];
  const fitnesses: number[] = [];
  for (const variant of parameterVariants(genome, s.mode, s.limits)) {
    const scores = scoreAssets(s, variant);
    rois.push(scores[0].metrics.search.roi);
    fitnesses.push(Math.max(UNFIT, fitnessOf(s, variant, scores).fitness));
  }
  const worstRoi = rois.length > 0 ? Math.min(...rois) : baseRoi;
  const worstFitness = fitnesses.length > 0 ? Math.min(...fitnesses) : base.fitness;

  // Cross-asset: the exact rules, untuned, on every other asset we have.
  const others = [...s.assets, ...s.extra];
  let crossAsset: RobustnessReport['crossAsset'] = null;
  if (others.length > 1) {
    const rows = others.map((asset) => {
      const m = evaluateSearch(asset.ind, genome, s.settings, asset.ws, asset.split, asset.periodsPerYear).search;
      return { symbol: asset.symbol, roi: m.roi, sharpe: m.sharpe, trades: m.trades, profitable: m.roi > 0 };
    });
    crossAsset = { rows, share: rows.filter((row) => row.profitable).length / rows.length };
  }

  return {
    sensitivity: {
      variants: rois.length,
      baseRoi,
      medianRoi: rois.length > 0 ? medianOf(rois) : baseRoi,
      worstRoi,
      baseFitness: base.fitness,
      medianFitness: fitnesses.length > 0 ? medianOf(fitnesses) : base.fitness,
      worstFitness,
      // Fragile: a 20% nudge of one parameter costs more than half of the return (or turns it negative).
      fragile: baseRoi > 0 ? worstRoi < 0.5 * baseRoi : true,
    },
    topTrades: { netPnl, netPnlWithout: netPnl - top, removed, profitable: netPnl - top > 0 },
    crossAsset,
  };
}

/**
 * Overall verdict.
 *   Overfit – it lost money on the holdout, or its deflated Sharpe says "probably luck" (< 0.5).
 *   Fragile – it survives the holdout but fails a robustness test or its deflated Sharpe is below 0.95.
 *   Robust  – everything holds.
 */
function verdictFor(report: RobustnessReport, result: StrategyResult): Verdict {
  if ((result.holdout !== null && result.holdout.roi <= 0) || result.dsr < 0.5) return 'Overfit';
  const crossOk = report.crossAsset === null || report.crossAsset.share >= 2 / 3 - 1e-9;
  if (report.sensitivity.fragile || !report.topTrades.profitable || !crossOk || result.dsr < 0.95) return 'Fragile';
  return 'Robust';
}

/* -------------------------------------------------------------------------- */
/*  Holdout reveal                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Evaluate a strategy on the WHOLE dataset, holdout included. Must only be
 * called after the search has ended.
 */
function revealHoldout(s: Session, base: StrategyResult): StrategyResult {
  if (!s.finished) throw new Error('The holdout cannot be evaluated while the search is running.');
  const assets: AssetResult[] = [];
  let equity = base.equity;
  for (let a = s.assets.length - 1; a >= 0; a--) {
    const asset = s.assets[a];
    runBacktest(asset.ind, base.genome, s.settings, asset.ws);
    const m = searchMetrics(asset.ws.equity, asset.split, s.settings.initialCapital, asset.periodsPerYear, asset.ws);
    assets[a] = {
      symbol: asset.symbol,
      ...m,
      holdout: holdoutMetrics(asset.ws.equity, asset.split, asset.periodsPerYear, asset.ws),
    };
    if (a === 0) equity = sampleCurve(asset.ws.equity, s.axis);
  }
  const primary = assets[0];
  const result: StrategyResult = {
    ...base,
    search: primary.search,
    train: primary.train,
    validation: primary.validation,
    holdout: primary.holdout,
    alpha: primary.search.roi - s.assets[0].benchmark.search.roi,
    equity,
    assets,
  };
  const robustness = analyze(s, base.genome);
  return { ...result, robustness, verdict: verdictFor(robustness, result) };
}

/** Final (holdout-revealed) result for a strategy, computing it when it is not on the final leaderboard. */
function finalResult(s: Session, id: string, genome: StrategyGenome): StrategyResult {
  const known = s.finals.get(id);
  if (known) return known;
  const scores = scoreAssets(s, genome);
  const { fitness, dsr } = fitnessOf(s, genome, scores);
  const primary = scores[0].metrics;
  const result = revealHoldout(s, {
    id,
    genome,
    fitness,
    dsr,
    params: countParameters(genome, s.limits),
    ...primary,
    holdout: null,
    alpha: primary.search.roi - s.assets[0].benchmark.search.roi,
    equity: new Float32Array(0),
    foundAt: 0,
    assets: [],
    robustness: null,
    verdict: null,
  });
  s.finals.set(id, result);
  return result;
}

function finalBenchmark(s: Session): BenchmarkResult {
  const primary = s.assets[0];
  const equity = benchmarkEquity(primary.data, s.settings);
  const { search, train, validation } = primary.benchmark;
  const holdout: PerformanceMetrics = {
    ...segmentMetrics(equity, primary.split.holdoutStart, primary.split.n, equity[primary.split.holdoutStart - 1], primary.periodsPerYear, null, null),
    exposure: 100,
  };
  return { search, train, validation, holdout, equity: sampleCurve(equity, s.axis) };
}

/* -------------------------------------------------------------------------- */
/*  Optimisation run                                                          */
/* -------------------------------------------------------------------------- */

async function run(payload: OptimizerStartPayload, token: number): Promise<void> {
  const { settings } = payload;
  if (payload.datasets.length === 0) throw new Error('No market data was supplied.');
  session = null;

  const budget = INDICATOR_BUDGET_BYTES / payload.datasets.length;
  const assets = payload.datasets.map((dataset) => buildAsset(dataset, settings, budget));
  const primary = assets[0];
  const axis = buildAxis(primary.data.t, primary.split);
  const s: Session = {
    assets,
    extra: [],
    settings,
    mode: settings.tradeMode,
    limits: deriveLimits(primary),
    axis,
    trials: new RunningStats(),
    hurdleTrials: 0,
    hurdleVariance: 0,
    board: [],
    finished: false,
    finals: new Map(),
  };
  session = s;
  const { mode, limits, board } = s;

  const requested = Math.round(payload.maxStagnation ?? DEFAULT_MAX_STAGNATION);
  const maxStagnation = Number.isFinite(requested)
    ? Math.max(MIN_MAX_STAGNATION, Math.min(MAX_MAX_STAGNATION, requested))
    : DEFAULT_MAX_STAGNATION;
  const rng = mulberry32(payload.seed ?? (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0);

  // The live benchmark stops at the holdout boundary, exactly like the strategy curves.
  scope.postMessage({
    type: 'ready',
    benchmark: {
      search: primary.benchmark.search,
      train: primary.benchmark.train,
      validation: primary.benchmark.validation,
      holdout: null,
      equity: sampleCurve(primary.benchmark.equity, axis, axis.holdoutPos),
    },
    axis,
    barsPerYear: primary.periodsPerYear,
  });

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
    bestRoi: board.length > 0 ? board[0].result.search.roi : null,
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
    const scores = scoreAssets(s, genome);
    const primaryMetrics = scores[0].metrics;
    // Every evaluated strategy is a trial, whether or not it makes the leaderboard.
    s.trials.push(primaryMetrics.validation.sharpeRaw);

    const { fitness, dsr } = fitnessOf(s, genome, scores);
    if (fitness <= UNFIT) return false;
    const best = board.length > 0 ? board[0].result.fitness : -Infinity;
    if (board.length >= LEADERBOARD_SIZE && fitness <= board[board.length - 1].result.fitness) return false;

    const { search, validation } = primaryMetrics;
    const fingerprint = `${search.trades}|${search.netPnl.toFixed(2)}|${search.maxDrawdown.toFixed(4)}|${validation.netPnl.toFixed(2)}`;
    if (board.some((e) => e.fingerprint === fingerprint)) return false;

    // Behavioural clones: a fitter twin already on the board wins; otherwise this one replaces its twins.
    // The curve stops at the holdout boundary – nothing beyond it has been simulated.
    const equity = sampleCurve(primary.ws.equity, axis, axis.holdoutPos);
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
      scores,
      result: {
        id: `S-${iteration}`,
        genome,
        fitness,
        dsr,
        params: countParameters(genome, limits),
        search: primaryMetrics.search,
        train: primaryMetrics.train,
        validation: primaryMetrics.validation,
        holdout: null,
        alpha: search.roi - primary.benchmark.search.roi,
        equity,
        foundAt: iteration,
        assets: searchAssetResults(s, scores),
        robustness: null,
        verdict: null,
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
      genome =
        board.length >= SEED_POPULATION && rng() < 0.7 ? mutateGenome(rng, genome, mode, limits, 1) : randomGenome(rng, mode, limits);
      signature = genomeSignature(genome);
    }
    const duplicate = seen.has(signature);
    if (seen.size > 250_000) seen.clear();
    seen.add(signature);
    const improved = duplicate ? false : consider(genome);
    stagnation = improved ? 0 : stagnation + 1;
  };

  const finish = (reason: StopReason): void => {
    // The search is over: only now is the holdout simulated, for the final leaderboard.
    rescoreBoard(s);
    s.finished = true;
    const strategies = board.map((entry) => {
      const final = revealHoldout(s, entry.result);
      s.finals.set(final.id, final);
      return final;
    });
    scope.postMessage({ type: 'done', reason, strategies, benchmark: finalBenchmark(s), ...snapshot() });
  };

  for (;;) {
    // The selection-bias hurdle is frozen for the batch so every comparison inside it is like for like.
    rescoreBoard(s);
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
