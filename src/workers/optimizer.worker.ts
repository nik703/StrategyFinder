/**
 * Optimisation engine – runs entirely inside a dedicated Web Worker.
 *
 * Indicator calculation, backtesting and the evolutionary search all happen
 * here, so the main thread stays responsive and nothing depends on (or is
 * limited by) a serverless function's execution timeout.
 *
 * Honest by construction
 * ----------------------
 * • The data is split Train 60% / Validation 20% / Holdout 20%. Every backtest of
 *   the search stops at the holdout boundary, and the holdout stays sealed when the
 *   run ends. The user then locks ONE strategy of the final leaderboard and only
 *   that strategy's holdout is simulated and revealed – once per run. Revealing
 *   the whole board would turn the holdout into one more selection set.
 * • Strategies with too few trades are disqualified outright (see the gates in
 *   `backtester.ts`).
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
  benchmarkEquity,
  evaluateSearch,
  holdoutMetrics,
  holdoutShortfalls,
  median,
  MIN_TRADES,
  MIN_TRADES_PER_YEAR,
  MIN_VALIDATION_TRADES,
  runBacktest,
  segmentMetrics,
  UNFIT,
} from '../engine/backtester';
import {
  buildAsset,
  createContext,
  evaluateCandidate,
  fitnessOf,
  rankFitness,
  sampleCurve,
  scoreAssets,
  scoreNeighbours,
  tradesPerYear,
  type Asset,
  type AssetScore,
  type CandidateResult,
  type EvalContext,
  type EvaluatorBatch,
  type EvaluatorReply,
  type NeighbourScore,
  type PortLike,
} from '../engine/evaluation';
import {
  countParameters,
  crossoverGenomes,
  genomeNiche,
  genomeSignature,
  HYPOTHESES,
  hypothesisUnsupported,
  mulberry32,
  mutateGenome,
  parameterAxes,
  parameterVariants,
  randomGenome,
  repairGenome,
  withParameterValues,
  type GenomeLimits,
  type Rng,
} from '../engine/genome';
import { expectedMaxSharpe, haircutSharpe, RunningStats } from '../lib/stats';
import {
  DEFAULT_RUN_EVALUATIONS,
  DSR_PATIENCE,
  EXHAUSTION_LIMIT,
  HYPOTHESIS_IDS,
  LEADERBOARD_SIZE,
  MIN_RUN_EVALUATIONS,
  PROGRESS_EVERY,
  WINDOW_TRIAL_BUDGET,
  type Condition,
  type Hypothesis,
  type StabilityGrid,
  type AssetResult,
  type BenchmarkResult,
  type NamedDataset,
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

/** Leaderboard snapshots are throttled so the UI is not flooded on fast datasets. */
const LEADERBOARD_INTERVAL_MS = 150;
/** Maximum number of leaderboard slots a single structural niche may occupy. */
const NICHE_CAP = 3;
/**
 * Two strategies whose sampled equity returns correlate at least this much trade
 * the same idea. Only the fitter twin is listed.
 */
const CLONE_CORRELATION = 0.9;
/**
 * With an evaluator pool, each evaluator scores this many blocks of 25 candidates per round.
 * Bigger rounds spend less time on messaging. On seeded comparisons, rounds of 300 found
 * strategies at least as good as rounds of 25, so the larger rounds cost no search quality.
 */
const ROUNDS_PER_EVALUATOR = 4;
/** Pure random search until the leaderboard holds this many strategies. */
const SEED_POPULATION = 10;
/** A candidate must beat the board's best deflated Sharpe by this much to reset the early-stopping clock. */
const DSR_MIN_GAIN = 0.001;
/** The stability map spans this many grid steps either side of the chosen value. */
const STABILITY_RADIUS = 2;
/** Trades removed by the "top trades" robustness test. */
const TOP_TRADES_REMOVED = 5;
/** Total memory budget for memoised indicator series, shared by all assets. */
const INDICATOR_BUDGET_BYTES = 256 * 1024 * 1024;

let activeRun = 0;
let stopRequested = false;
let session: Session | null = null;
/** The live run's evaluator pool, so a `pool-failed` message can release whatever it is waiting on. */
let activePool: EvaluatorPool | null = null;

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
  if (message.type === 'pool-failed') {
    // Rejects any batch in flight; the run catches that and continues single-threaded.
    activePool?.close();
    return;
  }
  if (message.type === 'analyze') {
    try {
      if (!session) throw new Error('No optimisation run is available to analyse.');
      if (message.universe) addUniverse(session, message.universe);
      const report = analyze(session, message.genome);
      // A verdict needs the holdout, and only the locked strategy's holdout is ever revealed.
      const locked = session.locked !== null && session.locked.id === message.strategyId ? session.finals.get(message.strategyId) : undefined;
      const verdict = locked ? judge(session, report, locked).verdict : null;
      scope.postMessage({ type: 'analysis', strategyId: message.strategyId, report, verdict });
    } catch (error) {
      fail(error);
    }
    return;
  }
  if (message.type === 'lock') {
    try {
      if (!session) throw new Error('No optimisation run is available.');
      const strategy = lockStrategy(session, message.strategyId, message.priorReveals, message.simulated);
      scope.postMessage({ type: 'locked', strategy, benchmark: finalBenchmark(session) });
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

interface Entry {
  result: StrategyResult;
  niche: string;
  /** Outcome fingerprint – strategies that produce identical results are treated as duplicates. */
  fingerprint: string;
  /** Sampled equity returns, used to detect behavioural clones. */
  returns: Float32Array;
  scores: AssetScore[];
  /** One-grid-step parameter neighbours, kept so the neighbourhood cap is re-applied whenever the hurdle moves. */
  neighbours: NeighbourScore[];
}

/** A run's state. The scoring context (`assets` = the universe, primary first) is shared with the evaluator pool. */
interface Session extends EvalContext {
  /** Additional assets used only by the cross-asset robustness test. */
  extra: Asset[];
  mode: TradeMode;
  /** The market hypothesis every strategy of this run is built from. */
  hypothesis: Hypothesis;
  /**
   * Validation Sharpe (per bar, primary asset) of every trial on this data window: the trials of
   * earlier runs (from the trial ledger) plus this run's.
   */
  trials: RunningStats;
  /** This run's own trials, as annualised Sharpe ratios (persisted to the trial ledger by the UI). */
  runTrials: RunningStats;
  /** Frozen copy of (trials, variance) used for scoring; refreshed between batches. */
  hurdleTrials: number;
  hurdleVariance: number;
  board: Entry[];
  finished: boolean;
  /** The one strategy whose holdout was revealed, and what was known about that holdout at the time. */
  locked: { id: string; priorReveals: number; simulated: boolean } | null;
  /** Final (holdout-revealed) results by id – only ever holds the locked strategy. */
  finals: Map<string, StrategyResult>;
  /** Candidates disqualified by the minimum-trade gates. */
  undersampled: number;
  /** Candidates dropped because their genome or trade log repeated an earlier one. */
  duplicates: number;
  /** Candidates that never traded. */
  inactive: number;
}

function addUniverse(s: Session, universe: NamedDataset[]): void {
  const known = new Set([...s.assets, ...s.extra].map((asset) => asset.symbol));
  const fresh = universe.filter((dataset) => !known.has(dataset.symbol));
  const budget = INDICATOR_BUDGET_BYTES / Math.max(1, s.assets.length + s.extra.length + fresh.length);
  for (const dataset of fresh) s.extra.push(buildAsset(dataset, s.settings, budget));
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

/** Textbook versions of the run's hypothesis: a sensible starting population that stays inside it. */
function seedGenomes(h: Hypothesis, mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome[] {
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
  const archetype = HYPOTHESES[h].archetype;
  const seed = (conditions: Condition[], exit: StrategyGenome['exit'], maxBarsInTrade: number | null = null): StrategyGenome => ({
    archetype,
    hypothesis: h,
    longEntry: { logic: 'AND', conditions },
    shortEntry: null,
    exit,
    maxBarsInTrade,
  });
  const adx = (op: 'gt' | 'lt', level: number): Condition => ({ ind: 'ADX', period: 14, variant: 'strength', op, level });
  const rsi = (period: number, op: 'lt' | 'cross_above', level: number): Condition => ({ ind: 'RSI', period, op, level });
  const slowTrend: Condition = { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' };
  let seeds: StrategyGenome[];
  switch (h) {
    case 'TREND_FOLLOWING':
      seeds = [
        // Long-term trend with a trailing ATR stop.
        seed([{ ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' }], dynamic(3, null, true)),
        // Medium-term trend entry when trend strength is present, closed when the trend breaks.
        seed([{ ind: 'PRICE_MA', maType: 'EMA', period: 56, op: 'cross_above' }, adx('gt', 20)], dynamic(2.5, null, true, true)),
        // Short-term trend with a fixed ATR bracket.
        seed([{ ind: 'PRICE_MA', maType: 'EMA', period: 20, op: 'cross_above' }], dynamic(2, 4, false)),
      ];
      break;
    case 'VOL_BREAKOUT':
      seeds = [
        seed([{ ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'cross_above', level: 0 }], dynamic(2, 4, false)),
        seed(
          [
            { ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'cross_above', level: 0 },
            { ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 },
          ],
          dynamic(2.5, null, true),
        ),
        seed([{ ind: 'BB', period: 40, mult: 2.5, variant: 'upper', op: 'cross_above', level: 0 }], dynamic(3, 6, false), 20),
      ];
      break;
    case 'DIP_IN_UPTREND':
      seeds = [
        seed([rsi(14, 'lt', 30), slowTrend], { mode: 'static', stopLossPct: 5, takeProfitPct: 10 }),
        seed([rsi(7, 'cross_above', 20), { ...slowTrend, period: 160 }], dynamic(2, 3, false), 10),
        seed([rsi(5, 'lt', 15), slowTrend], { mode: 'static', stopLossPct: 4, takeProfitPct: 7.5 }, 5),
      ];
      break;
    case 'RANGE_REVERSION':
      seeds = [
        seed([rsi(14, 'lt', 30)], { mode: 'static', stopLossPct: 4, takeProfitPct: 7.5 }),
        seed([rsi(14, 'lt', 30), adx('lt', 20)], dynamic(2, 3, false), 20),
        seed([rsi(7, 'cross_above', 25), adx('lt', 25)], { mode: 'static', stopLossPct: 3, takeProfitPct: 5 }, 10),
      ];
      break;
  }
  return seeds.map((g) => repairGenome(g, mode, limits, rng));
}

/* -------------------------------------------------------------------------- */
/*  Scoring                                                                   */
/* -------------------------------------------------------------------------- */

/** Re-score the leaderboard against the latest hurdle (it rises as more strategies are tried). */
function rescoreBoard(s: Session): void {
  s.hurdleTrials = s.trials.count;
  s.hurdleVariance = s.trials.variance;
  s.hurdle = expectedMaxSharpe(s.hurdleTrials, s.hurdleVariance);
  for (const entry of s.board) {
    const { fitness, point, neighbourMedian, dsr } = rankFitness(s, entry.result.genome, entry.scores, entry.neighbours);
    entry.result.fitness = fitness;
    entry.result.pointFitness = point;
    entry.result.neighbourMedian = neighbourMedian;
    entry.result.dsr = dsr;
    const haircut = haircutSharpe(entry.scores[0].metrics.validation, s.hurdleTrials, s.assets[0].periodsPerYear);
    entry.result.haircutSharpe = haircut.sharpe;
    entry.result.haircut = haircut.haircut;
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
  const baseScores = scoreAssets(s, genome).scores;
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
    const { scores } = scoreAssets(s, variant);
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
    stability: stabilityGrid(s, genome),
    crossAsset,
  };
}

/**
 * Parameter-stability map of the primary asset. The two axes are the parameters whose one-grid-step
 * neighbours move the search-period Sharpe the most (the Stage 2 neighbourhood, read per parameter);
 * the map spans `STABILITY_RADIUS` grid steps either side of the chosen values.
 */
function stabilityGrid(s: Session, genome: StrategyGenome): StabilityGrid | null {
  const axes = parameterAxes(genome, s.limits);
  if (axes.length === 0) return null;
  const primary = s.assets[0];
  const measure = (g: StrategyGenome): { sharpe: number; undersampled: boolean } => {
    const m = evaluateSearch(primary.ind, g, s.settings, primary.ws, primary.split, primary.periodsPerYear);
    return {
      sharpe: m.search.sharpe,
      undersampled:
        m.search.trades < MIN_TRADES || m.validation.trades < MIN_VALIDATION_TRADES || tradesPerYear(s, 0, m) < MIN_TRADES_PER_YEAR,
    };
  };
  const base = measure(genome).sharpe;
  // Sensitivity of each parameter: the largest Sharpe change one grid step away.
  const sensitivity = axes.map((axis, index) => {
    const at = axis.grid.indexOf(axis.value);
    let worst = 0;
    for (const step of [-1, 1]) {
      const value = axis.grid[at + step];
      if (value === undefined) continue;
      const neighbour = withParameterValues(genome, s.mode, s.limits, [[index, value]]);
      if (neighbour) worst = Math.max(worst, Math.abs(measure(neighbour).sharpe - base));
    }
    return { index, worst };
  });
  const ranked = sensitivity.filter((x) => axes[x.index].grid.length > 1).sort((a, b) => b.worst - a.worst || a.index - b.index);
  if (ranked.length === 0) return null;
  const window = (index: number): number[] => {
    const { grid, value } = axes[index];
    const at = grid.indexOf(value);
    return grid.slice(Math.max(0, at - STABILITY_RADIUS), Math.min(grid.length, at + STABILITY_RADIUS + 1));
  };
  const xi = ranked[0].index;
  const yi = ranked.length > 1 ? ranked[1].index : null;
  const xValues = window(xi);
  const yValues = yi === null ? [Number.NaN] : window(yi);
  const cells: (number | null)[][] = [];
  const undersampled: boolean[][] = [];
  for (const y of yValues) {
    const row: (number | null)[] = [];
    const flags: boolean[] = [];
    for (const x of xValues) {
      const values: [number, number][] = yi === null ? [[xi, x]] : [[xi, x], [yi, y]];
      const variant = withParameterValues(genome, s.mode, s.limits, values);
      if (variant === null) {
        row.push(null);
        flags.push(false);
      } else {
        const m = measure(variant);
        row.push(m.sharpe);
        flags.push(m.undersampled);
      }
    }
    cells.push(row);
    undersampled.push(flags);
  }
  return {
    xLabel: axes[xi].label,
    yLabel: yi === null ? null : axes[yi].label,
    xValues,
    yValues: yi === null ? [] : yValues,
    cells,
    undersampled,
    baseX: axes[xi].value,
    baseY: yi === null ? Number.NaN : axes[yi].value,
  };
}

/**
 * Verdict of the locked strategy, with the reasons it falls short of Robust.
 *
 *   Overfit – deflated Sharpe under 50% ("probably luck"), or the holdout lost money or had a
 *             non-positive Sharpe.
 *   Robust  – ALL of the following hold:
 *             • deflated Sharpe of the validation segment ≥ 95%;
 *             • holdout Sharpe 95% confidence interval entirely above zero;
 *             • holdout Sharpe above Buy & Hold's holdout Sharpe (risk-adjusted, not raw return);
 *             • at least `MIN_HOLDOUT_TRADES` holdout trades;
 *             • ±20% parameter sensitivity, top-trade removal and cross-asset tests all pass;
 *             • the holdout period had never been revealed before (untouched data);
 *             • the run used live market data, not the simulated fallback.
 *   Fragile – anything in between.
 */
function judge(s: Session, report: RobustnessReport, result: StrategyResult): { verdict: Verdict; notes: string[] } {
  const holdout = result.holdout;
  if (holdout === null) throw new Error('A verdict needs the revealed holdout.');
  const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
  const fixed = (x: number): string => x.toFixed(2);

  const overfit: string[] = [];
  if (result.dsr < 0.5) overfit.push(`Deflated Sharpe ${pct(result.dsr)} is under 50%: the validation result is most likely luck.`);
  if (holdout.roi <= 0) overfit.push(`Lost money on the holdout (${holdout.roi.toFixed(1)}%).`);
  else if (holdout.sharpe <= 0) overfit.push(`Holdout Sharpe ${fixed(holdout.sharpe)} is not positive.`);
  if (overfit.length > 0) return { verdict: 'Overfit', notes: overfit };

  const notes: string[] = [];
  if (result.dsr < 0.95) notes.push(`Deflated Sharpe ${pct(result.dsr)} is under 95%: not distinguishable from the luckiest of the strategies tried.`);
  notes.push(...holdoutShortfalls(holdout, finalBenchmark(s).holdout, s.assets[0].periodsPerYear));
  if (report.sensitivity.fragile) notes.push('A ±20% change of one parameter costs more than half of the return.');
  if (!report.topTrades.profitable) notes.push(`Unprofitable without its ${report.topTrades.removed} best trades.`);
  if (report.crossAsset !== null && report.crossAsset.share < 2 / 3 - 1e-9) {
    notes.push(`Profitable on only ${Math.round(report.crossAsset.share * 100)}% of the other assets (two thirds needed).`);
  }
  const lock = s.locked;
  if (lock !== null && lock.priorReveals > 0) {
    notes.push(`This holdout period had already been revealed ${lock.priorReveals} time${lock.priorReveals === 1 ? '' : 's'}: it is no longer untouched data.`);
  }
  if (lock !== null && lock.simulated) notes.push('The run used simulated fallback data, not real market data.');
  return { verdict: notes.length === 0 ? 'Robust' : 'Fragile', notes };
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
    // Train / validation figures stay exactly what the search ranked on: a backtest that ends at
    // the holdout boundary (any open trade is closed there, with costs) ...
    const m = evaluateSearch(asset.ind, base.genome, s.settings, asset.ws, asset.split, asset.periodsPerYear, asset.regimes);
    // ... and only the holdout comes from a backtest over the whole history.
    runBacktest(asset.ind, base.genome, s.settings, asset.ws);
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
    folds: primary.folds,
    regimes: primary.regimes,
    holdout: primary.holdout,
    alpha: primary.search.roi - s.assets[0].benchmark.search.roi,
    equity,
    assets,
  };
  const robustness = analyze(s, base.genome);
  const { verdict, notes } = judge(s, robustness, result);
  return { ...result, robustness, verdict, verdictNotes: notes };
}

/**
 * Lock `id` as the run's final strategy and reveal its holdout. Only one strategy per run, only
 * from the final leaderboard, only once the search has ended. Locking the same strategy again
 * returns the same result without a second reveal.
 */
function lockStrategy(s: Session, id: string, priorReveals: number, simulated: boolean): StrategyResult {
  if (!s.finished) throw new Error('The holdout can only be revealed after the run has ended.');
  if (s.locked !== null) {
    const known = s.finals.get(s.locked.id);
    if (s.locked.id === id && known) return known;
    throw new Error(`This run's holdout has already been spent on strategy ${s.locked.id}. Only one strategy per run can be revealed.`);
  }
  const entry = s.board.find((e) => e.result.id === id);
  if (!entry) throw new Error('Only a strategy on the final leaderboard can be locked.');
  s.locked = { id, priorReveals: Math.max(0, Math.floor(priorReveals)), simulated };
  const result = revealHoldout(s, entry.result);
  s.finals.set(id, result);
  return result;
}

/** Buy & Hold with the holdout still sealed: the curve stops at the holdout boundary. */
function sealedBenchmark(s: Session): BenchmarkResult {
  const primary = s.assets[0];
  return {
    search: primary.benchmark.search,
    train: primary.benchmark.train,
    validation: primary.benchmark.validation,
    holdout: null,
    equity: sampleCurve(primary.benchmark.equity, s.axis, s.axis.holdoutPos),
  };
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
/*  Evaluator pool                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Client side of the evaluation pool: the main thread connects one MessagePort
 * per evaluator worker. A batch is split into contiguous chunks, one per
 * evaluator, and the results come back in the original order – so the search
 * is identical whichever thread scored a candidate.
 */
class EvaluatorPool {
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (results: CandidateResult[]) => void; reject: (error: Error) => void }>();

  constructor(private ports: PortLike[]) {
    for (const port of ports) {
      port.onmessage = (event: MessageEvent) => {
        const reply = event.data as EvaluatorReply;
        const waiting = this.pending.get(reply.id);
        if (!waiting) return;
        this.pending.delete(reply.id);
        if (reply.type === 'results') waiting.resolve(reply.results);
        else waiting.reject(new Error(reply.message));
      };
    }
  }

  get size(): number {
    return this.ports.length;
  }

  async evaluate(genomes: StrategyGenome[], hurdle: number, floor: number, knownTrades: string[]): Promise<CandidateResult[]> {
    if (this.ports.length === 0) throw new Error('Evaluator pool closed.');
    const chunk = Math.ceil(genomes.length / this.ports.length);
    const requests: Promise<CandidateResult[]>[] = [];
    // Every evaluator keeps a full replica of the known trade logs, so each gets the delta.
    for (let w = 0; w < this.ports.length; w++) {
      const id = this.nextId++;
      const message: EvaluatorBatch = { type: 'batch', id, hurdle, floor, knownTrades, genomes: genomes.slice(w * chunk, (w + 1) * chunk) };
      requests.push(
        new Promise((resolve, reject) => {
          this.pending.set(id, { resolve, reject });
          this.ports[w].postMessage(message);
        }),
      );
    }
    return (await Promise.all(requests)).flat();
  }

  close(): void {
    for (const port of this.ports) {
      port.onmessage = null;
      port.close?.();
    }
    this.ports = [];
    for (const waiting of this.pending.values()) waiting.reject(new Error('Evaluator pool closed.'));
    this.pending.clear();
  }
}

/* -------------------------------------------------------------------------- */
/*  Optimisation run                                                          */
/* -------------------------------------------------------------------------- */

async function run(payload: OptimizerStartPayload, token: number): Promise<void> {
  const { settings } = payload;
  session = null;
  activePool?.close();
  const pool = new EvaluatorPool(payload.ports ?? []);
  activePool = pool;

  const hypothesis = payload.hypothesis;
  if (!hypothesis || !HYPOTHESIS_IDS.includes(hypothesis)) throw new Error('A run needs a market hypothesis to test.');
  const ctx = createContext(payload.datasets, settings, INDICATOR_BUDGET_BYTES);
  const unsupported = hypothesisUnsupported(hypothesis, ctx.limits);
  if (unsupported) throw new Error(unsupported);
  const { assets, axis } = ctx;
  const primary = assets[0];
  const annualiser = Math.sqrt(primary.periodsPerYear);

  // Trial budget: earlier runs on this data window count against it and seed the hurdle.
  const windowBudget = Math.max(0, Math.round(payload.windowBudget ?? WINDOW_TRIAL_BUDGET));
  const prior = payload.priorTrials && payload.priorTrials.count > 0 ? payload.priorTrials : { count: 0, mean: 0, m2: 0 };
  const remaining = windowBudget - prior.count;
  if (remaining <= 0) {
    throw new Error(
      `The trial budget of this data window is spent: ${prior.count.toLocaleString('en-US')} of ${windowBudget.toLocaleString('en-US')} ` +
        'evaluations have already been made on it. More search on the same data can only find luckier noise.',
    );
  }
  const requested = Math.round(payload.maxEvaluations ?? DEFAULT_RUN_EVALUATIONS);
  const maxEvaluations = Math.min(remaining, Number.isFinite(requested) ? Math.max(MIN_RUN_EVALUATIONS, requested) : DEFAULT_RUN_EVALUATIONS);

  const s: Session = {
    ...ctx,
    extra: [],
    mode: settings.tradeMode,
    hypothesis,
    // The ledger stores annualised Sharpe ratios; the hurdle works per bar of the primary asset.
    trials: RunningStats.from({ count: prior.count, mean: prior.mean / annualiser, m2: prior.m2 / (annualiser * annualiser) }),
    runTrials: new RunningStats(),
    hurdleTrials: 0,
    hurdleVariance: 0,
    board: [],
    finished: false,
    locked: null,
    finals: new Map(),
    undersampled: 0,
    duplicates: 0,
    inactive: 0,
  };
  session = s;
  const { mode, limits, board } = s;

  const rng = mulberry32(payload.seed ?? (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0);
  /**
   * Candidates proposed per round. They are all generated from the leaderboard as it
   * stood at the start of the round, scored (in parallel when there is a pool), then
   * applied one by one in proposal order. The size depends only on the pool size,
   * so a seeded run is reproducible.
   */
  const defaultBatch = pool.size > 0 ? PROGRESS_EVERY * ROUNDS_PER_EVALUATOR * pool.size : PROGRESS_EVERY;
  const batchSize = Math.max(1, Math.round(payload.batchSize ?? defaultBatch));

  // The live benchmark stops at the holdout boundary, exactly like the strategy curves.
  scope.postMessage({
    type: 'ready',
    benchmark: sealedBenchmark(s),
    axis,
    barsPerYear: primary.periodsPerYear,
  });

  const seen = new Set<string>();
  const seeds = seedGenomes(hypothesis, mode, limits, rng);
  let iteration = 0;
  /** Proposals since the top fitness last improved (informational). */
  let stagnation = 0;
  /** Unique, trading strategies evaluated by this run: its trials. */
  let evaluations = 0;
  /** Evaluations since a candidate last beat the best deflated Sharpe on the board. */
  let dsrStall = 0;
  /** Proposals since the last new (non-duplicate, trading) evaluation. */
  let sinceNew = 0;
  /** Best deflated Sharpe on the board, against the round's frozen hurdle. */
  let bestDsr = -Infinity;
  /** Trade-log fingerprints registered since the last batch was sent to the evaluators. */
  let knownDelta: string[] = [];
  let boardDirty = false;
  let lastBoardPost = 0;
  let itersPerSec = 0;
  const startedAt = performance.now();
  let lastBatchAt = startedAt;

  const snapshot = (): ProgressSnapshot => {
    let leader: StrategyResult | null = null;
    for (const entry of board) if (leader === null || entry.result.dsr > leader.dsr) leader = entry.result;
    return {
      iteration,
      stagnation,
      evaluations,
      windowTrials: prior.count + evaluations,
      dsrStall,
      duplicates: s.duplicates,
      runTrials: { count: s.runTrials.count, mean: s.runTrials.mean, m2: s.runTrials.m2 },
      bestFitness: board.length > 0 ? board[0].result.fitness : null,
      bestDsr: leader ? leader.dsr : null,
      bestDsrSharpe: leader ? leader.validation.sharpe : null,
      hurdleSharpe: s.hurdle * annualiser,
      undersampled: s.undersampled,
      itersPerSec,
      elapsedMs: performance.now() - startedAt,
    };
  };

  const postBoard = (): void => {
    scope.postMessage({ type: 'leaderboard', iteration, strategies: board.map((e) => e.result) });
    boardDirty = false;
    lastBoardPost = performance.now();
  };

  /** Rank-biased parent selection: quadratic bias towards the top of the leaderboard. */
  const pickParent = (): StrategyGenome => board[Math.floor(rng() * rng() * board.length)].result.genome;

  const propose = (): StrategyGenome => {
    if (seeds.length > 0) return seeds.shift() as StrategyGenome;
    if (board.length < SEED_POPULATION) return randomGenome(rng, mode, limits, hypothesis);
    const progress = Math.min(1, dsrStall / DSR_PATIENCE);
    const r = rng();
    // Exploration grows from 10% to 25% of proposals as the search stalls.
    const explore = 0.1 + 0.15 * progress;
    if (r < explore) return randomGenome(rng, mode, limits, hypothesis);
    if (r < explore + 0.12) return crossoverGenomes(rng, pickParent(), pickParent(), mode, limits);
    // Hill-climb the champion 40% of the time, otherwise a rank-biased parent.
    const parent = rng() < 0.4 ? board[0].result.genome : pickParent();
    return mutateGenome(rng, parent, mode, limits, 1 + Math.floor(progress * 3));
  };

  /** Primary-asset curve for a candidate whose curve was not captured during evaluation (rare: the board shrank). */
  const primaryCurve = (genome: StrategyGenome): Float32Array => {
    runBacktest(primary.ind, genome, settings, primary.ws, primary.split.holdoutStart);
    return sampleCurve(primary.ws.equity, axis, axis.holdoutPos);
  };

  /** True when a fully scored candidate fails a hard minimum-trade gate. */
  const isUndersampled = (scores: AssetScore[]): boolean => {
    const primaryMetrics = scores[0].metrics;
    if (primaryMetrics.search.trades < MIN_TRADES || primaryMetrics.validation.trades < MIN_VALIDATION_TRADES) return true;
    return scores.some((score, a) => tradesPerYear(s, a, score.metrics) < MIN_TRADES_PER_YEAR);
  };

  interface Outcome {
    /** False when the candidate was not a trial: it never traded, or repeated a known trade log. */
    trial: boolean;
    /** The top fitness improved. */
    improvedFitness: boolean;
    /** The candidate entered the board with a deflated Sharpe above the board's best. */
    improvedDsr: boolean;
  }
  const NOT_A_TRIAL: Outcome = { trial: false, improvedFitness: false, improvedDsr: false };

  /** Register an evaluated candidate and insert it into the leaderboard if it qualifies. */
  const consider = (genome: StrategyGenome, evaluated: CandidateResult): Outcome => {
    const { scores, complete, dsr, pointFitness, tradeHash, skip } = evaluated;
    let { fitness, neighbours } = evaluated;
    if (skip === 'inactive') {
      s.inactive++;
      return NOT_A_TRIAL;
    }
    // The evaluators only knew the trade logs registered before this round; repeats inside it are caught here.
    if (skip === 'duplicate' || (tradeHash !== null && s.knownTrades.has(tradeHash))) {
      s.duplicates++;
      return NOT_A_TRIAL;
    }
    if (tradeHash !== null) {
      s.knownTrades.add(tradeHash);
      knownDelta.push(tradeHash);
    }
    const primaryMetrics = scores[0].metrics;
    // Every new, trading strategy is a trial, whether or not it makes the leaderboard.
    s.trials.push(primaryMetrics.validation.sharpeRaw);
    s.runTrials.push(primaryMetrics.validation.sharpeRaw * annualiser);
    const trialOnly: Outcome = { trial: true, improvedFitness: false, improvedDsr: false };
    if (!complete) return trialOnly;
    if (fitness <= UNFIT) {
      if (isUndersampled(scores)) s.undersampled++;
      return trialOnly;
    }
    if (board.length >= LEADERBOARD_SIZE && pointFitness <= board[board.length - 1].result.fitness) return trialOnly;
    if (neighbours === null) {
      // Rare: the candidate did not beat the round's frozen floor, but the board has since shrunk.
      // Nothing enters the board without its neighbourhood cap.
      neighbours = scoreNeighbours(s, genome);
      fitness = rankFitness(s, genome, scores, neighbours).fitness;
    }
    const best = board.length > 0 ? board[0].result.fitness : -Infinity;
    if (board.length >= LEADERBOARD_SIZE && fitness <= board[board.length - 1].result.fitness) return trialOnly;

    const { search } = primaryMetrics;
    const neighbourValues = neighbours.map((nb) => fitnessOf(s, nb.genome, nb.scores).fitness);
    // Identical trade logs never reach this point, so the trade-log fingerprint identifies the board entry.
    const fingerprint = tradeHash ?? genomeSignature(genome);
    if (board.some((e) => e.fingerprint === fingerprint)) return trialOnly;

    // Behavioural clones: a fitter twin already on the board wins; otherwise this one replaces its twins.
    // The curve stops at the holdout boundary – nothing beyond it has been simulated.
    const equity = evaluated.curve ?? primaryCurve(genome);
    const returns = curveReturns(equity);
    const twins: number[] = [];
    for (let k = 0; k < board.length; k++) {
      if (correlation(returns, board[k].returns) >= CLONE_CORRELATION) twins.push(k);
    }
    if (twins.length > 0) {
      if (fitness <= board[twins[0]].result.fitness) return trialOnly;
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
      if (fitness <= board[nicheWorst].result.fitness) return trialOnly;
      board.splice(nicheWorst, 1);
    }

    const haircut = haircutSharpe(primaryMetrics.validation, s.hurdleTrials, primary.periodsPerYear);
    const entry: Entry = {
      niche,
      fingerprint,
      returns,
      scores,
      neighbours,
      result: {
        id: `S-${iteration}`,
        genome,
        fitness,
        pointFitness,
        neighbourMedian: neighbourValues.length > 0 ? median(neighbourValues) : null,
        neighbourCount: neighbourValues.length,
        dsr,
        haircutSharpe: haircut.sharpe,
        haircut: haircut.haircut,
        params: countParameters(genome, limits),
        search: primaryMetrics.search,
        train: primaryMetrics.train,
        validation: primaryMetrics.validation,
        folds: primaryMetrics.folds,
        regimes: primaryMetrics.regimes,
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
    const onBoard = board.includes(entry);
    const improvedDsr = onBoard && dsr > bestDsr + DSR_MIN_GAIN;
    if (onBoard && dsr > bestDsr) bestDsr = dsr;
    return { trial: true, improvedFitness: fitness > best + 1e-9, improvedDsr };
  };

  /** Propose a candidate the search has not proposed before (a few retries, then accept the repeat). */
  const proposeUnique = (): { genome: StrategyGenome; duplicate: boolean } => {
    let genome = propose();
    let signature = genomeSignature(genome);
    for (let attempt = 0; attempt < 8 && seen.has(signature); attempt++) {
      genome =
        board.length >= SEED_POPULATION && rng() < 0.7
          ? mutateGenome(rng, genome, mode, limits, 1)
          : randomGenome(rng, mode, limits, hypothesis);
      signature = genomeSignature(genome);
    }
    const duplicate = seen.has(signature);
    seen.add(signature);
    return { genome, duplicate };
  };

  /** Score a batch – on the evaluator pool when there is one, otherwise right here. */
  let usePool = pool.size > 0;
  const evaluateBatch = async (genomes: StrategyGenome[], floor: number): Promise<CandidateResult[]> => {
    const delta = knownDelta;
    knownDelta = [];
    if (usePool) {
      try {
        return await pool.evaluate(genomes, s.hurdle, floor, delta);
      } catch {
        // An evaluator failed: carry on single-threaded rather than abort the run.
        usePool = false;
        pool.close();
      }
    }
    return genomes.map((genome) => evaluateCandidate(s, genome, floor));
  };

  const postProgress = (): void => {
    scope.postMessage({ type: 'progress', ...snapshot() });
  };

  const finish = (reason: StopReason): void => {
    pool.close();
    // The search is over, but the holdout stays sealed: the user locks ONE strategy to reveal it.
    rescoreBoard(s);
    s.finished = true;
    const strategies = board.map((entry) => entry.result);
    scope.postMessage({ type: 'done', reason, strategies, benchmark: sealedBenchmark(s), ...snapshot() });
  };

  /** Why the run must end now, or null to carry on. */
  const stopReason = (): StopReason | null => {
    if (evaluations >= maxEvaluations) return prior.count + evaluations >= windowBudget ? 'window-budget' : 'budget';
    if (dsrStall >= DSR_PATIENCE) return 'converged';
    if (sinceNew >= EXHAUSTION_LIMIT) return 'exhausted';
    return null;
  };

  for (;;) {
    // The selection-bias hurdle and the leaderboard floor are frozen for the round, so every
    // candidate in it is judged like for like whichever evaluator scores it.
    rescoreBoard(s);
    bestDsr = board.reduce((max, entry) => Math.max(max, entry.result.dsr), -Infinity);
    const floor = board.length >= LEADERBOARD_SIZE ? board[board.length - 1].result.fitness : -Infinity;
    const batch: { genome: StrategyGenome; duplicate: boolean }[] = [];
    for (let k = 0; k < batchSize; k++) batch.push(proposeUnique());
    const results = await evaluateBatch(
      batch.filter((candidate) => !candidate.duplicate).map((candidate) => candidate.genome),
      floor,
    );
    if (token !== activeRun) {
      pool.close();
      return; // superseded by a newer run
    }

    // Apply in proposal order; candidates past a stop are discarded unseen.
    let considered = 0;
    let next = 0;
    let reason: StopReason | null = null;
    for (const candidate of batch) {
      if ((reason = stopReason()) !== null) break;
      iteration++;
      considered++;
      const outcome = candidate.duplicate ? NOT_A_TRIAL : consider(candidate.genome, results[next++]);
      if (candidate.duplicate) s.duplicates++;
      stagnation = outcome.improvedFitness ? 0 : stagnation + 1;
      if (outcome.trial) {
        evaluations++;
        sinceNew = 0;
        dsrStall = outcome.improvedDsr ? 0 : dsrStall + 1;
      } else {
        sinceNew++;
      }
      // Streamed every PROGRESS_EVERY (25) iterations.
      if (iteration % PROGRESS_EVERY === 0) postProgress();
    }
    reason ??= stopReason();

    const now = performance.now();
    const batchMs = now - lastBatchAt;
    lastBatchAt = now;
    if (batchMs > 0 && considered > 0) {
      const rate = (considered * 1000) / batchMs;
      itersPerSec = itersPerSec === 0 ? rate : itersPerSec * 0.9 + rate * 0.1;
    }
    if (boardDirty && now - lastBoardPost >= LEADERBOARD_INTERVAL_MS) postBoard();

    if (reason !== null) {
      if (iteration % PROGRESS_EVERY !== 0) postProgress();
      finish(reason);
      return;
    }

    await yieldToEventLoop();
    if (token !== activeRun) {
      pool.close();
      return; // superseded by a newer run
    }
    if (stopRequested) {
      finish('stopped');
      return;
    }
  }
}
