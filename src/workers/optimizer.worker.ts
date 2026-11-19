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
 * • Ranking uses one composite fitness (median-fold Sortino / Calmar, expectancy ×
 *   trade frequency, parameter-neighbourhood stability – see `assetBaseScore`). The
 *   Deflated Sharpe Ratio of the validation segment is a GATE: its hurdle rises with
 *   every strategy tried, on this run and on earlier runs over the same data.
 * • A multi-asset run trades every asset with the same rules; the trade gates apply to
 *   the pooled sample and the fitness is the median over the assets.
 *
 * Search: evolutionary hill-climbing (textbook seeds, mutation of the cycle's best or
 * a rank-biased parent, crossover, random immigrants) until the run finds its target
 * number of robust strategies, spends its evaluation budget, exhausts the search
 * space, or the UI sends `stop`. When the best fitness stalls for `FITNESS_PATIENCE`
 * evaluations the population restarts: the top `ELITE_SIZE` strategies are kept and
 * the rest is re-seeded with fresh random genomes.
 */

import {
  benchmarkEquity,
  evaluateSearch,
  holdoutMetrics,
  holdoutShortfalls,
  neighbourhoodMedian,
  passesTradeGates,
  ROBUST_MIN_DSR,
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
  robustChecks,
  sampleCurve,
  scoreNeighbours,
  sampleOf,
  tradesPerYear,
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
  randomGenome,
  repairGenome,
  withParameterValues,
  type GenomeLimits,
  type Rng,
} from '../engine/genome';
import { expectedMaxSharpe, RunningStats } from '../lib/stats';
import { monteCarloStress, summarise, tradesFrom, type McTrade } from '../engine/monteCarlo';
import {
  DEFAULT_ROBUST_TARGET,
  DEFAULT_RUN_EVALUATIONS,
  ELITE_SIZE,
  MAX_ROBUST_TARGET,
  MAX_RUN_EVALUATIONS,
  type GateFunnel,
  FITNESS_PATIENCE,
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
  type CrossMarketReport,
  type CrossMarketRow,
  type MonteCarloReport,
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
/** Evaluations a run makes before reaching its robust target may end it. */
const MIN_TARGET_EVALUATIONS = 1_000;
/** A candidate must beat the board's best fitness by this much to reset the restart clock. */
const FITNESS_MIN_GAIN = 0.001;
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
  if (message.type === 'cross-market') {
    try {
      if (!session) throw new Error('No optimisation run is available.');
      const report = crossMarketCheck(session, message.genome, message.datasets, message.runSymbols ?? []);
      session.crossMarket.set(message.strategyId, report);
      // The locked strategy's verdict depends on the check: judge it again with the new evidence.
      const final = session.locked?.id === message.strategyId ? session.finals.get(message.strategyId) : undefined;
      let locked: StrategyResult | undefined;
      if (final?.robustness) {
        const { verdict, notes } = judge(session, final.robustness, final);
        locked = { ...final, verdict, verdictNotes: notes };
        session.finals.set(final.id, locked);
      }
      scope.postMessage({ type: 'cross-market', strategyId: message.strategyId, report, ...(locked ? { locked } : {}) });
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

/** A run's state. The scoring context (`assets`, primary first) is shared with the evaluator pool. */
interface Session extends EvalContext {
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
  /**
   * Strategies that passed every robustness gate, by fingerprint – kept even when the leaderboard's
   * fitness ranking would push them off, and re-checked whenever the luck hurdle moves.
   */
  robust: Map<string, Entry>;
  /** How many candidates cleared each gate. */
  funnel: GateFunnel;
  /** Cross-market generalization checks the user ran, by strategy id. */
  crossMarket: Map<string, CrossMarketReport>;
}

/** Gates for one asset of a pooled run: the pooled gates shared out over the assets. */
function perAssetGates(s: Session): Session['gates'] {
  const k = Math.max(1, s.assets.length);
  return { ...s.gates, search: Math.ceil(s.gates.search / k), validation: Math.ceil(s.gates.validation / k), perYear: s.gates.perYear / k };
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
  const dynamic = (stop: number, target: number | null, trailing: boolean, reversalExit = false, breakeven = false): StrategyGenome['exit'] => ({
    mode: 'dynamic',
    atrPeriod: 14,
    stopAtrMult: stop,
    trailing,
    takeProfitAtrMult: target,
    reversalExit,
    longExit: null,
    shortExit: null,
    breakeven,
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
  const rsi = (period: number, level: number): Condition => ({ ind: 'RSI', period, op: 'cross_above', level });
  const emaCross = (fast: number, slow: number): Condition => ({ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: fast, slowType: 'EMA', slowPeriod: slow, op: 'cross_above' });
  const donchian = (period: number): Condition => ({ ind: 'DONCHIAN', period, variant: 'upper', op: 'cross_above' });
  const aboveEma = (period: number): Condition => ({ ind: 'PRICE_MA', maType: 'EMA', period, op: 'gt' });
  const rising = (period: number): Condition => ({ ind: 'MA_SLOPE', maType: 'EMA', period, op: 'gt' });
  const lowerBand: Condition = { ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'cross_above', level: 0 };
  const lowerKeltner: Condition = { ind: 'KELTNER', period: 20, mult: 2, variant: 'lower', op: 'cross_above' };
  const squeeze: Condition = { ind: 'SQUEEZE', period: 20, mult: 1.5, op: 'gt', level: 0.5 };
  const expansion: Condition = { ind: 'RANGE_EXP', period: 14, mult: 2, op: 'gt', level: 0.5 };
  const pullback = (period: number): Condition => ({ ind: 'PULLBACK', maType: 'EMA', period, op: 'gt', level: 0.5 });
  let seeds: StrategyGenome[];
  // Textbook versions with ONE indicator, or one trigger plus one filter from another family.
  switch (h) {
    case 'TREND_FOLLOWING':
      seeds = [
        // 20/56 EMA crossover with a trailing ATR stop.
        seed([emaCross(20, 56)], dynamic(3, null, true)),
        // Turtle-style 56-candle breakout while the slow EMA rises, closed on the opposite breakout.
        seed([donchian(56), rising(112)], dynamic(2.5, null, true, true)),
        // MACD signal cross while a trend is present, fixed ATR bracket.
        seed([{ ind: 'MACD', fast: 14, slow: 28, signal: 10, variant: 'signal', op: 'cross_above' }, adx('gt', 20)], dynamic(2, 4, false)),
      ];
      break;
    case 'VOL_BREAKOUT':
      seeds = [
        seed([donchian(20)], dynamic(2, 4, false)),
        seed([{ ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'cross_above', level: 0 }, { ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 }], dynamic(2.5, null, true)),
        seed([donchian(40), aboveEma(112)], dynamic(3, 6, false)),
        // TTM-style squeeze release ridden with a trailing stop, locked at breakeven after +1R.
        seed([squeeze], dynamic(2.5, null, true, false, true)),
        // Wide-range up candle in a slow uptrend, fixed ATR bracket.
        seed([expansion, aboveEma(200)], dynamic(2, 4, false, false, true)),
        seed([{ ind: 'KELTNER', period: 20, mult: 2, variant: 'upper', op: 'cross_above' }, rising(112)], dynamic(2.5, null, true)),
      ];
      break;
    case 'DIP_IN_UPTREND':
      seeds = [
        seed([rsi(14, 30), aboveEma(200)], dynamic(2, 3, false)),
        seed([rsi(7, 20), rising(112)], dynamic(2, 3, false)),
        seed([lowerBand, aboveEma(160)], dynamic(1.5, 2.5, false, true)),
        // Pullback that holds the 20 EMA inside a 200 EMA uptrend, breakeven after +1R.
        seed([pullback(20), aboveEma(200)], dynamic(2, 4, false, false, true)),
        seed([lowerKeltner, rising(112)], dynamic(1.5, 3, false)),
      ];
      break;
    case 'RANGE_REVERSION':
      seeds = [
        seed([rsi(14, 30)], dynamic(2, 3, false)),
        seed([rsi(14, 30), adx('lt', 20)], dynamic(2, 3, false, true)),
        seed([lowerBand], dynamic(1.5, 2.5, false)),
        seed([lowerKeltner, adx('lt', 20)], dynamic(1.5, 2.5, false)),
      ];
      break;
  }
  return seeds.map((g) => repairGenome(g, mode, limits, rng));
}

/* -------------------------------------------------------------------------- */
/*  Scoring                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Move the luck hurdle to the latest trial count (it rises as more strategies are tried) and
 * re-check every reported strategy's deflated Sharpe and robustness gates against it. The fitness
 * does not depend on the hurdle, so the leaderboard order only changes through new entries.
 */
function rescoreBoard(s: Session): void {
  s.hurdleTrials = s.trials.count;
  s.hurdleVariance = s.trials.variance;
  s.hurdle = expectedMaxSharpe(s.hurdleTrials, s.hurdleVariance);
  const entries = new Set([...s.board, ...s.robust.values()]);
  for (const entry of entries) {
    const { dsr } = fitnessOf(s, entry.scores);
    entry.result.dsr = dsr;
    // The deflated Sharpe falls as trials accrue: a strategy can drop out of the robust set.
    entry.result.robust = robustChecks(s, entry.scores, dsr, entry.neighbours);
  }
  for (const [key, entry] of s.robust) if (!entry.result.robust?.passed) s.robust.delete(key);
}

/**
 * Every strategy the run reports, in rank order: strategies passing every robustness gate first
 * (including robust ones that fell off the fitness board), then the rest of the leaderboard, each
 * group by fitness.
 */
function reportedEntries(s: Session): Entry[] {
  const out = [...s.board];
  for (const entry of s.robust.values()) if (!out.includes(entry)) out.push(entry);
  const passed = (entry: Entry): number => (entry.result.robust?.passed ? 1 : 0);
  return out.sort((a, b) => passed(b) - passed(a) || b.result.fitness - a.result.fitness);
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
 * Monte Carlo execution stress test of a strategy's search-period trades: the primary asset's, or
 * every asset's with a pooled universe (the pooled sample is what the strategy was gated on).
 * Backtests the search period again, so it overwrites the assets' workspaces.
 */
function stressTest(s: Session, genome: StrategyGenome): MonteCarloReport | null {
  const assets = s.pooled ? s.assets : s.assets.slice(0, 1);
  const trades: McTrade[] = [];
  for (const asset of assets) {
    runBacktest(asset.ind, genome, s.settings, asset.ws, asset.split.holdoutStart);
    trades.push(...tradesFrom(asset.ws, asset.split.holdoutStart));
  }
  // Seeded by the strategy itself, so the same strategy always gets the same report.
  let seed = 0x9e3779b9;
  for (const ch of genomeSignature(genome)) seed = Math.imul(seed ^ ch.charCodeAt(0), 0x01000193);
  return monteCarloStress(trades, s.settings, { seed: seed >>> 0 });
}

/** Unseen assets need at least this many candles before the holdout boundary to be tested. */
const CROSS_MARKET_MIN_BARS = 150;

/**
 * Cross-market generalization check: the exact (frozen) rules on assets the search never saw, over
 * their candles BEFORE the run's holdout boundary (by date), so the holdout stays sealed – a
 * correlated asset's holdout-period candles would leak the holdout.
 */
function crossMarketCheck(s: Session, genome: StrategyGenome, datasets: NamedDataset[], runSymbols: readonly string[]): CrossMarketReport {
  const primary = s.assets[0];
  const cutoff = primary.data.t[primary.split.holdoutStart];
  const inRun = new Set([...s.assets.map((asset) => asset.symbol), ...runSymbols].map((symbol) => symbol.toUpperCase()));
  const rows: CrossMarketRow[] = [];
  for (const dataset of datasets) {
    const { data, symbol } = dataset;
    let bars = 0;
    while (bars < data.t.length && data.t[bars] < cutoff) bars++;
    const empty: CrossMarketRow = { symbol, unseen: !inRun.has(symbol.toUpperCase()), bars, roi: 0, sharpe: 0, maxDrawdown: 0, trades: 0, holdRoi: 0, profitable: false, note: null };
    if (bars < CROSS_MARKET_MIN_BARS) {
      rows.push({ ...empty, note: `Only ${bars} candles before ${new Date(cutoff).toISOString().slice(0, 10)} (${CROSS_MARKET_MIN_BARS} needed).` });
      continue;
    }
    const asset = buildAsset(dataset, s.settings, 32 * 1024 * 1024);
    runBacktest(asset.ind, genome, s.settings, asset.ws, bars);
    const m = segmentMetrics(asset.ws.equity, 0, bars, s.settings.initialCapital, asset.periodsPerYear, asset.ws);
    const hold = benchmarkEquity(data, s.settings, bars);
    rows.push({
      ...empty,
      roi: m.roi,
      sharpe: m.sharpe,
      maxDrawdown: m.maxDrawdown,
      trades: m.trades,
      holdRoi: (hold[bars - 1] / s.settings.initialCapital - 1) * 100,
      profitable: m.netPnl > 0,
    });
  }
  const tested = rows.filter((row) => row.unseen && row.note === null);
  const share = tested.length > 0 ? tested.filter((row) => row.profitable).length / tested.length : 0;
  const medianSharpe = medianOf(tested.map((row) => row.sharpe));
  return {
    rows,
    tested: tested.length,
    share,
    medianSharpe,
    passed: tested.length >= 2 && share >= 2 / 3 - 1e-9 && medianSharpe > 0,
    cutoff,
  };
}

/**
 * Robustness tests on the search period only (the holdout is not touched): removal of the best
 * trades, the parameter-stability map and the Monte Carlo execution stress test. (Parameter
 * stability as a gate is the one-step neighbourhood of `RobustChecks`; generalisation to other
 * markets is the cross-market check on unseen assets.)
 */
function analyze(s: Session, genome: StrategyGenome): RobustnessReport {
  // Top-trade removal over the sample the strategy was gated on: the primary asset's search-period
  // trades, or every asset's in a pooled run.
  const pnls: number[] = [];
  for (const asset of s.pooled ? s.assets : s.assets.slice(0, 1)) {
    runBacktest(asset.ind, genome, s.settings, asset.ws, asset.split.holdoutStart);
    for (let k = 0; k < asset.ws.tradeCount; k++) if (asset.ws.tradeEntryBar[k] < asset.split.holdoutStart) pnls.push(asset.ws.tradePnl[k]);
  }
  pnls.sort((x, y) => y - x);
  const removed = Math.min(TOP_TRADES_REMOVED, pnls.length);
  let netPnl = 0;
  for (const pnl of pnls) netPnl += pnl;
  let top = 0;
  for (let k = 0; k < removed; k++) top += pnls[k];

  return {
    topTrades: { netPnl, netPnlWithout: netPnl - top, removed, profitable: netPnl - top > 0 },
    stability: stabilityGrid(s, genome),
    monteCarlo: stressTest(s, genome),
  };
}

/**
 * Parameter-stability map of the primary asset. The two axes are the parameters whose one-grid-step
 * neighbours move the search-period Sharpe the most (the ranking neighbourhood, read per parameter);
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
      // The map reads the primary asset alone, so it is gated per asset even in a pooled run.
      undersampled: !passesTradeGates({
        trades: m.search.trades,
        validationTrades: m.validation.trades,
        tradesPerYear: tradesPerYear(s, 0, m),
        gates: s.pooled ? perAssetGates(s) : s.gates,
      }),
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
 *             • the search-period gates: validation, fold consistency and a stable one-step
 *               parameter neighbourhood (see `RobustChecks`);
 *             • holdout Sharpe 95% confidence interval entirely above zero;
 *             • holdout Sharpe above Buy & Hold's holdout Sharpe (risk-adjusted, not raw return);
 *             • at least `gates.holdout` holdout trades (summed over the assets when pooled);
 *             • profitable without its best trades, and the Monte Carlo stress test passed;
 *             • generalisation shown: a pooled multi-asset run, or a passed cross-market check;
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
  const gates = result.robust;
  if (gates && !gates.validation) notes.push('Validation gate failed: the validation Sharpe is not above 0.3 with a profit after costs.');
  if (gates && !gates.folds) notes.push('Fold gate failed: fewer than 3 of 5 search-period slices made money, or one lost more than 35%.');
  if (gates && !gates.neighbours) notes.push('Parameter neighbourhood: the one-step neighbours do not keep a positive median profit per trade.');
  // In a pooled run the holdout sample is pooled too.
  const holdoutTrades = result.sample?.holdoutTrades ?? holdout.trades;
  notes.push(...holdoutShortfalls(holdout, finalBenchmark(s).holdout, s.assets[0].periodsPerYear, s.gates.holdout, holdoutTrades));
  if (!report.topTrades.profitable) notes.push(`Unprofitable without its ${report.topTrades.removed} best trades.`);
  const mc = report.monteCarlo;
  if (mc && !mc.passed) {
    notes.push(
      `Monte Carlo stress test: ${Math.round(mc.probProfit * 100)}% of ${mc.simulations} perturbed paths end in profit (80% needed), ` +
        `95th-percentile drawdown ${mc.drawdownP95.toFixed(1)}% (under 50% needed).`,
    );
  }
  const cross = s.crossMarket.get(result.id);
  if (cross && cross.tested > 0 && !cross.passed) {
    notes.push(`Cross-market check: profitable on ${Math.round(cross.share * 100)}% of ${cross.tested} unseen assets, median Sharpe ${cross.medianSharpe.toFixed(2)}.`);
  } else if (!s.pooled && !(cross && cross.passed)) {
    notes.push('Generalisation not shown yet: run the cross-market check on unseen markets (or pool several assets in the run).');
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
    const m = evaluateSearch(asset.ind, base.genome, s.settings, asset.ws, asset.split, asset.periodsPerYear);
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
    holdout: primary.holdout,
    alpha: primary.search.roi - s.assets[0].benchmark.search.roi,
    equity,
    assets,
  };
  if (result.sample) {
    const holdoutTrades = s.pooled ? assets.reduce((sum, asset) => sum + (asset.holdout?.trades ?? 0), 0) : (primary.holdout?.trades ?? 0);
    result.sample = { ...result.sample, holdoutTrades };
  }
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
  const entry = reportedEntries(s).find((e) => e.result.id === id);
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
  const maxEvaluations = Math.min(
    remaining,
    MAX_RUN_EVALUATIONS,
    Number.isFinite(requested) ? Math.max(MIN_RUN_EVALUATIONS, requested) : DEFAULT_RUN_EVALUATIONS,
  );
  // 0 = no target: scan the whole budget.
  const robustTarget = Math.max(0, Math.min(MAX_ROBUST_TARGET, Math.round(payload.robustTarget ?? DEFAULT_ROBUST_TARGET)));

  const s: Session = {
    ...ctx,
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
    robust: new Map(),
    funnel: { scored: 0, trades: 0, validation: 0, folds: 0, dsr: 0, neighboursChecked: 0, neighbours: 0 },
    crossMarket: new Map(),
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
    gates: s.gates,
    pooled: s.pooled,
  });

  const seen = new Set<string>();
  const seeds = seedGenomes(hypothesis, mode, limits, rng);
  let iteration = 0;
  /** Unique, trading strategies evaluated by this run: its trials. */
  let evaluations = 0;
  /** Evaluations since a candidate last improved the best fitness on the board (a restart at `FITNESS_PATIENCE`). */
  let fitnessStall = 0;
  /** Population cycle: 1, then +1 at every diversity restart. */
  let cycle = 1;
  /** The best `ELITE_SIZE` strategies of the board at the last restart: they survive it. */
  let elite: StrategyGenome[] = [];
  /**
   * After a restart, the re-seeded part of the population: the best candidates scored since the
   * restart (by fitness, whether or not they made the leaderboard), so fresh genomes are evolved
   * rather than only sampled.
   */
  let cyclePool: { genome: StrategyGenome; fitness: number }[] = [];
  /** Proposals since the last new (non-duplicate, trading) evaluation. */
  let sinceNew = 0;
  /** Trade-log fingerprints registered since the last batch was sent to the evaluators. */
  let knownDelta: string[] = [];
  let boardDirty = false;
  let lastBoardPost = 0;
  let itersPerSec = 0;
  const startedAt = performance.now();
  let lastBatchAt = startedAt;

  const snapshot = (): ProgressSnapshot => {
    let leader: StrategyResult | null = null;
    for (const entry of [...board, ...s.robust.values()]) if (leader === null || entry.result.dsr > leader.dsr) leader = entry.result;
    return {
      iteration,
      evaluations,
      windowTrials: prior.count + evaluations,
      fitnessStall,
      duplicates: s.duplicates,
      runTrials: { count: s.runTrials.count, mean: s.runTrials.mean, m2: s.runTrials.m2 },
      bestFitness: board.length > 0 ? board[0].result.fitness : null,
      bestDsr: leader ? leader.dsr : null,
      bestDsrSharpe: leader ? leader.validation.sharpe : null,
      hurdleSharpe: s.hurdle * annualiser,
      undersampled: s.undersampled,
      robustFound: s.robust.size,
      robustTarget,
      cycle,
      funnel: { ...s.funnel },
      itersPerSec,
      elapsedMs: performance.now() - startedAt,
    };
  };

  const postBoard = (): void => {
    scope.postMessage({ type: 'leaderboard', iteration, strategies: reportedEntries(s).map((e) => e.result) });
    boardDirty = false;
    lastBoardPost = performance.now();
  };

  /**
   * The parent population of the current cycle, best first. In the first cycle it is the
   * leaderboard. After a restart it is the cycle's own best candidates (the re-seeded part), then
   * the elite: the fresh genomes lead and are hill-climbed, while the elite still breed through
   * crossover and rank-biased picks without pulling every mutation back to the optimum the run had
   * stalled on.
   */
  let parents: StrategyGenome[] = [];
  const refreshParents = (): void => {
    parents = cycle === 1 ? board.map((entry) => entry.result.genome) : [...cyclePool.map((member) => member.genome), ...elite];
  };
  /** Offer a fully scored candidate to the re-seeded part of the population (cycles after the first). */
  const joinCyclePool = (genome: StrategyGenome, fitness: number): void => {
    if (cycle === 1 || !(fitness > UNFIT)) return;
    const capacity = LEADERBOARD_SIZE - ELITE_SIZE;
    if (cyclePool.length >= capacity && fitness <= cyclePool[cyclePool.length - 1].fitness) return;
    let at = cyclePool.length;
    while (at > 0 && cyclePool[at - 1].fitness < fitness) at--;
    cyclePool.splice(at, 0, { genome, fitness });
    if (cyclePool.length > capacity) cyclePool.length = capacity;
  };
  /** Rank-biased parent selection: quadratic bias towards the top of the parent population. */
  const pickParent = (): StrategyGenome => parents[Math.floor(rng() * rng() * parents.length)];

  const propose = (): StrategyGenome => {
    if (seeds.length > 0) return seeds.shift() as StrategyGenome;
    if (parents.length < SEED_POPULATION) return randomGenome(rng, mode, limits, hypothesis);
    const progress = Math.min(1, fitnessStall / FITNESS_PATIENCE);
    const r = rng();
    // Exploration grows from 10% to 25% of proposals as the cycle stalls.
    const explore = 0.1 + 0.15 * progress;
    if (r < explore) return randomGenome(rng, mode, limits, hypothesis);
    if (r < explore + 0.12) return crossoverGenomes(rng, pickParent(), pickParent(), mode, limits);
    // Hill-climb the cycle's best 40% of the time, otherwise a rank-biased parent.
    const parent = rng() < 0.4 ? parents[0] : pickParent();
    return mutateGenome(rng, parent, mode, limits, 1 + Math.floor(progress * 3));
  };

  /**
   * Diversity restart: keep the best `ELITE_SIZE` strategies of the board as the elite and re-seed
   * the rest of the parent population with fresh random single- and dual-indicator genomes, so a
   * long scan escapes the local optimum it has stalled on instead of looping on minor tweaks – and
   * keeps scanning instead of stopping. The leaderboard and the robust archive are untouched.
   */
  const restart = (): void => {
    cycle++;
    fitnessStall = 0;
    elite = board.slice(0, ELITE_SIZE).map((entry) => entry.result.genome);
    cyclePool = [];
    for (let k = 0; k < 2 * SEED_POPULATION; k++) seeds.push(randomGenome(rng, mode, limits, hypothesis));
    refreshParents();
  };

  /** Primary-asset curve for a candidate whose curve was not captured during evaluation (rare: the board shrank). */
  const primaryCurve = (genome: StrategyGenome): Float32Array => {
    runBacktest(primary.ind, genome, settings, primary.ws, primary.split.holdoutStart);
    return sampleCurve(primary.ws.equity, axis, axis.holdoutPos);
  };

  /** True when a fully scored candidate fails a hard minimum-trade gate. */
  const isUndersampled = (scores: AssetScore[]): boolean => !passesTradeGates({ ...sampleOf(s, scores), gates: s.gates });

  interface Outcome {
    /** False when the candidate was not a trial: it never traded, or repeated a known trade log. */
    trial: boolean;
    /** The best fitness on the board improved by at least `FITNESS_MIN_GAIN` (resets the restart clock). */
    improved: boolean;
  }
  const NOT_A_TRIAL: Outcome = { trial: false, improved: false };

  /** A leaderboard / robust-archive entry for a fully scored candidate. */
  const buildEntry = (
    genome: StrategyGenome,
    evaluated: CandidateResult,
    fitness: number,
    neighbours: NeighbourScore[],
    fingerprint: string,
    robust: StrategyResult['robust'],
    curve: Float32Array | null = null,
  ): Entry => {
    const { scores, pointFitness, dsr } = evaluated;
    const primaryMetrics = scores[0].metrics;
    const equity = curve ?? evaluated.curve ?? primaryCurve(genome);
    const monteCarlo = summarise(stressTest(s, genome));
    const neighbourValues = neighbours.map((nb) => fitnessOf(s, nb.scores).fitness);
    return {
      niche: genomeNiche(genome),
      fingerprint,
      returns: curveReturns(equity),
      scores,
      neighbours,
      result: {
        id: `S-${iteration}`,
        genome,
        fitness,
        pointFitness,
        neighbourMedian: neighbourhoodMedian(pointFitness, neighbourValues),
        neighbourCount: neighbourValues.length,
        dsr,
        params: countParameters(genome, limits),
        sample: { ...sampleOf(s, scores), holdoutTrades: null, gates: s.gates, pooled: s.pooled },
        robust,
        search: primaryMetrics.search,
        train: primaryMetrics.train,
        validation: primaryMetrics.validation,
        folds: primaryMetrics.folds,
        holdout: null,
        alpha: primaryMetrics.search.roi - primary.benchmark.search.roi,
        equity,
        foundAt: iteration,
        assets: searchAssetResults(s, scores),
        robustness: null,
        verdict: null,
        monteCarlo,
      },
    };
  };

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
    const trialOnly: Outcome = { trial: true, improved: false };
    if (!complete) return trialOnly;

    // Robustness funnel: which gates this candidate clears on its own (the neighbourhood comes later).
    const cheap = robustChecks(s, scores, dsr, null);
    const funnel = s.funnel;
    funnel.scored++;
    if (!isUndersampled(scores)) funnel.trades++;
    if (cheap.validation) funnel.validation++;
    if (cheap.folds) funnel.folds++;
    if (dsr >= ROBUST_MIN_DSR) funnel.dsr++;

    if (fitness <= UNFIT) {
      if (isUndersampled(scores)) s.undersampled++;
      return trialOnly;
    }
    joinCyclePool(genome, fitness);
    const fingerprint = tradeHash ?? genomeSignature(genome);
    // A candidate that clears every other gate always gets its neighbourhood checked, even when its
    // fitness would not make the leaderboard: it may be one of the robust strategies the run is after.
    if (cheap.validation && cheap.folds && cheap.significance && !s.robust.has(fingerprint)) {
      if (neighbours === null) {
        neighbours = scoreNeighbours(s, genome);
        fitness = rankFitness(s, scores, neighbours).fitness;
      }
      funnel.neighboursChecked++;
      const checks = robustChecks(s, scores, dsr, neighbours);
      if (checks.neighbours) funnel.neighbours++;
      if (checks.passed) {
        s.robust.set(fingerprint, buildEntry(genome, evaluated, fitness, neighbours, fingerprint, checks));
        boardDirty = true;
      }
    } else if (neighbours !== null) {
      funnel.neighboursChecked++;
      if (robustChecks(s, scores, dsr, neighbours).neighbours) funnel.neighbours++;
    }

    if (board.length >= LEADERBOARD_SIZE && pointFitness <= board[board.length - 1].result.fitness) return trialOnly;
    if (neighbours === null) {
      // Rare: the candidate did not beat the round's frozen floor, but the board has since shrunk.
      // Nothing enters the board without its neighbourhood cap.
      neighbours = scoreNeighbours(s, genome);
      fitness = rankFitness(s, scores, neighbours).fitness;
    }
    const best = board.length > 0 ? board[0].result.fitness : -Infinity;
    if (board.length >= LEADERBOARD_SIZE && fitness <= board[board.length - 1].result.fitness) return trialOnly;

    // Identical trade logs never reach this point, so the trade-log fingerprint identifies the board entry.
    if (board.some((e) => e.fingerprint === fingerprint)) return trialOnly;

    // Behavioural clones: a fitter twin already on the board wins; otherwise this one replaces its twins.
    // The curve stops at the holdout boundary – nothing beyond it has been simulated.
    const archived = s.robust.get(fingerprint);
    const equity = archived?.result.equity ?? evaluated.curve ?? primaryCurve(genome);
    const returns = archived?.returns ?? curveReturns(equity);
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

    const entry = archived ?? buildEntry(genome, evaluated, fitness, neighbours, fingerprint, robustChecks(s, scores, dsr, neighbours), equity);
    let at = board.length;
    while (at > 0 && board[at - 1].result.fitness < fitness) at--;
    board.splice(at, 0, entry);
    if (board.length > LEADERBOARD_SIZE) board.length = LEADERBOARD_SIZE;
    boardDirty = true;
    const onBoard = board.includes(entry);
    return { trial: true, improved: onBoard && fitness > best + FITNESS_MIN_GAIN };
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
    const strategies = reportedEntries(s).map((entry) => entry.result);
    scope.postMessage({ type: 'done', reason, strategies, benchmark: sealedBenchmark(s), ...snapshot() });
  };

  /** Why the run must end now, or null to carry on. */
  const stopReason = (): StopReason | null => {
    // The target ends a run only after a minimum scan: with a handful of trials the deflated
    // Sharpe's hurdle rests on too few Sharpe ratios to be trusted.
    if (robustTarget > 0 && s.robust.size >= robustTarget && evaluations >= Math.min(MIN_TARGET_EVALUATIONS, maxEvaluations)) return 'target';
    if (evaluations >= maxEvaluations) return prior.count + evaluations >= windowBudget ? 'window-budget' : 'budget';
    if (sinceNew >= EXHAUSTION_LIMIT) return 'exhausted';
    return null;
  };

  for (;;) {
    // The selection-bias hurdle and the leaderboard floor are frozen for the round, so every
    // candidate in it is judged like for like whichever evaluator scores it.
    rescoreBoard(s);
    // A stall is no reason to stop while robust strategies are still missing: restart instead.
    if (fitnessStall >= FITNESS_PATIENCE) restart();
    else refreshParents();
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
      if (outcome.trial) {
        evaluations++;
        sinceNew = 0;
        fitnessStall = outcome.improved ? 0 : fitnessStall + 1;
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
