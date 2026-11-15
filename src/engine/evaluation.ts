/**
 * Strategy evaluation shared by the optimiser worker and the evaluator pool.
 *
 * Both sides build identical `Asset`s from the same data and score a genome
 * with exactly the same code, so a strategy gets the same fitness whichever
 * thread backtested it. `attachEvaluator` turns any MessagePort into a remote
 * evaluator: the optimiser sends batches of genomes and gets scores back.
 */

import {
  assetBaseScore,
  BacktestWorkspace,
  barsPerYear,
  benchmarkSearch,
  combineFitness,
  deflatedTerm,
  evaluateSearch,
  isSingleRegime,
  plateauFitness,
  regimeMask,
  splitFor,
  tradeLogHash,
  UNFIT,
  type Regimes,
  type SearchMetrics,
  type Split,
} from './backtester';
import { countParameters, rewardToRisk, stepNeighbours, type GenomeLimits } from './genome';
import { IndicatorCache } from './indicators';
import { deflatedSharpeAgainst } from '../lib/stats';
import type { BacktestSettings, ChartAxis, NamedDataset, OHLCV, StrategyGenome } from '../types/trading';

/** Minimum number of candles required for a meaningful three-way split. */
export const MIN_BARS = 150;
/** Maximum number of points per equity curve sent to the UI. */
export const MAX_CHART_POINTS = 1200;

/** Everything needed to backtest one asset. */
export interface Asset {
  symbol: string;
  data: OHLCV;
  ind: IndicatorCache;
  ws: BacktestWorkspace;
  split: Split;
  periodsPerYear: number;
  /** Buy & Hold over the search period. */
  benchmark: SearchMetrics & { equity: Float64Array };
  /** Bull / bear / high-volatility label of every search-period bar. */
  regimes: Regimes;
}

/** Per-asset ingredients of a fitness score that do not depend on the number of trials. */
export interface AssetScore {
  base: number;
  metrics: SearchMetrics;
}

/** What scoring needs: the universe (primary asset first), the run settings and the current selection-bias hurdle. */
export interface EvalContext {
  assets: Asset[];
  settings: BacktestSettings;
  limits: GenomeLimits;
  axis: ChartAxis;
  /** Per-bar Sharpe a strategy must beat to count as more than luck; frozen for a batch. */
  hurdle: number;
  /** Trade-log fingerprints already evaluated in this run (frozen for a batch, see `tradeLogHash`). */
  knownTrades: Set<string>;
}

/**
 * Why a candidate was not scored:
 *   inactive  – it never traded on the primary asset (nothing to rank, not a trial);
 *   duplicate – its primary-asset trade log repeats one already evaluated (not a new trial).
 */
export type SkipReason = 'inactive' | 'duplicate';

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

export function buildAsset(dataset: NamedDataset, settings: BacktestSettings, budgetBytes: number): Asset {
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
    regimes: regimeMask(data.c, split, periodsPerYear),
  };
}

/** Evenly spaced sample of bar indices that always contains bar 0, both segment boundaries and the last bar. */
export function buildAxis(t: Float64Array, split: Split): ChartAxis {
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
export function sampleCurve(equity: Float64Array, axis: ChartAxis, points: number = axis.barIndex.length): Float32Array {
  const out = new Float32Array(points);
  for (let k = 0; k < points; k++) out[k] = equity[axis.barIndex[k]];
  return out;
}

export function deriveLimits(primary: Asset): GenomeLimits {
  const volume = primary.data.v;
  let withVolume = 0;
  for (let i = 0; i < volume.length; i++) if (volume[i] > 0) withVolume++;
  return {
    // Lookbacks are capped so indicators are warmed up for most of the training segment.
    maxLookback: Math.max(30, Math.min(200, Math.floor(primary.split.validationStart / 3))),
    hasVolume: withVolume > volume.length * 0.5,
  };
}

/* -------------------------------------------------------------------------- */
/*  Scoring                                                                   */
/* -------------------------------------------------------------------------- */

/** Stand-in for "as good as it could possibly get" when bounding the fitness of a partly scored strategy. */
const OPTIMISTIC = 1e9;

/**
 * Backtest `genome` on the search period of the universe, primary asset first.
 *
 * With `floor` set (the weakest fitness on a full leaderboard) the remaining
 * assets are skipped as soon as the strategy provably cannot beat it: the
 * fitness is the MEDIAN of the per-asset scores, so once more than half of the
 * assets are scored, the best case for the rest still leaves the median capped.
 * The bound is exact, so pruning never changes which strategies are accepted.
 */
export function scoreAssets(
  ctx: EvalContext,
  genome: StrategyGenome,
  floor = -Infinity,
  dedupe = false,
): { scores: AssetScore[]; complete: boolean; tradeHash: string | null; skip: SkipReason | null } {
  const total = ctx.assets.length;
  const scores: AssetScore[] = [];
  let tradeHash: string | null = null;
  for (let a = 0; a < total; a++) {
    const asset = ctx.assets[a];
    const metrics = evaluateSearch(asset.ind, genome, ctx.settings, asset.ws, asset.split, asset.periodsPerYear, asset.regimes);
    scores.push({ base: assetBaseScore(metrics), metrics });
    if (a === 0 && dedupe) {
      // The primary asset decides whether this is a new bet at all, before any other work is spent on it.
      tradeHash = tradeLogHash(asset.ws, asset.split.holdoutStart);
      if (metrics.search.trades === 0) return { scores, complete: false, tradeHash, skip: 'inactive' };
      if (ctx.knownTrades.has(tradeHash)) return { scores, complete: false, tradeHash, skip: 'duplicate' };
    }
    // The median of `total` values is settled by its lower half: bounding needs more than half of them.
    if (floor > -Infinity && a + 1 < total && a + 1 > total >> 1 && fitnessCeiling(ctx, genome, scores) <= floor) {
      return { scores, complete: false, tradeHash, skip: null };
    }
  }
  return { scores, complete: true, tradeHash, skip: null };
}

/** Average trades per year that asset `a` opened over its search period. */
export function tradesPerYear(ctx: EvalContext, a: number, metrics: SearchMetrics): number {
  const asset = ctx.assets[a];
  const years = asset.split.holdoutStart / asset.periodsPerYear;
  return years > 0 ? metrics.search.trades / years : 0;
}

/** Lowest trades-per-year rate over the scored assets (Infinity when none is scored). */
function minTradesPerYear(ctx: EvalContext, scores: AssetScore[]): number {
  let min = Infinity;
  scores.forEach((score, a) => {
    const rate = tradesPerYear(ctx, a, score.metrics);
    if (rate < min) min = rate;
  });
  return min;
}

/** Highest fitness a partly scored strategy can still reach if every remaining asset turns out perfect. */
function fitnessCeiling(ctx: EvalContext, genome: StrategyGenome, scores: AssetScore[]): number {
  const missing = ctx.assets.length - scores.length;
  let profitable = missing;
  const assetScores = scores.map((score) => {
    if (score.metrics.validation.roi > 0) profitable++;
    return score.base + deflatedTerm(deflatedSharpeAgainst(score.metrics.validation, ctx.hurdle));
  });
  for (let k = 0; k < missing; k++) assetScores.push(OPTIMISTIC);
  // The primary asset is always scored first, so its trade counts are exact; unscored assets
  // cannot lower the minimum trade rate below what the scored ones already show.
  return combineFitness({
    assetScores,
    primaryTrades: scores[0].metrics.search.trades,
    primaryValidationTrades: scores[0].metrics.validation.trades,
    minTradesPerYear: minTradesPerYear(ctx, scores),
    singleRegime: isSingleRegime(scores[0].metrics.regimes),
    profitableShare: profitable / ctx.assets.length,
    params: countParameters(genome, ctx.limits),
    rewardToRisk: rewardToRisk(genome),
    validationWinRate: scores[0].metrics.validation.winRate,
  });
}

/** Fitness of a fully scored genome against the current selection-bias hurdle. */
export function fitnessOf(ctx: EvalContext, genome: StrategyGenome, scores: AssetScore[]): { fitness: number; dsr: number } {
  let profitable = 0;
  const assetScores = scores.map((score) => {
    if (score.metrics.validation.roi > 0) profitable++;
    return score.base + deflatedTerm(deflatedSharpeAgainst(score.metrics.validation, ctx.hurdle));
  });
  const fitness = combineFitness({
    assetScores,
    primaryTrades: scores[0].metrics.search.trades,
    primaryValidationTrades: scores[0].metrics.validation.trades,
    minTradesPerYear: minTradesPerYear(ctx, scores),
    singleRegime: isSingleRegime(scores[0].metrics.regimes),
    profitableShare: profitable / scores.length,
    params: countParameters(genome, ctx.limits),
    rewardToRisk: rewardToRisk(genome),
    validationWinRate: scores[0].metrics.validation.winRate,
  });
  const dsr = deflatedSharpeAgainst(scores[0].metrics.validation, ctx.hurdle).probability;
  return { fitness, dsr };
}

/** A one-grid-step neighbour of a candidate and its per-asset scores. */
export interface NeighbourScore {
  genome: StrategyGenome;
  scores: AssetScore[];
}

/** Backtest every one-grid-step parameter neighbour of `genome` on the whole universe. */
export function scoreNeighbours(ctx: EvalContext, genome: StrategyGenome): NeighbourScore[] {
  return stepNeighbours(genome, ctx.settings.tradeMode, ctx.limits).map((neighbour) => ({
    genome: neighbour,
    scores: scoreAssets(ctx, neighbour).scores,
  }));
}

/**
 * Rank fitness: the candidate's own fitness capped at the median fitness of its neighbours, all
 * scored against the current hurdle. Returns the point fitness and the neighbour median as well.
 */
export function rankFitness(
  ctx: EvalContext,
  genome: StrategyGenome,
  scores: AssetScore[],
  neighbours: readonly NeighbourScore[] | null,
): { fitness: number; point: number; neighbourMedian: number | null; dsr: number } {
  const { fitness: point, dsr } = fitnessOf(ctx, genome, scores);
  if (neighbours === null || neighbours.length === 0) return { fitness: point, point, neighbourMedian: null, dsr };
  const values = neighbours.map((nb) => fitnessOf(ctx, nb.genome, nb.scores).fitness);
  const fitness = plateauFitness(point, values);
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const neighbourMedian = sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { fitness, point, neighbourMedian, dsr };
}

/** Outcome of evaluating one candidate during the search. */
export interface CandidateResult {
  scores: AssetScore[];
  /** False when scoring stopped early: the candidate provably cannot beat `floor`, or it was skipped. */
  complete: boolean;
  /** Fingerprint of the primary-asset trade log. */
  tradeHash: string | null;
  /** Set when the candidate was not scored at all (it is then not a trial). */
  skip: SkipReason | null;
  /** Rank fitness: `pointFitness` capped by the neighbourhood (equal to it when no neighbours were scored). */
  fitness: number;
  /** Fitness of the exact parameter values. */
  pointFitness: number;
  dsr: number;
  /** One-grid-step neighbours – scored only for candidates whose point fitness beats `floor`. */
  neighbours: NeighbourScore[] | null;
  /** Primary-asset equity curve on the chart axis (search period only) – only for candidates that beat `floor`. */
  curve: Float32Array | null;
}

/**
 * Evaluate one candidate against the frozen hurdle in `ctx` and the leaderboard floor.
 *
 * Candidates whose own fitness beats the floor – the only ones that can enter the leaderboard –
 * also have their one-grid-step parameter neighbourhood backtested, and are ranked by their
 * fitness capped at the neighbours' median (`plateauFitness`). The cap only ever lowers a score,
 * so candidates that cannot beat the floor on their own need no neighbours, and pruning stays exact.
 */
export function evaluateCandidate(ctx: EvalContext, genome: StrategyGenome, floor: number): CandidateResult {
  const { scores, complete, tradeHash, skip } = scoreAssets(ctx, genome, floor, true);
  if (!complete) return { scores, complete, tradeHash, skip, fitness: -Infinity, pointFitness: -Infinity, dsr: 0, neighbours: null, curve: null };
  const { fitness: point, dsr } = fitnessOf(ctx, genome, scores);
  if (!(point > floor) || point <= UNFIT) {
    return { scores, complete, tradeHash, skip, fitness: point, pointFitness: point, dsr, neighbours: null, curve: null };
  }
  // Sample the curve BEFORE the neighbours' backtests overwrite the primary workspace.
  const curve = sampleCurve(ctx.assets[0].ws.equity, ctx.axis, ctx.axis.holdoutPos);
  const neighbours = scoreNeighbours(ctx, genome);
  const { fitness } = rankFitness(ctx, genome, scores, neighbours);
  return { scores, complete, tradeHash, skip, fitness, pointFitness: point, dsr, neighbours, curve: fitness > floor ? curve : null };
}

/** Build a scoring context from raw datasets (the primary asset first). */
export function createContext(datasets: NamedDataset[], settings: BacktestSettings, budgetBytes: number): EvalContext {
  if (datasets.length === 0) throw new Error('No market data was supplied.');
  const perAsset = budgetBytes / datasets.length;
  const assets = datasets.map((dataset) => buildAsset(dataset, settings, perAsset));
  const primary = assets[0];
  return {
    assets,
    settings,
    limits: deriveLimits(primary),
    axis: buildAxis(primary.data.t, primary.split),
    hurdle: 0,
    knownTrades: new Set(),
  };
}

/* -------------------------------------------------------------------------- */
/*  Remote evaluator protocol                                                 */
/* -------------------------------------------------------------------------- */

export interface EvaluatorInit {
  datasets: NamedDataset[];
  settings: BacktestSettings;
  /** Memory budget for this evaluator's indicator cache, in bytes. */
  budgetBytes: number;
}

/** Sent by the main thread to a fresh evaluator worker: its data and the port the optimiser will talk on. */
export interface EvaluatorConnect extends EvaluatorInit {
  type: 'connect';
  port: MessagePort;
}

export interface EvaluatorBatch {
  type: 'batch';
  id: number;
  hurdle: number;
  floor: number;
  /** Trade-log fingerprints registered since the previous batch: every evaluator keeps a full replica. */
  knownTrades: string[];
  genomes: StrategyGenome[];
}

export type EvaluatorReply =
  | { type: 'results'; id: number; results: CandidateResult[] }
  | { type: 'error'; id: number; message: string };

/** Minimal port surface shared by browser MessagePorts and Node's worker_threads ports. */
export interface PortLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent) => void) | null;
  close?(): void;
}

/**
 * Serve evaluation batches on `port`. The context is built on the first batch,
 * so connecting is instant and the data copy happens off the critical path.
 */
export function attachEvaluator(port: PortLike, init: EvaluatorInit): void {
  let ctx: EvalContext | null = null;
  port.onmessage = (event: MessageEvent) => {
    const message = event.data as EvaluatorBatch;
    if (message?.type !== 'batch') return;
    try {
      ctx ??= createContext(init.datasets, init.settings, init.budgetBytes);
      ctx.hurdle = message.hurdle;
      for (const hash of message.knownTrades) ctx.knownTrades.add(hash);
      const results = message.genomes.map((genome) => evaluateCandidate(ctx as EvalContext, genome, message.floor));
      const transfer: Transferable[] = [];
      for (const result of results) if (result.curve) transfer.push(result.curve.buffer as ArrayBuffer);
      const reply: EvaluatorReply = { type: 'results', id: message.id, results };
      port.postMessage(reply, transfer);
    } catch (error) {
      const reply: EvaluatorReply = { type: 'error', id: message.id, message: error instanceof Error ? error.message : String(error) };
      port.postMessage(reply);
    }
  };
}
