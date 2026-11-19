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
  expectancy,
  median,
  neighbourhoodMedian,
  passesFoldGate,
  passesTradeGates,
  passesValidationGate,
  ROBUST_MIN_DSR,
  ROBUST_MIN_VALIDATION_SHARPE,
  evaluateSearch,
  plateauFitness,
  splitFor,
  tradeGatesFor,
  tradeLogHash,
  UNFIT,
  type SearchMetrics,
  type Split,
} from './backtester';
import { stepNeighbours, type GenomeLimits } from './genome';
import { IndicatorCache } from './indicators';
import { deflatedSharpeAgainst } from '../lib/stats';
import type { BacktestSettings, ChartAxis, NamedDataset, OHLCV, RobustChecks, StrategyGenome, TradeGates } from '../types/trading';

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
}

/** One asset's search-period metrics and its score (see `assetBaseScore`). */
export interface AssetScore {
  base: number;
  metrics: SearchMetrics;
}

/** What scoring needs: the run's assets (primary first), the settings and the current selection-bias hurdle. */
export interface EvalContext {
  assets: Asset[];
  settings: BacktestSettings;
  limits: GenomeLimits;
  axis: ChartAxis;
  /** Per-bar Sharpe a strategy must beat to count as more than luck (the deflated-Sharpe gate); frozen for a batch. */
  hurdle: number;
  /** Trade-log fingerprints already evaluated in this run (frozen for a batch, see `tradeLogHash`). */
  knownTrades: Set<string>;
  /** Minimum-trade gates, from the primary asset's timeframe and history. */
  gates: TradeGates;
  /** True for a multi-asset run: every asset is traded with the same rules and the trade gates apply to the pooled trades. */
  pooled: boolean;
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

export function deriveLimits(primary: Asset, settings?: Pick<BacktestSettings, 'maPeriods'>): GenomeLimits {
  const volume = primary.data.v;
  let withVolume = 0;
  for (let i = 0; i < volume.length; i++) if (volume[i] > 0) withVolume++;
  return {
    // Lookbacks are capped so indicators are warmed up for most of the training segment.
    maxLookback: Math.max(30, Math.min(200, Math.floor(primary.split.validationStart / 3))),
    hasVolume: withVolume > volume.length * 0.5,
    barMs: medianBarMs(primary.data.t),
    maPeriods: settings?.maPeriods ?? 'grid',
  };
}

/** Typical candle length: the median gap between consecutive open times (robust to weekends and gaps). */
function medianBarMs(t: Float64Array): number {
  const n = Math.min(t.length - 1, 2000);
  if (n < 1) return 86_400_000;
  const gaps = new Float64Array(n);
  for (let i = 0; i < n; i++) gaps[i] = t[t.length - n + i] - t[t.length - n + i - 1];
  gaps.sort();
  return gaps[n >> 1];
}

/* -------------------------------------------------------------------------- */
/*  Scoring                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Backtest `genome` on the search period of every asset, primary asset first. With `dedupe` the
 * primary asset decides first whether this is a new bet at all: a strategy that never trades, or
 * repeats a trade log already evaluated, is skipped before any other asset is backtested.
 */
export function scoreAssets(
  ctx: EvalContext,
  genome: StrategyGenome,
  dedupe = false,
): { scores: AssetScore[]; complete: boolean; tradeHash: string | null; skip: SkipReason | null } {
  const scores: AssetScore[] = [];
  let tradeHash: string | null = null;
  for (let a = 0; a < ctx.assets.length; a++) {
    const asset = ctx.assets[a];
    const metrics = evaluateSearch(asset.ind, genome, ctx.settings, asset.ws, asset.split, asset.periodsPerYear);
    scores.push({ base: assetBaseScore(metrics, asset.periodsPerYear), metrics });
    if (a === 0 && dedupe) {
      tradeHash = tradeLogHash(asset.ws, asset.split.holdoutStart);
      if (metrics.search.trades === 0) return { scores, complete: false, tradeHash, skip: 'inactive' };
      if (ctx.knownTrades.has(tradeHash)) return { scores, complete: false, tradeHash, skip: 'duplicate' };
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

/**
 * The sample a strategy is gated on: the primary asset's trades, or – in a pooled multi-asset run –
 * the trades summed over every asset.
 */
export function sampleOf(ctx: EvalContext, scores: AssetScore[]): { trades: number; validationTrades: number; tradesPerYear: number } {
  let trades = 0;
  let validationTrades = 0;
  let perYear = 0;
  const counted = ctx.pooled ? scores : scores.slice(0, 1);
  counted.forEach((score, a) => {
    trades += score.metrics.search.trades;
    validationTrades += score.metrics.validation.trades;
    perYear += tradesPerYear(ctx, a, score.metrics);
  });
  return { trades, validationTrades, tradesPerYear: perYear };
}

/**
 * Point fitness of a fully scored genome (see `combineFitness`) and its deflated Sharpe ratio
 * against the current selection-bias hurdle – the primary asset's, or in a pooled run the MEDIAN
 * over the assets, so pooling raises the sample size for the gates without letting one lucky asset
 * carry the statistic. The fitness itself does not depend on the hurdle.
 */
export function fitnessOf(ctx: EvalContext, scores: AssetScore[]): { fitness: number; dsr: number } {
  const fitness = combineFitness({ assetScores: scores.map((score) => score.base), ...sampleOf(ctx, scores), gates: ctx.gates });
  const dsrs = (ctx.pooled ? scores : scores.slice(0, 1)).map((score) => deflatedSharpeAgainst(score.metrics.validation, ctx.hurdle).probability);
  return { fitness, dsr: median(dsrs) };
}

/** A one-grid-step neighbour of a candidate and its per-asset scores. */
export interface NeighbourScore {
  genome: StrategyGenome;
  scores: AssetScore[];
}

/** Backtest every one-grid-step parameter neighbour of `genome` on every asset of the run. */
export function scoreNeighbours(ctx: EvalContext, genome: StrategyGenome): NeighbourScore[] {
  return stepNeighbours(genome, ctx.settings.tradeMode, ctx.limits).map((neighbour) => ({
    genome: neighbour,
    scores: scoreAssets(ctx, neighbour).scores,
  }));
}

/**
 * Rank fitness: the candidate's own fitness blended towards the median fitness of its one-step
 * neighbours when they are worse (see `plateauFitness`). Returns the point fitness, the neighbour
 * median and the deflated Sharpe as well.
 */
export function rankFitness(
  ctx: EvalContext,
  scores: AssetScore[],
  neighbours: readonly NeighbourScore[] | null,
): { fitness: number; point: number; neighbourMedian: number | null; dsr: number } {
  const { fitness: point, dsr } = fitnessOf(ctx, scores);
  if (neighbours === null || neighbours.length === 0) return { fitness: point, point, neighbourMedian: null, dsr };
  const values = neighbours.map((nb) => fitnessOf(ctx, nb.scores).fitness);
  return { fitness: plateauFitness(point, values), point, neighbourMedian: neighbourhoodMedian(point, values), dsr };
}

/**
 * The robustness gates of a fully scored candidate (see `RobustChecks`). In a pooled run the
 * validation gate uses the median asset's Sharpe and the pooled net profit, the fold gate must hold
 * on two thirds of the assets, and the neighbours' expectancy is pooled.
 * `neighbours` null means "not checked yet": the neighbourhood gate then fails.
 */
export function robustChecks(
  ctx: EvalContext,
  scores: AssetScore[],
  dsr: number,
  neighbours: readonly NeighbourScore[] | null,
): RobustChecks {
  const assets = ctx.pooled ? scores : scores.slice(0, 1);
  const validation = ctx.pooled
    ? median(assets.map((score) => score.metrics.validation.sharpe)) > ROBUST_MIN_VALIDATION_SHARPE &&
      assets.reduce((sum, score) => sum + score.metrics.validation.netPnl, 0) > 0
    : passesValidationGate(assets[0].metrics.validation);
  const folds = assets.filter((score) => passesFoldGate(score.metrics.folds)).length >= Math.ceil((2 / 3) * assets.length - 1e-9);
  const pooledExpectancy = (list: AssetScore[]): number => {
    if (!ctx.pooled) return expectancy(list[0].metrics.search);
    const trades = list.reduce((sum, score) => sum + score.metrics.search.trades, 0);
    return trades > 0 ? list.reduce((sum, score) => sum + score.metrics.search.netPnl, 0) / trades : 0;
  };
  const neighbourhood =
    neighbours !== null && neighbours.length > 0 && median(neighbours.map((nb) => pooledExpectancy(nb.scores))) > 0;
  const significance = passesTradeGates({ ...sampleOf(ctx, scores), gates: ctx.gates }) && dsr >= ROBUST_MIN_DSR;
  return { validation, folds, neighbours: neighbourhood, significance, passed: validation && folds && neighbourhood && significance };
}

/** Outcome of evaluating one candidate during the search. */
export interface CandidateResult {
  scores: AssetScore[];
  /** False when the candidate was skipped (see `skip`). */
  complete: boolean;
  /** Fingerprint of the primary-asset trade log. */
  tradeHash: string | null;
  /** Set when the candidate was not scored at all (it is then not a trial). */
  skip: SkipReason | null;
  /** Rank fitness: `pointFitness` blended with the neighbourhood (equal to it when no neighbours were scored). */
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
 * Every asset is scored – the robustness gates are not fitness-based, so a candidate that cannot
 * make the leaderboard may still be a robust strategy. Candidates whose own fitness beats the floor
 * – the only ones that can enter the leaderboard – also have their one-grid-step parameter
 * neighbourhood backtested, and are ranked by their fitness blended with the neighbours' median
 * (`plateauFitness`). The blend only ever lowers a score, so candidates that cannot beat the floor
 * on their own need no neighbours.
 */
export function evaluateCandidate(ctx: EvalContext, genome: StrategyGenome, floor: number): CandidateResult {
  const { scores, complete, tradeHash, skip } = scoreAssets(ctx, genome, true);
  if (!complete) return { scores, complete, tradeHash, skip, fitness: -Infinity, pointFitness: -Infinity, dsr: 0, neighbours: null, curve: null };
  const { fitness: point, dsr } = fitnessOf(ctx, scores);
  if (!(point > floor) || point <= UNFIT) {
    return { scores, complete, tradeHash, skip, fitness: point, pointFitness: point, dsr, neighbours: null, curve: null };
  }
  // Sample the curve BEFORE the neighbours' backtests overwrite the primary workspace.
  const curve = sampleCurve(ctx.assets[0].ws.equity, ctx.axis, ctx.axis.holdoutPos);
  const neighbours = scoreNeighbours(ctx, genome);
  const { fitness } = rankFitness(ctx, scores, neighbours);
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
    limits: deriveLimits(primary, settings),
    axis: buildAxis(primary.data.t, primary.split),
    hurdle: 0,
    knownTrades: new Set(),
    gates: tradeGatesFor(primary.split, primary.periodsPerYear),
    // Several assets are always one pooled sample: the same rules on each, the trade gates on the total.
    pooled: assets.length > 1,
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
