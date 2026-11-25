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
  designEffects,
  expectancy,
  median,
  passesFoldGate,
  passesTradeGates,
  ROBUST_MIN_DSR,
  evaluateSearch,
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
import {
  MIN_BORROW_PCT_YEAR,
  MIN_FUNDING_PCT_8H,
  MIN_SLIPPAGE_ATR_MULT,
  type BacktestSettings,
  type ChartAxis,
  type MonteCarloSummary,
  type NamedDataset,
  type OHLCV,
  type RobustChecks,
  type StrategyGenome,
  type TradeGates,
} from '../types/trading';

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
  /**
   * Pooled runs only: in-market flag of every search-period bar of the PRIMARY asset's time axis
   * (the basket's common clock), for the effective number of independent bets.
   */
  inMarket?: Uint8Array;
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
  /**
   * Pooled runs: for every asset, its bar index at each search-period bar of the primary asset's
   * time axis (−1 where the asset has no candle at that time).
   */
  alignment: Int32Array[];
  /** Pooled runs: correlation of bar returns between every pair of assets over the common search period. */
  correlation: number[][];
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
    const score: AssetScore = { base: assetBaseScore(metrics, asset.periodsPerYear), metrics };
    if (ctx.pooled) score.inMarket = inMarketOnAxis(ctx, a);
    scores.push(score);
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

/** In-market flags of asset `a`'s last backtest, on the primary asset's search-period time axis. */
function inMarketOnAxis(ctx: EvalContext, a: number): Uint8Array {
  const asset = ctx.assets[a];
  const map = ctx.alignment[a];
  const out = new Uint8Array(map.length);
  const limit = asset.split.holdoutStart;
  const flags = asset.ws.inMarket;
  for (let i = 0; i < map.length; i++) {
    const k = map[i];
    if (k >= 0 && k < limit) out[i] = flags[k];
  }
  return out;
}

/** The sample a strategy is gated on, raw and effective. */
export interface Sample {
  /** Effective train + validation trades (independent bets): the trade gates use these. */
  trades: number;
  validationTrades: number;
  tradesPerYear: number;
  /** Trades actually opened (summed over the assets of a pooled run). */
  rawTrades: number;
  rawValidationTrades: number;
}

/**
 * The sample a strategy is gated on: the primary asset's trades, or – in a pooled multi-asset run –
 * the trades of every asset, each divided by its design effect (overlap with correlated assets'
 * trades, see `designEffects`), i.e. the effective number of independent bets.
 */
export function sampleOf(ctx: EvalContext, scores: AssetScore[]): Sample {
  const counted = ctx.pooled ? scores : scores.slice(0, 1);
  const design =
    ctx.pooled && counted.every((score) => score.inMarket)
      ? designEffects(
          counted.map((score) => ({ trades: score.metrics.search.trades, inMarket: score.inMarket as Uint8Array })),
          (a, b) => ctx.correlation[a]?.[b] ?? 0,
        )
      : counted.map(() => 1);
  const out: Sample = { trades: 0, validationTrades: 0, tradesPerYear: 0, rawTrades: 0, rawValidationTrades: 0 };
  counted.forEach((score, a) => {
    const d = design[a];
    out.rawTrades += score.metrics.search.trades;
    out.rawValidationTrades += score.metrics.validation.trades;
    out.trades += score.metrics.search.trades / d;
    out.validationTrades += score.metrics.validation.trades / d;
    out.tradesPerYear += tradesPerYear(ctx, a, score.metrics) / d;
  });
  return out;
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

/** Gate-3 inputs, computed in the second evaluation tier for the top contenders only. */
export interface StressInputs {
  monteCarlo: MonteCarloSummary | null;
  /** Median expectancy per trade of the one-step neighbours (pooled over the assets in a pooled run). */
  neighbourExpectancy: number | null;
}

/** Median expectancy per trade over a strategy's one-step neighbours (null without neighbours). */
export function neighbourExpectancy(ctx: EvalContext, neighbours: readonly NeighbourScore[]): number | null {
  if (neighbours.length === 0) return null;
  const pooledExpectancy = (list: AssetScore[]): number => {
    if (!ctx.pooled) return expectancy(list[0].metrics.search);
    const trades = list.reduce((sum, score) => sum + score.metrics.search.trades, 0);
    return trades > 0 ? list.reduce((sum, score) => sum + score.metrics.search.netPnl, 0) / trades : 0;
  };
  return median(neighbours.map((nb) => pooledExpectancy(nb.scores)));
}

/**
 * The three robustness gates of a fully scored candidate (see `RobustChecks`). In a pooled run the
 * fold half of gate 1 must hold on two thirds of the assets and the sample is the effective number
 * of independent bets. `stress` null means "not through the second tier yet": gate 3 is then open
 * (null) and the strategy cannot pass.
 */
export function robustChecks(ctx: EvalContext, scores: AssetScore[], dsr: number, stress: StressInputs | null): RobustChecks {
  const assets = ctx.pooled ? scores : scores.slice(0, 1);
  const folds = assets.filter((score) => passesFoldGate(score.metrics.folds)).length >= Math.ceil((2 / 3) * assets.length - 1e-9);
  const sample = folds && passesTradeGates({ ...sampleOf(ctx, scores), gates: ctx.gates });
  const significance = dsr >= ROBUST_MIN_DSR;
  const stressed =
    stress === null ? null : stress.monteCarlo !== null && stress.monteCarlo.passed && stress.neighbourExpectancy !== null && stress.neighbourExpectancy > 0;
  return { sample, dsr: significance, stress: stressed, passed: sample && significance && stressed === true };
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
  /** Fitness (see `combineFitness`); it does not depend on the neighbourhood or the trial count. */
  fitness: number;
  dsr: number;
  /** Primary-asset equity curve on the chart axis (search period only) – only for candidates that beat `floor`. */
  curve: Float32Array | null;
}

/**
 * FIRST evaluation tier – what the evolutionary loop runs on every candidate, on the pooled
 * evaluator workers: one backtest per asset with the single-pass multi-fold metrics, the fitness and
 * the deflated Sharpe against the frozen hurdle. Nothing expensive: the one-step neighbourhood grid
 * and the Monte Carlo stress test are the SECOND tier, run by the optimiser only on the top
 * contenders (see `optimizer.worker.ts`). Candidates whose fitness beats the leaderboard floor also
 * return their sampled equity curve.
 */
export function evaluateCandidate(ctx: EvalContext, genome: StrategyGenome, floor: number): CandidateResult {
  const { scores, complete, tradeHash, skip } = scoreAssets(ctx, genome, true);
  if (!complete) return { scores, complete, tradeHash, skip, fitness: -Infinity, dsr: 0, curve: null };
  const { fitness, dsr } = fitnessOf(ctx, scores);
  // The primary asset was backtested first, so its curve must be sampled before nothing else ran on it:
  // with several assets the primary workspace is untouched by the later assets' backtests.
  const curve = fitness > floor && fitness > UNFIT ? sampleCurve(ctx.assets[0].ws.equity, ctx.axis, ctx.axis.holdoutPos) : null;
  return { scores, complete, tradeHash, skip, fitness, dsr, curve };
}

/** Index of each of `t`'s bars at the times of `axis` (−1 where `t` has no candle at that time). */
function alignTo(axis: Float64Array, length: number, t: Float64Array): Int32Array {
  const out = new Int32Array(length).fill(-1);
  let k = 0;
  for (let i = 0; i < length; i++) {
    while (k < t.length && t[k] < axis[i]) k++;
    if (k < t.length && t[k] === axis[i]) out[i] = k;
  }
  return out;
}

/** Correlation of two assets' close-to-close returns over the bars both have (0 when too few). */
function returnCorrelation(axisLength: number, a: { map: Int32Array; c: Float64Array }, b: { map: Int32Array; c: Float64Array }): number {
  let n = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (let i = 1; i < axisLength; i++) {
    const a0 = a.map[i - 1];
    const a1 = a.map[i];
    const b0 = b.map[i - 1];
    const b1 = b.map[i];
    if (a0 < 0 || a1 < 0 || b0 < 0 || b1 < 0) continue;
    const ra = a.c[a1] / a.c[a0] - 1;
    const rb = b.c[b1] / b.c[b0] - 1;
    n++;
    sa += ra;
    sb += rb;
    saa += ra * ra;
    sbb += rb * rb;
    sab += ra * rb;
  }
  if (n < 30) return 0;
  const cov = sab / n - (sa / n) * (sb / n);
  const va = saa / n - (sa / n) ** 2;
  const vb = sbb / n - (sb / n) ** 2;
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

/**
 * The settings a search, analysis or forecast really runs with: the execution-realism floors of
 * `MIN_SLIPPAGE_ATR_MULT` / `MIN_FUNDING_PCT_8H` / `MIN_BORROW_PCT_YEAR` applied, so volatility-scaled
 * slippage and short financing can never be switched off. `barMs` is the candle length (financing is
 * charged per candle). A Long/Short run that carries neither funding nor borrow is charged the
 * borrow floor (shorts are never free).
 */
export function enforceRealism(settings: BacktestSettings, barMs: number): BacktestSettings {
  const out: BacktestSettings = { ...settings, slippageAtrMult: Math.max(MIN_SLIPPAGE_ATR_MULT, settings.slippageAtrMult ?? 0) };
  if (settings.tradeMode !== 'long-only') {
    const fundingFloor = MIN_FUNDING_PCT_8H * (barMs / (8 * 3_600_000));
    const borrowFloor = MIN_BORROW_PCT_YEAR * (barMs / (365.25 * 86_400_000));
    const funding = settings.fundingPctPerBar ?? 0;
    const borrow = settings.borrowPctPerBar ?? 0;
    if (funding > 0) out.fundingPctPerBar = Math.max(funding, fundingFloor);
    else out.borrowPctPerBar = Math.max(borrow, borrowFloor);
  }
  return out;
}

/** Build a scoring context from raw datasets (the primary asset first). */
export function createContext(datasets: NamedDataset[], rawSettings: BacktestSettings, budgetBytes: number): EvalContext {
  if (datasets.length === 0) throw new Error('No market data was supplied.');
  const settings = enforceRealism(rawSettings, medianBarMs(datasets[0].data.t));
  const perAsset = budgetBytes / datasets.length;
  const assets = datasets.map((dataset) => buildAsset(dataset, settings, perAsset));
  const primary = assets[0];
  const pooled = assets.length > 1;
  // The basket's common clock: the primary asset's search-period candles.
  const axisLength = primary.split.holdoutStart;
  const alignment = pooled ? assets.map((asset) => alignTo(primary.data.t, axisLength, asset.data.t)) : [];
  const correlation = assets.map((_, a) =>
    assets.map((__, b) =>
      !pooled ? 0 : a === b ? 1 : returnCorrelation(axisLength, { map: alignment[a], c: assets[a].data.c }, { map: alignment[b], c: assets[b].data.c }),
    ),
  );
  return {
    assets,
    settings,
    limits: deriveLimits(primary, settings),
    axis: buildAxis(primary.data.t, primary.split),
    hurdle: 0,
    knownTrades: new Set(),
    gates: tradeGatesFor(primary.split, primary.periodsPerYear),
    // Several assets are always one pooled sample: the same rules on each, the trade gates on the
    // effective number of independent bets.
    pooled,
    alignment,
    correlation,
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
