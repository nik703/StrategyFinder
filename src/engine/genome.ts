/**
 * Strategy genome: random generation, mutation, crossover and repair.
 *
 * A genome is plain JSON (see `StrategyGenome`), so it can be structured-cloned
 * between the worker and the UI, rendered as plain English and exported to Pine.
 * Every operator in this file returns a *valid* genome: parameters are clamped
 * to their documented ranges and structural constraints (fast < slow, at least
 * one exit path, no shorts in Long-Only mode, ...) are enforced by `repairGenome`.
 */

import type {
  CmpOp,
  Condition,
  DynamicExit,
  ExitGene,
  IndicatorFamily,
  MaType,
  RuleGroup,
  SizingGene,
  StrategyGenome,
  TradeMode,
} from '../types/trading';

/* -------------------------------------------------------------------------- */
/*  Random number generation                                                  */
/* -------------------------------------------------------------------------- */

export type Rng = () => number;

/** Small, fast, seedable PRNG (uniform in [0, 1)). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample (Box–Muller). */
export function gaussian(rng: Rng): number {
  let u = rng();
  if (u < 1e-12) u = 1e-12;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/* -------------------------------------------------------------------------- */
/*  Parameter ranges                                                          */
/* -------------------------------------------------------------------------- */

/** Search-space bounds. Lookbacks are additionally capped by `GenomeLimits.maxLookback`. */
export const RANGES = {
  rsiPeriod: [7, 28],
  rsiLevel: [10, 90],
  macdFast: [5, 20],
  macdSlow: [18, 60],
  macdSignal: [5, 15],
  maPeriod: [5, 250],
  maFast: [5, 60],
  maSlow: [20, 250],
  bbPeriod: [10, 50],
  bbMult: [1.5, 3],
  bbPercentB: [-0.2, 1.2],
  bbBandwidth: [0.2, 100],
  atrPeriod: [7, 28],
  atrPercent: [0.05, 30],
  stochK: [5, 21],
  stochSmooth: [1, 5],
  stochD: [3, 7],
  stochLevel: [5, 95],
  adxPeriod: [7, 28],
  adxLevel: [10, 50],
  volumePeriod: [10, 50],
  volumeRatio: [0.5, 3.5],
  supertrendPeriod: [7, 21],
  supertrendMult: [1, 5],
  cciPeriod: [10, 40],
  cciLevel: [-250, 250],
  stopLossPct: [0.5, 15],
  takeProfitPct: [1, 60],
  stopAtrMult: [1, 4],
  takeProfitAtrMult: [1.5, 8],
  riskPct: [0.5, 5],
} as const satisfies Record<string, readonly [number, number]>;

export const MAX_CONDITIONS = 4;
const MAX_EXIT_CONDITIONS = 2;

/** Dataset-dependent limits for the search space. */
export interface GenomeLimits {
  /** Longest indicator lookback allowed (keeps warm-up short relative to the sample). */
  maxLookback: number;
  /** False when the dataset has no usable volume (volume conditions are then excluded). */
  hasVolume: boolean;
  /** Typical in-sample range (5th–95th percentile) of ATR as % of price. */
  atrPercentRange: readonly [number, number];
  /** Typical in-sample range (5th–95th percentile) of Bollinger Bandwidth %. */
  bandwidthRange: readonly [number, number];
}

export const DEFAULT_LIMITS: GenomeLimits = {
  maxLookback: 250,
  hasVolume: true,
  atrPercentRange: [0.5, 6],
  bandwidthRange: [2, 30],
};

/* -------------------------------------------------------------------------- */
/*  Numeric helpers                                                           */
/* -------------------------------------------------------------------------- */

const clamp = (v: number, min: number, max: number): number => (v < min ? min : v > max ? max : v);

function quantize(v: number, step: number): number {
  const q = Math.round(v / step) * step;
  return Number(q.toFixed(6));
}

const randInt = (rng: Rng, min: number, max: number): number =>
  max <= min ? min : min + Math.floor(rng() * (max - min + 1));

const randStep = (rng: Rng, min: number, max: number, step: number): number =>
  clamp(quantize(min + rng() * (max - min), step), min, max);

/** Log-uniform integer – favours short lookbacks while still reaching long ones. */
function randLogInt(rng: Rng, min: number, max: number): number {
  if (max <= min) return min;
  return clamp(Math.round(Math.exp(Math.log(min) + rng() * (Math.log(max) - Math.log(min)))), min, max);
}

function randLogStep(rng: Rng, min: number, max: number, step: number): number {
  const lo = Math.max(min, step);
  return clamp(quantize(Math.exp(Math.log(lo) + rng() * (Math.log(max) - Math.log(lo))), step), lo, max);
}

function nudgeInt(rng: Rng, v: number, min: number, max: number): number {
  if (max <= min) return min;
  let delta = Math.round(gaussian(rng) * Math.max(1, (max - min) * 0.12));
  if (delta === 0) delta = rng() < 0.5 ? -1 : 1;
  const next = clamp(v + delta, min, max);
  return next === v ? clamp(v - delta, min, max) : next;
}

function nudgeStep(rng: Rng, v: number, min: number, max: number, step: number): number {
  let delta = quantize(gaussian(rng) * (max - min) * 0.12, step);
  if (delta === 0) delta = rng() < 0.5 ? -step : step;
  const next = clamp(quantize(v + delta, step), min, max);
  return next === v ? clamp(quantize(v - delta, step), min, max) : next;
}

/** Multiplicative nudge for strictly positive, scale-free parameters. */
function nudgeLog(rng: Rng, v: number, min: number, max: number, step: number): number {
  const next = clamp(quantize(v * Math.exp(gaussian(rng) * 0.25), step), min, max);
  if (next !== v) return next;
  return clamp(quantize(v + (rng() < 0.5 ? -step : step), step), min, max);
}

function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)];
}

function pickWeighted<T>(rng: Rng, items: readonly (readonly [T, number])[]): T {
  let total = 0;
  for (const [, w] of items) total += w;
  let r = rng() * total;
  for (const [item, w] of items) {
    r -= w;
    if (r < 0) return item;
  }
  return items[items.length - 1][0];
}

const cap = (max: number, limits: GenomeLimits): number => Math.max(5, Math.min(max, limits.maxLookback));

/* -------------------------------------------------------------------------- */
/*  Condition semantics                                                       */
/* -------------------------------------------------------------------------- */

const ALL_OPS: readonly CmpOp[] = ['gt', 'lt', 'cross_above', 'cross_below'];
const LEVEL_OPS: readonly CmpOp[] = ['gt', 'lt'];
const MA_TYPES: readonly MaType[] = ['EMA', 'SMA'];

export function flipOp(op: CmpOp): CmpOp {
  switch (op) {
    case 'gt':
      return 'lt';
    case 'lt':
      return 'gt';
    case 'cross_above':
      return 'cross_below';
    case 'cross_below':
      return 'cross_above';
  }
}

/** Filter-style conditions are direction-agnostic and only support `>` / `<`. */
export function isFilter(c: Condition): boolean {
  return (
    c.ind === 'ATR' ||
    c.ind === 'VOLUME' ||
    (c.ind === 'ADX' && c.variant === 'strength') ||
    (c.ind === 'BB' && c.variant === 'bandwidth')
  );
}

function allowedOps(c: Condition): readonly CmpOp[] {
  return isFilter(c) ? LEVEL_OPS : ALL_OPS;
}

/**
 * Mirror a condition into its bearish/bullish opposite. Used to derive short
 * entries from long entries and to build indicator-reversal exit rules.
 * Direction-agnostic filters are returned unchanged.
 */
export function mirrorCondition(c: Condition): Condition {
  switch (c.ind) {
    case 'RSI':
      return { ...c, op: flipOp(c.op), level: 100 - c.level };
    case 'MACD':
    case 'PRICE_MA':
    case 'MA_CROSS':
    case 'SUPERTREND':
      return { ...c, op: flipOp(c.op) };
    case 'BB':
      if (c.variant === 'pctb') return { ...c, op: flipOp(c.op), level: quantize(1 - c.level, 0.05) };
      if (c.variant === 'upper') return { ...c, variant: 'lower', op: flipOp(c.op) };
      if (c.variant === 'lower') return { ...c, variant: 'upper', op: flipOp(c.op) };
      return { ...c };
    case 'STOCH':
      return c.variant === 'level' ? { ...c, op: flipOp(c.op), level: 100 - c.level } : { ...c, op: flipOp(c.op) };
    case 'ADX':
      return c.variant === 'di' ? { ...c, op: flipOp(c.op) } : { ...c };
    case 'CCI':
      return { ...c, op: flipOp(c.op), level: -c.level };
    case 'ATR':
    case 'VOLUME':
      return { ...c };
  }
}

export function mirrorGroup(group: RuleGroup): RuleGroup {
  return { logic: group.logic, conditions: group.conditions.map(mirrorCondition) };
}

/* -------------------------------------------------------------------------- */
/*  Random conditions                                                         */
/* -------------------------------------------------------------------------- */

const FAMILY_WEIGHTS: readonly (readonly [IndicatorFamily, number])[] = [
  ['RSI', 1.2],
  ['MACD', 1],
  ['PRICE_MA', 1.2],
  ['MA_CROSS', 1],
  ['BB', 1],
  ['ATR', 0.45],
  ['STOCH', 0.9],
  ['ADX', 0.8],
  ['VOLUME', 0.7],
  ['SUPERTREND', 1],
  ['CCI', 0.8],
];

function randomFamily(rng: Rng, limits: GenomeLimits, directionalOnly: boolean): IndicatorFamily {
  const pool = FAMILY_WEIGHTS.filter(([family]) => {
    if (family === 'VOLUME') return limits.hasVolume && !directionalOnly;
    if (family === 'ATR') return !directionalOnly;
    return true;
  });
  return pickWeighted(rng, pool);
}

/** Level that makes sense for a bounded 0–100 oscillator given the operator. */
function oscillatorLevel(rng: Rng, op: CmpOp, low: number, high: number): number {
  const mid = Math.round((low + high) / 2);
  if (op === 'lt') return randInt(rng, low, mid);
  if (op === 'gt') return randInt(rng, mid, high);
  return randInt(rng, low, high);
}

export function randomCondition(
  rng: Rng,
  limits: GenomeLimits,
  options: { family?: IndicatorFamily; directionalOnly?: boolean } = {},
): Condition {
  const directionalOnly = options.directionalOnly ?? false;
  const family = options.family ?? randomFamily(rng, limits, directionalOnly);
  switch (family) {
    case 'RSI': {
      const op = pickWeighted<CmpOp>(rng, [
        ['lt', 0.35],
        ['gt', 0.35],
        ['cross_above', 0.15],
        ['cross_below', 0.15],
      ]);
      return {
        ind: 'RSI',
        period: randInt(rng, RANGES.rsiPeriod[0], RANGES.rsiPeriod[1]),
        op,
        level: oscillatorLevel(rng, op, 15, 85),
      };
    }
    case 'MACD': {
      const fast = randInt(rng, RANGES.macdFast[0], RANGES.macdFast[1]);
      const slowMax = Math.max(fast + 5, cap(RANGES.macdSlow[1], limits));
      const slow = randInt(rng, Math.max(fast + 5, RANGES.macdSlow[0]), slowMax);
      return {
        ind: 'MACD',
        fast,
        slow,
        signal: randInt(rng, RANGES.macdSignal[0], RANGES.macdSignal[1]),
        variant: pickWeighted(rng, [
          ['signal', 0.5],
          ['zero', 0.25],
          ['hist', 0.25],
        ] as const),
        op: pick(rng, ALL_OPS),
      };
    }
    case 'PRICE_MA':
      return {
        ind: 'PRICE_MA',
        maType: pick(rng, MA_TYPES),
        period: randLogInt(rng, RANGES.maPeriod[0], cap(RANGES.maPeriod[1], limits)),
        op: pick(rng, ALL_OPS),
      };
    case 'MA_CROSS': {
      const slowMax = cap(RANGES.maSlow[1], limits);
      const fastPeriod = randLogInt(rng, RANGES.maFast[0], Math.min(RANGES.maFast[1], slowMax - 5));
      return {
        ind: 'MA_CROSS',
        fastType: pick(rng, MA_TYPES),
        fastPeriod,
        slowType: pick(rng, MA_TYPES),
        slowPeriod: randLogInt(rng, Math.max(fastPeriod + 5, RANGES.maSlow[0]), slowMax),
        op: pick(rng, ALL_OPS),
      };
    }
    case 'BB': {
      const variant = directionalOnly
        ? pickWeighted(rng, [
            ['pctb', 0.5],
            ['upper', 0.25],
            ['lower', 0.25],
          ] as const)
        : pickWeighted(rng, [
            ['pctb', 0.4],
            ['bandwidth', 0.15],
            ['upper', 0.2],
            ['lower', 0.25],
          ] as const);
      const period = randInt(rng, RANGES.bbPeriod[0], cap(RANGES.bbPeriod[1], limits));
      const mult = randStep(rng, RANGES.bbMult[0], RANGES.bbMult[1], 0.1);
      if (variant === 'bandwidth') {
        return {
          ind: 'BB',
          period,
          mult,
          variant,
          op: pick(rng, LEVEL_OPS),
          level: randLogStep(rng, limits.bandwidthRange[0], limits.bandwidthRange[1], 0.1),
        };
      }
      return {
        ind: 'BB',
        period,
        mult,
        variant,
        op: pick(rng, ALL_OPS),
        level: variant === 'pctb' ? randStep(rng, RANGES.bbPercentB[0], RANGES.bbPercentB[1], 0.05) : 0,
      };
    }
    case 'ATR':
      return {
        ind: 'ATR',
        period: randInt(rng, RANGES.atrPeriod[0], RANGES.atrPeriod[1]),
        op: pick(rng, LEVEL_OPS),
        level: randLogStep(rng, limits.atrPercentRange[0], limits.atrPercentRange[1], 0.05),
      };
    case 'STOCH': {
      const variant = pickWeighted(rng, [
        ['level', 0.55],
        ['kd', 0.45],
      ] as const);
      const op = pick(rng, ALL_OPS);
      return {
        ind: 'STOCH',
        kPeriod: randInt(rng, RANGES.stochK[0], RANGES.stochK[1]),
        smooth: randInt(rng, RANGES.stochSmooth[0], RANGES.stochSmooth[1]),
        dPeriod: randInt(rng, RANGES.stochD[0], RANGES.stochD[1]),
        variant,
        op,
        level: oscillatorLevel(rng, op, 10, 90),
      };
    }
    case 'ADX': {
      const variant = directionalOnly
        ? 'di'
        : pickWeighted(rng, [
            ['strength', 0.6],
            ['di', 0.4],
          ] as const);
      return {
        ind: 'ADX',
        period: randInt(rng, RANGES.adxPeriod[0], RANGES.adxPeriod[1]),
        variant,
        op:
          variant === 'strength'
            ? pickWeighted<CmpOp>(rng, [
                ['gt', 0.75],
                ['lt', 0.25],
              ])
            : pick(rng, ALL_OPS),
        level: randInt(rng, 15, 40),
      };
    }
    case 'VOLUME': {
      const op = pickWeighted<CmpOp>(rng, [
        ['gt', 0.8],
        ['lt', 0.2],
      ]);
      return {
        ind: 'VOLUME',
        period: randInt(rng, RANGES.volumePeriod[0], cap(RANGES.volumePeriod[1], limits)),
        op,
        level: op === 'gt' ? randStep(rng, 1, 2.5, 0.05) : randStep(rng, 0.5, 1.2, 0.05),
      };
    }
    case 'SUPERTREND':
      return {
        ind: 'SUPERTREND',
        period: randInt(rng, RANGES.supertrendPeriod[0], RANGES.supertrendPeriod[1]),
        mult: randStep(rng, RANGES.supertrendMult[0], RANGES.supertrendMult[1], 0.1),
        op: pick(rng, ALL_OPS),
      };
    case 'CCI':
      return {
        ind: 'CCI',
        period: randInt(rng, RANGES.cciPeriod[0], cap(RANGES.cciPeriod[1], limits)),
        op: pick(rng, ALL_OPS),
        level: randStep(rng, -200, 200, 5),
      };
  }
}

/* -------------------------------------------------------------------------- */
/*  Condition mutation                                                        */
/* -------------------------------------------------------------------------- */

/** Nudge one numeric parameter (lookback, threshold or multiplier) of a condition. */
export function tweakCondition(rng: Rng, c: Condition, limits: GenomeLimits): Condition {
  switch (c.ind) {
    case 'RSI':
      return rng() < 0.5
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.rsiPeriod[0], RANGES.rsiPeriod[1]) }
        : { ...c, level: nudgeInt(rng, c.level, RANGES.rsiLevel[0], RANGES.rsiLevel[1]) };
    case 'MACD': {
      const r = rng();
      if (r < 0.4) return { ...c, fast: nudgeInt(rng, c.fast, RANGES.macdFast[0], RANGES.macdFast[1]) };
      if (r < 0.8) return { ...c, slow: nudgeInt(rng, c.slow, RANGES.macdSlow[0], cap(RANGES.macdSlow[1], limits)) };
      return { ...c, signal: nudgeInt(rng, c.signal, RANGES.macdSignal[0], RANGES.macdSignal[1]) };
    }
    case 'PRICE_MA':
      return rng() < 0.85
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.maPeriod[0], cap(RANGES.maPeriod[1], limits)) }
        : { ...c, maType: c.maType === 'EMA' ? 'SMA' : 'EMA' };
    case 'MA_CROSS': {
      const r = rng();
      if (r < 0.4) return { ...c, fastPeriod: nudgeInt(rng, c.fastPeriod, RANGES.maFast[0], RANGES.maFast[1]) };
      if (r < 0.8) {
        return { ...c, slowPeriod: nudgeInt(rng, c.slowPeriod, RANGES.maSlow[0], cap(RANGES.maSlow[1], limits)) };
      }
      return r < 0.9
        ? { ...c, fastType: c.fastType === 'EMA' ? 'SMA' : 'EMA' }
        : { ...c, slowType: c.slowType === 'EMA' ? 'SMA' : 'EMA' };
    }
    case 'BB': {
      const r = rng();
      const hasLevel = c.variant === 'pctb' || c.variant === 'bandwidth';
      if (hasLevel && r < 0.45) {
        return c.variant === 'pctb'
          ? { ...c, level: nudgeStep(rng, c.level, RANGES.bbPercentB[0], RANGES.bbPercentB[1], 0.05) }
          : { ...c, level: nudgeLog(rng, c.level, RANGES.bbBandwidth[0], RANGES.bbBandwidth[1], 0.1) };
      }
      return r < 0.75
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.bbPeriod[0], cap(RANGES.bbPeriod[1], limits)) }
        : { ...c, mult: nudgeStep(rng, c.mult, RANGES.bbMult[0], RANGES.bbMult[1], 0.1) };
    }
    case 'ATR':
      return rng() < 0.4
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.atrPeriod[0], RANGES.atrPeriod[1]) }
        : { ...c, level: nudgeLog(rng, c.level, RANGES.atrPercent[0], RANGES.atrPercent[1], 0.05) };
    case 'STOCH': {
      const r = rng();
      if (c.variant === 'level' && r < 0.45) {
        return { ...c, level: nudgeInt(rng, c.level, RANGES.stochLevel[0], RANGES.stochLevel[1]) };
      }
      if (c.variant === 'kd' && r < 0.25) {
        return { ...c, dPeriod: nudgeInt(rng, c.dPeriod, RANGES.stochD[0], RANGES.stochD[1]) };
      }
      return r < 0.8
        ? { ...c, kPeriod: nudgeInt(rng, c.kPeriod, RANGES.stochK[0], RANGES.stochK[1]) }
        : { ...c, smooth: nudgeInt(rng, c.smooth, RANGES.stochSmooth[0], RANGES.stochSmooth[1]) };
    }
    case 'ADX':
      return c.variant === 'strength' && rng() < 0.5
        ? { ...c, level: nudgeInt(rng, c.level, RANGES.adxLevel[0], RANGES.adxLevel[1]) }
        : { ...c, period: nudgeInt(rng, c.period, RANGES.adxPeriod[0], RANGES.adxPeriod[1]) };
    case 'VOLUME':
      return rng() < 0.4
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.volumePeriod[0], cap(RANGES.volumePeriod[1], limits)) }
        : { ...c, level: nudgeStep(rng, c.level, RANGES.volumeRatio[0], RANGES.volumeRatio[1], 0.05) };
    case 'SUPERTREND':
      return rng() < 0.5
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.supertrendPeriod[0], RANGES.supertrendPeriod[1]) }
        : { ...c, mult: nudgeStep(rng, c.mult, RANGES.supertrendMult[0], RANGES.supertrendMult[1], 0.1) };
    case 'CCI':
      return rng() < 0.4
        ? { ...c, period: nudgeInt(rng, c.period, RANGES.cciPeriod[0], cap(RANGES.cciPeriod[1], limits)) }
        : { ...c, level: nudgeStep(rng, c.level, RANGES.cciLevel[0], RANGES.cciLevel[1], 5) };
  }
}

/** Change the operator, or switch to another variant of the same indicator. */
export function alterCondition(rng: Rng, c: Condition, limits: GenomeLimits): Condition {
  const changeVariant = rng() < 0.4;
  if (changeVariant) {
    switch (c.ind) {
      case 'MACD': {
        const variant = pick(rng, (['signal', 'zero', 'hist'] as const).filter((v) => v !== c.variant));
        return { ...c, variant };
      }
      case 'BB': {
        const variant = pick(rng, (['pctb', 'bandwidth', 'upper', 'lower'] as const).filter((v) => v !== c.variant));
        const fresh = randomCondition(rng, limits, { family: 'BB' });
        const level =
          variant === 'pctb'
            ? randStep(rng, RANGES.bbPercentB[0], RANGES.bbPercentB[1], 0.05)
            : variant === 'bandwidth'
              ? randLogStep(rng, limits.bandwidthRange[0], limits.bandwidthRange[1], 0.1)
              : 0;
        const op = variant === 'bandwidth' ? pick(rng, LEVEL_OPS) : fresh.op;
        return { ...c, variant, op, level };
      }
      case 'STOCH':
        return { ...c, variant: c.variant === 'level' ? 'kd' : 'level' };
      case 'ADX': {
        const variant = c.variant === 'strength' ? 'di' : 'strength';
        return { ...c, variant, op: variant === 'strength' ? pick(rng, LEVEL_OPS) : c.op };
      }
      default:
        break;
    }
  }
  const ops = allowedOps(c).filter((op) => op !== c.op);
  return { ...c, op: pick(rng, ops) };
}

/** Clamp every parameter into range and fix structural constraints. */
export function repairCondition(c: Condition, limits: GenomeLimits): Condition {
  const int = (v: number, range: readonly [number, number], maxOverride?: number): number =>
    clamp(Math.round(v), range[0], Math.max(range[0], maxOverride ?? range[1]));
  const op: CmpOp = isFilter(c) && (c.op === 'cross_above' || c.op === 'cross_below') ? 'gt' : c.op;
  switch (c.ind) {
    case 'RSI':
      return { ...c, op, period: int(c.period, RANGES.rsiPeriod), level: int(c.level, RANGES.rsiLevel) };
    case 'MACD': {
      const fast = int(c.fast, RANGES.macdFast);
      const slowMax = Math.max(fast + 3, cap(RANGES.macdSlow[1], limits));
      const slow = clamp(Math.round(c.slow), Math.max(fast + 3, RANGES.macdSlow[0]), slowMax);
      return { ...c, op, fast, slow, signal: int(c.signal, RANGES.macdSignal) };
    }
    case 'PRICE_MA':
      return { ...c, op, period: int(c.period, RANGES.maPeriod, cap(RANGES.maPeriod[1], limits)) };
    case 'MA_CROSS': {
      const slowMax = cap(RANGES.maSlow[1], limits);
      const fastPeriod = clamp(Math.round(c.fastPeriod), RANGES.maFast[0], Math.min(RANGES.maFast[1], slowMax - 3));
      const slowPeriod = clamp(Math.round(c.slowPeriod), Math.max(fastPeriod + 3, RANGES.maSlow[0]), slowMax);
      return { ...c, op, fastPeriod, slowPeriod };
    }
    case 'BB': {
      const level =
        c.variant === 'pctb'
          ? clamp(quantize(c.level, 0.05), RANGES.bbPercentB[0], RANGES.bbPercentB[1])
          : c.variant === 'bandwidth'
            ? clamp(quantize(c.level, 0.1), RANGES.bbBandwidth[0], RANGES.bbBandwidth[1])
            : 0;
      return {
        ...c,
        op,
        period: int(c.period, RANGES.bbPeriod, cap(RANGES.bbPeriod[1], limits)),
        mult: clamp(quantize(c.mult, 0.1), RANGES.bbMult[0], RANGES.bbMult[1]),
        level,
      };
    }
    case 'ATR':
      return {
        ...c,
        op,
        period: int(c.period, RANGES.atrPeriod),
        level: clamp(quantize(c.level, 0.05), RANGES.atrPercent[0], RANGES.atrPercent[1]),
      };
    case 'STOCH':
      return {
        ...c,
        op,
        kPeriod: int(c.kPeriod, RANGES.stochK),
        smooth: int(c.smooth, RANGES.stochSmooth),
        dPeriod: int(c.dPeriod, RANGES.stochD),
        level: int(c.level, RANGES.stochLevel),
      };
    case 'ADX':
      return { ...c, op, period: int(c.period, RANGES.adxPeriod), level: int(c.level, RANGES.adxLevel) };
    case 'VOLUME':
      return {
        ...c,
        op,
        period: int(c.period, RANGES.volumePeriod, cap(RANGES.volumePeriod[1], limits)),
        level: clamp(quantize(c.level, 0.05), RANGES.volumeRatio[0], RANGES.volumeRatio[1]),
      };
    case 'SUPERTREND':
      return {
        ...c,
        op,
        period: int(c.period, RANGES.supertrendPeriod),
        mult: clamp(quantize(c.mult, 0.1), RANGES.supertrendMult[0], RANGES.supertrendMult[1]),
      };
    case 'CCI':
      return {
        ...c,
        op,
        period: int(c.period, RANGES.cciPeriod, cap(RANGES.cciPeriod[1], limits)),
        level: clamp(quantize(c.level, 5), RANGES.cciLevel[0], RANGES.cciLevel[1]),
      };
  }
}

/* -------------------------------------------------------------------------- */
/*  Rule groups                                                               */
/* -------------------------------------------------------------------------- */

function conditionKey(c: Condition): string {
  const record = c as unknown as Record<string, string | number>;
  return Object.keys(record)
    .sort()
    .map((k) => `${k}=${record[k]}`)
    .join(',');
}

function groupKey(group: RuleGroup | null): string {
  if (group === null) return '-';
  return `${group.logic}[${group.conditions.map(conditionKey).sort().join(';')}]`;
}

function cloneGroup(group: RuleGroup): RuleGroup {
  return { logic: group.logic, conditions: group.conditions.map((c) => ({ ...c })) };
}

export function randomGroup(rng: Rng, limits: GenomeLimits): RuleGroup {
  const count = pickWeighted(rng, [
    [1, 0.22],
    [2, 0.42],
    [3, 0.26],
    [4, 0.1],
  ] as const);
  const conditions: Condition[] = [];
  // The first condition is always directional so a group never consists of filters only.
  conditions.push(randomCondition(rng, limits, { directionalOnly: true }));
  for (let i = 1; i < count; i++) conditions.push(randomCondition(rng, limits));
  return { logic: count > 1 && rng() < 0.25 ? 'OR' : 'AND', conditions };
}

/**
 * Indicator-reversal exit for a position opened by `entry`: the mirrored
 * directional conditions of the entry (falling back to a fresh directional
 * condition when the entry only contains filters).
 */
export function reversalExitFor(rng: Rng, entry: RuleGroup, limits: GenomeLimits): RuleGroup {
  const mirrored = entry.conditions.filter((c) => !isFilter(c)).map(mirrorCondition);
  const conditions =
    mirrored.length > 0
      ? mirrored.slice(0, MAX_EXIT_CONDITIONS)
      : [mirrorCondition(randomCondition(rng, limits, { directionalOnly: true }))];
  return { logic: conditions.length > 1 && rng() < 0.7 ? 'OR' : 'AND', conditions };
}

function dedupeGroup(group: RuleGroup, maxConditions: number, limits: GenomeLimits): RuleGroup {
  const seen = new Set<string>();
  const conditions: Condition[] = [];
  for (const raw of group.conditions) {
    const c = repairCondition(raw, limits);
    if (c.ind === 'VOLUME' && !limits.hasVolume) continue;
    const key = conditionKey(c);
    if (seen.has(key)) continue;
    seen.add(key);
    conditions.push(c);
    if (conditions.length >= maxConditions) break;
  }
  return { logic: conditions.length > 1 ? group.logic : 'AND', conditions };
}

/* -------------------------------------------------------------------------- */
/*  Exit and sizing genes                                                     */
/* -------------------------------------------------------------------------- */

function randomStaticExit(rng: Rng): ExitGene {
  return {
    mode: 'static',
    stopLossPct: randLogStep(rng, RANGES.stopLossPct[0], RANGES.stopLossPct[1], 0.1),
    takeProfitPct: randLogStep(rng, RANGES.takeProfitPct[0], RANGES.takeProfitPct[1], 0.5),
  };
}

function randomDynamicExit(
  rng: Rng,
  limits: GenomeLimits,
  longEntry: RuleGroup,
  shortEntry: RuleGroup | null,
): DynamicExit {
  const trailing = rng() < 0.45;
  const useSignalExit = rng() < 0.35;
  const canDropTarget = trailing || useSignalExit;
  return {
    mode: 'dynamic',
    atrPeriod: randInt(rng, RANGES.atrPeriod[0], RANGES.atrPeriod[1]),
    stopAtrMult: randStep(rng, RANGES.stopAtrMult[0], RANGES.stopAtrMult[1], 0.1),
    trailing,
    takeProfitAtrMult:
      canDropTarget && rng() < 0.4
        ? null
        : randStep(rng, RANGES.takeProfitAtrMult[0], RANGES.takeProfitAtrMult[1], 0.1),
    longExit: useSignalExit ? reversalExitFor(rng, longEntry, limits) : null,
    shortExit: useSignalExit && shortEntry !== null ? reversalExitFor(rng, shortEntry, limits) : null,
  };
}

function randomExit(rng: Rng, limits: GenomeLimits, longEntry: RuleGroup, shortEntry: RuleGroup | null): ExitGene {
  return rng() < 0.45 ? randomStaticExit(rng) : randomDynamicExit(rng, limits, longEntry, shortEntry);
}

function randomSizing(rng: Rng): SizingGene {
  return rng() < 0.85
    ? { mode: 'full' }
    : { mode: 'risk', riskPct: randStep(rng, RANGES.riskPct[0], RANGES.riskPct[1], 0.25) };
}

function tweakExit(rng: Rng, exit: ExitGene): ExitGene {
  if (exit.mode === 'static') {
    return rng() < 0.5
      ? { ...exit, stopLossPct: nudgeLog(rng, exit.stopLossPct, RANGES.stopLossPct[0], RANGES.stopLossPct[1], 0.1) }
      : {
          ...exit,
          takeProfitPct: nudgeLog(rng, exit.takeProfitPct, RANGES.takeProfitPct[0], RANGES.takeProfitPct[1], 0.5),
        };
  }
  const r = rng();
  if (r < 0.4 || exit.takeProfitAtrMult === null) {
    if (r < 0.75) {
      return {
        ...exit,
        stopAtrMult: nudgeStep(rng, exit.stopAtrMult, RANGES.stopAtrMult[0], RANGES.stopAtrMult[1], 0.1),
      };
    }
    return { ...exit, atrPeriod: nudgeInt(rng, exit.atrPeriod, RANGES.atrPeriod[0], RANGES.atrPeriod[1]) };
  }
  if (r < 0.8) {
    return {
      ...exit,
      takeProfitAtrMult: nudgeStep(
        rng,
        exit.takeProfitAtrMult,
        RANGES.takeProfitAtrMult[0],
        RANGES.takeProfitAtrMult[1],
        0.1,
      ),
    };
  }
  return { ...exit, atrPeriod: nudgeInt(rng, exit.atrPeriod, RANGES.atrPeriod[0], RANGES.atrPeriod[1]) };
}

/** Switch static ↔ dynamic, or toggle the trailing stop / profit target / indicator exit. */
function restructureExit(rng: Rng, genome: StrategyGenome, limits: GenomeLimits): ExitGene {
  const exit = genome.exit;
  if (exit.mode === 'static') return randomDynamicExit(rng, limits, genome.longEntry, genome.shortEntry);
  const r = rng();
  if (r < 0.25) return randomStaticExit(rng);
  if (r < 0.5) return { ...exit, trailing: !exit.trailing };
  if (r < 0.75) {
    return {
      ...exit,
      takeProfitAtrMult:
        exit.takeProfitAtrMult === null
          ? randStep(rng, RANGES.takeProfitAtrMult[0], RANGES.takeProfitAtrMult[1], 0.1)
          : null,
    };
  }
  if (exit.longExit !== null) return { ...exit, longExit: null, shortExit: null };
  return {
    ...exit,
    longExit: reversalExitFor(rng, genome.longEntry, limits),
    shortExit: genome.shortEntry !== null ? reversalExitFor(rng, genome.shortEntry, limits) : null,
  };
}

/* -------------------------------------------------------------------------- */
/*  Whole-genome operators                                                    */
/* -------------------------------------------------------------------------- */

export function cloneGenome(g: StrategyGenome): StrategyGenome {
  const exit: ExitGene =
    g.exit.mode === 'static'
      ? { ...g.exit }
      : {
          ...g.exit,
          longExit: g.exit.longExit ? cloneGroup(g.exit.longExit) : null,
          shortExit: g.exit.shortExit ? cloneGroup(g.exit.shortExit) : null,
        };
  return {
    longEntry: cloneGroup(g.longEntry),
    shortEntry: g.shortEntry ? cloneGroup(g.shortEntry) : null,
    exit,
    sizing: { ...g.sizing },
  };
}

/** Enforce every structural invariant. Always returns a fresh, valid genome. */
export function repairGenome(g: StrategyGenome, mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome {
  let longEntry = dedupeGroup(g.longEntry, MAX_CONDITIONS, limits);
  if (longEntry.conditions.length === 0) longEntry = randomGroup(rng, limits);

  let shortEntry: RuleGroup | null = null;
  if (mode === 'long-short') {
    shortEntry = g.shortEntry ? dedupeGroup(g.shortEntry, MAX_CONDITIONS, limits) : mirrorGroup(longEntry);
    if (shortEntry.conditions.length === 0) shortEntry = mirrorGroup(longEntry);
  }

  let exit: ExitGene;
  if (g.exit.mode === 'static') {
    exit = {
      mode: 'static',
      stopLossPct: clamp(quantize(g.exit.stopLossPct, 0.1), RANGES.stopLossPct[0], RANGES.stopLossPct[1]),
      takeProfitPct: clamp(quantize(g.exit.takeProfitPct, 0.5), RANGES.takeProfitPct[0], RANGES.takeProfitPct[1]),
    };
  } else {
    let longExit = g.exit.longExit ? dedupeGroup(g.exit.longExit, MAX_EXIT_CONDITIONS, limits) : null;
    if (longExit !== null && longExit.conditions.length === 0) longExit = null;
    let shortExit: RuleGroup | null = null;
    if (shortEntry !== null && longExit !== null) {
      shortExit = g.exit.shortExit ? dedupeGroup(g.exit.shortExit, MAX_EXIT_CONDITIONS, limits) : null;
      if (shortExit === null || shortExit.conditions.length === 0) shortExit = mirrorGroup(longExit);
    }
    let takeProfitAtrMult =
      g.exit.takeProfitAtrMult === null
        ? null
        : clamp(quantize(g.exit.takeProfitAtrMult, 0.1), RANGES.takeProfitAtrMult[0], RANGES.takeProfitAtrMult[1]);
    // Without a target there must be another way to bank profits.
    if (takeProfitAtrMult === null && !g.exit.trailing && longExit === null) {
      takeProfitAtrMult = randStep(rng, RANGES.takeProfitAtrMult[0], RANGES.takeProfitAtrMult[1], 0.1);
    }
    exit = {
      mode: 'dynamic',
      atrPeriod: clamp(Math.round(g.exit.atrPeriod), RANGES.atrPeriod[0], RANGES.atrPeriod[1]),
      stopAtrMult: clamp(quantize(g.exit.stopAtrMult, 0.1), RANGES.stopAtrMult[0], RANGES.stopAtrMult[1]),
      trailing: g.exit.trailing,
      takeProfitAtrMult,
      longExit,
      shortExit,
    };
  }

  const sizing: SizingGene =
    g.sizing.mode === 'risk'
      ? { mode: 'risk', riskPct: clamp(quantize(g.sizing.riskPct, 0.25), RANGES.riskPct[0], RANGES.riskPct[1]) }
      : { mode: 'full' };

  return { longEntry, shortEntry, exit, sizing };
}

export function randomGenome(rng: Rng, mode: TradeMode, limits: GenomeLimits): StrategyGenome {
  const longEntry = randomGroup(rng, limits);
  let shortEntry: RuleGroup | null = null;
  if (mode === 'long-short') {
    // Symmetric (mirrored) short rules are a strong prior; independent rules add diversity.
    shortEntry = rng() < 0.6 ? mirrorGroup(longEntry) : randomGroup(rng, limits);
  }
  const genome: StrategyGenome = {
    longEntry,
    shortEntry,
    exit: randomExit(rng, limits, longEntry, shortEntry),
    sizing: randomSizing(rng),
  };
  return repairGenome(genome, mode, limits, rng);
}

type MutationKind =
  | 'tweak'
  | 'alter'
  | 'replace'
  | 'add'
  | 'remove'
  | 'logic'
  | 'exit-tweak'
  | 'exit-structure'
  | 'mirror'
  | 'sizing';

const MUTATION_WEIGHTS: readonly (readonly [MutationKind, number])[] = [
  ['tweak', 0.34],
  ['alter', 0.08],
  ['replace', 0.1],
  ['add', 0.07],
  ['remove', 0.07],
  ['logic', 0.05],
  ['exit-tweak', 0.16],
  ['exit-structure', 0.07],
  ['mirror', 0.03],
  ['sizing', 0.03],
];

/** Every rule group in the genome, with entry groups weighted above exit groups. */
function mutableGroups(g: StrategyGenome): (readonly [RuleGroup, number])[] {
  const groups: (readonly [RuleGroup, number])[] = [[g.longEntry, 1]];
  if (g.shortEntry) groups.push([g.shortEntry, 1]);
  if (g.exit.mode === 'dynamic') {
    if (g.exit.longExit) groups.push([g.exit.longExit, 0.5]);
    if (g.exit.shortExit) groups.push([g.exit.shortExit, 0.5]);
  }
  return groups;
}

function isExitGroup(g: StrategyGenome, group: RuleGroup): boolean {
  return g.exit.mode === 'dynamic' && (group === g.exit.longExit || group === g.exit.shortExit);
}

function applyMutation(rng: Rng, g: StrategyGenome, mode: TradeMode, limits: GenomeLimits): void {
  const kind = pickWeighted(rng, MUTATION_WEIGHTS);
  const group = pickWeighted(rng, mutableGroups(g));
  const index = Math.floor(rng() * group.conditions.length);
  const maxConditions = isExitGroup(g, group) ? MAX_EXIT_CONDITIONS : MAX_CONDITIONS;

  switch (kind) {
    case 'tweak':
      group.conditions[index] = tweakCondition(rng, group.conditions[index], limits);
      return;
    case 'alter':
      group.conditions[index] = alterCondition(rng, group.conditions[index], limits);
      return;
    case 'replace':
      group.conditions[index] = randomCondition(rng, limits, { directionalOnly: isExitGroup(g, group) });
      return;
    case 'add':
      if (group.conditions.length < maxConditions) {
        group.conditions.push(randomCondition(rng, limits, { directionalOnly: isExitGroup(g, group) }));
      } else {
        group.conditions[index] = tweakCondition(rng, group.conditions[index], limits);
      }
      return;
    case 'remove':
      if (group.conditions.length > 1) group.conditions.splice(index, 1);
      else group.conditions[index] = tweakCondition(rng, group.conditions[index], limits);
      return;
    case 'logic':
      if (group.conditions.length > 1) group.logic = group.logic === 'AND' ? 'OR' : 'AND';
      else group.conditions[index] = alterCondition(rng, group.conditions[index], limits);
      return;
    case 'exit-tweak':
      g.exit = tweakExit(rng, g.exit);
      return;
    case 'exit-structure':
      g.exit = restructureExit(rng, g, limits);
      return;
    case 'mirror':
      if (mode === 'long-short' && g.shortEntry) {
        if (rng() < 0.5) g.shortEntry = mirrorGroup(g.longEntry);
        else g.longEntry = mirrorGroup(g.shortEntry);
      } else {
        group.conditions[index] = tweakCondition(rng, group.conditions[index], limits);
      }
      return;
    case 'sizing':
      if (g.sizing.mode === 'full') {
        g.sizing = { mode: 'risk', riskPct: randStep(rng, RANGES.riskPct[0], RANGES.riskPct[1], 0.25) };
      } else if (rng() < 0.5) {
        g.sizing = { mode: 'full' };
      } else {
        g.sizing = {
          mode: 'risk',
          riskPct: nudgeStep(rng, g.sizing.riskPct, RANGES.riskPct[0], RANGES.riskPct[1], 0.25),
        };
      }
      return;
  }
}

/**
 * Produce a mutated child. `intensity` (≥ 1) adds extra mutation steps so the
 * search can take larger jumps when it is stagnating.
 */
export function mutateGenome(
  rng: Rng,
  parent: StrategyGenome,
  mode: TradeMode,
  limits: GenomeLimits,
  intensity = 1,
): StrategyGenome {
  const child = cloneGenome(parent);
  let steps = 1 + (rng() < 0.35 ? 1 : 0) + (rng() < 0.12 ? 1 : 0);
  steps += Math.max(0, Math.round(intensity) - 1);
  for (let i = 0; i < steps; i++) applyMutation(rng, child, mode, limits);
  return repairGenome(child, mode, limits, rng);
}

/** Block-level crossover with optional condition mixing on the long entry. */
export function crossoverGenomes(
  rng: Rng,
  a: StrategyGenome,
  b: StrategyGenome,
  mode: TradeMode,
  limits: GenomeLimits,
): StrategyGenome {
  const pa = cloneGenome(a);
  const pb = cloneGenome(b);
  let longEntry = rng() < 0.5 ? pa.longEntry : pb.longEntry;
  if (rng() < 0.3) {
    const pool = [...pa.longEntry.conditions, ...pb.longEntry.conditions];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    longEntry = { logic: longEntry.logic, conditions: pool.slice(0, randInt(rng, 1, Math.min(MAX_CONDITIONS, pool.length))) };
  }
  const child: StrategyGenome = {
    longEntry,
    shortEntry: rng() < 0.5 ? pa.shortEntry : pb.shortEntry,
    exit: rng() < 0.5 ? pa.exit : pb.exit,
    sizing: rng() < 0.5 ? pa.sizing : pb.sizing,
  };
  return repairGenome(child, mode, limits, rng);
}

/* -------------------------------------------------------------------------- */
/*  Identity                                                                  */
/* -------------------------------------------------------------------------- */

/** Canonical string identity of a genome (order of conditions inside a group is irrelevant). */
export function genomeSignature(g: StrategyGenome): string {
  const exit =
    g.exit.mode === 'static'
      ? `S:${g.exit.stopLossPct}:${g.exit.takeProfitPct}`
      : `D:${g.exit.atrPeriod}:${g.exit.stopAtrMult}:${g.exit.trailing ? 1 : 0}:${g.exit.takeProfitAtrMult ?? 'x'}:${groupKey(g.exit.longExit)}:${groupKey(g.exit.shortExit)}`;
  const sizing = g.sizing.mode === 'full' ? 'F' : `R:${g.sizing.riskPct}`;
  return `${groupKey(g.longEntry)}|${groupKey(g.shortEntry)}|${exit}|${sizing}`;
}

/**
 * Structural "niche" of a genome: which indicator families drive the entries
 * and which exit mode is used. The leaderboard caps the number of strategies
 * per niche so the Top 25 is not 25 parameter tweaks of the same idea.
 */
export function genomeNiche(g: StrategyGenome): string {
  const families = new Set<string>();
  for (const c of g.longEntry.conditions) families.add(c.ind);
  if (g.shortEntry) for (const c of g.shortEntry.conditions) families.add(c.ind);
  return `${[...families].sort().join('+')}|${g.exit.mode}`;
}
