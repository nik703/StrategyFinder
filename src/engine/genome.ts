/**
 * Strategy genome: archetypes, parameter grids, random generation, mutation,
 * crossover and repair.
 *
 * The search space is deliberately small and coherent, so the optimiser cannot
 * assemble a pile of unrelated, finely tuned rules:
 *
 *   • every strategy follows one archetype (TREND or MEAN_REVERSION);
 *   • an entry is ONE trigger plus at most ONE filter, combined with AND;
 *   • short rules are always the mirror image of the long rules;
 *   • every number comes from a coarse, round-number grid;
 *   • a strategy may tune at most `MAX_PARAMETERS` numbers in total;
 *   • position sizing is a run setting, not a gene.
 */

import type {
  Archetype,
  CmpOp,
  Condition,
  ConditionRole,
  DynamicExit,
  ExitGene,
  IndicatorFamily,
  RuleGroup,
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

/* -------------------------------------------------------------------------- */
/*  Parameter grids                                                           */
/* -------------------------------------------------------------------------- */

/** Every indicator lookback comes from this list. */
export const LOOKBACK_GRID: readonly number[] = [5, 7, 10, 14, 20, 21, 26, 30, 50, 100, 150, 200];
/** Static stop-loss / take-profit percentages. */
export const STATIC_PCT_GRID: readonly number[] = [1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30];
export const TIME_STOP_GRID: readonly number[] = [5, 10, 20, 40];

const range = (from: number, to: number, step: number): number[] => {
  const out: number[] = [];
  for (let v = from; v <= to + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
};

export const GRIDS = {
  stopLossPct: STATIC_PCT_GRID.filter((v) => v <= 15),
  takeProfitPct: STATIC_PCT_GRID,
  atrPeriod: [7, 10, 14, 20, 21],
  stopAtrMult: range(1, 4, 0.5),
  takeProfitAtrMult: range(1.5, 8, 0.5),
  macdSignal: [5, 7, 10, 14],
  bbMult: range(1.5, 3, 0.5),
  supertrendMult: range(1, 5, 0.5),
  /** Oscillator levels move in steps of 5; the low zone is "oversold", the high zone "overbought". */
  rsiLow: range(10, 45, 5),
  rsiHigh: range(55, 90, 5),
  stochLow: range(5, 45, 5),
  stochHigh: range(55, 95, 5),
  cciLow: [-200, -150, -100, -50],
  cciHigh: [50, 100, 150, 200],
  percentBLow: range(0, 0.4, 0.05),
  percentBHigh: range(0.6, 1, 0.05),
  bandwidth: [2, 3, 4, 5, 7.5, 10, 15, 20, 30, 40],
  atrPercent: [0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10],
  adxLevel: range(15, 40, 5),
  volumeRatio: [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3],
} as const;

/** One trigger + one filter per entry. */
export const MAX_CONDITIONS = 2;
/** Indicator-reversal exits are a single condition. */
export const MAX_EXIT_CONDITIONS = 1;
/** Hard cap on the number of tunable numbers in a strategy. */
export const MAX_PARAMETERS = 8;
/** Parameters above this count cost fitness. */
export const FREE_PARAMETERS = 4;
/** "Short-term" triggers look back at most this many bars; regime filters at least `SLOW_LOOKBACK`. */
export const SHORT_LOOKBACK = 30;
export const SLOW_LOOKBACK = 100;

/** Stochastic smoothing is fixed at the textbook 3 / 3 so it is not a tunable parameter. */
const STOCH_SMOOTH = 3;
const STOCH_D = 3;

/** Dataset-dependent limits for the search space. */
export interface GenomeLimits {
  /** Longest indicator lookback allowed (keeps warm-up short relative to the sample). */
  maxLookback: number;
  /** False when the dataset has no usable volume (volume conditions are then excluded). */
  hasVolume: boolean;
}

export const DEFAULT_LIMITS: GenomeLimits = { maxLookback: 200, hasVolume: true };

function lookbacks(min: number, max: number, limits: GenomeLimits): number[] {
  const cap = Math.min(max, Math.max(SHORT_LOOKBACK, limits.maxLookback));
  return LOOKBACK_GRID.filter((v) => v >= min && v <= cap);
}

/** Nearest value of `grid` to `value`. */
export function snap(value: number, grid: readonly number[]): number {
  let best = grid[0];
  let distance = Infinity;
  for (const g of grid) {
    const d = Math.abs(g - value);
    if (d < distance) {
      distance = d;
      best = g;
    }
  }
  return best;
}

/* -------------------------------------------------------------------------- */
/*  Condition semantics                                                       */
/* -------------------------------------------------------------------------- */

const ALL_OPS: readonly CmpOp[] = ['gt', 'lt', 'cross_above', 'cross_below'];

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

/** Direction-agnostic conditions: volatility, volume and trend strength. */
export function isFilter(c: Condition): boolean {
  return (
    c.ind === 'ATR' ||
    c.ind === 'VOLUME' ||
    (c.ind === 'ADX' && c.variant === 'strength') ||
    (c.ind === 'BB' && c.variant === 'bandwidth')
  );
}

/** Midpoint that separates the "low" (oversold) from the "high" (overbought) zone of an oscillator. */
function oscillatorMid(c: Condition): number | null {
  if (c.ind === 'RSI') return 50;
  if (c.ind === 'STOCH' && c.variant === 'level') return 50;
  if (c.ind === 'CCI') return 0;
  if (c.ind === 'BB' && c.variant === 'pctb') return 0.5;
  return null;
}

/**
 * What a condition says about the market.
 *
 *   FILTER      ATR %, volume ratio, ADX strength, Bollinger bandwidth
 *   OVERSOLD    an oscillator in its low zone, or price at the lower Bollinger band
 *   OVERBOUGHT  an oscillator in its high zone, or price at the upper Bollinger band
 *   TREND_UP    price / momentum above its reference (`>` or `crosses above`)
 *   TREND_DOWN  price / momentum below its reference (`<` or `crosses below`)
 */
export function conditionRole(c: Condition): ConditionRole {
  if (isFilter(c)) return 'FILTER';
  const mid = oscillatorMid(c);
  if (mid !== null) return (c as { level: number }).level < mid ? 'OVERSOLD' : 'OVERBOUGHT';
  if (c.ind === 'BB') return c.variant === 'lower' ? 'OVERSOLD' : 'OVERBOUGHT';
  return c.op === 'gt' || c.op === 'cross_above' ? 'TREND_UP' : 'TREND_DOWN';
}

/** The lookback that decides whether a trend condition is short-term or slow. */
export function conditionLookback(c: Condition): number {
  switch (c.ind) {
    case 'MACD':
      return c.slow;
    case 'MA_CROSS':
      return c.slowPeriod;
    case 'STOCH':
      return c.kPeriod;
    default:
      return c.period;
  }
}

/**
 * Mirror a condition into its opposite (bullish ↔ bearish, oversold ↔ overbought).
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
      if (c.variant === 'pctb') return { ...c, op: flipOp(c.op), level: Number((1 - c.level).toFixed(2)) };
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
  return { logic: 'AND', conditions: group.conditions.map(mirrorCondition) };
}

/* -------------------------------------------------------------------------- */
/*  Slots, parameter specs and normalisation                                  */
/* -------------------------------------------------------------------------- */

/** Where a condition sits inside an entry – this decides which lookbacks it may use. */
type Slot = 'trend-trigger' | 'reversion-trigger' | 'filter' | 'regime';

function slotOf(archetype: Archetype, index: number, c: Condition): Slot {
  if (index === 0) return archetype === 'TREND' ? 'trend-trigger' : 'reversion-trigger';
  return isFilter(c) ? 'filter' : 'regime';
}

type Spec = readonly [key: string, grid: readonly number[]];

/** Lookback window allowed for trend-style indicators in a given slot. */
function trendWindow(slot: Slot, min: number): readonly [number, number] {
  if (slot === 'reversion-trigger') return [min, SHORT_LOOKBACK];
  if (slot === 'regime') return [SLOW_LOOKBACK, 200];
  return [min, 200];
}

/** The tunable numeric parameters of a condition, each with the grid it lives on. */
function conditionSpecs(c: Condition, slot: Slot, limits: GenomeLimits): Spec[] {
  switch (c.ind) {
    case 'RSI':
      return [
        ['period', lookbacks(5, 30, limits)],
        ['level', c.level < 50 ? GRIDS.rsiLow : GRIDS.rsiHigh],
      ];
    case 'MACD': {
      const [, max] = trendWindow(slot, 14);
      const slowGrid = lookbacks(slot === 'reversion-trigger' ? 14 : 20, Math.min(max, 100), limits);
      const specs: Spec[] = [
        ['fast', lookbacks(5, 26, limits)],
        ['slow', slowGrid],
      ];
      if (c.variant !== 'zero') specs.push(['signal', GRIDS.macdSignal]);
      return specs;
    }
    case 'PRICE_MA': {
      const [min, max] = trendWindow(slot, 5);
      return [['period', lookbacks(min, max, limits)]];
    }
    case 'MA_CROSS': {
      const [min, max] = trendWindow(slot, 10);
      return [
        ['fastPeriod', lookbacks(5, 50, limits)],
        ['slowPeriod', lookbacks(min, max, limits)],
      ];
    }
    case 'BB': {
      const specs: Spec[] = [
        ['period', lookbacks(10, 50, limits)],
        ['mult', GRIDS.bbMult],
      ];
      if (c.variant === 'pctb') specs.push(['level', c.level < 0.5 ? GRIDS.percentBLow : GRIDS.percentBHigh]);
      if (c.variant === 'bandwidth') specs.push(['level', GRIDS.bandwidth]);
      return specs;
    }
    case 'ATR':
      return [
        ['period', lookbacks(7, 30, limits)],
        ['level', GRIDS.atrPercent],
      ];
    case 'STOCH': {
      const specs: Spec[] = [['kPeriod', lookbacks(5, 21, limits)]];
      if (c.variant === 'level') specs.push(['level', c.level < 50 ? GRIDS.stochLow : GRIDS.stochHigh]);
      return specs;
    }
    case 'ADX': {
      const specs: Spec[] = [['period', lookbacks(7, 30, limits)]];
      if (c.variant === 'strength') specs.push(['level', GRIDS.adxLevel]);
      return specs;
    }
    case 'VOLUME':
      return [
        ['period', lookbacks(10, 50, limits)],
        ['level', GRIDS.volumeRatio],
      ];
    case 'SUPERTREND':
      return [
        ['period', lookbacks(7, 21, limits)],
        ['mult', GRIDS.supertrendMult],
      ];
    case 'CCI':
      return [
        ['period', lookbacks(10, 50, limits)],
        ['level', c.level < 0 ? GRIDS.cciLow : GRIDS.cciHigh],
      ];
  }
}

type Fields = Record<string, number | string>;

/** Snap every parameter onto its grid and make the operator consistent with the condition's kind. */
function normalizeCondition(input: Condition, slot: Slot, limits: GenomeLimits): Condition {
  const c = { ...input } as Condition;
  const fields = c as unknown as Fields;

  // Operators.
  if (isFilter(c)) {
    if (c.op === 'cross_above') c.op = 'gt';
    if (c.op === 'cross_below') c.op = 'lt';
  } else {
    const mid = oscillatorMid(c);
    if (mid !== null) {
      // Low zone: "< level" (or a cross of that level). High zone: "> level".
      const low = (fields.level as number) < mid;
      if (low && c.op === 'gt') c.op = 'lt';
      if (!low && c.op === 'lt') c.op = 'gt';
    } else if (c.ind === 'BB') {
      if (c.variant === 'lower' && c.op === 'gt') c.op = 'lt';
      if (c.variant === 'upper' && c.op === 'lt') c.op = 'gt';
    }
  }
  if (slot === 'regime') c.op = c.op === 'lt' || c.op === 'cross_below' ? 'lt' : 'gt';

  // Fixed, non-tunable fields.
  if (c.ind === 'STOCH') {
    c.smooth = STOCH_SMOOTH;
    c.dPeriod = STOCH_D;
    if (c.variant === 'kd') c.level = 50;
  }
  if (c.ind === 'BB' && (c.variant === 'upper' || c.variant === 'lower')) c.level = 0;
  if (c.ind === 'ADX' && c.variant === 'di') c.level = 25;
  if (c.ind === 'MACD' && c.variant === 'zero') c.signal = 10;

  for (const [key, grid] of conditionSpecs(c, slot, limits)) {
    if (grid.length > 0) fields[key] = snap(fields[key] as number, grid);
  }

  // Fast must stay below slow.
  if (c.ind === 'MACD' && c.fast >= c.slow) {
    const below = lookbacks(5, 26, limits).filter((v) => v < c.slow);
    c.fast = below.length > 0 ? below[below.length - 1] : 5;
  }
  if (c.ind === 'MA_CROSS' && c.fastPeriod >= c.slowPeriod) {
    const below = lookbacks(5, 50, limits).filter((v) => v < c.slowPeriod);
    c.fastPeriod = below.length > 0 ? below[below.length - 1] : 5;
  }
  return c;
}

function isValidTrigger(c: Condition, archetype: Archetype, limits: GenomeLimits): boolean {
  if (c.ind === 'VOLUME' && !limits.hasVolume) return false;
  const role = conditionRole(c);
  if (archetype === 'TREND') return role === 'TREND_UP';
  return role === 'OVERSOLD' || (role === 'TREND_DOWN' && conditionLookback(c) <= SHORT_LOOKBACK);
}

function isValidSecond(c: Condition, archetype: Archetype, limits: GenomeLimits): boolean {
  if (c.ind === 'VOLUME' && !limits.hasVolume) return false;
  if (isFilter(c)) return true;
  if (archetype === 'TREND') return false;
  // Mean reversion may be gated by a slow up-trend regime.
  return (
    (c.ind === 'PRICE_MA' || c.ind === 'MA_CROSS') &&
    c.op === 'gt' &&
    conditionLookback(c) >= SLOW_LOOKBACK &&
    limits.maxLookback >= SLOW_LOOKBACK
  );
}

/** True when the long entry obeys the genome's archetype. */
export function isCoherent(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  const { conditions } = g.longEntry;
  if (conditions.length < 1 || conditions.length > MAX_CONDITIONS) return false;
  if (!isValidTrigger(conditions[0], g.archetype, limits)) return false;
  return conditions.length === 1 || isValidSecond(conditions[1], g.archetype, limits);
}

/* -------------------------------------------------------------------------- */
/*  Random conditions                                                         */
/* -------------------------------------------------------------------------- */

function blank(ind: IndicatorFamily): Condition {
  switch (ind) {
    case 'RSI':
      return { ind, period: 14, op: 'lt', level: 30 };
    case 'MACD':
      return { ind, fast: 10, slow: 26, signal: 10, variant: 'signal', op: 'gt' };
    case 'PRICE_MA':
      return { ind, maType: 'SMA', period: 50, op: 'gt' };
    case 'MA_CROSS':
      return { ind, fastType: 'SMA', fastPeriod: 20, slowType: 'SMA', slowPeriod: 50, op: 'gt' };
    case 'BB':
      return { ind, period: 20, mult: 2, variant: 'lower', op: 'lt', level: 0 };
    case 'ATR':
      return { ind, period: 14, op: 'gt', level: 2 };
    case 'STOCH':
      return { ind, kPeriod: 14, smooth: STOCH_SMOOTH, dPeriod: STOCH_D, variant: 'level', op: 'lt', level: 20 };
    case 'ADX':
      return { ind, period: 14, variant: 'strength', op: 'gt', level: 25 };
    case 'VOLUME':
      return { ind, period: 20, op: 'gt', level: 1.5 };
    case 'SUPERTREND':
      return { ind, period: 10, mult: 3, op: 'gt' };
    case 'CCI':
      return { ind, period: 20, op: 'lt', level: -100 };
  }
}

/** Draw every tunable parameter of `c` uniformly from its grid. */
function randomizeParams(rng: Rng, c: Condition, slot: Slot, limits: GenomeLimits): Condition {
  const fields = c as unknown as Fields;
  for (const [key, grid] of conditionSpecs(c, slot, limits)) {
    if (grid.length > 0) fields[key] = pick(rng, grid);
  }
  return normalizeCondition(c, slot, limits);
}

const MA_TYPES = ['EMA', 'SMA'] as const;

function randomTrendCondition(rng: Rng, slot: Slot, up: boolean, limits: GenomeLimits): Condition {
  const ind = pickWeighted<IndicatorFamily>(rng, [
    ['PRICE_MA', 1.2],
    ['MA_CROSS', 1],
    ['MACD', 1],
    ['SUPERTREND', 1.2],
    ['ADX', 0.5],
    ['STOCH', 0.5],
  ]);
  const c = blank(ind);
  const event = rng() < 0.5;
  c.op = up ? (event ? 'cross_above' : 'gt') : event ? 'cross_below' : 'lt';
  if (c.ind === 'MACD') c.variant = pick(rng, ['signal', 'zero', 'hist'] as const);
  if (c.ind === 'PRICE_MA') c.maType = pick(rng, MA_TYPES);
  if (c.ind === 'MA_CROSS') {
    c.fastType = pick(rng, MA_TYPES);
    c.slowType = pick(rng, MA_TYPES);
  }
  if (c.ind === 'ADX') c.variant = 'di';
  if (c.ind === 'STOCH') c.variant = 'kd';
  return randomizeParams(rng, c, slot, limits);
}

function randomOversoldCondition(rng: Rng, limits: GenomeLimits): Condition {
  const kind = pickWeighted(rng, [
    ['RSI', 1.2],
    ['STOCH', 0.8],
    ['CCI', 0.8],
    ['PCTB', 0.7],
    ['LOWER', 0.9],
  ] as const);
  const op = pick<CmpOp>(rng, ['lt', 'cross_above', 'cross_below']);
  let c: Condition;
  if (kind === 'RSI') c = { ...blank('RSI'), op, level: 30 } as Condition;
  else if (kind === 'STOCH') c = { ...blank('STOCH'), op, level: 20 } as Condition;
  else if (kind === 'CCI') c = { ...blank('CCI'), op, level: -100 } as Condition;
  else if (kind === 'PCTB') c = { ind: 'BB', period: 20, mult: 2, variant: 'pctb', op, level: 0.1 };
  else c = { ind: 'BB', period: 20, mult: 2, variant: 'lower', op, level: 0 };
  return randomizeParams(rng, c, 'reversion-trigger', limits);
}

function randomFilterCondition(rng: Rng, limits: GenomeLimits): Condition {
  const kinds: (readonly ['ATR' | 'VOLUME' | 'ADX' | 'BW', number])[] = [
    ['ATR', 0.8],
    ['ADX', 1],
    ['BW', 0.7],
  ];
  if (limits.hasVolume) kinds.push(['VOLUME', 1]);
  const kind = pickWeighted(rng, kinds);
  const op: CmpOp = rng() < 0.7 ? 'gt' : 'lt';
  let c: Condition;
  if (kind === 'ATR') c = { ...blank('ATR'), op } as Condition;
  else if (kind === 'VOLUME') c = { ...blank('VOLUME'), op } as Condition;
  else if (kind === 'ADX') c = { ...blank('ADX'), op } as Condition;
  else c = { ind: 'BB', period: 20, mult: 2, variant: 'bandwidth', op, level: 10 };
  return randomizeParams(rng, c, 'filter', limits);
}

function randomRegimeCondition(rng: Rng, limits: GenomeLimits): Condition {
  const c = blank(rng() < 0.7 ? 'PRICE_MA' : 'MA_CROSS');
  c.op = 'gt';
  if (c.ind === 'PRICE_MA') c.maType = pick(rng, MA_TYPES);
  if (c.ind === 'MA_CROSS') {
    c.fastType = pick(rng, MA_TYPES);
    c.slowType = pick(rng, MA_TYPES);
  }
  return randomizeParams(rng, c, 'regime', limits);
}

export function randomTrigger(rng: Rng, archetype: Archetype, limits: GenomeLimits): Condition {
  if (archetype === 'TREND') return randomTrendCondition(rng, 'trend-trigger', true, limits);
  return rng() < 0.6
    ? randomOversoldCondition(rng, limits)
    : randomTrendCondition(rng, 'reversion-trigger', false, limits);
}

export function randomSecond(rng: Rng, archetype: Archetype, limits: GenomeLimits): Condition {
  if (archetype === 'MEAN_REVERSION' && limits.maxLookback >= SLOW_LOOKBACK && rng() < 0.6) {
    return randomRegimeCondition(rng, limits);
  }
  return randomFilterCondition(rng, limits);
}

function randomEntry(rng: Rng, archetype: Archetype, limits: GenomeLimits): RuleGroup {
  const conditions = [randomTrigger(rng, archetype, limits)];
  if (rng() < 0.55) conditions.push(randomSecond(rng, archetype, limits));
  return { logic: 'AND', conditions };
}

/* -------------------------------------------------------------------------- */
/*  Exits                                                                     */
/* -------------------------------------------------------------------------- */

function randomStaticExit(rng: Rng): ExitGene {
  return {
    mode: 'static',
    stopLossPct: pick(rng, GRIDS.stopLossPct),
    takeProfitPct: pick(rng, GRIDS.takeProfitPct),
  };
}

function randomDynamicExit(rng: Rng, archetype: Archetype): DynamicExit {
  const trailing = rng() < 0.4;
  const reversalExit = archetype === 'TREND' && rng() < 0.35;
  return {
    mode: 'dynamic',
    atrPeriod: pick(rng, GRIDS.atrPeriod),
    stopAtrMult: pick(rng, GRIDS.stopAtrMult),
    trailing,
    takeProfitAtrMult: (trailing || reversalExit) && rng() < 0.45 ? null : pick(rng, GRIDS.takeProfitAtrMult),
    reversalExit,
    longExit: null,
    shortExit: null,
  };
}

/** Reward-to-risk ratio implied by the exit, or null when there is no fixed target. */
export function rewardToRisk(g: StrategyGenome): number | null {
  if (g.exit.mode === 'static') return g.exit.takeProfitPct / g.exit.stopLossPct;
  return g.exit.takeProfitAtrMult === null ? null : g.exit.takeProfitAtrMult / g.exit.stopAtrMult;
}

/* -------------------------------------------------------------------------- */
/*  Whole-genome helpers                                                      */
/* -------------------------------------------------------------------------- */

function cloneGroup(group: RuleGroup): RuleGroup {
  return { logic: group.logic, conditions: group.conditions.map((c) => ({ ...c })) };
}

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
    archetype: g.archetype,
    longEntry: cloneGroup(g.longEntry),
    shortEntry: g.shortEntry ? cloneGroup(g.shortEntry) : null,
    exit,
    maxBarsInTrade: g.maxBarsInTrade,
  };
}

/** A tunable number inside a genome, with the grid it lives on. */
interface Handle {
  grid: readonly number[];
  get: () => number;
  set: (value: number) => void;
}

/** Handles for every tunable number of `g`. They read and write `g` in place. */
function parameterHandles(g: StrategyGenome, limits: GenomeLimits): Handle[] {
  const handles: Handle[] = [];
  g.longEntry.conditions.forEach((c, index) => {
    const fields = c as unknown as Fields;
    for (const [key, grid] of conditionSpecs(c, slotOf(g.archetype, index, c), limits)) {
      if (grid.length === 0) continue;
      handles.push({ grid, get: () => fields[key] as number, set: (v) => (fields[key] = v) });
    }
  });
  const exit = g.exit;
  if (exit.mode === 'static') {
    handles.push({ grid: GRIDS.stopLossPct, get: () => exit.stopLossPct, set: (v) => (exit.stopLossPct = v) });
    handles.push({ grid: GRIDS.takeProfitPct, get: () => exit.takeProfitPct, set: (v) => (exit.takeProfitPct = v) });
  } else {
    handles.push({ grid: GRIDS.atrPeriod, get: () => exit.atrPeriod, set: (v) => (exit.atrPeriod = v) });
    handles.push({ grid: GRIDS.stopAtrMult, get: () => exit.stopAtrMult, set: (v) => (exit.stopAtrMult = v) });
    if (exit.takeProfitAtrMult !== null) {
      handles.push({
        grid: GRIDS.takeProfitAtrMult,
        get: () => exit.takeProfitAtrMult as number,
        set: (v) => (exit.takeProfitAtrMult = v),
      });
    }
  }
  if (g.maxBarsInTrade !== null) {
    handles.push({ grid: TIME_STOP_GRID, get: () => g.maxBarsInTrade as number, set: (v) => (g.maxBarsInTrade = v) });
  }
  return handles;
}

/**
 * Number of tunable numeric values in a strategy: every lookback, threshold and
 * multiplier of the long entry, the exit distances and the time stop. Mirrored
 * short rules and derived reversal exits reuse the same numbers and are not
 * counted twice.
 */
export function countParameters(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): number {
  return parameterHandles(g, limits).length;
}

/**
 * Deterministic clean-up: snap every number onto its grid, derive the mirrored
 * short rules and reversal exits, and enforce the exit sanity rules and the
 * parameter cap. Does NOT fix an entry that violates its archetype – see `repairGenome`.
 */
export function normalizeGenome(input: StrategyGenome, mode: TradeMode, limits: GenomeLimits): StrategyGenome {
  const archetype: Archetype = input.archetype === 'MEAN_REVERSION' ? 'MEAN_REVERSION' : 'TREND';
  const conditions = input.longEntry.conditions
    .slice(0, MAX_CONDITIONS)
    .map((c, index) => normalizeCondition(c, slotOf(archetype, index, c), limits));
  const longEntry: RuleGroup = { logic: 'AND', conditions };

  let maxBarsInTrade = input.maxBarsInTrade === null ? null : snap(input.maxBarsInTrade, TIME_STOP_GRID);

  let exit: ExitGene;
  if (input.exit.mode === 'static') {
    exit = {
      mode: 'static',
      stopLossPct: snap(input.exit.stopLossPct, GRIDS.stopLossPct),
      takeProfitPct: snap(input.exit.takeProfitPct, GRIDS.takeProfitPct),
    };
  } else {
    const stopAtrMult = snap(input.exit.stopAtrMult, GRIDS.stopAtrMult);
    let takeProfitAtrMult =
      input.exit.takeProfitAtrMult === null ? null : snap(input.exit.takeProfitAtrMult, GRIDS.takeProfitAtrMult);
    let trailing = input.exit.trailing;
    const reversalExit = archetype === 'TREND' && input.exit.reversalExit && conditions.length > 0;
    // A trail can only lock in profit if the target sits at least 1 ATR beyond the stop distance.
    if (trailing && takeProfitAtrMult !== null && takeProfitAtrMult < stopAtrMult + 1) trailing = false;
    // Without a target there must be another way out with a profit.
    if (takeProfitAtrMult === null && !trailing && !reversalExit && maxBarsInTrade === null) {
      takeProfitAtrMult = snap(stopAtrMult * 2, GRIDS.takeProfitAtrMult);
    }
    exit = {
      mode: 'dynamic',
      atrPeriod: snap(input.exit.atrPeriod, GRIDS.atrPeriod),
      stopAtrMult,
      trailing,
      takeProfitAtrMult,
      reversalExit,
      longExit: null,
      shortExit: null,
    };
  }

  const genome: StrategyGenome = { archetype, longEntry, shortEntry: null, exit, maxBarsInTrade };

  // Parameter cap: shed the least essential pieces first.
  while (countParameters(genome, limits) > MAX_PARAMETERS) {
    if (genome.maxBarsInTrade !== null && (genome.exit.mode === 'static' || genome.exit.takeProfitAtrMult !== null)) {
      genome.maxBarsInTrade = null;
      maxBarsInTrade = null;
    } else if (genome.longEntry.conditions.length > 1) {
      genome.longEntry.conditions.pop();
    } else {
      break;
    }
  }

  // Derived parts.
  if (mode === 'long-short') genome.shortEntry = mirrorGroup(genome.longEntry);
  if (genome.exit.mode === 'dynamic' && genome.exit.reversalExit && genome.longEntry.conditions.length > 0) {
    const trigger = genome.longEntry.conditions[0];
    genome.exit.longExit = { logic: 'AND', conditions: [mirrorCondition(trigger)] };
    // The short entry trigger is the mirrored long trigger, so its reversal is the long trigger itself.
    genome.exit.shortExit = mode === 'long-short' ? { logic: 'AND', conditions: [{ ...trigger }] } : null;
  }
  return genome;
}

/** Enforce every invariant. Incoherent entries are replaced by fresh random ones. */
export function repairGenome(g: StrategyGenome, mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome {
  let genome = normalizeGenome(g, mode, limits);
  const { conditions } = genome.longEntry;
  let changed = false;
  if (conditions.length === 0 || !isValidTrigger(conditions[0], genome.archetype, limits)) {
    conditions[0] = randomTrigger(rng, genome.archetype, limits);
    changed = true;
  }
  if (conditions.length > 1 && !isValidSecond(conditions[1], genome.archetype, limits)) {
    conditions.pop();
    changed = true;
  }
  if (changed) genome = normalizeGenome(genome, mode, limits);
  return genome;
}

export function randomGenome(rng: Rng, mode: TradeMode, limits: GenomeLimits): StrategyGenome {
  const archetype: Archetype = rng() < 0.5 ? 'TREND' : 'MEAN_REVERSION';
  const genome: StrategyGenome = {
    archetype,
    longEntry: randomEntry(rng, archetype, limits),
    shortEntry: null,
    exit: rng() < 0.45 ? randomStaticExit(rng) : randomDynamicExit(rng, archetype),
    maxBarsInTrade: rng() < 0.2 ? pick(rng, TIME_STOP_GRID) : null,
  };
  return repairGenome(genome, mode, limits, rng);
}

/* -------------------------------------------------------------------------- */
/*  Mutation and crossover                                                    */
/* -------------------------------------------------------------------------- */

type MutationKind = 'tweak' | 'operator' | 'trigger' | 'second' | 'exit' | 'time' | 'archetype';

const MUTATION_WEIGHTS: readonly (readonly [MutationKind, number])[] = [
  ['tweak', 0.45],
  ['operator', 0.1],
  ['trigger', 0.1],
  ['second', 0.13],
  ['exit', 0.12],
  ['time', 0.06],
  ['archetype', 0.04],
];

/** Move one parameter to a neighbouring grid value. */
function tweak(rng: Rng, g: StrategyGenome, limits: GenomeLimits): void {
  const handles = parameterHandles(g, limits).filter((h) => h.grid.length > 1);
  if (handles.length === 0) return;
  const handle = pick(rng, handles);
  const at = handle.grid.indexOf(snap(handle.get(), handle.grid));
  const stride = rng() < 0.85 ? 1 : 2;
  let next = at + (rng() < 0.5 ? -stride : stride);
  if (next < 0 || next >= handle.grid.length) next = at + (next < 0 ? stride : -stride);
  handle.set(handle.grid[Math.max(0, Math.min(handle.grid.length - 1, next))]);
}

function mutateOperator(rng: Rng, g: StrategyGenome): void {
  const c = pick(rng, g.longEntry.conditions);
  if (rng() < 0.35) {
    if (c.ind === 'MACD') c.variant = pick(rng, (['signal', 'zero', 'hist'] as const).filter((v) => v !== c.variant));
    else if (c.ind === 'BB' && c.variant === 'lower') Object.assign(c, { variant: 'pctb', level: 0.1 });
    else if (c.ind === 'BB' && c.variant === 'pctb' && c.level < 0.5) Object.assign(c, { variant: 'lower', level: 0 });
    else if (c.ind === 'PRICE_MA') c.maType = c.maType === 'EMA' ? 'SMA' : 'EMA';
    else if (c.ind === 'MA_CROSS') {
      if (rng() < 0.5) c.fastType = c.fastType === 'EMA' ? 'SMA' : 'EMA';
      else c.slowType = c.slowType === 'EMA' ? 'SMA' : 'EMA';
    }
    return;
  }
  // Keep the direction, change between "state" and "event" (or the filter's side).
  const role = conditionRole(c);
  let options: CmpOp[];
  if (role === 'FILTER') options = ['gt', 'lt'];
  else if (role === 'TREND_UP') options = ['gt', 'cross_above'];
  else if (role === 'TREND_DOWN') options = ['lt', 'cross_below'];
  else options = ALL_OPS.filter((op) => op !== (role === 'OVERSOLD' ? 'gt' : 'lt'));
  const others = options.filter((op) => op !== c.op);
  if (others.length > 0) c.op = pick(rng, others);
}

function mutateExit(rng: Rng, g: StrategyGenome): void {
  const exit = g.exit;
  if (exit.mode === 'static') {
    g.exit = randomDynamicExit(rng, g.archetype);
    return;
  }
  const r = rng();
  if (r < 0.25) g.exit = randomStaticExit(rng);
  else if (r < 0.5) exit.trailing = !exit.trailing;
  else if (r < 0.75) exit.takeProfitAtrMult = exit.takeProfitAtrMult === null ? pick(rng, GRIDS.takeProfitAtrMult) : null;
  else exit.reversalExit = !exit.reversalExit;
}

function applyMutation(rng: Rng, g: StrategyGenome, limits: GenomeLimits): void {
  const kind = pickWeighted(rng, MUTATION_WEIGHTS);
  const { conditions } = g.longEntry;
  switch (kind) {
    case 'tweak':
      tweak(rng, g, limits);
      return;
    case 'operator':
      mutateOperator(rng, g);
      return;
    case 'trigger':
      conditions[0] = randomTrigger(rng, g.archetype, limits);
      return;
    case 'second':
      if (conditions.length > 1 && rng() < 0.4) conditions.pop();
      else conditions[1] = randomSecond(rng, g.archetype, limits);
      return;
    case 'exit':
      mutateExit(rng, g);
      return;
    case 'time':
      g.maxBarsInTrade = g.maxBarsInTrade === null || rng() < 0.5 ? pick(rng, [null, ...TIME_STOP_GRID]) : null;
      return;
    case 'archetype':
      g.archetype = g.archetype === 'TREND' ? 'MEAN_REVERSION' : 'TREND';
      g.longEntry = randomEntry(rng, g.archetype, limits);
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
  let child = cloneGenome(parent);
  let steps = 1 + (rng() < 0.3 ? 1 : 0) + (rng() < 0.1 ? 1 : 0);
  steps += Math.max(0, Math.round(intensity) - 1);
  for (let i = 0; i < steps; i++) {
    applyMutation(rng, child, limits);
    // Repair after every step so later steps always see a coherent genome.
    child = repairGenome(child, mode, limits, rng);
  }
  return child;
}

/** Block-level crossover: the entry of one parent with the exit of the other. */
export function crossoverGenomes(
  rng: Rng,
  a: StrategyGenome,
  b: StrategyGenome,
  mode: TradeMode,
  limits: GenomeLimits,
): StrategyGenome {
  const entryFromA = rng() < 0.5;
  const entrySource = cloneGenome(entryFromA ? a : b);
  const exitSource = cloneGenome(entryFromA ? b : a);
  const child: StrategyGenome = {
    archetype: entrySource.archetype,
    longEntry: entrySource.longEntry,
    shortEntry: null,
    exit: exitSource.exit,
    maxBarsInTrade: rng() < 0.5 ? entrySource.maxBarsInTrade : exitSource.maxBarsInTrade,
  };
  return repairGenome(child, mode, limits, rng);
}

/**
 * How a neighbour moves one parameter:
 *   pct20 – ±20%, snapped to the grid (the adjacent grid value when ±20% rounds back) – the
 *           post-run sensitivity report;
 *   step  – exactly one grid step down and one up – the neighbourhood used for ranking.
 */
export type NeighbourKind = 'pct20' | 'step';

/**
 * Every single-parameter neighbour of a strategy (see `NeighbourKind`). Each tunable
 * parameter is moved down and up once while every other parameter stays put.
 * Variants that would break the archetype, or that normalise back to an already
 * listed genome, are skipped.
 */
export function parameterVariants(g: StrategyGenome, mode: TradeMode, limits: GenomeLimits, kind: NeighbourKind = 'pct20'): StrategyGenome[] {
  const base = normalizeGenome(g, mode, limits);
  const baseSignature = genomeSignature(base);
  const seen = new Set<string>([baseSignature]);
  const variants: StrategyGenome[] = [];
  const count = parameterHandles(cloneGenome(base), limits).length;
  for (let index = 0; index < count; index++) {
    for (const factor of [0.8, 1.2]) {
      const copy = cloneGenome(base);
      const handle = parameterHandles(copy, limits)[index];
      if (!handle || handle.grid.length < 2) continue;
      const current = snap(handle.get(), handle.grid);
      const at = handle.grid.indexOf(current);
      let next: number;
      if (kind === 'step') {
        const neighbour = at + (factor < 1 ? -1 : 1);
        if (neighbour < 0 || neighbour >= handle.grid.length) continue;
        next = handle.grid[neighbour];
      } else {
        const target = current * factor;
        next = snap(target, handle.grid);
        if (next === current) {
          const neighbour = at + (target < current ? -1 : 1);
          if (neighbour < 0 || neighbour >= handle.grid.length) continue;
          next = handle.grid[neighbour];
        }
      }
      handle.set(next);
      const variant = normalizeGenome(copy, mode, limits);
      const signature = genomeSignature(variant);
      if (seen.has(signature) || !isCoherent(variant, limits)) continue;
      seen.add(signature);
      variants.push(variant);
    }
  }
  return variants;
}

/** The one-grid-step parameter neighbourhood used to rank strategies by plateau, not spike. */
export function stepNeighbours(g: StrategyGenome, mode: TradeMode, limits: GenomeLimits): StrategyGenome[] {
  return parameterVariants(g, mode, limits, 'step');
}

/* -------------------------------------------------------------------------- */
/*  Identity                                                                  */
/* -------------------------------------------------------------------------- */

function conditionKey(c: Condition): string {
  const record = c as unknown as Fields;
  return Object.keys(record)
    .sort()
    .map((k) => `${k}=${record[k]}`)
    .join(',');
}

/** Canonical string identity of a genome (derived parts are implied and not included). */
export function genomeSignature(g: StrategyGenome): string {
  const exit =
    g.exit.mode === 'static'
      ? `S:${g.exit.stopLossPct}:${g.exit.takeProfitPct}`
      : `D:${g.exit.atrPeriod}:${g.exit.stopAtrMult}:${g.exit.trailing ? 1 : 0}:${g.exit.takeProfitAtrMult ?? 'x'}:${g.exit.reversalExit ? 1 : 0}`;
  return `${g.archetype}|${g.longEntry.conditions.map(conditionKey).join(';')}|${g.shortEntry ? 'LS' : 'L'}|${exit}|${g.maxBarsInTrade ?? 'x'}`;
}

/**
 * Structural "niche" of a genome: archetype, indicator families and exit mode.
 * The leaderboard caps the number of strategies per niche.
 */
export function genomeNiche(g: StrategyGenome): string {
  const families = g.longEntry.conditions.map((c) => c.ind).sort();
  return `${g.archetype}|${families.join('+')}|${g.exit.mode}`;
}
