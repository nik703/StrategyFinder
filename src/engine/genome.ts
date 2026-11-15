/**
 * Strategy genome: market hypotheses, parameter grids, random generation,
 * mutation, crossover and repair.
 *
 * The search space is coherent but expressive enough to hold real multi-condition strategies:
 *
 *   • every run tests ONE market hypothesis (see `HYPOTHESES`), which lists the
 *     primitives each slot of an entry may use and fixes their direction;
 *   • an entry has up to three slots, combined with AND:
 *       1. a required ENTRY TRIGGER – a discrete event (a crossover, a channel
 *          breakout, an oscillator crossing back out of its extreme zone). It
 *          fires once per setup, so a strategy cannot re-enter the same trend
 *          leg on every candle after a stop or target is hit;
 *       2. an optional REGIME filter – a state (price above a slow EMA, a rising
 *          slow EMA) that says which market the trigger is meant for;
 *       3. an optional CONFIRMATION filter – a state (trend strength, volume);
 *   • exits: fixed or ATR brackets, a trailing stop, a time stop and an
 *     opposite-signal exit (the mirrored trigger closes the trade);
 *   • every strategy follows its hypothesis's archetype (TREND or MEAN_REVERSION);
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
  Hypothesis,
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

/**
 * Every indicator lookback comes from this list: a geometric grid in steps of about √2, so one
 * grid step is the same relative change everywhere and no two values are near-duplicates.
 */
export const LOOKBACK_GRID: readonly number[] = [5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200];
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
  atrPeriod: [7, 10, 14, 20, 28],
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

/** Trigger + regime filter + confirmation filter. */
export const MAX_CONDITIONS = 3;
/** Indicator-reversal exits are a single condition. */
export const MAX_EXIT_CONDITIONS = 1;
/** Hard cap on the number of tunable numbers in a strategy. */
export const MAX_PARAMETERS = 10;
/** Parameters above this count cost fitness. */
export const FREE_PARAMETERS = 6;
/** "Short-term" triggers look back at most this many bars; regime filters at least `SLOW_LOOKBACK`. */
export const SHORT_LOOKBACK = 30;
export const SLOW_LOOKBACK = 100;
/** Shortest EMA whose slope may serve as a regime filter. */
export const MIN_SLOPE_LOOKBACK = 56;

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
    case 'DONCHIAN':
      return { ...c, variant: c.variant === 'upper' ? 'lower' : 'upper', op: flipOp(c.op) };
    case 'MA_SLOPE':
      return { ...c, op: flipOp(c.op) };
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
    case 'DONCHIAN':
      return [['period', lookbacks(10, 112, limits)]];
    case 'MA_SLOPE':
      return [['period', lookbacks(MIN_SLOPE_LOOKBACK, 200, limits)]];
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
    } else if (c.ind === 'BB' || c.ind === 'DONCHIAN') {
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
    (c.ind === 'PRICE_MA' || c.ind === 'MA_CROSS' || c.ind === 'MA_SLOPE') &&
    c.op === 'gt' &&
    conditionLookback(c) >= SLOW_LOOKBACK &&
    limits.maxLookback >= SLOW_LOOKBACK
  );
}

/**
 * True when the long entry obeys the genome's hypothesis (see `conformsTo`), or – for a hand-built
 * genome without one – its archetype.
 */
export function isCoherent(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  if (g.hypothesis !== undefined) return conformsTo(g, limits);
  const { conditions } = g.longEntry;
  if (conditions.length < 1 || conditions.length > MAX_CONDITIONS) return false;
  if (!isValidTrigger(conditions[0], g.archetype, limits)) return false;
  return conditions.slice(1).every((c) => isValidSecond(c, g.archetype, limits));
}

/* -------------------------------------------------------------------------- */
/*  Market hypotheses                                                         */
/* -------------------------------------------------------------------------- */

export interface HypothesisInfo {
  id: Hypothesis;
  label: string;
  /** One-word name for compact UI. */
  short: string;
  archetype: Archetype;
  /** Why the effect should exist at all. */
  rationale: string;
  /** The entry-trigger events the search may use. */
  trigger: string;
  /** The regime and confirmation filters (and their fixed direction). */
  filter: string;
  /** True when every strategy must carry a regime filter. */
  filterRequired: boolean;
  /** What the slots replace – collinear indicators that measure the same thing. */
  replaces: string;
}

/** Entry slot a primitive fills. */
export type EntrySlot = 'trigger' | 'regime' | 'confirm';

/** One building block of an entry: a template the search tunes, and the test that recognises it. */
interface Primitive {
  id: string;
  slot: EntrySlot;
  /** Template condition (its numbers are redrawn from the grids), or null when the data cannot support it. */
  template: (limits: GenomeLimits) => Condition | null;
  matches: (c: Condition, limits: GenomeLimits) => boolean;
}

const hasGrid = (min: number, max: number, limits: GenomeLimits): boolean => lookbacks(min, max, limits).length > 0;

const PRIMITIVES = {
  // ---- Entry triggers: discrete events ----------------------------------------------------------
  EMA_CROSS: {
    id: 'EMA_CROSS',
    slot: 'trigger',
    template: () => ({ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'EMA', slowPeriod: 56, op: 'cross_above' }),
    matches: (c) => c.ind === 'MA_CROSS' && c.fastType === 'EMA' && c.slowType === 'EMA' && c.op === 'cross_above',
  },
  DONCHIAN_BREAKOUT: {
    id: 'DONCHIAN_BREAKOUT',
    slot: 'trigger',
    template: () => ({ ind: 'DONCHIAN', period: 20, variant: 'upper', op: 'cross_above' }),
    matches: (c) => c.ind === 'DONCHIAN' && c.variant === 'upper' && c.op === 'cross_above',
  },
  MACD_CROSS: {
    id: 'MACD_CROSS',
    slot: 'trigger',
    template: () => ({ ind: 'MACD', fast: 14, slow: 28, signal: 10, variant: 'signal', op: 'cross_above' }),
    matches: (c) => c.ind === 'MACD' && c.variant === 'signal' && c.op === 'cross_above',
  },
  BB_BREAKOUT: {
    id: 'BB_BREAKOUT',
    slot: 'trigger',
    template: () => ({ ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'cross_above', level: 0 }),
    matches: (c) => c.ind === 'BB' && c.variant === 'upper' && c.op === 'cross_above',
  },
  RSI_REJECTION: {
    id: 'RSI_REJECTION',
    slot: 'trigger',
    template: () => ({ ind: 'RSI', period: 14, op: 'cross_above', level: 30 }),
    matches: (c) => c.ind === 'RSI' && c.op === 'cross_above' && c.level < 50,
  },
  BB_REJECTION: {
    id: 'BB_REJECTION',
    slot: 'trigger',
    template: () => ({ ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'cross_above', level: 0 }),
    matches: (c) => c.ind === 'BB' && c.variant === 'lower' && c.op === 'cross_above',
  },
  // ---- Regime filters: slow states ---------------------------------------------------------------
  ABOVE_SLOW_EMA: {
    id: 'ABOVE_SLOW_EMA',
    slot: 'regime',
    template: (limits) => (hasGrid(SLOW_LOOKBACK, 200, limits) ? { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' } : null),
    matches: (c) => c.ind === 'PRICE_MA' && c.maType === 'EMA' && c.op === 'gt' && c.period >= SLOW_LOOKBACK,
  },
  SLOW_EMA_RISING: {
    id: 'SLOW_EMA_RISING',
    slot: 'regime',
    template: (limits) => (hasGrid(MIN_SLOPE_LOOKBACK, 200, limits) ? { ind: 'MA_SLOPE', maType: 'EMA', period: 112, op: 'gt' } : null),
    matches: (c) => c.ind === 'MA_SLOPE' && c.maType === 'EMA' && c.op === 'gt' && c.period >= MIN_SLOPE_LOOKBACK,
  },
  // ---- Confirmation filters: fast states ---------------------------------------------------------
  ADX_TRENDING: {
    id: 'ADX_TRENDING',
    slot: 'confirm',
    template: () => ({ ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 }),
    matches: (c) => c.ind === 'ADX' && c.variant === 'strength' && c.op === 'gt',
  },
  ADX_RANGING: {
    id: 'ADX_RANGING',
    slot: 'confirm',
    template: () => ({ ind: 'ADX', period: 14, variant: 'strength', op: 'lt', level: 20 }),
    matches: (c) => c.ind === 'ADX' && c.variant === 'strength' && c.op === 'lt',
  },
  VOLUME_SURGE: {
    id: 'VOLUME_SURGE',
    slot: 'confirm',
    template: (limits) => (limits.hasVolume ? { ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 } : null),
    matches: (c, limits) => c.ind === 'VOLUME' && c.op === 'gt' && limits.hasVolume,
  },
} satisfies Record<string, Primitive>;

type PrimitiveId = keyof typeof PRIMITIVES;

interface HypothesisSlots {
  triggers: readonly PrimitiveId[];
  regimes: readonly PrimitiveId[];
  confirms: readonly PrimitiveId[];
}

/** The primitives each hypothesis may put in each entry slot. */
const SLOTS: Readonly<Record<Hypothesis, HypothesisSlots>> = {
  TREND_FOLLOWING: { triggers: ['EMA_CROSS', 'DONCHIAN_BREAKOUT', 'MACD_CROSS'], regimes: ['ABOVE_SLOW_EMA', 'SLOW_EMA_RISING'], confirms: ['ADX_TRENDING'] },
  VOL_BREAKOUT: { triggers: ['DONCHIAN_BREAKOUT', 'BB_BREAKOUT'], regimes: ['ABOVE_SLOW_EMA', 'SLOW_EMA_RISING'], confirms: ['VOLUME_SURGE'] },
  DIP_IN_UPTREND: { triggers: ['RSI_REJECTION', 'BB_REJECTION'], regimes: ['ABOVE_SLOW_EMA', 'SLOW_EMA_RISING'], confirms: ['ADX_TRENDING'] },
  RANGE_REVERSION: { triggers: ['RSI_REJECTION', 'BB_REJECTION'], regimes: [], confirms: ['ADX_RANGING'] },
};

/**
 * The market hypotheses a run can test. Each is a distinct, economically motivated idea; its
 * slots list the event triggers that express it and the state filters that say when it applies:
 *
 *   TREND_FOLLOWING  EMA cross · Donchian breakout · MACD cross  | close > slow EMA · slow EMA rising | ADX > level
 *   VOL_BREAKOUT     Donchian breakout · upper Bollinger break   | close > slow EMA · slow EMA rising | volume surge
 *   DIP_IN_UPTREND   RSI back above oversold · back inside lower band | REQUIRED: close > slow EMA or slow EMA rising | ADX > level
 *   RANGE_REVERSION  RSI back above oversold · back inside lower band | –                                  | ADX < level
 */
export const HYPOTHESES: Readonly<Record<Hypothesis, HypothesisInfo>> = {
  TREND_FOLLOWING: {
    id: 'TREND_FOLLOWING',
    label: 'Trend following',
    short: 'Trend',
    archetype: 'TREND',
    rationale:
      'Time-series momentum: markets that have been rising tend to keep rising for a while, because information spreads slowly and money chases performance.',
    trigger: 'An EMA crossover, a Donchian-channel breakout or a MACD signal cross (one event per setup).',
    filter: 'Optional regime: close above a slow EMA or a rising slow EMA. Optional confirmation: ADX above a level.',
    filterRequired: false,
    replaces: 'Supertrend, +DI/−DI and price-vs-MA states, which all measure the same trend.',
  },
  VOL_BREAKOUT: {
    id: 'VOL_BREAKOUT',
    label: 'Volatility breakout',
    short: 'Breakout',
    archetype: 'TREND',
    rationale:
      'A close beyond the recent range or volatility envelope starts a move the old range no longer contains; heavy volume shows real participation behind it.',
    trigger: 'Close breaks above the prior Donchian high or the upper Bollinger band.',
    filter: 'Optional regime: close above a slow EMA or a rising slow EMA. Optional confirmation: volume above its average.',
    filterRequired: false,
    replaces: 'Band-width and ATR "squeeze" filters, which describe the same volatility state.',
  },
  DIP_IN_UPTREND: {
    id: 'DIP_IN_UPTREND',
    label: 'Buy the dip in an uptrend',
    short: 'Dip-buy',
    archetype: 'MEAN_REVERSION',
    rationale:
      'Short-term overreaction: sharp pullbacks inside a long-term uptrend tend to be partly reversed, as buyers are paid to absorb forced selling.',
    trigger: 'Pullback rejection: RSI crosses back above an oversold level, or the close crosses back above the lower Bollinger band.',
    filter: 'Required regime: close above a slow EMA or a rising slow EMA. Optional confirmation: ADX above a level.',
    filterRequired: true,
    replaces: 'Stochastic, CCI and %B, which all measure "short-horizon oversold".',
  },
  RANGE_REVERSION: {
    id: 'RANGE_REVERSION',
    label: 'Range reversion',
    short: 'Range',
    archetype: 'MEAN_REVERSION',
    rationale: 'In trendless markets prices oscillate around a mean, so oversold readings tend to revert.',
    trigger: 'RSI crosses back above an oversold level, or the close crosses back above the lower Bollinger band.',
    filter: 'Optional confirmation: ADX below a level – only trade when no trend is present.',
    filterRequired: false,
    replaces: 'Stochastic, CCI and %B, which all measure "short-horizon oversold".',
  },
};

/** Every hypothesis may close a trade on the opposite signal (the mirrored trigger); hand-built MEAN_REVERSION genomes may not. */
export function allowsReversalExit(h: Hypothesis | undefined, archetype: Archetype): boolean {
  return h === undefined ? archetype === 'TREND' : true;
}

/** Why `h` cannot be tested on data with these limits, or null when it can. */
export function hypothesisUnsupported(h: Hypothesis, limits: GenomeLimits): string | null {
  if (HYPOTHESES[h].filterRequired && SLOTS[h].regimes.every((id) => PRIMITIVES[id].template(limits) === null)) {
    return `${HYPOTHESES[h].label} needs a slow trend filter of ${MIN_SLOPE_LOOKBACK}+ candles, which needs at least about ${MIN_SLOPE_LOOKBACK * 5} candles of history.`;
  }
  return null;
}

function slotPrimitive(c: Condition, ids: readonly PrimitiveId[], limits: GenomeLimits): PrimitiveId | null {
  return ids.find((id) => PRIMITIVES[id].matches(c, limits)) ?? null;
}

/** True when `c` is one of the entry-trigger events of `h`. */
export function triggerConforms(c: Condition, h: Hypothesis, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  return slotPrimitive(c, SLOTS[h].triggers, limits) !== null;
}

/** The filter slot `c` fills in `h` (regime or confirmation), or null when it is not one of its filters. */
export function filterSlot(c: Condition, h: Hypothesis, limits: GenomeLimits): 'regime' | 'confirm' | null {
  if (slotPrimitive(c, SLOTS[h].regimes, limits) !== null) return 'regime';
  if (slotPrimitive(c, SLOTS[h].confirms, limits) !== null) return 'confirm';
  return null;
}

/** True when `c` is a regime or confirmation filter of `h`, pointing in the hypothesis's direction. */
export function filterConforms(c: Condition, h: Hypothesis, limits: GenomeLimits): boolean {
  return filterSlot(c, h, limits) !== null;
}

/** True when a genome is built exactly from its hypothesis's slots: trigger, then at most one regime and one confirmation filter. */
export function conformsTo(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  const h = g.hypothesis;
  if (h === undefined) return false;
  const spec = HYPOTHESES[h];
  const { conditions } = g.longEntry;
  if (g.archetype !== spec.archetype || conditions.length < 1 || conditions.length > MAX_CONDITIONS) return false;
  if (!triggerConforms(conditions[0], h, limits)) return false;
  const slots = conditions.slice(1).map((c) => filterSlot(c, h, limits));
  if (slots.some((slot) => slot === null)) return false;
  // Canonical order: regime before confirmation, one of each at most.
  const order = slots.map((slot) => (slot === 'regime' ? 0 : 1));
  for (let k = 1; k < order.length; k++) if (order[k] <= order[k - 1]) return false;
  if (spec.filterRequired && !slots.includes('regime')) return false;
  if (g.exit.mode === 'dynamic' && g.exit.reversalExit && !allowsReversalExit(h, g.archetype)) return false;
  return true;
}

/**
 * The hypothesis a hand-built genome (one without `hypothesis`) belongs to: the first one of its
 * archetype whose triggers it already uses, else the archetype's default.
 */
export function inferHypothesis(g: StrategyGenome): Hypothesis {
  const trigger = g.longEntry.conditions[0];
  const filters = g.longEntry.conditions.slice(1);
  const candidates: Hypothesis[] =
    g.archetype === 'TREND' ? ['TREND_FOLLOWING', 'VOL_BREAKOUT'] : ['DIP_IN_UPTREND', 'RANGE_REVERSION'];
  if (trigger) {
    for (const h of candidates) {
      if (!triggerConforms(trigger, h)) continue;
      // An oversold rejection inside a slow uptrend filter is the dip-buy idea.
      if (h === 'DIP_IN_UPTREND' && !filters.some((c) => filterSlot(c, h, DEFAULT_LIMITS) === 'regime')) continue;
      return h;
    }
  }
  if (g.archetype === 'TREND') return 'TREND_FOLLOWING';
  // A mean-reversion entry gated by a slow up-trend is the dip-buy idea, whatever oscillator it used.
  const slowUptrend = filters.some(
    (c) => (c.ind === 'PRICE_MA' || c.ind === 'MA_CROSS' || c.ind === 'MA_SLOPE') && c.op === 'gt' && conditionLookback(c) >= SLOW_LOOKBACK,
  );
  return slowUptrend ? 'DIP_IN_UPTREND' : 'RANGE_REVERSION';
}

/** Draw every tunable parameter of `c` uniformly from its grid. */
function randomizeParams(rng: Rng, c: Condition, slot: Slot, limits: GenomeLimits): Condition {
  const fields = c as unknown as Fields;
  for (const [key, grid] of conditionSpecs(c, slot, limits)) {
    if (grid.length > 0) fields[key] = pick(rng, grid);
  }
  return normalizeCondition(c, slot, limits);
}

/** A random instance of primitive `id` with its numbers drawn from the grids, or null when the data cannot support it. */
function drawPrimitive(rng: Rng, id: PrimitiveId, archetype: Archetype, limits: GenomeLimits): Condition | null {
  const primitive: Primitive = PRIMITIVES[id];
  const template = primitive.template(limits);
  if (template === null) return null;
  const slot: Slot = primitive.slot === 'trigger' ? (archetype === 'TREND' ? 'trend-trigger' : 'reversion-trigger') : primitive.slot === 'regime' ? 'regime' : 'filter';
  for (let attempt = 0; attempt < 8; attempt++) {
    const c = randomizeParams(rng, { ...template } as Condition, slot, limits);
    if (primitive.matches(c, limits)) return c;
  }
  return normalizeCondition(template, slot, limits);
}

/** A random entry trigger of `h` (optionally a different primitive from `avoid`). */
export function hypothesisTrigger(rng: Rng, h: Hypothesis, limits: GenomeLimits, avoid: Condition | null = null): Condition {
  const archetype = HYPOTHESES[h].archetype;
  let ids = SLOTS[h].triggers;
  if (avoid !== null && ids.length > 1) {
    const current = slotPrimitive(avoid, ids, limits);
    ids = ids.filter((id) => id !== current);
  }
  const shuffled = [...ids].sort(() => rng() - 0.5);
  for (const id of shuffled) {
    const c = drawPrimitive(rng, id, archetype, limits);
    if (c) return c;
  }
  throw new Error(`No entry trigger of ${h} fits this data.`);
}

/** A random filter of `h` for `slot`, or null when the hypothesis has none (or the data cannot support it). */
export function hypothesisSlotFilter(rng: Rng, h: Hypothesis, slot: 'regime' | 'confirm', limits: GenomeLimits): Condition | null {
  const ids = slot === 'regime' ? SLOTS[h].regimes : SLOTS[h].confirms;
  const shuffled = [...ids].sort(() => rng() - 0.5);
  for (const id of shuffled) {
    const c = drawPrimitive(rng, id, HYPOTHESES[h].archetype, limits);
    if (c) return c;
  }
  return null;
}

/** A random filter of `h`: its regime filter when one is required, otherwise either kind. */
export function hypothesisFilter(rng: Rng, h: Hypothesis, limits: GenomeLimits): Condition | null {
  if (HYPOTHESES[h].filterRequired) return hypothesisSlotFilter(rng, h, 'regime', limits);
  const first: 'regime' | 'confirm' = SLOTS[h].regimes.length > 0 && rng() < 0.5 ? 'regime' : 'confirm';
  return hypothesisSlotFilter(rng, h, first, limits) ?? hypothesisSlotFilter(rng, h, first === 'regime' ? 'confirm' : 'regime', limits);
}

/** The primitives of every slot of `h` (for the UI and tests). */
export function hypothesisPrimitives(h: Hypothesis): HypothesisSlots {
  return SLOTS[h];
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

function randomDynamicExit(rng: Rng, allowReversal: boolean): DynamicExit {
  const trailing = rng() < 0.4;
  const reversalExit = allowReversal && rng() < 0.35;
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
    ...(g.hypothesis !== undefined ? { hypothesis: g.hypothesis } : {}),
    longEntry: cloneGroup(g.longEntry),
    shortEntry: g.shortEntry ? cloneGroup(g.shortEntry) : null,
    exit,
    maxBarsInTrade: g.maxBarsInTrade,
  };
}

/** A tunable number inside a genome, with the grid it lives on. */
interface Handle {
  /** Human-readable name, e.g. "RSI level" or "Stop × ATR". */
  label: string;
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
      handles.push({ label: parameterLabel(c, key, index), grid, get: () => fields[key] as number, set: (v) => (fields[key] = v) });
    }
  });
  const exit = g.exit;
  if (exit.mode === 'static') {
    handles.push({ label: 'Stop loss %', grid: GRIDS.stopLossPct, get: () => exit.stopLossPct, set: (v) => (exit.stopLossPct = v) });
    handles.push({ label: 'Take profit %', grid: GRIDS.takeProfitPct, get: () => exit.takeProfitPct, set: (v) => (exit.takeProfitPct = v) });
  } else {
    handles.push({ label: 'ATR period', grid: GRIDS.atrPeriod, get: () => exit.atrPeriod, set: (v) => (exit.atrPeriod = v) });
    handles.push({ label: 'Stop × ATR', grid: GRIDS.stopAtrMult, get: () => exit.stopAtrMult, set: (v) => (exit.stopAtrMult = v) });
    if (exit.takeProfitAtrMult !== null) {
      handles.push({
        label: 'Target × ATR',
        grid: GRIDS.takeProfitAtrMult,
        get: () => exit.takeProfitAtrMult as number,
        set: (v) => (exit.takeProfitAtrMult = v),
      });
    }
  }
  if (g.maxBarsInTrade !== null) {
    handles.push({ label: 'Time stop (candles)', grid: TIME_STOP_GRID, get: () => g.maxBarsInTrade as number, set: (v) => (g.maxBarsInTrade = v) });
  }
  return handles;
}

const KEY_NAMES: Record<string, string> = {
  period: 'period',
  level: 'level',
  mult: 'width (σ)',
  fast: 'fast period',
  slow: 'slow period',
  signal: 'signal period',
  fastPeriod: 'fast period',
  slowPeriod: 'slow period',
  kPeriod: '%K period',
};

/** "RSI level", "EMA period", "Filter ADX level", … */
function parameterLabel(c: Condition, key: string, index: number): string {
  const name =
    c.ind === 'PRICE_MA'
      ? c.maType
      : c.ind === 'MA_SLOPE'
        ? `${c.maType} slope`
        : c.ind === 'DONCHIAN'
          ? 'Channel'
          : c.ind === 'BB'
            ? 'Band'
            : c.ind === 'VOLUME'
              ? 'Volume'
              : c.ind === 'MA_CROSS'
                ? 'MA'
                : c.ind;
  const what = c.ind === 'VOLUME' && key === 'level' ? 'ratio' : (KEY_NAMES[key] ?? key);
  return `${index > 0 ? (isFilter(c) ? 'Confirm ' : 'Regime ') : ''}${name} ${what}`;
}

/** One tunable parameter of a genome: its name, its grid and its current value. */
export interface ParameterAxis {
  label: string;
  grid: readonly number[];
  value: number;
}

/** Every tunable parameter of `g`, in a stable order (entry conditions, then exit, then time stop). */
export function parameterAxes(g: StrategyGenome, limits: GenomeLimits): ParameterAxis[] {
  return parameterHandles(cloneGenome(g), limits).map((h) => ({ label: h.label, grid: h.grid, value: snap(h.get(), h.grid) }));
}

/**
 * `g` with some parameters (by `parameterAxes` index) set to other grid values, normalised.
 * Null when the result is not a coherent strategy or normalisation would move a requested value.
 */
export function withParameterValues(
  g: StrategyGenome,
  mode: TradeMode,
  limits: GenomeLimits,
  values: readonly (readonly [index: number, value: number])[],
): StrategyGenome | null {
  const copy = cloneGenome(g);
  const handles = parameterHandles(copy, limits);
  for (const [index, value] of values) {
    if (!handles[index]) return null;
    handles[index].set(value);
  }
  const result = normalizeGenome(copy, mode, limits);
  if (!isCoherent(result, limits)) return null;
  const after = parameterHandles(cloneGenome(result), limits);
  for (const [index, value] of values) if (!after[index] || after[index].get() !== value) return null;
  return result;
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
  const hypothesis = input.hypothesis;
  const archetype: Archetype =
    hypothesis !== undefined ? HYPOTHESES[hypothesis].archetype : input.archetype === 'MEAN_REVERSION' ? 'MEAN_REVERSION' : 'TREND';
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
    const reversalExit = allowsReversalExit(hypothesis, archetype) && input.exit.reversalExit && conditions.length > 0;
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

  const genome: StrategyGenome = {
    archetype,
    ...(hypothesis !== undefined ? { hypothesis } : {}),
    longEntry,
    shortEntry: null,
    exit,
    maxBarsInTrade,
  };

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

/**
 * Enforce every invariant, including the hypothesis: a trigger that is not the hypothesis's
 * primitive is redrawn, a filter that is not its filter (or points the wrong way) is dropped, and a
 * required filter is added. A hand-built genome without a hypothesis is assigned one first (see
 * `inferHypothesis`), so everything that leaves this function conforms to exactly one hypothesis.
 */
export function repairGenome(g: StrategyGenome, mode: TradeMode, limits: GenomeLimits, rng: Rng): StrategyGenome {
  const hypothesis = g.hypothesis ?? inferHypothesis(g);
  const spec = HYPOTHESES[hypothesis];
  let genome = normalizeGenome({ ...g, hypothesis, archetype: spec.archetype }, mode, limits);
  const { conditions } = genome.longEntry;
  const { trigger, regime, confirm } = entrySlots(genome, limits);
  const slots = {
    trigger: trigger ?? hypothesisTrigger(rng, hypothesis, limits),
    regime: regime ?? (spec.filterRequired ? hypothesisSlotFilter(rng, hypothesis, 'regime', limits) : null),
    confirm,
  };
  const next = [slots.trigger, slots.regime, slots.confirm].filter((c): c is Condition => c !== null);
  if (next.length !== conditions.length || next.some((c, k) => c !== conditions[k])) {
    genome.longEntry = { logic: 'AND', conditions: next };
    genome = normalizeGenome(genome, mode, limits);
  }
  return genome;
}

/**
 * The entry slots of a genome built from a hypothesis: its trigger (when it is one of the
 * hypothesis's events) and its first regime and confirmation filter. Anything else is dropped.
 */
export function entrySlots(g: StrategyGenome, limits: GenomeLimits): { trigger: Condition | null; regime: Condition | null; confirm: Condition | null } {
  const h = g.hypothesis ?? inferHypothesis(g);
  const [first, ...rest] = g.longEntry.conditions;
  let regime: Condition | null = null;
  let confirm: Condition | null = null;
  for (const c of rest) {
    const slot = filterSlot(c, h, limits);
    if (slot === 'regime' && regime === null) regime = c;
    else if (slot === 'confirm' && confirm === null) confirm = c;
  }
  return { trigger: first !== undefined && triggerConforms(first, h, limits) ? first : null, regime, confirm };
}

/** Probability that a random strategy carries a regime filter / a confirmation filter (when optional). */
const REGIME_RATE = 0.6;
const CONFIRM_RATE = 0.4;

/** A random strategy built from the slots of hypothesis `h`. */
export function randomGenome(rng: Rng, mode: TradeMode, limits: GenomeLimits, h: Hypothesis): StrategyGenome {
  const spec = HYPOTHESES[h];
  const conditions = [hypothesisTrigger(rng, h, limits)];
  if (spec.filterRequired || rng() < REGIME_RATE) {
    const regime = hypothesisSlotFilter(rng, h, 'regime', limits);
    if (regime) conditions.push(regime);
  }
  if (rng() < CONFIRM_RATE) {
    const confirm = hypothesisSlotFilter(rng, h, 'confirm', limits);
    if (confirm) conditions.push(confirm);
  }
  const genome: StrategyGenome = {
    archetype: spec.archetype,
    hypothesis: h,
    longEntry: { logic: 'AND', conditions },
    shortEntry: null,
    exit: rng() < 0.45 ? randomStaticExit(rng) : randomDynamicExit(rng, allowsReversalExit(h, spec.archetype)),
    maxBarsInTrade: rng() < 0.2 ? pick(rng, TIME_STOP_GRID) : null,
  };
  return repairGenome(genome, mode, limits, rng);
}

/* -------------------------------------------------------------------------- */
/*  Mutation and crossover                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Mutations stay inside the hypothesis: they tune numbers, swap or redraw the entry trigger, add /
 * drop / swap the regime or confirmation filter, or change the exit. They never flip a filter's
 * direction or change the archetype.
 */
type MutationKind = 'tweak' | 'trigger' | 'regime' | 'confirm' | 'exit' | 'time';

const MUTATION_WEIGHTS: readonly (readonly [MutationKind, number])[] = [
  ['tweak', 0.45],
  ['trigger', 0.1],
  ['regime', 0.12],
  ['confirm', 0.1],
  ['exit', 0.15],
  ['time', 0.08],
];

/** Attempts at a random mutation before falling back to a guaranteed-distinct neighbour. */
const MUTATION_ATTEMPTS = 16;

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

function mutateExit(rng: Rng, g: StrategyGenome, allowReversal: boolean): void {
  const exit = g.exit;
  if (exit.mode === 'static') {
    g.exit = randomDynamicExit(rng, allowReversal);
    return;
  }
  const r = rng();
  const options = allowReversal ? 4 : 3;
  const k = Math.floor(r * options);
  if (k === 0) g.exit = randomStaticExit(rng);
  else if (k === 1) exit.trailing = !exit.trailing;
  else if (k === 2) exit.takeProfitAtrMult = exit.takeProfitAtrMult === null ? pick(rng, GRIDS.takeProfitAtrMult) : null;
  else exit.reversalExit = !exit.reversalExit;
}

function applyMutation(rng: Rng, g: StrategyGenome, h: Hypothesis, limits: GenomeLimits): void {
  const kind = pickWeighted(rng, MUTATION_WEIGHTS);
  const spec = HYPOTHESES[h];
  const slots = entrySlots(g, limits);
  const rebuild = (): void => {
    const trigger = slots.trigger ?? hypothesisTrigger(rng, h, limits);
    g.longEntry = { logic: 'AND', conditions: [trigger, slots.regime, slots.confirm].filter((c): c is Condition => c !== null) };
  };
  switch (kind) {
    case 'tweak':
      tweak(rng, g, limits);
      return;
    case 'trigger':
      // Half the time another event of the hypothesis, otherwise fresh numbers for any of them.
      slots.trigger = hypothesisTrigger(rng, h, limits, rng() < 0.5 ? slots.trigger : null);
      rebuild();
      return;
    case 'regime':
    case 'confirm': {
      const slot = kind;
      const required = slot === 'regime' && spec.filterRequired;
      if (slots[slot] !== null && !required && rng() < 0.4) {
        slots[slot] = null;
      } else {
        const filter = hypothesisSlotFilter(rng, h, slot, limits);
        if (filter === null) {
          tweak(rng, g, limits);
          return;
        }
        slots[slot] = filter;
      }
      rebuild();
      return;
    }
    case 'exit':
      mutateExit(rng, g, allowsReversalExit(h, spec.archetype));
      return;
    case 'time': {
      if (g.maxBarsInTrade === null) g.maxBarsInTrade = pick(rng, TIME_STOP_GRID);
      else if (rng() < 0.5) g.maxBarsInTrade = null;
      else g.maxBarsInTrade = pick(rng, TIME_STOP_GRID.filter((v) => v !== g.maxBarsInTrade));
      return;
    }
  }
}

/**
 * Produce a mutated child that is GUARANTEED to differ from its parent: random mutations are
 * retried until repair no longer maps the child back onto the parent, then a random one-grid-step
 * neighbour is used, and as a last resort the exit mode is switched. A no-op mutation would cost a
 * full evaluation and teach the search nothing. `intensity` (≥ 1) adds extra mutation steps so the
 * search can take larger jumps when it is stagnating.
 */
export function mutateGenome(
  rng: Rng,
  parent: StrategyGenome,
  mode: TradeMode,
  limits: GenomeLimits,
  intensity = 1,
): StrategyGenome {
  const h = parent.hypothesis;
  if (h === undefined) throw new Error('Only genomes built from a market hypothesis can be mutated.');
  const parentSignature = genomeSignature(normalizeGenome(parent, mode, limits));
  for (let attempt = 0; attempt < MUTATION_ATTEMPTS; attempt++) {
    let child = cloneGenome(parent);
    let steps = 1 + (rng() < 0.3 ? 1 : 0) + (rng() < 0.1 ? 1 : 0);
    steps += Math.max(0, Math.round(intensity) - 1);
    for (let i = 0; i < steps; i++) {
      applyMutation(rng, child, h, limits);
      // Repair after every step so later steps always see a coherent genome.
      child = repairGenome(child, mode, limits, rng);
    }
    if (genomeSignature(child) !== parentSignature) return child;
  }
  const neighbours = stepNeighbours(parent, mode, limits);
  if (neighbours.length > 0) return neighbours[Math.floor(rng() * neighbours.length)];
  const child = cloneGenome(parent);
  child.exit = parent.exit.mode === 'static' ? randomDynamicExit(rng, allowsReversalExit(h, parent.archetype)) : randomStaticExit(rng);
  return repairGenome(child, mode, limits, rng);
}

/** Block-level crossover: the entry of one parent with the exit of the other (both share the run's hypothesis). */
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
    ...(entrySource.hypothesis !== undefined ? { hypothesis: entrySource.hypothesis } : {}),
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
  return `${g.hypothesis ?? '-'}|${g.archetype}|${g.longEntry.conditions.map(conditionKey).join(';')}|${g.shortEntry ? 'LS' : 'L'}|${exit}|${g.maxBarsInTrade ?? 'x'}`;
}

/**
 * Structural "niche" of a genome: archetype, indicator families and exit mode.
 * The leaderboard caps the number of strategies per niche.
 */
export function genomeNiche(g: StrategyGenome): string {
  // Within one hypothesis the structure is: filter or not, state or event trigger, and the exit design.
  const families = g.longEntry.conditions.map((c) => c.ind).sort();
  const trigger = g.longEntry.conditions[0]?.op ?? '';
  const exit =
    g.exit.mode === 'static'
      ? 'S'
      : `D${g.exit.trailing ? 't' : ''}${g.exit.takeProfitAtrMult !== null ? 'p' : ''}${g.exit.reversalExit ? 'r' : ''}`;
  return `${g.hypothesis ?? g.archetype}|${families.join('+')}|${trigger}|${exit}${g.maxBarsInTrade === null ? '' : 'T'}`;
}
