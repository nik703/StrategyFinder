/**
 * Strategy genome: market hypotheses, parameter grids, random generation,
 * mutation, crossover and repair.
 *
 * The search space is deliberately small and coherent, so the optimiser cannot
 * assemble a pile of unrelated, finely tuned rules:
 *
 *   • every run tests ONE market hypothesis (see `HYPOTHESES`): it fixes the
 *     single trigger primitive, the filter and the filter's direction. Collinear
 *     indicators are not separate ideas – RSI, Stochastic, CCI and %B all measure
 *     "short-horizon oversold", and Supertrend, MA crosses and MACD all measure
 *     "trend" – so each idea is expressed by one primitive and the search only
 *     tunes its numbers;
 *   • every strategy follows its hypothesis's archetype (TREND or MEAN_REVERSION);
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

/**
 * True when the long entry obeys the genome's hypothesis (see `conformsTo`), or – for a hand-built
 * genome without one – its archetype.
 */
export function isCoherent(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  if (g.hypothesis !== undefined) return conformsTo(g, limits);
  const { conditions } = g.longEntry;
  if (conditions.length < 1 || conditions.length > MAX_CONDITIONS) return false;
  if (!isValidTrigger(conditions[0], g.archetype, limits)) return false;
  return conditions.length === 1 || isValidSecond(conditions[1], g.archetype, limits);
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
  /** The one trigger primitive the search tunes. */
  trigger: string;
  /** The filter (and its fixed direction), or why there is none. */
  filter: string;
  filterRequired: boolean;
  /** The indicators this hypothesis replaces – they measure the same thing. */
  replaces: string;
}

/**
 * The market hypotheses a run can test. Each one is a distinct, economically motivated idea,
 * expressed by exactly one trigger primitive and one filter whose direction is part of the idea:
 *
 *   TREND_FOLLOWING  close vs. its EMA                          + optional ADX above a level
 *   VOL_BREAKOUT     close crosses above the upper Bollinger band + optional volume above average
 *   DIP_IN_UPTREND   RSI oversold                               + REQUIRED close above a slow EMA
 *   RANGE_REVERSION  RSI oversold                               + optional ADX below a level
 */
export const HYPOTHESES: Readonly<Record<Hypothesis, HypothesisInfo>> = {
  TREND_FOLLOWING: {
    id: 'TREND_FOLLOWING',
    label: 'Trend following',
    short: 'Trend',
    archetype: 'TREND',
    rationale:
      'Time-series momentum: markets that have been rising tend to keep rising for a while, because information spreads slowly and money chases performance.',
    trigger: 'Close above its EMA (a state) or crossing above it (an event).',
    filter: 'Optional: ADX above a level – only trade when a trend is actually present.',
    filterRequired: false,
    replaces: 'Supertrend, moving-average crosses, MACD and +DI/−DI, which all measure the same trend.',
  },
  VOL_BREAKOUT: {
    id: 'VOL_BREAKOUT',
    label: 'Volatility breakout',
    short: 'Breakout',
    archetype: 'TREND',
    rationale:
      'A close beyond the recent volatility envelope starts a move the old range no longer contains; heavy volume shows real participation behind it.',
    trigger: 'Close crosses above the upper Bollinger band.',
    filter: 'Optional: volume above its average – participation confirms the breakout.',
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
    trigger: 'RSI oversold – below a level, or crossing back above it.',
    filter: 'Required: close above a slow EMA (100+ bars) – the long-term trend must be up.',
    filterRequired: true,
    replaces: 'Stochastic, CCI, %B and lower-band touches, which all measure "short-horizon oversold".',
  },
  RANGE_REVERSION: {
    id: 'RANGE_REVERSION',
    label: 'Range reversion',
    short: 'Range',
    archetype: 'MEAN_REVERSION',
    rationale: 'In trendless markets prices oscillate around a mean, so oversold readings tend to revert.',
    trigger: 'RSI oversold – below a level, or crossing back above it.',
    filter: 'Optional: ADX below a level – only trade when no trend is present.',
    filterRequired: false,
    replaces: 'Stochastic, CCI, %B and lower-band touches, which all measure "short-horizon oversold".',
  },
};

/** Operators the trigger of each hypothesis may use. */
const TRIGGER_OPS: Readonly<Record<Hypothesis, readonly CmpOp[]>> = {
  TREND_FOLLOWING: ['gt', 'cross_above'],
  VOL_BREAKOUT: ['cross_above'],
  DIP_IN_UPTREND: ['lt', 'cross_above'],
  RANGE_REVERSION: ['lt', 'cross_above'],
};

/** Only trend following may close on a reversal of its trigger (for a breakout or an oscillator it has no meaning). */
export function allowsReversalExit(h: Hypothesis | undefined, archetype: Archetype): boolean {
  return h === undefined ? archetype === 'TREND' : h === 'TREND_FOLLOWING';
}

/** Why `h` cannot be tested on data with these limits, or null when it can. */
export function hypothesisUnsupported(h: Hypothesis, limits: GenomeLimits): string | null {
  if (h === 'DIP_IN_UPTREND' && lookbacks(SLOW_LOOKBACK, 200, limits).length === 0) {
    return 'Buy the dip in an uptrend needs a slow trend filter of 100+ candles, which needs at least about 560 candles of history.';
  }
  return null;
}

/** True when `c` is the trigger primitive of `h`. */
export function triggerConforms(c: Condition, h: Hypothesis): boolean {
  if (!TRIGGER_OPS[h].includes(c.op)) return false;
  switch (h) {
    case 'TREND_FOLLOWING':
      return c.ind === 'PRICE_MA' && c.maType === 'EMA';
    case 'VOL_BREAKOUT':
      return c.ind === 'BB' && c.variant === 'upper';
    case 'DIP_IN_UPTREND':
    case 'RANGE_REVERSION':
      return c.ind === 'RSI' && c.level < 50;
  }
}

/** True when `c` is the filter of `h`, pointing in the hypothesis's direction. */
export function filterConforms(c: Condition, h: Hypothesis, limits: GenomeLimits): boolean {
  switch (h) {
    case 'TREND_FOLLOWING':
      return c.ind === 'ADX' && c.variant === 'strength' && c.op === 'gt';
    case 'VOL_BREAKOUT':
      return c.ind === 'VOLUME' && c.op === 'gt' && limits.hasVolume;
    case 'DIP_IN_UPTREND':
      return c.ind === 'PRICE_MA' && c.maType === 'EMA' && c.op === 'gt' && c.period >= SLOW_LOOKBACK;
    case 'RANGE_REVERSION':
      return c.ind === 'ADX' && c.variant === 'strength' && c.op === 'lt';
  }
}

/** True when a genome is built exactly from its hypothesis's primitives. */
export function conformsTo(g: StrategyGenome, limits: GenomeLimits = DEFAULT_LIMITS): boolean {
  const h = g.hypothesis;
  if (h === undefined) return false;
  const spec = HYPOTHESES[h];
  const { conditions } = g.longEntry;
  if (g.archetype !== spec.archetype || conditions.length < 1 || conditions.length > MAX_CONDITIONS) return false;
  if (!triggerConforms(conditions[0], h)) return false;
  if (conditions.length > 1 && !filterConforms(conditions[1], h, limits)) return false;
  if (conditions.length === 1 && spec.filterRequired) return false;
  if (g.exit.mode === 'dynamic' && g.exit.reversalExit && !allowsReversalExit(h, g.archetype)) return false;
  return true;
}

/**
 * The hypothesis a hand-built genome (one without `hypothesis`) belongs to: the first one of its
 * archetype whose trigger it already uses, else the archetype's default.
 */
export function inferHypothesis(g: StrategyGenome): Hypothesis {
  const trigger = g.longEntry.conditions[0];
  const candidates: Hypothesis[] =
    g.archetype === 'TREND' ? ['TREND_FOLLOWING', 'VOL_BREAKOUT'] : ['DIP_IN_UPTREND', 'RANGE_REVERSION'];
  if (trigger) {
    for (const h of candidates) {
      if (!triggerConforms(trigger, h)) continue;
      // An oscillator dip inside a slow uptrend filter is the dip-buy idea.
      if (h === 'DIP_IN_UPTREND' && !(g.longEntry.conditions[1] && filterConforms(g.longEntry.conditions[1], h, DEFAULT_LIMITS))) continue;
      return h;
    }
  }
  if (g.archetype === 'TREND') return 'TREND_FOLLOWING';
  // A mean-reversion entry gated by a slow up-trend is the dip-buy idea, whatever oscillator it used.
  const second = g.longEntry.conditions[1];
  const slowUptrend =
    second !== undefined && (second.ind === 'PRICE_MA' || second.ind === 'MA_CROSS') && second.op === 'gt' && conditionLookback(second) >= SLOW_LOOKBACK;
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

/** A random instance of the trigger primitive of `h`. */
export function hypothesisTrigger(rng: Rng, h: Hypothesis, limits: GenomeLimits): Condition {
  const op = pick(rng, TRIGGER_OPS[h]);
  switch (h) {
    case 'TREND_FOLLOWING':
      return randomizeParams(rng, { ind: 'PRICE_MA', maType: 'EMA', period: 56, op }, 'trend-trigger', limits);
    case 'VOL_BREAKOUT':
      return randomizeParams(rng, { ind: 'BB', period: 20, mult: 2, variant: 'upper', op, level: 0 }, 'trend-trigger', limits);
    case 'DIP_IN_UPTREND':
    case 'RANGE_REVERSION':
      return randomizeParams(rng, { ind: 'RSI', period: 14, op, level: 30 }, 'reversion-trigger', limits);
  }
}

/** A random instance of the filter of `h`, or null when the data cannot support it. */
export function hypothesisFilter(rng: Rng, h: Hypothesis, limits: GenomeLimits): Condition | null {
  switch (h) {
    case 'TREND_FOLLOWING':
      return randomizeParams(rng, { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 }, 'filter', limits);
    case 'VOL_BREAKOUT':
      return limits.hasVolume ? randomizeParams(rng, { ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 }, 'filter', limits) : null;
    case 'DIP_IN_UPTREND':
      return hypothesisUnsupported(h, limits) === null
        ? randomizeParams(rng, { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' }, 'regime', limits)
        : null;
    case 'RANGE_REVERSION':
      return randomizeParams(rng, { ind: 'ADX', period: 14, variant: 'strength', op: 'lt', level: 20 }, 'filter', limits);
  }
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
    c.ind === 'PRICE_MA' ? c.maType : c.ind === 'BB' ? 'Band' : c.ind === 'VOLUME' ? 'Volume' : c.ind === 'MA_CROSS' ? 'MA' : c.ind;
  const what = c.ind === 'VOLUME' && key === 'level' ? 'ratio' : (KEY_NAMES[key] ?? key);
  return `${index > 0 ? 'Filter ' : ''}${name} ${what}`;
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
  let changed = false;
  if (conditions.length === 0 || !triggerConforms(conditions[0], hypothesis)) {
    conditions[0] = hypothesisTrigger(rng, hypothesis, limits);
    changed = true;
  }
  if (conditions.length > 1 && !filterConforms(conditions[1], hypothesis, limits)) {
    conditions.length = 1;
    changed = true;
  }
  if (spec.filterRequired && conditions.length === 1) {
    const filter = hypothesisFilter(rng, hypothesis, limits);
    if (filter) {
      conditions.push(filter);
      changed = true;
    }
  }
  if (changed) genome = normalizeGenome(genome, mode, limits);
  return genome;
}

/** A random strategy built from the primitives of hypothesis `h`. */
export function randomGenome(rng: Rng, mode: TradeMode, limits: GenomeLimits, h: Hypothesis): StrategyGenome {
  const spec = HYPOTHESES[h];
  const conditions = [hypothesisTrigger(rng, h, limits)];
  if (spec.filterRequired || rng() < 0.5) {
    const filter = hypothesisFilter(rng, h, limits);
    if (filter) conditions.push(filter);
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
 * Mutations stay inside the hypothesis: they tune numbers, switch the trigger between its state
 * and event form, add / drop / redraw the hypothesis's filter, or change the exit. They never swap
 * the indicator family, flip a filter's direction or change the archetype.
 */
type MutationKind = 'tweak' | 'operator' | 'redraw' | 'filter' | 'exit' | 'time';

const MUTATION_WEIGHTS: readonly (readonly [MutationKind, number])[] = [
  ['tweak', 0.5],
  ['operator', 0.1],
  ['redraw', 0.05],
  ['filter', 0.13],
  ['exit', 0.14],
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
  const { conditions } = g.longEntry;
  const spec = HYPOTHESES[h];
  switch (kind) {
    case 'tweak':
      tweak(rng, g, limits);
      return;
    case 'operator': {
      // State ↔ event form of the trigger (e.g. "close above EMA" ↔ "close crosses above EMA").
      const others = TRIGGER_OPS[h].filter((op) => op !== conditions[0].op);
      if (others.length > 0) conditions[0].op = pick(rng, others);
      else tweak(rng, g, limits);
      return;
    }
    case 'redraw':
      conditions[0] = hypothesisTrigger(rng, h, limits);
      return;
    case 'filter': {
      if (!spec.filterRequired && conditions.length > 1 && rng() < 0.5) {
        conditions.length = 1;
        return;
      }
      const filter = hypothesisFilter(rng, h, limits);
      if (filter) conditions[1] = filter;
      else tweak(rng, g, limits);
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
