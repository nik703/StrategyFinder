/**
 * Next-bar entry & price solver.
 *
 * Answers, for a strategy and the latest closed candles:
 *   • Is an entry already pending for the next open?
 *   • Otherwise, at what CLOSE of the next candle would the long (or short) entry
 *     rule fire, what would the fill be, and where would the stop and target sit?
 *   • How ready is each side (0–100%), and which way does the strategy lean?
 *
 * Exact rule parity: a synthetic next candle (open = last close, high/low spanning
 * the last close and P, close = P) is appended to the trailing candles and run
 * through the very same `IndicatorCache`, `evaluateCondition` and `evaluateGroup`
 * the backtester uses. Nothing here re-implements indicator or rule maths.
 *
 * The trigger price is found with a bracketed grid scan of P across ±10 × ATR(14)
 * (capped at ±30%) around the last close, then bisection (48 steps) between the
 * nearest grid point where the rule is true and its false neighbour.
 */

import { BacktestWorkspace, evaluateCondition, evaluateGroup, fillSlippage, resolveOperands, runBacktest } from './backtester';
import { conditionOperands, conditionFormula } from './describe';
import { exitSpec } from './strategyAst';
import { conditionRole } from './genome';
import { IndicatorCache } from './indicators';
import { allowsShorts, type BacktestSettings, type CmpOp, type Condition, type OHLCV, type RuleGroup, type StrategyGenome } from '../types/trading';

/** Trailing candles used when probing the next bar (enough for every 200-bar lookback to converge). */
export const SOLVER_WINDOW = Number.POSITIVE_INFINITY;
/** Grid points across the scan range (odd, so the last close is a grid point). */
export const SCAN_POINTS = 241;
/** Bisection steps after the grid scan. */
export const BISECTION_STEPS = 48;
/** Scan range: ±10 × ATR(14) around the last close, capped at ±30%. */
export const SCAN_ATR_MULTIPLE = 10;
export const SCAN_MAX_FRACTION = 0.3;

/* -------------------------------------------------------------------------- */
/*  Result types                                                              */
/* -------------------------------------------------------------------------- */

export type Side = 'long' | 'short';

/**
 * pending      the rule fired on the last closed candle – the entry happens at the next open
 * now          the rule fires if the next candle closes at (about) the current price
 * trigger      the rule fires once the next candle closes beyond `triggerPrice`
 * blocked      price alone is not enough: a filter is false at every price in range
 *              (`triggerPrice`, when present, is where the rest of the rule would fire)
 * unreachable  no close within the scan range fires the rule
 * holding      a position on this side is already open
 * disabled     the side is not traded (Long-Only mode)
 */
export type SideStatus = 'pending' | 'now' | 'trigger' | 'blocked' | 'unreachable' | 'holding' | 'disabled';

export type CheckState = 'met' | 'pending' | 'blocked' | 'reset';

export interface ConditionCheck {
  formula: string;
  /** Role of the condition inside its rule. */
  job: 'Trigger' | 'Filter' | 'Regime filter' | 'Exit';
  state: CheckState;
  leftLabel: string;
  rightLabel: string;
  left: number;
  right: number;
  op: CmpOp;
  /** How far the indicator is from satisfying the condition, in `unit`s (null when met). */
  distance: number | null;
  unit: string;
  /** 0 (far) … 1 (met). */
  proximity: number;
  /** Plain-English status, e.g. "EMA(5) vs EMA(112): 0.42 ATR away from bullish cross [Pending]". */
  text: string;
}

export interface Blocker {
  formula: string;
  /** e.g. "ADX(7) 12.1 < 15". */
  detail: string;
}

export interface SideForecast {
  side: Side;
  status: SideStatus;
  /** Close of the next candle needed to fire the rule (the current close when pending / now). */
  triggerPrice: number | null;
  /** Price move from the last close to the trigger, in percent. */
  movePct: number | null;
  /** For `now`: the band of next closes that keeps the rule true (null bound = beyond the scan range). */
  activeRange: [number | null, number | null] | null;
  /** Estimated fill at the following open, including the configured slippage. */
  fillPrice: number | null;
  /** Initial stop at the fill (the backtester's level: ATR of the signal candle, or % of the fill). */
  stopLoss: number | null;
  /** Null when the exit has no fixed target (trailing / reversal / time stop). */
  takeProfit: number | null;
  /** 'trailing': the stop ratchets `trailAtrMult` × ATR behind the best price after every close; null without a stop. */
  stopMode: 'fixed' | 'trailing' | null;
  /** Trailing distance in ATRs (null for a fixed stop). */
  trailAtrMult: number | null;
  /** Breakeven lock: the price at +1R that moves the stop to the entry (null without a lock). */
  breakevenAt: number | null;
  blockers: Blocker[];
  /**
   * Conditions no next-candle close can change – a lagging filter (ADX, a slow average, volume), a
   * squeeze that is not on yet, a time-stop cooldown – reported alongside the trigger price: the
   * trigger alone is not enough while any of them stands.
   */
  nonPriceBlockers: Blocker[];
  /** Why the side cannot enter (unreachable / disabled / holding). */
  reason: string | null;
  /** Another rule that would suppress the entry at the trigger price. */
  conflict: string | null;
  conditions: ConditionCheck[];
  /** 0–100. */
  readiness: number;
  /** One-line summary for compact tables. */
  summary: string;
}

export interface PositionForecast {
  dir: Side;
  entryTime: number;
  entryPrice: number;
  barsHeld: number;
  /** Levels that apply to the next candle. */
  stopLoss: number | null;
  takeProfit: number | null;
  unrealisedPct: number;
  /** Set when an exit is already scheduled for the next open. */
  exitAtNextOpen: string | null;
  /** Bars left before the time stop closes the trade (null = no time stop). */
  timeStopIn: number | null;
  /** Breakeven lock: price at +1R that moves the stop to the entry (null without a lock, or once locked). */
  breakevenAt: number | null;
  /** True once the breakeven lock has moved the stop to the entry price. */
  breakevenLocked: boolean;
  /**
   * The stop after the NEXT candle closes, if it closes at the last close (a flat candle): the
   * engine's own update order applied to the carried state – breakeven lock first, then the
   * trailing ratchet behind the best price since entry, never moving against the trade.
   */
  projectedStop: number | null;
  /** How the stop is managed right now. */
  stopState: 'fixed' | 'trailing' | 'breakeven';
  /** Next-candle close that would trigger an indicator exit or a reversal, with its source. */
  signalExitPrice: number | null;
  signalExitSource: string | null;
}

export type Verdict = 'Strong Long' | 'Leaning Long' | 'Neutral / Waiting' | 'Leaning Short' | 'Strong Short' | 'In Position';

export interface NextEntryForecast {
  lastClose: number;
  lastTime: number;
  /** Open time of the next (forming) candle. */
  nextTime: number;
  atr: number;
  scanRange: [number, number];
  long: SideForecast;
  short: SideForecast;
  position: PositionForecast | null;
  verdict: Verdict;
  /** Notes about assumptions that affect the forecast. */
  notes: string[];
}

/**
 * The complete path-dependent state at the last closed candle, carried into the hypothetical next
 * candle by the solver. Indicator state (EMAs, Wilder averages, Supertrend, squeeze on / off) is
 * exact because the probe recomputes every indicator over the full history; the backtest state below
 * comes from the engine's own run over that history.
 */
export interface SolverState {
  /** The open position with its trailing-stop anchor, breakeven status and age, or null when flat. */
  position: (NonNullable<BacktestWorkspace['openPosition']> & { barsHeld: number }) | null;
  /** The time stop closes the position at the next open; that open cannot re-enter the same side (cooldown). */
  timeStopAtNextOpen: boolean;
  /** Consecutive closed candles each squeeze condition's Bollinger-inside-Keltner state has been on (by formula). */
  squeezeRun: Map<string, number>;
}

/** Consecutive candles up to `last` on which Bollinger(period, 2) sat inside Keltner(period, mult). */
function squeezeRunLength(ind: IndicatorCache, period: number, mult: number, last: number): number {
  const bbU = ind.bbUpper(period, 2);
  const bbL = ind.bbLower(period, 2);
  const kcU = ind.kcUpper(period, mult);
  const kcL = ind.kcLower(period, mult);
  let run = 0;
  for (let i = last; i >= 0 && bbU[i] < kcU[i] && bbL[i] > kcL[i]; i--) run++;
  return run;
}

/** The state vector of `genome` at the last closed candle of `data` (see `SolverState`). */
export function solverState(full: IndicatorCache, ws: BacktestWorkspace, genome: StrategyGenome): SolverState {
  const n = full.n;
  const open = ws.openPosition;
  const position = open ? { ...open, barsHeld: n - open.entryBar } : null;
  const maxBars = exitSpec(genome).timeStop;
  const squeezeRun = new Map<string, number>();
  for (const group of [genome.longEntry, genome.shortEntry]) {
    for (const c of group?.conditions ?? []) {
      if (c.ind === 'SQUEEZE') squeezeRun.set(conditionFormula(c), squeezeRunLength(full, c.period, c.mult, n - 1));
    }
  }
  return { position, timeStopAtNextOpen: position !== null && maxBars !== null && position.barsHeld >= maxBars, squeezeRun };
}

/**
 * The stop after a next candle with this `high` / `low` and ATR `atrNext` closes, from the carried
 * state – the backtester's order: breakeven lock (+R reached) first, then the trailing ratchet behind
 * the best price since entry; the stop never moves against the trade. Assumes the stop itself is not
 * hit by that candle.
 */
export function projectStop(state: SolverState, genome: StrategyGenome, high: number, low: number, atrNext: number): { stop: number; locked: boolean } | null {
  const p = state.position;
  if (!p) return null;
  const spec = exitSpec(genome);
  let stop = p.stop;
  let locked = p.breakevenLocked;
  const dir = p.dir;
  if (spec.breakevenR !== null && !locked && p.risk > 0 && (dir === 1 ? high >= p.entryPrice + spec.breakevenR * p.risk : low <= p.entryPrice - spec.breakevenR * p.risk)) {
    locked = true;
    if (dir === 1 ? p.entryPrice > stop : p.entryPrice < stop) stop = p.entryPrice;
  }
  if (spec.trailing !== null && atrNext === atrNext) {
    const extreme = dir === 1 ? Math.max(p.extreme, high) : Math.min(p.extreme, low);
    const next = dir === 1 ? extreme - spec.trailing.mult * atrNext : extreme + spec.trailing.mult * atrNext;
    if (dir === 1 ? next > stop : next < stop) stop = next;
  }
  return { stop, locked };
}

export interface SolverInput {
  data: OHLCV;
  genome: StrategyGenome;
  settings: BacktestSettings;
  /** Override the trailing window (tests). */
  window?: number;
}

/* -------------------------------------------------------------------------- */
/*  Probing the next candle                                                   */
/* -------------------------------------------------------------------------- */

/** The trailing candles plus one synthetic next candle whose close can be moved. */
class NextBarProbe {
  readonly data: OHLCV;
  /** Index of the synthetic candle. */
  readonly at: number;
  private readonly out: Uint8Array;
  private readonly scratch: Uint8Array;

  constructor(
    full: OHLCV,
    window: number,
    readonly lastClose: number,
    readonly nextTime: number,
    readonly assumedVolume: number,
  ) {
    const n = full.c.length;
    const w = Math.min(n, window);
    const from = n - w;
    const copy = (src: Float64Array): Float64Array => {
      const out = new Float64Array(w + 1);
      out.set(src.subarray(from, n));
      return out;
    };
    this.data = { t: copy(full.t), o: copy(full.o), h: copy(full.h), l: copy(full.l), c: copy(full.c), v: copy(full.v) };
    this.at = w;
    this.out = new Uint8Array(w + 1);
    this.scratch = new Uint8Array(w + 1);
  }

  /** Indicators with the next candle closing at `price`. */
  cache(price: number): IndicatorCache {
    const { data, at } = this;
    data.t[at] = this.nextTime;
    data.o[at] = this.lastClose;
    data.h[at] = Math.max(this.lastClose, price);
    data.l[at] = Math.min(this.lastClose, price);
    data.c[at] = price;
    data.v[at] = this.assumedVolume;
    // A fresh cache every time: the data underneath has changed.
    return new IndicatorCache(data, 64 * 1024 * 1024);
  }

  /** Is `c` true on the synthetic candle? (The backtester's own evaluator.) */
  condition(c: Condition, ind: IndicatorCache): boolean {
    evaluateCondition(c, ind, this.out, this.at + 1);
    return this.out[this.at] === 1;
  }

  /** Is `group` true on the synthetic candle? (The backtester's own evaluator.) */
  group(group: RuleGroup, ind: IndicatorCache): boolean {
    evaluateGroup(group, ind, this.out, this.scratch, this.at + 1);
    return this.out[this.at] === 1;
  }
}

/** A rule group restricted to some of its conditions (used to look past blocked filters). */
function subgroup(group: RuleGroup, keep: readonly boolean[]): RuleGroup {
  return { logic: group.logic, conditions: group.conditions.filter((_, i) => keep[i]) };
}

interface Scan {
  prices: Float64Array;
  centre: number;
  /** truth[k][i]: condition i of the group is true when the next candle closes at prices[k]. */
  truth: boolean[][];
}

function scanGroup(probe: NextBarProbe, group: RuleGroup, prices: Float64Array): Scan {
  const truth: boolean[][] = [];
  for (let k = 0; k < prices.length; k++) {
    const ind = probe.cache(prices[k]);
    truth.push(group.conditions.map((c) => probe.condition(c, ind)));
  }
  return { prices, centre: (prices.length - 1) >> 1, truth };
}

function combine(row: readonly boolean[], logic: RuleGroup['logic'], keep: readonly boolean[]): boolean {
  let any = false;
  let all = true;
  let used = 0;
  row.forEach((value, i) => {
    if (!keep[i]) return;
    used++;
    any ||= value;
    all &&= value;
  });
  if (used === 0) return false;
  return logic === 'AND' ? all : any;
}

/** Bisection between a price where `group` is false and one where it is true; returns the true side. */
function bisect(probe: NextBarProbe, group: RuleGroup, falseAt: number, trueAt: number): number {
  let a = falseAt;
  let b = trueAt;
  for (let i = 0; i < BISECTION_STEPS; i++) {
    const mid = (a + b) / 2;
    if (mid === a || mid === b) break;
    if (probe.group(group, probe.cache(mid))) b = mid;
    else a = mid;
  }
  return b;
}

interface Solved {
  /** 'now' when the rule is true at the last close. */
  kind: 'now' | 'trigger' | 'none';
  price: number | null;
  activeRange: [number | null, number | null] | null;
}

/** Nearest close (from the last close outwards) at which `group` (restricted to `keep`) becomes true. */
function solveGroup(probe: NextBarProbe, group: RuleGroup, scan: Scan, keep: readonly boolean[]): Solved {
  const sub = subgroup(group, keep);
  const { prices, centre, truth } = scan;
  const at = (k: number): boolean => combine(truth[k], group.logic, keep);

  if (at(centre)) {
    // Already true at the current price: find the band of closes that keeps it true.
    let lower: number | null = null;
    let upper: number | null = null;
    for (let k = centre - 1; k >= 0; k--) {
      if (!at(k)) {
        lower = bisect(probe, sub, prices[k], prices[k + 1]);
        break;
      }
    }
    for (let k = centre + 1; k < prices.length; k++) {
      if (!at(k)) {
        upper = bisect(probe, sub, prices[k], prices[k - 1]);
        break;
      }
    }
    return { kind: 'now', price: probe.lastClose, activeRange: [lower, upper] };
  }

  for (let step = 1; step <= centre; step++) {
    const up = centre + step;
    const down = centre - step;
    const upTrue = up < prices.length && at(up);
    const downTrue = down >= 0 && at(down);
    if (!upTrue && !downTrue) continue;
    // Equidistant on the grid: refine both and keep the closer one.
    const candidates: number[] = [];
    if (upTrue) candidates.push(bisect(probe, sub, prices[up - 1], prices[up]));
    if (downTrue) candidates.push(bisect(probe, sub, prices[down + 1], prices[down]));
    candidates.sort((x, y) => Math.abs(x - probe.lastClose) - Math.abs(y - probe.lastClose));
    return { kind: 'trigger', price: candidates[0], activeRange: null };
  }
  return { kind: 'none', price: null, activeRange: null };
}

/* -------------------------------------------------------------------------- */
/*  Condition status                                                          */
/* -------------------------------------------------------------------------- */

const OP_SYMBOL: Record<CmpOp, string> = { gt: '>', lt: '<', cross_above: 'crosses above', cross_below: 'crosses below' };

/** Units used to judge how far an indicator is from its threshold. */
function scaleOf(c: Condition, atr: number, right: number): { scale: number; unit: string; price: boolean } {
  const priceUnit = { scale: atr > 0 ? atr : 1, unit: 'ATR', price: true };
  switch (c.ind) {
    case 'PRICE_MA':
    case 'MA_CROSS':
    case 'SUPERTREND':
    case 'MACD':
    case 'DONCHIAN':
    case 'MA_SLOPE':
    case 'KELTNER':
      return priceUnit;
    case 'SQUEEZE':
    case 'RANGE_EXP':
    case 'PULLBACK':
      // One-candle events: there is no distance to measure, only whether the next candle can fire them.
      return { scale: 1, unit: 'event', price: false };
    case 'BB':
      if (c.variant === 'pctb') return { scale: 0.1, unit: '%B', price: false };
      if (c.variant === 'bandwidth') return { scale: Math.max(0.1, 0.2 * Math.abs(right)), unit: 'pts', price: false };
      return priceUnit;
    case 'RSI':
    case 'STOCH':
    case 'ADX':
      return { scale: 10, unit: 'pts', price: false };
    case 'CCI':
      return { scale: 50, unit: 'pts', price: false };
    case 'ATR':
      return { scale: Math.max(0.1, 0.2 * Math.abs(right)), unit: 'pts', price: false };
    case 'VOLUME':
      return { scale: 0.25, unit: '× avg', price: false };
  }
}

function fmtValue(value: number, price: boolean): string {
  if (!Number.isFinite(value)) return 'n/a';
  const abs = Math.abs(value);
  if (price) {
    if (abs >= 1000) return value.toFixed(2);
    if (abs >= 1) return String(Number(value.toFixed(4)));
    return value.toPrecision(4);
  }
  return String(Number(value.toFixed(abs >= 100 ? 1 : abs >= 1 ? 2 : 3)));
}

/** The signed event series (+1 / −1 / 0): a squeeze release, a range expansion, a pullback that holds. */
const isEvent = (c: Condition): boolean => c.ind === 'SQUEEZE' || c.ind === 'RANGE_EXP' || c.ind === 'PULLBACK';

/** Short name of an event condition for the checklist. */
function eventName(c: Condition): string {
  if (c.ind === 'SQUEEZE') return `Squeeze release ${c.level > 0 ? 'up' : 'down'} (BB ${c.period} inside KC ${c.mult})`;
  if (c.ind === 'RANGE_EXP') return `Range expansion ${c.level > 0 ? 'up' : 'down'} (TR > ${c.mult} × ATR ${c.period})`;
  if (c.ind === 'PULLBACK') return `Pullback ${c.level > 0 ? 'holding above' : 'rejected below'} ${c.maType}(${c.period})`;
  return conditionFormula(c);
}

function jobOf(group: RuleGroup, index: number, exit: boolean): ConditionCheck['job'] {
  if (exit) return 'Exit';
  if (index === 0) return 'Trigger';
  return conditionRole(group.conditions[index]) === 'FILTER' ? 'Filter' : 'Regime filter';
}

/**
 * Status of every condition of a rule on the last closed candle, plus whether
 * price alone can make it true on the next candle (from the scan).
 */
function checkConditions(
  group: RuleGroup,
  full: IndicatorCache,
  lastIndex: number,
  atr: number,
  scan: Scan,
  exit: boolean,
): ConditionCheck[] {
  return group.conditions.map((c, i) => {
    const { left: leftSeries, right: rightSeries, level } = resolveOperands(c, full);
    const left = leftSeries[lastIndex];
    const right = rightSeries ? rightSeries[lastIndex] : level;
    const out = new Uint8Array(lastIndex + 1);
    evaluateCondition(c, full, out, lastIndex + 1);
    const metNow = out[lastIndex] === 1;
    const reachable = scan.truth.some((row) => row[i]);
    const { scale, unit, price } = scaleOf(c, atr, right);
    const labels = conditionOperands(c);
    const rising = c.op === 'gt' || c.op === 'cross_above';
    // How far `left` must still move to satisfy the comparison.
    const gap = rising ? right - left : left - right;
    const crossing = c.op === 'cross_above' || c.op === 'cross_below';

    let state: CheckState;
    if (metNow) state = 'met';
    else if (reachable) state = 'pending';
    else if (crossing && gap < 0) state = 'reset';
    else state = 'blocked';

    if (isEvent(c)) {
      // An event either can or cannot fire on the next candle; there is no partial distance.
      const name = eventName(c);
      const text =
        state === 'met'
          ? `${name}: fired on the last candle [Met]`
          : state === 'pending'
            ? `${name}: can fire on the next candle at the right close [Pending]`
            : `${name}: cannot fire on the next candle${c.ind === 'SQUEEZE' ? ' (no squeeze is in progress)' : ''} [Blocked]`;
      return {
        formula: conditionFormula(c),
        job: jobOf(group, i, exit),
        state,
        leftLabel: name,
        rightLabel: '',
        left,
        right,
        op: c.op,
        distance: null,
        unit: 'event',
        proximity: state === 'met' ? 1 : state === 'pending' ? 0.5 : 0.05,
        text,
      };
    }

    const distance = state === 'met' ? null : Math.max(0, gap) / scale;
    let proximity: number;
    if (state === 'met') proximity = 1;
    else if (state === 'reset') proximity = 0.1;
    else proximity = Math.exp(-(distance ?? 0)) * (state === 'blocked' ? 0.5 : 1);

    const lv = fmtValue(left, price);
    const rv = rightSeries ? fmtValue(right, price) : fmtValue(level, false);
    const rightText = rightSeries ? `${labels.right} (${rv})` : rv;
    const phrase =
      c.op === 'cross_above' ? 'bullish cross' : c.op === 'cross_below' ? 'bearish cross' : rising ? 'moving above' : 'moving below';
    const away = distance === null ? '' : `${distance.toFixed(2)} ${unit}`;
    const volumeNote = c.ind === 'VOLUME' ? ' (the next candle is assumed to trade at average volume)' : '';

    let text: string;
    switch (state) {
      case 'met':
        text = crossing
          ? `${labels.left} ${c.op === 'cross_above' ? 'crossed above' : 'crossed below'} ${rightText} on the last candle [Met]`
          : `${labels.left} = ${lv} ${OP_SYMBOL[c.op]} ${rightText} [Met]`;
        break;
      case 'pending':
        text = rightSeries
          ? `${labels.left} vs ${labels.right}: ${away} away from ${phrase} [Pending]`
          : `${labels.left} = ${lv}, needs ${OP_SYMBOL[c.op]} ${rv} (${away} away) [Pending]`;
        break;
      case 'reset':
        text = `${labels.left} is already ${c.op === 'cross_above' ? 'above' : 'below'} ${rightText}: it must cross back before a new ${phrase} [Needs reset]`;
        break;
      case 'blocked':
        text = `${labels.left} = ${lv}, needs ${OP_SYMBOL[c.op]} ${rightText}${volumeNote} – not reachable on the next candle (${away} away) [Blocked]`;
        break;
    }

    return {
      formula: conditionFormula(c),
      job: jobOf(group, i, exit),
      state,
      leftLabel: labels.left,
      rightLabel: labels.right,
      left,
      right,
      op: c.op,
      distance,
      unit,
      proximity,
      text,
    };
  });
}

function blockerDetail(check: ConditionCheck): string {
  const price = check.unit === 'ATR';
  const relation = check.left < check.right ? '<' : check.left > check.right ? '>' : '=';
  if (check.state === 'reset') return `${check.leftLabel} already ${check.op === 'cross_above' ? 'above' : 'below'} ${check.rightLabel}`;
  return `${check.leftLabel} ${fmtValue(check.left, price)} ${relation} ${fmtValue(check.right, price)}`;
}

/* -------------------------------------------------------------------------- */
/*  Exit levels at an entry                                                   */
/* -------------------------------------------------------------------------- */

/**
 * The exit levels an entry filled at `fill` starts with – exactly the backtester's: the stop and
 * target from the ATR of the signal candle (or a % of the fill), how the stop is then managed
 * (fixed, or trailing k × ATR behind the best price) and where the breakeven lock (+1R) would
 * move it to the entry.
 */
export function exitLevels(
  genome: StrategyGenome,
  side: Side,
  fill: number,
  atrAtSignal: number,
): Pick<SideForecast, 'stopLoss' | 'takeProfit' | 'stopMode' | 'trailAtrMult' | 'breakevenAt'> {
  const dir = side === 'long' ? 1 : -1;
  const { exit } = genome;
  let stopLoss: number | null = null;
  let takeProfit: number | null = null;
  if (exit.mode === 'static') {
    stopLoss = fill * (1 - (dir * exit.stopLossPct) / 100);
    takeProfit = fill * (1 + (dir * exit.takeProfitPct) / 100);
  } else if (atrAtSignal > 0) {
    stopLoss = fill - dir * exit.stopAtrMult * atrAtSignal;
    takeProfit = exit.takeProfitAtrMult === null ? null : fill + dir * exit.takeProfitAtrMult * atrAtSignal;
  }
  const trailing = exit.mode === 'dynamic' && exit.trailing && stopLoss !== null;
  return {
    stopLoss,
    takeProfit,
    stopMode: stopLoss === null ? null : trailing ? 'trailing' : 'fixed',
    trailAtrMult: trailing && exit.mode === 'dynamic' ? exit.stopAtrMult : null,
    // 1R is the initial stop distance.
    breakevenAt: exit.breakeven === true && stopLoss !== null ? fill + (fill - stopLoss) : null,
  };
}

/* -------------------------------------------------------------------------- */
/*  Readiness and verdict                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Readiness (0–100):
 *   35% condition checklist (share of conditions met on the last candle)
 *   35% normalised distance (mean proximity of every condition to its threshold)
 *   30% price distance to the trigger (exp(−|move| / 2 ATR); half weight when blocked)
 * A pending entry is 100; a side that cannot trade is 0.
 */
export function readinessScore(side: Pick<SideForecast, 'status' | 'conditions' | 'triggerPrice'>, lastClose: number, atr: number): number {
  if (side.status === 'pending') return 100;
  if (side.status === 'disabled' || side.conditions.length === 0) return 0;
  const met = side.conditions.filter((c) => c.state === 'met').length / side.conditions.length;
  const proximity = side.conditions.reduce((sum, c) => sum + c.proximity, 0) / side.conditions.length;
  let price = 0;
  if (side.triggerPrice !== null && atr > 0) {
    price = Math.exp(-Math.abs(side.triggerPrice - lastClose) / (2 * atr));
    if (side.status === 'blocked') price *= 0.5;
  }
  if (side.status === 'now') price = 1;
  const score = 100 * (0.35 * met + 0.35 * proximity + 0.3 * price);
  return Math.max(0, Math.min(99, Math.round(score * 10) / 10));
}

/** Directional verdict from the two readiness scores. */
export function verdictFor(long: number, short: number, shortsEnabled: boolean, inPosition: boolean): Verdict {
  if (inPosition) return 'In Position';
  if (!shortsEnabled) {
    if (long >= 75) return 'Strong Long';
    if (long >= 45) return 'Leaning Long';
    return 'Neutral / Waiting';
  }
  const diff = long - short;
  const top = Math.max(long, short);
  if (top >= 75 && Math.abs(diff) >= 20) return diff > 0 ? 'Strong Long' : 'Strong Short';
  if (top >= 40 && Math.abs(diff) >= 10) return diff > 0 ? 'Leaning Long' : 'Leaning Short';
  return 'Neutral / Waiting';
}

/* -------------------------------------------------------------------------- */
/*  Solver                                                                    */
/* -------------------------------------------------------------------------- */

function medianStep(t: Float64Array): number {
  const n = t.length;
  const steps: number[] = [];
  for (let i = Math.max(1, n - 50); i < n; i++) steps.push(t[i] - t[i - 1]);
  steps.sort((a, b) => a - b);
  return steps.length > 0 ? steps[steps.length >> 1] : 86_400_000;
}

function sideSummary(f: SideForecast): string {
  const price = f.triggerPrice === null ? '' : fmtValue(f.triggerPrice, true);
  switch (f.status) {
    case 'pending':
      return 'Entry at the next open';
    case 'now':
      return 'Fires on a flat next candle';
    case 'trigger':
      return `Close ${f.movePct !== null && f.movePct >= 0 ? '≥' : '≤'} ${price}`;
    case 'blocked':
      return price ? `${price} (blocked)` : 'Blocked';
    case 'holding':
      return 'Position open';
    case 'disabled':
      return 'Not traded';
    case 'unreachable':
      return 'Out of range';
  }
}

/** Compute the next-entry forecast for one asset. Pure: no I/O, no globals. */
export function solveNextEntry(input: SolverInput): NextEntryForecast {
  const { data, genome, settings } = input;
  const n = data.c.length;
  if (n < 3) throw new Error('Not enough candles to forecast the next entry.');
  const last = n - 1;
  const lastClose = data.c[last];
  const shortsEnabled = allowsShorts(settings.tradeMode) && genome.shortEntry !== null;
  const notes: string[] = [];

  // Exact current state from the full history, with the backtester itself.
  const full = new IndicatorCache(data);
  const atr = full.atr(14)[last];
  const ws = new BacktestWorkspace(n);
  runBacktest(full, genome, settings, ws);
  const open = ws.openPosition;
  // The full state vector at the last close, carried into every hypothetical next candle.
  const state = solverState(full, ws, genome);

  const lastSignal = (group: RuleGroup | null): boolean => {
    if (!group) return false;
    const out = new Uint8Array(n);
    evaluateGroup(group, full, out, new Uint8Array(n), n);
    return out[last] === 1;
  };
  const exit = genome.exit;
  const longExit = exit.mode === 'dynamic' ? exit.longExit : null;
  const shortExit = exit.mode === 'dynamic' && shortsEnabled ? exit.shortExit : null;
  const longNow = lastSignal(genome.longEntry);
  const shortNow = shortsEnabled ? lastSignal(genome.shortEntry) : false;
  const longExitNow = lastSignal(longExit);
  const shortExitNow = lastSignal(shortExit);

  // Synthetic next candle.
  const span = Math.min(SCAN_ATR_MULTIPLE * (atr > 0 ? atr : lastClose * 0.02), SCAN_MAX_FRACTION * lastClose);
  const lo = Math.max(lastClose * 0.01, lastClose - span);
  const hi = lastClose + span;
  const prices = new Float64Array(SCAN_POINTS);
  const centre = (SCAN_POINTS - 1) >> 1;
  for (let k = 0; k < SCAN_POINTS; k++) {
    prices[k] = k < centre ? lo + ((lastClose - lo) * k) / centre : lastClose + ((hi - lastClose) * (k - centre)) / centre;
  }
  prices[centre] = lastClose;
  let avgVolume = 0;
  const volumeBars = Math.min(20, n);
  for (let i = n - volumeBars; i < n; i++) avgVolume += data.v[i];
  avgVolume /= volumeBars;
  const nextTime = data.t[last] + medianStep(data.t);
  const probe = new NextBarProbe(data, input.window ?? SOLVER_WINDOW, lastClose, nextTime, avgVolume);
  if (genome.longEntry.conditions.some((c) => c.ind === 'VOLUME')) {
    notes.push('Volume conditions assume the next candle trades at its 20-candle average volume.');
  }

  const forecastSide = (side: Side): SideForecast => {
    const group = side === 'long' ? genome.longEntry : genome.shortEntry;
    const base: SideForecast = {
      side,
      status: 'disabled',
      triggerPrice: null,
      movePct: null,
      activeRange: null,
      fillPrice: null,
      stopLoss: null,
      takeProfit: null,
      stopMode: null,
      trailAtrMult: null,
      breakevenAt: null,
      blockers: [],
      nonPriceBlockers: [],
      reason: null,
      conflict: null,
      conditions: [],
      readiness: 0,
      summary: '',
    };
    if (!group || (side === 'short' && !shortsEnabled)) {
      base.reason = !allowsShorts(settings.tradeMode)
        ? 'Long-Only mode: this strategy never sells short.'
        : 'No short rule: the Asymmetric short side needs a slow EMA/SMA for its bear-regime filter, and this history is too short for one.';
      base.summary = sideSummary(base);
      return base;
    }

    const scan = scanGroup(probe, group, prices);
    const conditions = checkConditions(group, full, last, atr, scan, false);
    base.conditions = conditions;
    const dir = side === 'long' ? 1 : -1;
    // Conditions whose truth is the same at every next close in range are not price-driven: report
    // the false ones next to the trigger price.
    conditions.forEach((check, i) => {
      const first = scan.truth[0]?.[i] ?? false;
      if (first || !scan.truth.every((row) => row[i] === first)) return;
      const c = group.conditions[i];
      const detail =
        c.ind === 'SQUEEZE'
          ? `Squeeze not on: Bollinger(${c.period}, 2) must sit inside Keltner(${c.period}, ${c.mult}) before it can release – on for ${state.squeezeRun.get(check.formula) ?? 0} closed candles`
          : `${blockerDetail(check)} – no next-candle close can change it`;
      base.nonPriceBlockers.push({ formula: check.formula, detail });
    });
    if (state.timeStopAtNextOpen && state.position && state.position.dir === dir) {
      base.nonPriceBlockers.push({
        formula: 'Time-stop cooldown',
        detail: 'The time stop closes this position at the next open, and that open cannot re-enter the same side',
      });
    }
    const opposite = side === 'long' ? genome.shortEntry : genome.longEntry;
    const ownExit = side === 'long' ? longExit : shortExit;
    const signalNow = side === 'long' ? longNow : shortNow;
    const oppositeNow = side === 'long' ? shortNow : longNow;
    const exitNow = side === 'long' ? longExitNow : shortExitNow;

    if (open && open.dir === dir) {
      base.status = 'holding';
      base.reason = `A ${side} position is already open (no pyramiding).`;
    } else if (signalNow && !(shortsEnabled && oppositeNow) && !exitNow) {
      // Fired on the last closed candle: the backtester enters at the next open.
      base.status = 'pending';
      base.triggerPrice = lastClose;
      base.movePct = 0;
      const atrSignal = exit.mode === 'dynamic' ? full.atr(exit.atrPeriod)[last] : 0;
      // Fill at the next open (assumed ≈ the last close) with the backtester's slippage rule.
      base.fillPrice = lastClose * (1 + dir * fillSlippage(settings, full, last));
      Object.assign(base, exitLevels(genome, side, base.fillPrice, atrSignal));
    } else {
      if (signalNow) {
        base.conflict = exitNow
          ? 'The rule fired on the last candle, but its own exit signal fired too, so no entry is taken.'
          : 'The rule fired on the last candle, but so did the opposite entry, so neither side enters.';
      }
      // Conditions that no next-candle close can satisfy.
      const blocked = conditions.map((c) => c.state === 'blocked' || c.state === 'reset');
      const keepAll = conditions.map(() => true);
      let solved = solveGroup(probe, group, scan, keepAll);
      if (solved.kind === 'none' && blocked.some(Boolean)) {
        base.blockers = conditions.filter((_, i) => blocked[i]).map((c) => ({ formula: c.formula, detail: blockerDetail(c) }));
        if (blocked[0]) {
          // The trigger itself cannot fire on the next candle.
          base.status = 'unreachable';
          base.reason = `Trigger not reachable on the next candle: ${blockerDetail(conditions[0])}.`;
        } else {
          // Where would the rest of the rule fire, if the filters allowed it?
          solved = solveGroup(
            probe,
            group,
            scan,
            blocked.map((b) => !b),
          );
          base.status = 'blocked';
          if (solved.kind === 'none') base.reason = 'Even without the blocked filters, no close within the scan range fires the rule.';
        }
      } else if (solved.kind === 'none') {
        base.status = 'unreachable';
        base.reason = `No close within ${fmtValue(lo, true)}–${fmtValue(hi, true)} fires the rule on the next candle.`;
      } else {
        base.status = solved.kind === 'now' ? 'now' : 'trigger';
      }

      if (solved.price !== null) {
        const p = solved.price;
        base.triggerPrice = p;
        base.movePct = ((p - lastClose) / lastClose) * 100;
        base.activeRange = solved.activeRange;
        // The signal fires on the next candle's close; the fill is the open after it (assumed ≈ that close),
        // slipped by the backtester's own rule, which reads that candle's ATR%.
        const ind = probe.cache(p);
        base.fillPrice = p * (1 + dir * fillSlippage(settings, ind, probe.at));
        const atrSignal = exit.mode === 'dynamic' ? ind.atr(exit.atrPeriod)[probe.at] : 0;
        Object.assign(base, exitLevels(genome, side, base.fillPrice, atrSignal));
        if (base.status !== 'blocked') {
          if (shortsEnabled && opposite && probe.group(opposite, ind)) {
            base.conflict = 'At this price the opposite entry fires too, so neither side would enter.';
          } else if (ownExit && probe.group(ownExit, ind)) {
            base.conflict = 'At this price the exit signal fires too, which cancels the entry.';
          }
        }
      }
    }

    base.readiness = readinessScore(base, lastClose, atr);
    base.summary = sideSummary(base);
    return base;
  };

  const long = forecastSide('long');
  const short = forecastSide('short');

  // Open position: expected exits.
  let position: PositionForecast | null = null;
  if (open) {
    const side: Side = open.dir === 1 ? 'long' : 'short';
    const dir = open.dir;
    const barsHeld = n - open.entryBar;
    const maxBars = genome.maxBarsInTrade;
    const timeStopIn = maxBars === null ? null : Math.max(0, maxBars - barsHeld);
    let exitAtNextOpen: string | null = null;
    if (timeStopIn === 0) exitAtNextOpen = `Time stop: ${maxBars} candles in the trade.`;
    // A trailing ratchet or breakeven lock can leave the stop beyond the last close: a next candle
    // that opens where the last one closed is stopped out at its open.
    else if (Number.isFinite(open.stop) && (dir === 1 ? open.stop >= lastClose : open.stop <= lastClose)) {
      exitAtNextOpen = `The stop (${open.stop.toPrecision(6)}) is already ${dir === 1 ? 'above' : 'below'} the last close: it fills at the next open unless the price opens ${dir === 1 ? 'above' : 'below'} it.`;
    } else if (dir === 1 && shortsEnabled && shortNow) exitAtNextOpen = 'The short entry fired: reverse at the next open.';
    else if (dir === -1 && longNow) exitAtNextOpen = 'The long entry fired: reverse at the next open.';
    else if (dir === 1 && longExitNow) exitAtNextOpen = 'The exit signal fired: close at the next open.';
    else if (dir === -1 && shortExitNow) exitAtNextOpen = 'The exit signal fired: close at the next open.';

    // Nearest close that would trigger an indicator exit or a reversal.
    let signalExitPrice: number | null = null;
    let signalExitSource: string | null = null;
    const exitGroup = dir === 1 ? longExit : shortExit;
    if (exitGroup) {
      const scan = scanGroup(probe, exitGroup, prices);
      const solved = solveGroup(probe, exitGroup, scan, exitGroup.conditions.map(() => true));
      if (solved.price !== null) {
        signalExitPrice = solved.price;
        signalExitSource = 'trigger-reversal exit';
      }
    }
    const reverse = dir === 1 ? short : long;
    if (reverse.triggerPrice !== null && (reverse.status === 'trigger' || reverse.status === 'now')) {
      if (signalExitPrice === null || Math.abs(reverse.triggerPrice - lastClose) < Math.abs(signalExitPrice - lastClose)) {
        signalExitPrice = reverse.triggerPrice;
        signalExitSource = `${reverse.side} entry (reversal)`;
      }
    }

    const breakeven = genome.exit.breakeven === true;
    const spec = exitSpec(genome);
    // The stop after a flat next candle at the last close, from the carried state (trailing anchor, lock).
    const flat = probe.cache(lastClose);
    const atrNext = flat.atr(spec.trailing?.period ?? (spec.stop.kind === 'atr' ? spec.stop.period : 14))[probe.at];
    const projected = projectStop(state, genome, lastClose, lastClose, atrNext);
    position = {
      dir: side,
      entryTime: data.t[open.entryBar],
      entryPrice: open.entryPrice,
      barsHeld,
      stopLoss: Number.isFinite(open.stop) ? open.stop : null,
      takeProfit: Number.isFinite(open.target) ? open.target : null,
      unrealisedPct: ((lastClose - open.entryPrice) / open.entryPrice) * 100 * dir,
      exitAtNextOpen,
      timeStopIn,
      signalExitPrice,
      signalExitSource,
      breakevenAt: breakeven && !open.breakevenLocked && open.risk > 0 ? open.entryPrice + dir * open.risk : null,
      breakevenLocked: breakeven && open.breakevenLocked,
      projectedStop: projected ? projected.stop : null,
      stopState: open.breakevenLocked ? 'breakeven' : spec.trailing !== null ? 'trailing' : 'fixed',
    };
  }

  return {
    lastClose,
    lastTime: data.t[last],
    nextTime,
    atr,
    scanRange: [lo, hi],
    long,
    short,
    position,
    verdict: verdictFor(long.readiness, short.readiness, shortsEnabled, position !== null),
    notes,
  };
}

/** Sort key for multi-asset tables: entries closest to firing first. */
export function proximityRank(f: NextEntryForecast): number {
  const sides = [f.long, f.short].filter((s) => s.status !== 'disabled' && s.status !== 'holding');
  let best = Infinity;
  for (const s of sides) {
    if (s.status === 'pending' || s.status === 'now') best = Math.min(best, -1);
    else if (s.status === 'trigger' && s.movePct !== null) best = Math.min(best, Math.abs(s.movePct));
    else if (s.status === 'blocked' && s.movePct !== null) best = Math.min(best, 1000 + Math.abs(s.movePct));
  }
  return best;
}
