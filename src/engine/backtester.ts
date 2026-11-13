/**
 * Event-driven backtester.
 *
 * Execution model
 * ---------------
 * • Signals are evaluated on the CLOSE of candle `t` and executed at the OPEN of
 *   candle `t + 1`. Nothing in a decision can see data it would not have had live.
 * • Every fill pays the configured fee and suffers adverse slippage.
 * • Stop-loss / take-profit orders are simulated intrabar from the entry candle
 *   onwards. A gap through the level fills at the open. When a candle touches
 *   both the stop and the target, the stop is assumed to fill first.
 * • ATR-based levels use the ATR of the signal candle, so they are known at entry.
 *   A trailing ATR stop is updated only after a candle has closed.
 * • The equity curve is marked to market at every close, and any position still
 *   open on the last candle is liquidated at that close (costs included).
 */

import type {
  BacktestSettings,
  Condition,
  PerformanceMetrics,
  RuleGroup,
  StrategyGenome,
} from '../types/trading';
import type { IndicatorCache } from './indicators';

/* -------------------------------------------------------------------------- */
/*  Signal evaluation                                                         */
/* -------------------------------------------------------------------------- */

interface Operands {
  left: Float64Array;
  /** Right-hand series, or null when the right-hand side is the constant `level`. */
  right: Float64Array | null;
  level: number;
}

/** Resolve a condition into `left <op> right` operands. */
export function resolveOperands(c: Condition, ind: IndicatorCache): Operands {
  switch (c.ind) {
    case 'RSI':
      return { left: ind.rsi(c.period), right: null, level: c.level };
    case 'MACD':
      switch (c.variant) {
        case 'signal':
          return { left: ind.macdLine(c.fast, c.slow), right: ind.macdSignal(c.fast, c.slow, c.signal), level: 0 };
        case 'zero':
          return { left: ind.macdLine(c.fast, c.slow), right: null, level: 0 };
        case 'hist':
          return {
            left: ind.macdHist(c.fast, c.slow, c.signal),
            right: ind.macdHistPrev(c.fast, c.slow, c.signal),
            level: 0,
          };
      }
      break;
    case 'PRICE_MA':
      return { left: ind.data.c, right: ind.ma(c.maType, c.period), level: 0 };
    case 'MA_CROSS':
      return { left: ind.ma(c.fastType, c.fastPeriod), right: ind.ma(c.slowType, c.slowPeriod), level: 0 };
    case 'BB':
      switch (c.variant) {
        case 'pctb':
          return { left: ind.bbPercentB(c.period, c.mult), right: null, level: c.level };
        case 'bandwidth':
          return { left: ind.bbBandwidth(c.period, c.mult), right: null, level: c.level };
        case 'upper':
          return { left: ind.data.c, right: ind.bbUpper(c.period, c.mult), level: 0 };
        case 'lower':
          return { left: ind.data.c, right: ind.bbLower(c.period, c.mult), level: 0 };
      }
      break;
    case 'ATR':
      return { left: ind.atrPercent(c.period), right: null, level: c.level };
    case 'STOCH':
      return c.variant === 'level'
        ? { left: ind.stochK(c.kPeriod, c.smooth), right: null, level: c.level }
        : { left: ind.stochK(c.kPeriod, c.smooth), right: ind.stochD(c.kPeriod, c.smooth, c.dPeriod), level: 0 };
    case 'ADX':
      return c.variant === 'strength'
        ? { left: ind.adx(c.period), right: null, level: c.level }
        : { left: ind.plusDI(c.period), right: ind.minusDI(c.period), level: 0 };
    case 'VOLUME':
      return { left: ind.volumeRatio(c.period), right: null, level: c.level };
    case 'SUPERTREND':
      return { left: ind.data.c, right: ind.supertrend(c.period, c.mult), level: 0 };
    case 'CCI':
      return { left: ind.cci(c.period), right: null, level: c.level };
  }
  throw new Error(`Unsupported condition: ${JSON.stringify(c)}`);
}

/**
 * Evaluate one condition on every bar. `out[i]` is 1 when the condition is true
 * on the close of bar `i`. Comparisons against NaN (indicator warm-up) are false.
 */
export function evaluateCondition(c: Condition, ind: IndicatorCache, out: Uint8Array): void {
  const { left, right, level } = resolveOperands(c, ind);
  const n = out.length;
  if (n === 0) return;
  switch (c.op) {
    case 'gt':
      if (right) for (let i = 0; i < n; i++) out[i] = left[i] > right[i] ? 1 : 0;
      else for (let i = 0; i < n; i++) out[i] = left[i] > level ? 1 : 0;
      return;
    case 'lt':
      if (right) for (let i = 0; i < n; i++) out[i] = left[i] < right[i] ? 1 : 0;
      else for (let i = 0; i < n; i++) out[i] = left[i] < level ? 1 : 0;
      return;
    case 'cross_above':
      out[0] = 0;
      if (right) for (let i = 1; i < n; i++) out[i] = left[i] > right[i] && left[i - 1] <= right[i - 1] ? 1 : 0;
      else for (let i = 1; i < n; i++) out[i] = left[i] > level && left[i - 1] <= level ? 1 : 0;
      return;
    case 'cross_below':
      out[0] = 0;
      if (right) for (let i = 1; i < n; i++) out[i] = left[i] < right[i] && left[i - 1] >= right[i - 1] ? 1 : 0;
      else for (let i = 1; i < n; i++) out[i] = left[i] < level && left[i - 1] >= level ? 1 : 0;
      return;
  }
}

/** Combine the conditions of a rule group with AND / OR into `out`. */
export function evaluateGroup(group: RuleGroup, ind: IndicatorCache, out: Uint8Array, scratch: Uint8Array): void {
  const { conditions, logic } = group;
  if (conditions.length === 0) {
    out.fill(0);
    return;
  }
  evaluateCondition(conditions[0], ind, out);
  const n = out.length;
  for (let k = 1; k < conditions.length; k++) {
    evaluateCondition(conditions[k], ind, scratch);
    if (logic === 'AND') for (let i = 0; i < n; i++) out[i] &= scratch[i];
    else for (let i = 0; i < n; i++) out[i] |= scratch[i];
  }
}

/* -------------------------------------------------------------------------- */
/*  Workspace                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Reusable scratch buffers for one dataset. The optimiser runs tens of
 * thousands of backtests, so nothing is allocated inside the hot loop.
 */
export class BacktestWorkspace {
  readonly goLong: Uint8Array;
  readonly goShort: Uint8Array;
  readonly exitLong: Uint8Array;
  readonly exitShort: Uint8Array;
  readonly scratch: Uint8Array;
  /** Mark-to-market equity at the close of every bar. */
  readonly equity: Float64Array;
  readonly tradeEntryBar: Int32Array;
  readonly tradeExitBar: Int32Array;
  /** Net profit of each trade after all fees and slippage. */
  readonly tradePnl: Float64Array;
  /** +1 for long trades, −1 for short trades. */
  readonly tradeDir: Int8Array;
  tradeCount = 0;
  /** In-market flag per bar (1 when a position is open at the close). */
  readonly inMarket: Uint8Array;

  constructor(readonly n: number) {
    this.goLong = new Uint8Array(n);
    this.goShort = new Uint8Array(n);
    this.exitLong = new Uint8Array(n);
    this.exitShort = new Uint8Array(n);
    this.scratch = new Uint8Array(n);
    this.equity = new Float64Array(n);
    this.inMarket = new Uint8Array(n);
    // A trade can open and close inside a single candle, so at most one per bar (+1 for safety).
    this.tradeEntryBar = new Int32Array(n + 1);
    this.tradeExitBar = new Int32Array(n + 1);
    this.tradePnl = new Float64Array(n + 1);
    this.tradeDir = new Int8Array(n + 1);
  }
}

/* -------------------------------------------------------------------------- */
/*  Core simulation                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Simulate `genome` over the whole dataset. Results are written into `ws`
 * (`ws.equity`, `ws.trade*`, `ws.tradeCount`, `ws.inMarket`).
 */
export function runBacktest(
  ind: IndicatorCache,
  genome: StrategyGenome,
  settings: BacktestSettings,
  ws: BacktestWorkspace,
): void {
  const { o: O, h: H, l: L, c: C } = ind.data;
  const n = ind.n;
  const fee = settings.feePct / 100;
  const slip = settings.slippagePct / 100;
  const exit = genome.exit;
  const dynamic = exit.mode === 'dynamic';
  const allowShort = settings.tradeMode === 'long-short' && genome.shortEntry !== null;

  const { goLong, goShort, exitLong, exitShort, scratch, equity: eq, inMarket } = ws;
  const { tradeEntryBar, tradeExitBar, tradePnl, tradeDir } = ws;

  /* ---- 1. Signals (vectorised) ------------------------------------------ */
  evaluateGroup(genome.longEntry, ind, goLong, scratch);
  if (allowShort && genome.shortEntry) evaluateGroup(genome.shortEntry, ind, goShort, scratch);
  else goShort.fill(0);

  const hasLongExit = exit.mode === 'dynamic' && exit.longExit !== null;
  const hasShortExit = exit.mode === 'dynamic' && allowShort && exit.shortExit !== null;
  if (exit.mode === 'dynamic' && exit.longExit) evaluateGroup(exit.longExit, ind, exitLong, scratch);
  else exitLong.fill(0);
  if (exit.mode === 'dynamic' && allowShort && exit.shortExit) evaluateGroup(exit.shortExit, ind, exitShort, scratch);
  else exitShort.fill(0);

  // Resolve conflicts once: an entry is only actionable when the opposite entry
  // and the same-side exit signal are both silent on that candle.
  for (let i = 0; i < n; i++) {
    const l = goLong[i];
    const s = goShort[i];
    goLong[i] = l === 1 && s === 0 && exitLong[i] === 0 ? 1 : 0;
    goShort[i] = s === 1 && l === 0 && exitShort[i] === 0 ? 1 : 0;
  }

  /* ---- 2. Exit parameters ------------------------------------------------ */
  const atr = exit.mode === 'dynamic' ? ind.atr(exit.atrPeriod) : null;
  const slPct = exit.mode === 'static' ? exit.stopLossPct / 100 : 0;
  const tpPct = exit.mode === 'static' ? exit.takeProfitPct / 100 : 0;
  const slMult = exit.mode === 'dynamic' ? exit.stopAtrMult : 0;
  const tpMult = exit.mode === 'dynamic' && exit.takeProfitAtrMult !== null ? exit.takeProfitAtrMult : 0;
  const trailing = exit.mode === 'dynamic' && exit.trailing;
  const riskFrac = genome.sizing.mode === 'risk' ? genome.sizing.riskPct / 100 : 0;

  /* ---- 3. Bar-by-bar simulation ------------------------------------------ */
  let equity = settings.initialCapital;
  let pos = 0; // +1 long, −1 short, 0 flat
  let qty = 0;
  let entryPrice = 0;
  let entryBar = 0;
  let equityBeforeEntry = 0;
  let equityAfterEntryFee = 0;
  let stop = 0;
  let target = 0;
  let extreme = 0;
  let trades = 0;
  let busted = false;

  for (let i = 0; i < n; i++) {
    const o = O[i];
    const h = H[i];
    const l = L[i];
    const c = C[i];

    if (i > 0 && !busted) {
      const p = i - 1; // signal candle

      /* -- 3a. Signal-driven exit / reversal at the open ------------------- */
      if (pos !== 0) {
        const leave =
          pos === 1
            ? goShort[p] === 1 || (hasLongExit && exitLong[p] === 1)
            : goLong[p] === 1 || (hasShortExit && exitShort[p] === 1);
        if (leave) {
          const px = pos === 1 ? o * (1 - slip) : o * (1 + slip);
          equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
          tradeEntryBar[trades] = entryBar;
          tradeExitBar[trades] = i;
          tradePnl[trades] = equity - equityBeforeEntry;
          tradeDir[trades] = pos;
          trades++;
          pos = 0;
        }
      }

      /* -- 3b. Entries at the open ---------------------------------------- */
      if (pos === 0 && equity > 0) {
        const dir = goLong[p] === 1 ? 1 : goShort[p] === 1 ? -1 : 0;
        if (dir !== 0) {
          const a = atr !== null ? atr[p] : 0;
          // Dynamic exits need a valid ATR on the signal candle.
          if (!dynamic || (a === a && a > 0)) {
            const px = dir === 1 ? o * (1 + slip) : o * (1 - slip);
            const stopDist = dynamic ? slMult * a : px * slPct;
            let fraction = 1;
            if (riskFrac > 0) {
              // Volatility sizing: being stopped out costs `riskFrac` of equity (no leverage).
              const stopFrac = dynamic ? stopDist / C[p] : slPct;
              fraction = stopFrac > 0 ? riskFrac / stopFrac : 1;
              if (fraction > 1) fraction = 1;
            }
            equityBeforeEntry = equity;
            qty = (equity * fraction) / (px * (1 + fee));
            equityAfterEntryFee = equity - qty * px * fee;
            entryPrice = px;
            entryBar = i;
            pos = dir;
            extreme = px;
            if (dir === 1) {
              stop = px - stopDist;
              target = dynamic ? (tpMult > 0 ? px + tpMult * a : Infinity) : px * (1 + tpPct);
            } else {
              stop = px + stopDist;
              target = dynamic ? (tpMult > 0 ? px - tpMult * a : -Infinity) : px * (1 - tpPct);
            }
          }
        }
      }
    }

    /* -- 3c. Intrabar stop-loss / take-profit ------------------------------ */
    if (pos !== 0) {
      let px = NaN;
      if (pos === 1) {
        if (l <= stop) px = (o < stop ? o : stop) * (1 - slip);
        else if (h >= target) px = (o > target ? o : target) * (1 - slip);
      } else if (h >= stop) px = (o > stop ? o : stop) * (1 + slip);
      else if (l <= target) px = (o < target ? o : target) * (1 + slip);

      if (px === px) {
        equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
        tradeEntryBar[trades] = entryBar;
        tradeExitBar[trades] = i;
        tradePnl[trades] = equity - equityBeforeEntry;
        tradeDir[trades] = pos;
        trades++;
        pos = 0;
      } else if (trailing && atr !== null) {
        /* -- 3d. Trailing ATR stop, updated after the candle has closed ---- */
        const a = atr[i];
        if (a === a) {
          if (pos === 1) {
            if (h > extreme) extreme = h;
            const next = extreme - slMult * a;
            if (next > stop) stop = next;
          } else {
            if (l < extreme) extreme = l;
            const next = extreme + slMult * a;
            if (next < stop) stop = next;
          }
        }
      }
    }

    /* -- 3e. Mark to market ------------------------------------------------ */
    let mark = pos === 0 ? equity : equityAfterEntryFee + pos * qty * (c - entryPrice);
    if (pos !== 0 && mark <= 0) {
      // Account wiped out (only reachable for shorts through an extreme gap).
      tradeEntryBar[trades] = entryBar;
      tradeExitBar[trades] = i;
      tradePnl[trades] = -equityBeforeEntry;
      tradeDir[trades] = pos;
      trades++;
      pos = 0;
      equity = 0;
      mark = 0;
      busted = true;
    }
    eq[i] = mark;
    inMarket[i] = pos !== 0 ? 1 : 0;
  }

  /* ---- 4. Liquidate whatever is still open at the final close ------------ */
  if (pos !== 0 && n > 0) {
    const last = n - 1;
    const px = pos === 1 ? C[last] * (1 - slip) : C[last] * (1 + slip);
    equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
    if (equity < 0) equity = 0;
    tradeEntryBar[trades] = entryBar;
    tradeExitBar[trades] = last;
    tradePnl[trades] = equity - equityBeforeEntry;
    tradeDir[trades] = pos;
    trades++;
    eq[last] = equity;
  }

  ws.tradeCount = trades;
}

/* -------------------------------------------------------------------------- */
/*  Metrics                                                                   */
/* -------------------------------------------------------------------------- */

const MS_PER_YEAR = 365.25 * 86_400_000;

/** Empirical number of bars per calendar year (handles 24/7 crypto and session-based stocks alike). */
export function barsPerYear(t: Float64Array): number {
  const n = t.length;
  if (n < 2) return 252;
  const span = t[n - 1] - t[0];
  if (!(span > 0)) return 252;
  return ((n - 1) * MS_PER_YEAR) / span;
}

export function emptyMetrics(): PerformanceMetrics {
  return {
    netPnl: 0,
    roi: 0,
    cagr: 0,
    maxDrawdown: 0,
    sharpe: 0,
    sortino: 0,
    winRate: 0,
    profitFactor: 0,
    trades: 0,
    exposure: 0,
  };
}

/** Largest Sortino reported when a segment has gains but no losing bars at all. */
const SORTINO_CAP = 99;

/**
 * Metrics for the bars `[from, to)` of an equity curve.
 *
 * @param startEquity Equity immediately before bar `from` (the segment's capital base).
 * @param ws          Optional workspace – when provided, trade statistics are computed from
 *                    the trades opened inside the segment.
 */
export function segmentMetrics(
  equity: Float64Array,
  from: number,
  to: number,
  startEquity: number,
  periodsPerYear: number,
  ws: BacktestWorkspace | null,
): PerformanceMetrics {
  const m = emptyMetrics();
  const bars = to - from;
  if (bars <= 0 || !(startEquity > 0)) {
    if (!(startEquity > 0)) {
      m.roi = -100;
      m.cagr = -100;
      m.maxDrawdown = 100;
    }
    return m;
  }

  let prev = startEquity;
  let peak = startEquity;
  let maxDd = 0;
  let sum = 0;
  let sumSq = 0;
  let downSq = 0;
  let exposed = 0;
  for (let i = from; i < to; i++) {
    const value = equity[i];
    const r = prev > 0 ? value / prev - 1 : 0;
    sum += r;
    sumSq += r * r;
    if (r < 0) downSq += r * r;
    if (value > peak) peak = value;
    else if (peak > 0) {
      const dd = 1 - value / peak;
      if (dd > maxDd) maxDd = dd;
    }
    prev = value;
    if (ws !== null && ws.inMarket[i] === 1) exposed++;
  }

  const endEquity = equity[to - 1];
  const growth = endEquity / startEquity;
  const mean = sum / bars;
  const variance = Math.max(0, sumSq / bars - mean * mean);
  const std = Math.sqrt(variance);
  const downside = Math.sqrt(downSq / bars);
  const annualiser = Math.sqrt(periodsPerYear);
  const years = bars / periodsPerYear;

  m.netPnl = endEquity - startEquity;
  m.roi = (growth - 1) * 100;
  m.cagr = growth <= 0 ? -100 : years > 0 ? (Math.pow(growth, 1 / years) - 1) * 100 : m.roi;
  m.maxDrawdown = maxDd * 100;
  m.sharpe = std > 1e-12 ? (mean / std) * annualiser : 0;
  m.sortino =
    downside > 1e-12
      ? Math.max(-SORTINO_CAP, Math.min(SORTINO_CAP, (mean / downside) * annualiser))
      : mean > 0
        ? SORTINO_CAP
        : 0;

  if (ws !== null) {
    let trades = 0;
    let wins = 0;
    let grossProfit = 0;
    let grossLoss = 0;
    for (let k = 0; k < ws.tradeCount; k++) {
      const entry = ws.tradeEntryBar[k];
      if (entry < from || entry >= to) continue;
      trades++;
      const pnl = ws.tradePnl[k];
      if (pnl > 0) {
        wins++;
        grossProfit += pnl;
      } else {
        grossLoss -= pnl;
      }
    }
    m.trades = trades;
    m.winRate = trades > 0 ? (wins / trades) * 100 : 0;
    m.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
    m.exposure = (exposed / bars) * 100;
  }
  return m;
}

export interface SegmentedMetrics {
  full: PerformanceMetrics;
  inSample: PerformanceMetrics;
  outOfSample: PerformanceMetrics;
}

/**
 * Split an equity curve chronologically at `splitIndex` (first out-of-sample bar)
 * and report full-period, in-sample and out-of-sample metrics. The out-of-sample
 * segment is rebased on the equity at the end of the in-sample segment, so
 * `inSample.netPnl + outOfSample.netPnl === full.netPnl`.
 */
export function splitMetrics(
  equity: Float64Array,
  splitIndex: number,
  initialCapital: number,
  periodsPerYear: number,
  ws: BacktestWorkspace | null,
): SegmentedMetrics {
  const n = equity.length;
  const split = Math.max(1, Math.min(n - 1, splitIndex));
  return {
    full: segmentMetrics(equity, 0, n, initialCapital, periodsPerYear, ws),
    inSample: segmentMetrics(equity, 0, split, initialCapital, periodsPerYear, ws),
    outOfSample: segmentMetrics(equity, split, n, equity[split - 1], periodsPerYear, ws),
  };
}

/** Run a backtest and compute segmented metrics. The equity curve is left in `ws.equity`. */
export function evaluateGenome(
  ind: IndicatorCache,
  genome: StrategyGenome,
  settings: BacktestSettings,
  ws: BacktestWorkspace,
  splitIndex: number,
  periodsPerYear: number,
): SegmentedMetrics {
  runBacktest(ind, genome, settings, ws);
  return splitMetrics(ws.equity, splitIndex, settings.initialCapital, periodsPerYear, ws);
}

/* -------------------------------------------------------------------------- */
/*  Buy & Hold benchmark                                                      */
/* -------------------------------------------------------------------------- */

export interface Benchmark extends SegmentedMetrics {
  equity: Float64Array;
}

/**
 * Buy & Hold over exactly the same bars and starting capital: the whole
 * account buys at the first candle's open (paying the same fee and slippage as
 * the strategies) and is marked to market at every close.
 */
export function buildBenchmark(
  data: { o: Float64Array; c: Float64Array },
  settings: BacktestSettings,
  splitIndex: number,
  periodsPerYear: number,
): Benchmark {
  const n = data.c.length;
  const equity = new Float64Array(n);
  if (n === 0) {
    const empty = emptyMetrics();
    return { equity, full: empty, inSample: empty, outOfSample: empty };
  }
  const fee = settings.feePct / 100;
  const slip = settings.slippagePct / 100;
  const entryPrice = data.o[0] * (1 + slip);
  const units = settings.initialCapital / (entryPrice * (1 + fee));
  for (let i = 0; i < n; i++) equity[i] = units * data.c[i];
  const metrics = splitMetrics(equity, splitIndex, settings.initialCapital, periodsPerYear, null);
  const decorate = (m: PerformanceMetrics, from: number, to: number): PerformanceMetrics => ({
    ...m,
    trades: from === 0 ? 1 : 0,
    exposure: to > from ? 100 : 0,
    winRate: 0,
    profitFactor: 0,
  });
  const split = Math.max(1, Math.min(n - 1, splitIndex));
  return {
    equity,
    full: decorate(metrics.full, 0, n),
    inSample: decorate(metrics.inSample, 0, split),
    outOfSample: decorate(metrics.outOfSample, split, n),
  };
}

/* -------------------------------------------------------------------------- */
/*  Composite fitness                                                         */
/* -------------------------------------------------------------------------- */

/** Minimum number of trades before the low-activity penalty disappears. */
export const MIN_TRADES = 20;
/** Minimum number of out-of-sample trades before the low-activity penalty disappears. */
export const MIN_OOS_TRADES = 6;
/** Fitness assigned to strategies that never trade. */
export const UNFIT = -1000;

const clampTo = (v: number, min: number, max: number): number => (v < min ? min : v > max ? max : v);

/**
 * Blend of Sortino (60%) and Sharpe (40%) on a Sharpe-like scale.
 * Both ratios are clamped: beyond a Sharpe of 3 (Sortino 4.5) nothing more is
 * earned, so the search has no incentive to chase implausible, curve-fitted ratios.
 */
function riskAdjusted(m: PerformanceMetrics): number {
  return 0.6 * (clampTo(m.sortino, -3, 4.5) / 1.5) + 0.4 * clampTo(m.sharpe, -3, 3);
}

/** Excess return squashed into (−1, 1); the scale adapts to how far the benchmark itself moved. */
function excessReturnScore(strategyRoi: number, benchmarkRoi: number): number {
  const scale = 15 + 0.5 * Math.abs(benchmarkRoi);
  return Math.tanh((strategyRoi - benchmarkRoi) / scale);
}

/**
 * Composite fitness used to rank strategies (higher is better).
 *
 *   + 1.00 × out-of-sample risk-adjusted return (0.6 Sortino + 0.4 Sharpe)
 *   + 0.35 × in-sample risk-adjusted return
 *   + 1.50 × out-of-sample excess return over Buy & Hold (squashed to ±1)
 *   + 0.75 × full-period excess return over Buy & Hold (squashed to ±1)
 *   − up to 4.0 for fewer than 20 trades, up to 2.0 for fewer than 6 out-of-sample trades
 *   − drawdown penalty beyond 20%, steeper beyond 45%
 *   − 0.75 when the strategy lost money in-sample (out-of-sample luck)
 *   − consistency penalty when in-sample and out-of-sample behaviour diverge
 */
export function computeFitness(strategy: SegmentedMetrics, benchmark: SegmentedMetrics): number {
  const { full, inSample, outOfSample } = strategy;
  if (full.trades === 0) return UNFIT;

  const oosRisk = riskAdjusted(outOfSample);
  const isRisk = riskAdjusted(inSample);
  let score =
    1.0 * oosRisk +
    0.35 * isRisk +
    1.5 * excessReturnScore(outOfSample.roi, benchmark.outOfSample.roi) +
    0.75 * excessReturnScore(full.roi, benchmark.full.roi);

  if (full.trades < MIN_TRADES) score -= 4 * (1 - full.trades / MIN_TRADES);
  if (outOfSample.trades < MIN_OOS_TRADES) score -= 2 * (1 - outOfSample.trades / MIN_OOS_TRADES);

  const dd = full.maxDrawdown / 100;
  score -= 4 * Math.max(0, dd - 0.2) + 6 * Math.max(0, dd - 0.45);

  if (inSample.roi <= 0) score -= 0.75;
  score -= 0.4 * Math.max(0, Math.abs(oosRisk - isRisk) - 1);

  return Number.isFinite(score) ? score : UNFIT;
}
