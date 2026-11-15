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
 * • Position size is a run setting: risk a fixed share of capital at the stop
 *   (default 2%, no compounding) or deploy 100% of equity. Exposure never exceeds 1×.
 * • The equity curve is marked to market at every close, and any position still
 *   open on the last candle is liquidated at that close (costs included).
 * • Drawdown uses the intrabar worst case: the low of every candle a long is
 *   open, the high for a short.
 */

import {
  TRAIN_FRACTION,
  VALIDATION_FRACTION,
  type BacktestSettings,
  type Condition,
  type PerformanceMetrics,
  type RuleGroup,
  type StrategyGenome,
} from '../types/trading';
import type { DeflatedSharpe } from '../lib/stats';
import { FREE_PARAMETERS } from './genome';
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
 * Evaluate one condition on the first `n` bars. `out[i]` is 1 when the condition is
 * true on the close of bar `i`. Comparisons against NaN (indicator warm-up) are false.
 */
export function evaluateCondition(c: Condition, ind: IndicatorCache, out: Uint8Array, n: number = out.length): void {
  const { left, right, level } = resolveOperands(c, ind);
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

/**
 * AND a further condition into `out` in place. Only bars that are still set are
 * looked at, so a sparse trigger makes the second condition almost free.
 */
function refineCondition(c: Condition, ind: IndicatorCache, out: Uint8Array, n: number): void {
  const { left, right, level } = resolveOperands(c, ind);
  if (n === 0) return;
  switch (c.op) {
    case 'gt':
      if (right) {
        for (let i = 0; i < n; i++) if (out[i] === 1 && !(left[i] > right[i])) out[i] = 0;
      } else {
        for (let i = 0; i < n; i++) if (out[i] === 1 && !(left[i] > level)) out[i] = 0;
      }
      return;
    case 'lt':
      if (right) {
        for (let i = 0; i < n; i++) if (out[i] === 1 && !(left[i] < right[i])) out[i] = 0;
      } else {
        for (let i = 0; i < n; i++) if (out[i] === 1 && !(left[i] < level)) out[i] = 0;
      }
      return;
    case 'cross_above':
      out[0] = 0;
      if (right) {
        for (let i = 1; i < n; i++) if (out[i] === 1 && !(left[i] > right[i] && left[i - 1] <= right[i - 1])) out[i] = 0;
      } else {
        for (let i = 1; i < n; i++) if (out[i] === 1 && !(left[i] > level && left[i - 1] <= level)) out[i] = 0;
      }
      return;
    case 'cross_below':
      out[0] = 0;
      if (right) {
        for (let i = 1; i < n; i++) if (out[i] === 1 && !(left[i] < right[i] && left[i - 1] >= right[i - 1])) out[i] = 0;
      } else {
        for (let i = 1; i < n; i++) if (out[i] === 1 && !(left[i] < level && left[i - 1] >= level)) out[i] = 0;
      }
      return;
  }
}

/** Combine the conditions of a rule group with AND / OR into the first `n` entries of `out`. */
export function evaluateGroup(
  group: RuleGroup,
  ind: IndicatorCache,
  out: Uint8Array,
  scratch: Uint8Array,
  n: number = out.length,
): void {
  const { conditions, logic } = group;
  if (conditions.length === 0) {
    out.fill(0, 0, n);
    return;
  }
  evaluateCondition(conditions[0], ind, out, n);
  for (let k = 1; k < conditions.length; k++) {
    if (logic === 'AND') {
      refineCondition(conditions[k], ind, out, n);
    } else {
      evaluateCondition(conditions[k], ind, scratch, n);
      for (let i = 0; i < n; i++) out[i] |= scratch[i];
    }
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
  /** Worst equity reached inside every bar (adverse extreme of the candle while a position is open). */
  readonly equityLow: Float64Array;
  readonly tradeEntryBar: Int32Array;
  readonly tradeExitBar: Int32Array;
  /** Net profit of each trade after all fees, slippage and funding. */
  readonly tradePnl: Float64Array;
  /** +1 for long trades, −1 for short trades. */
  readonly tradeDir: Int8Array;
  /** 1 when the trade was closed on a candle that touched both its stop and its target. */
  readonly tradeAmbiguous: Uint8Array;
  tradeCount = 0;
  /** Number of bars simulated by the last run. */
  bars = 0;
  /** In-market flag per bar (1 when a position is open at the close). */
  readonly inMarket: Uint8Array;

  constructor(readonly n: number) {
    this.goLong = new Uint8Array(n);
    this.goShort = new Uint8Array(n);
    this.exitLong = new Uint8Array(n);
    this.exitShort = new Uint8Array(n);
    this.scratch = new Uint8Array(n);
    this.equity = new Float64Array(n);
    this.equityLow = new Float64Array(n);
    this.inMarket = new Uint8Array(n);
    // A trade can open and close inside a single candle, so at most one per bar (+1 for safety).
    this.tradeEntryBar = new Int32Array(n + 1);
    this.tradeExitBar = new Int32Array(n + 1);
    this.tradePnl = new Float64Array(n + 1);
    this.tradeDir = new Int8Array(n + 1);
    this.tradeAmbiguous = new Uint8Array(n + 1);
  }
}

/* -------------------------------------------------------------------------- */
/*  Core simulation                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Simulate `genome` over the first `bars` candles of the dataset (default: all).
 * Results are written into `ws`. Candles at or beyond `bars` are never read, so a
 * run limited to the search period cannot see the holdout.
 */
export function runBacktest(
  ind: IndicatorCache,
  genome: StrategyGenome,
  settings: BacktestSettings,
  ws: BacktestWorkspace,
  bars: number = ind.n,
): void {
  const { o: O, h: H, l: L, c: C } = ind.data;
  const n = Math.max(0, Math.min(bars, ind.n));
  const fee = settings.feePct / 100;
  const slip = settings.slippagePct / 100;
  const funding = settings.fundingPctPerBar / 100;
  const exit = genome.exit;
  const dynamic = exit.mode === 'dynamic';
  const allowShort = settings.tradeMode === 'long-short' && genome.shortEntry !== null;

  const { goLong, goShort, exitLong, exitShort, scratch, equity: eq, equityLow: eqLow, inMarket } = ws;
  const { tradeEntryBar, tradeExitBar, tradePnl, tradeDir, tradeAmbiguous } = ws;

  /* ---- 1. Signals (vectorised, search window only) ------------------------- */
  // Only the arrays the simulation will actually read are computed; the others stay stale and untouched.
  evaluateGroup(genome.longEntry, ind, goLong, scratch, n);
  if (allowShort && genome.shortEntry) evaluateGroup(genome.shortEntry, ind, goShort, scratch, n);

  const hasLongExit = exit.mode === 'dynamic' && exit.longExit !== null;
  const hasShortExit = exit.mode === 'dynamic' && allowShort && exit.shortExit !== null;
  if (exit.mode === 'dynamic' && exit.longExit) evaluateGroup(exit.longExit, ind, exitLong, scratch, n);
  if (exit.mode === 'dynamic' && allowShort && exit.shortExit) evaluateGroup(exit.shortExit, ind, exitShort, scratch, n);

  // Resolve conflicts once: an entry is only actionable when the opposite entry
  // and the same-side exit signal are both silent on that candle.
  if (allowShort) {
    for (let i = 0; i < n; i++) {
      const l = goLong[i];
      const s = goShort[i];
      if (l === 0 && s === 0) continue;
      goLong[i] = l === 1 && s === 0 && !(hasLongExit && exitLong[i] === 1) ? 1 : 0;
      goShort[i] = s === 1 && l === 0 && !(hasShortExit && exitShort[i] === 1) ? 1 : 0;
    }
  } else if (hasLongExit) {
    for (let i = 0; i < n; i++) if (goLong[i] === 1 && exitLong[i] === 1) goLong[i] = 0;
  }

  /* ---- 2. Exit and sizing parameters -------------------------------------- */
  const atr = exit.mode === 'dynamic' ? ind.atr(exit.atrPeriod) : null;
  const slPct = exit.mode === 'static' ? exit.stopLossPct / 100 : 0;
  const tpPct = exit.mode === 'static' ? exit.takeProfitPct / 100 : 0;
  const slMult = exit.mode === 'dynamic' ? exit.stopAtrMult : 0;
  const tpMult = exit.mode === 'dynamic' && exit.takeProfitAtrMult !== null ? exit.takeProfitAtrMult : 0;
  const trailing = exit.mode === 'dynamic' && exit.trailing;
  const maxBars = genome.maxBarsInTrade ?? 0;
  const riskSizing = settings.sizing.mode === 'risk';
  const riskFrac = settings.sizing.riskPct / 100;
  const compounding = settings.sizing.compounding;
  const startCapital = settings.initialCapital;

  /* ---- 3. Bar-by-bar simulation ------------------------------------------ */
  let equity = startCapital;
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
    // Fast path: flat, and the previous candle gave no entry signal → nothing can happen on this bar.
    if (pos === 0 && (i === 0 || (goLong[i - 1] === 0 && (!allowShort || goShort[i - 1] === 0)))) {
      eq[i] = equity;
      eqLow[i] = equity;
      inMarket[i] = 0;
      continue;
    }
    const o = O[i];
    const h = H[i];
    const l = L[i];
    const c = C[i];
    let worst = equity;
    let exitedAtOpen = false;

    if (i > 0 && !busted) {
      const p = i - 1; // signal candle

      /* -- 3a. Signal-driven exit, reversal or time stop at the open -------- */
      if (pos !== 0) {
        const timedOut = maxBars > 0 && i - entryBar >= maxBars;
        const leave =
          timedOut ||
          (pos === 1
            ? (allowShort && goShort[p] === 1) || (hasLongExit && exitLong[p] === 1)
            : goLong[p] === 1 || (hasShortExit && exitShort[p] === 1));
        if (leave) {
          const px = pos === 1 ? o * (1 - slip) : o * (1 + slip);
          equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
          tradeEntryBar[trades] = entryBar;
          tradeExitBar[trades] = i;
          tradePnl[trades] = equity - equityBeforeEntry;
          tradeDir[trades] = pos;
          tradeAmbiguous[trades] = 0;
          trades++;
          pos = 0;
          worst = equity;
          // A pure time-stop exit does not re-enter on the same open.
          exitedAtOpen = timedOut;
        }
      }

      /* -- 3b. Entries at the open ---------------------------------------- */
      if (pos === 0 && equity > 0 && !exitedAtOpen) {
        const dir = goLong[p] === 1 ? 1 : allowShort && goShort[p] === 1 ? -1 : 0;
        if (dir !== 0) {
          const a = atr !== null ? atr[p] : 0;
          // Dynamic exits need a valid ATR on the signal candle.
          if (!dynamic || (a === a && a > 0)) {
            const px = dir === 1 ? o * (1 + slip) : o * (1 - slip);
            const stopDist = dynamic ? slMult * a : px * slPct;
            // Notional to deploy: risk-based or everything, never more than current equity.
            let notional = equity;
            if (riskSizing && stopDist > 0) {
              const riskCapital = compounding ? equity : startCapital;
              notional = (riskCapital * riskFrac * px) / stopDist;
              if (notional > equity) notional = equity;
            }
            equityBeforeEntry = equity;
            qty = notional / (px * (1 + fee));
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
      let ambiguous = 0;
      let hitStop: boolean;
      if (pos === 1) {
        hitStop = l <= stop;
        const hitTarget = h >= target;
        if (hitStop) px = (o < stop ? o : stop) * (1 - slip);
        else if (hitTarget) px = (o > target ? o : target) * (1 - slip);
        if (hitStop && hitTarget) ambiguous = 1;
      } else {
        hitStop = h >= stop;
        const hitTarget = l <= target;
        if (hitStop) px = (o > stop ? o : stop) * (1 + slip);
        else if (hitTarget) px = (o < target ? o : target) * (1 + slip);
        if (hitStop && hitTarget) ambiguous = 1;
      }

      // Worst mark while the position was open inside this candle.
      const adverse = equityAfterEntryFee + pos * qty * ((pos === 1 ? l : h) - entryPrice);

      if (px === px) {
        equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
        tradeEntryBar[trades] = entryBar;
        tradeExitBar[trades] = i;
        tradePnl[trades] = equity - equityBeforeEntry;
        tradeDir[trades] = pos;
        tradeAmbiguous[trades] = ambiguous;
        trades++;
        // A stop realises the worst case; a target exit may still have dipped first.
        if (!hitStop && adverse < worst) worst = adverse;
        if (equity < worst) worst = equity;
        pos = 0;
      } else {
        if (adverse < worst) worst = adverse;
        if (trailing && atr !== null) {
          /* -- 3d. Trailing ATR stop, updated after the candle has closed -- */
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
        /* -- 3e. Perpetual funding on the open notional -------------------- */
        if (funding > 0) equityAfterEntryFee -= qty * c * funding;
      }
    }

    /* -- 3f. Mark to market ------------------------------------------------ */
    let mark = pos === 0 ? equity : equityAfterEntryFee + pos * qty * (c - entryPrice);
    if (pos !== 0 && mark <= 0) {
      // Account wiped out (only reachable through an extreme gap).
      tradeEntryBar[trades] = entryBar;
      tradeExitBar[trades] = i;
      tradePnl[trades] = -equityBeforeEntry;
      tradeDir[trades] = pos;
      tradeAmbiguous[trades] = 0;
      trades++;
      pos = 0;
      equity = 0;
      mark = 0;
      busted = true;
    }
    eq[i] = mark;
    if (worst < 0) worst = 0;
    eqLow[i] = worst < mark ? worst : mark;
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
    tradeAmbiguous[trades] = 0;
    trades++;
    eq[last] = equity;
    if (equity < eqLow[last]) eqLow[last] = equity;
  }

  ws.tradeCount = trades;
  ws.bars = n;
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
    bars: 0,
    sharpeRaw: 0,
    skew: 0,
    kurtosis: 3,
    maxConsecLosses: 0,
    payoffRatio: 0,
    breakevenWinRate: 0,
    tradeSkew: 0,
    ambiguousTrades: 0,
  };
}

/** Largest Sortino reported when a segment has gains but no losing bars at all. */
const SORTINO_CAP = 99;

/** Chronological segment boundaries for a dataset of `n` bars. */
export interface Split {
  /** First validation bar. */
  validationStart: number;
  /** First holdout bar; the search only ever simulates bars `[0, holdoutStart)`. */
  holdoutStart: number;
  n: number;
}

export function splitFor(n: number): Split {
  const validationStart = Math.max(1, Math.min(n - 2, Math.floor(n * TRAIN_FRACTION)));
  const holdoutStart = Math.max(validationStart + 1, Math.min(n - 1, Math.floor(n * (TRAIN_FRACTION + VALIDATION_FRACTION))));
  return { validationStart, holdoutStart, n };
}

/**
 * Metrics for the bars `[from, to)` of an equity curve.
 *
 * @param startEquity Equity immediately before bar `from` (the segment's capital base).
 * @param equityLow   Optional intrabar worst-case equity, used for the drawdown.
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
  equityLow: Float64Array | null = ws ? ws.equityLow : null,
): PerformanceMetrics {
  const m = emptyMetrics();
  const bars = to - from;
  m.bars = Math.max(0, bars);
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
    // Drawdown is measured from the highest close to the worst intrabar mark.
    const low = equityLow !== null && equityLow[i] < value ? equityLow[i] : value;
    if (peak > 0) {
      const dd = 1 - low / peak;
      if (dd > maxDd) maxDd = dd;
    }
    if (value > peak) peak = value;
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

  // Higher moments of per-bar returns (needed by the deflated Sharpe ratio).
  if (std > 1e-12) {
    let m3 = 0;
    let m4 = 0;
    prev = startEquity;
    for (let i = from; i < to; i++) {
      const value = equity[i];
      const d = (prev > 0 ? value / prev - 1 : 0) - mean;
      m3 += d * d * d;
      m4 += d * d * d * d;
      prev = value;
    }
    m.skew = m3 / bars / (std * std * std);
    m.kurtosis = m4 / bars / (variance * variance);
  }

  m.netPnl = endEquity - startEquity;
  m.roi = (growth - 1) * 100;
  m.cagr = growth <= 0 ? -100 : years > 0 ? (Math.pow(growth, 1 / years) - 1) * 100 : m.roi;
  m.maxDrawdown = Math.min(100, maxDd * 100);
  m.sharpeRaw = std > 1e-12 ? mean / std : 0;
  m.sharpe = m.sharpeRaw * annualiser;
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
    let streak = 0;
    let worstStreak = 0;
    let ambiguous = 0;
    let pnlSum = 0;
    let pnlSq = 0;
    let pnlCube = 0;
    for (let k = 0; k < ws.tradeCount; k++) {
      const entry = ws.tradeEntryBar[k];
      if (entry < from || entry >= to) continue;
      trades++;
      const pnl = ws.tradePnl[k];
      pnlSum += pnl;
      pnlSq += pnl * pnl;
      pnlCube += pnl * pnl * pnl;
      ambiguous += ws.tradeAmbiguous[k];
      if (pnl > 0) {
        wins++;
        grossProfit += pnl;
        streak = 0;
      } else {
        grossLoss -= pnl;
        streak++;
        if (streak > worstStreak) worstStreak = streak;
      }
    }
    const losses = trades - wins;
    m.trades = trades;
    m.winRate = trades > 0 ? (wins / trades) * 100 : 0;
    m.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
    m.exposure = (exposed / bars) * 100;
    m.maxConsecLosses = worstStreak;
    m.ambiguousTrades = ambiguous;
    const avgWin = wins > 0 ? grossProfit / wins : 0;
    const avgLoss = losses > 0 ? grossLoss / losses : 0;
    m.payoffRatio = avgLoss > 0 ? avgWin / avgLoss : avgWin > 0 ? Infinity : 0;
    m.breakevenWinRate = m.payoffRatio === Infinity ? 0 : m.payoffRatio > 0 ? 100 / (1 + m.payoffRatio) : 100;
    if (trades > 2) {
      const tm = pnlSum / trades;
      const tv = Math.max(0, pnlSq / trades - tm * tm);
      const ts = Math.sqrt(tv);
      if (ts > 1e-12) m.tradeSkew = (pnlCube / trades - 3 * tm * tv - tm * tm * tm) / (ts * ts * ts);
    }
  }
  return m;
}

/** Metrics of everything the search may look at. */
export interface SearchMetrics {
  /** Train + Validation combined. */
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
}

/** Running totals of one segment. Raw power sums let every moment be derived without a second pass. */
interface Tally {
  bars: number;
  start: number;
  end: number;
  s1: number;
  s2: number;
  s3: number;
  s4: number;
  down: number;
  exposed: number;
  maxDd: number;
  trades: number;
  wins: number;
  grossProfit: number;
  grossLoss: number;
  worstStreak: number;
  ambiguous: number;
  pnl1: number;
  pnl2: number;
  pnl3: number;
}

const newTally = (): Tally => ({
  bars: 0, start: 0, end: 0, s1: 0, s2: 0, s3: 0, s4: 0, down: 0, exposed: 0, maxDd: 0,
  trades: 0, wins: 0, grossProfit: 0, grossLoss: 0, worstStreak: 0, ambiguous: 0, pnl1: 0, pnl2: 0, pnl3: 0,
});

function resetTally(t: Tally, bars: number, start: number): void {
  t.bars = bars;
  t.start = start;
  t.end = start;
  t.s1 = t.s2 = t.s3 = t.s4 = t.down = t.exposed = t.maxDd = 0;
  t.trades = t.wins = t.grossProfit = t.grossLoss = t.worstStreak = t.ambiguous = t.pnl1 = t.pnl2 = t.pnl3 = 0;
}

// Scratch tallies: `searchMetrics` is synchronous and never re-entered.
const TRAIN = newTally();
const VALIDATION = newTally();
const SEARCH = newTally();

/** Turn a tally into metrics. Produces the same numbers as `segmentMetrics`. */
function finishTally(t: Tally, periodsPerYear: number, withTrades: boolean): PerformanceMetrics {
  const m = emptyMetrics();
  const { bars } = t;
  m.bars = bars;
  const growth = t.end / t.start;
  const mean = t.s1 / bars;
  const meanSq = t.s2 / bars;
  const variance = Math.max(0, meanSq - mean * mean);
  const std = Math.sqrt(variance);
  const downside = Math.sqrt(t.down / bars);
  const annualiser = Math.sqrt(periodsPerYear);
  const years = bars / periodsPerYear;

  if (std > 1e-12) {
    // Central moments from the raw power sums.
    const meanCube = t.s3 / bars;
    const m3 = meanCube - 3 * mean * meanSq + 2 * mean * mean * mean;
    const m4 = t.s4 / bars - 4 * mean * meanCube + 6 * mean * mean * meanSq - 3 * mean * mean * mean * mean;
    m.skew = m3 / (std * std * std);
    m.kurtosis = m4 / (variance * variance);
  }

  m.netPnl = t.end - t.start;
  m.roi = (growth - 1) * 100;
  m.cagr = growth <= 0 ? -100 : years > 0 ? (Math.pow(growth, 1 / years) - 1) * 100 : m.roi;
  m.maxDrawdown = Math.min(100, t.maxDd * 100);
  m.sharpeRaw = std > 1e-12 ? mean / std : 0;
  m.sharpe = m.sharpeRaw * annualiser;
  m.sortino =
    downside > 1e-12
      ? Math.max(-SORTINO_CAP, Math.min(SORTINO_CAP, (mean / downside) * annualiser))
      : mean > 0
        ? SORTINO_CAP
        : 0;

  if (withTrades) {
    const { trades, wins, grossProfit, grossLoss } = t;
    const losses = trades - wins;
    m.trades = trades;
    m.winRate = trades > 0 ? (wins / trades) * 100 : 0;
    m.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
    m.exposure = (t.exposed / bars) * 100;
    m.maxConsecLosses = t.worstStreak;
    m.ambiguousTrades = t.ambiguous;
    const avgWin = wins > 0 ? grossProfit / wins : 0;
    const avgLoss = losses > 0 ? grossLoss / losses : 0;
    m.payoffRatio = avgLoss > 0 ? avgWin / avgLoss : avgWin > 0 ? Infinity : 0;
    m.breakevenWinRate = m.payoffRatio === Infinity ? 0 : m.payoffRatio > 0 ? 100 / (1 + m.payoffRatio) : 100;
    if (trades > 2) {
      const tm = t.pnl1 / trades;
      const tv = Math.max(0, t.pnl2 / trades - tm * tm);
      const ts = Math.sqrt(tv);
      if (ts > 1e-12) m.tradeSkew = (t.pnl3 / trades - 3 * tm * tv - tm * tm * tm) / (ts * ts * ts);
    }
  }
  return m;
}

/**
 * Train / validation / combined metrics of a run that simulated bars `[0, split.holdoutStart)`
 * (or more – anything from the holdout onwards is ignored).
 *
 * This is the optimiser's hottest function after the simulation itself, so it
 * does not walk the whole curve. Outside a trade equity cannot move (no
 * position, no funding, the intrabar low equals the close), so such bars add
 * nothing to returns, drawdown or exposure. Only the bars from each trade's
 * entry to its exit are visited, all three segments are filled in that single
 * pass (the combined segment is the sum of the other two), and the results
 * equal three `segmentMetrics` calls.
 */
export function searchMetrics(
  equity: Float64Array,
  split: Split,
  initialCapital: number,
  periodsPerYear: number,
  ws: BacktestWorkspace | null,
  equityLow: Float64Array | null = ws ? ws.equityLow : null,
): SearchMetrics {
  const { validationStart, holdoutStart } = split;
  const validationCapital = equity[validationStart - 1];
  if (ws === null || equity !== ws.equity || equityLow !== ws.equityLow || !(initialCapital > 0) || !(validationCapital > 0)) {
    // No trade list to walk (the Buy & Hold benchmark), a foreign curve, or a wiped-out
    // account: use the reference implementation, which handles every case.
    return {
      search: segmentMetrics(equity, 0, holdoutStart, initialCapital, periodsPerYear, ws, equityLow),
      train: segmentMetrics(equity, 0, validationStart, initialCapital, periodsPerYear, ws, equityLow),
      validation: segmentMetrics(equity, validationStart, holdoutStart, validationCapital, periodsPerYear, ws, equityLow),
    };
  }

  resetTally(TRAIN, validationStart, initialCapital);
  resetTally(VALIDATION, holdoutStart - validationStart, validationCapital);
  resetTally(SEARCH, holdoutStart, initialCapital);

  const low = ws.equityLow;
  const { inMarket, tradeEntryBar, tradeExitBar, tradePnl, tradeAmbiguous, tradeCount } = ws;

  let seg = TRAIN;
  let inValidation = false;
  let prev = initialCapital;
  let peak = initialCapital;
  // The combined segment's drawdown equals the train drawdown until validation starts.
  let searchPeak = initialCapital;
  let searchDd = 0;
  let streak = 0;
  let streakOwner = TRAIN;
  let searchStreak = 0;
  let searchWorst = 0;
  /** First bar not yet visited (consecutive trades can share the exit / re-entry bar). */
  let cursor = 0;

  for (let k = 0; k <= tradeCount; k++) {
    // k === tradeCount is a sentinel that only closes the train segment if no trade reached validation.
    const entry = k < tradeCount ? tradeEntryBar[k] : holdoutStart;
    const stop = k < tradeCount ? Math.min(tradeExitBar[k], holdoutStart - 1) : -1;
    let from = entry > cursor ? entry : cursor;

    for (;;) {
      // Cross into the validation segment when this trade (or the end) lies beyond it.
      if (!inValidation && (from >= validationStart || entry >= validationStart)) {
        TRAIN.end = validationCapital;
        TRAIN.maxDd = seg.maxDd;
        searchPeak = peak;
        searchDd = TRAIN.maxDd;
        seg = VALIDATION;
        inValidation = true;
        prev = validationCapital;
        peak = validationCapital;
      }
      const to = !inValidation && stop >= validationStart ? validationStart - 1 : stop;
      let maxDd = seg.maxDd;
      for (let i = from; i <= to; i++) {
        const value = equity[i];
        if (value !== prev) {
          const r = prev > 0 ? value / prev - 1 : 0;
          const r2 = r * r;
          seg.s1 += r;
          seg.s2 += r2;
          seg.s3 += r2 * r;
          seg.s4 += r2 * r2;
          if (r < 0) seg.down += r2;
          prev = value;
        }
        // Drawdown: from the highest close so far to the worst intrabar mark.
        const worst = low[i] < value ? low[i] : value;
        if (worst < peak) {
          const dd = 1 - worst / peak;
          if (dd > maxDd) maxDd = dd;
        }
        if (value > peak) peak = value;
        if (inValidation) {
          if (worst < searchPeak) {
            const dd = 1 - worst / searchPeak;
            if (dd > searchDd) searchDd = dd;
          }
          if (value > searchPeak) searchPeak = value;
        }
        seg.exposed += inMarket[i];
      }
      seg.maxDd = maxDd;
      if (to >= from && to + 1 > cursor) cursor = to + 1;
      // A trade that straddles the boundary continues in the validation segment.
      if (!inValidation && stop >= validationStart) {
        from = validationStart;
        continue;
      }
      break;
    }

    if (k === tradeCount || entry >= holdoutStart) break; // trades are stored in entry order

    /* -- Trade statistics: a trade belongs to the segment it was opened in -- */
    const owner = entry < validationStart ? TRAIN : VALIDATION;
    if (owner !== streakOwner) {
      // Losing streaks are counted per segment (the combined streak keeps running).
      streak = 0;
      streakOwner = owner;
    }
    const pnl = tradePnl[k];
    owner.trades++;
    owner.pnl1 += pnl;
    owner.pnl2 += pnl * pnl;
    owner.pnl3 += pnl * pnl * pnl;
    owner.ambiguous += tradeAmbiguous[k];
    if (pnl > 0) {
      owner.wins++;
      owner.grossProfit += pnl;
      streak = 0;
      searchStreak = 0;
    } else {
      owner.grossLoss -= pnl;
      streak++;
      searchStreak++;
      if (streak > owner.worstStreak) owner.worstStreak = streak;
      if (searchStreak > searchWorst) searchWorst = searchStreak;
    }
  }

  if (!inValidation) {
    // Every trade was in the training segment (or there were none): validation is flat.
    TRAIN.end = validationCapital;
    TRAIN.maxDd = seg.maxDd;
    searchDd = TRAIN.maxDd;
  }
  VALIDATION.end = equity[holdoutStart - 1];

  SEARCH.s1 = TRAIN.s1 + VALIDATION.s1;
  SEARCH.s2 = TRAIN.s2 + VALIDATION.s2;
  SEARCH.s3 = TRAIN.s3 + VALIDATION.s3;
  SEARCH.s4 = TRAIN.s4 + VALIDATION.s4;
  SEARCH.down = TRAIN.down + VALIDATION.down;
  SEARCH.exposed = TRAIN.exposed + VALIDATION.exposed;
  SEARCH.maxDd = searchDd;
  SEARCH.end = VALIDATION.end;
  SEARCH.trades = TRAIN.trades + VALIDATION.trades;
  SEARCH.wins = TRAIN.wins + VALIDATION.wins;
  SEARCH.grossProfit = TRAIN.grossProfit + VALIDATION.grossProfit;
  SEARCH.grossLoss = TRAIN.grossLoss + VALIDATION.grossLoss;
  SEARCH.ambiguous = TRAIN.ambiguous + VALIDATION.ambiguous;
  SEARCH.pnl1 = TRAIN.pnl1 + VALIDATION.pnl1;
  SEARCH.pnl2 = TRAIN.pnl2 + VALIDATION.pnl2;
  SEARCH.pnl3 = TRAIN.pnl3 + VALIDATION.pnl3;
  SEARCH.worstStreak = searchWorst;

  return {
    search: finishTally(SEARCH, periodsPerYear, true),
    train: finishTally(TRAIN, periodsPerYear, true),
    validation: finishTally(VALIDATION, periodsPerYear, true),
  };
}

/** Holdout metrics of a run that simulated the whole dataset. Only call this once the search has ended. */
export function holdoutMetrics(
  equity: Float64Array,
  split: Split,
  periodsPerYear: number,
  ws: BacktestWorkspace | null,
  equityLow: Float64Array | null = ws ? ws.equityLow : null,
): PerformanceMetrics {
  return segmentMetrics(equity, split.holdoutStart, split.n, equity[split.holdoutStart - 1], periodsPerYear, ws, equityLow);
}

/** Backtest the search period only (the holdout is never simulated) and return its metrics. */
export function evaluateSearch(
  ind: IndicatorCache,
  genome: StrategyGenome,
  settings: BacktestSettings,
  ws: BacktestWorkspace,
  split: Split,
  periodsPerYear: number,
): SearchMetrics {
  runBacktest(ind, genome, settings, ws, split.holdoutStart);
  return searchMetrics(ws.equity, split, settings.initialCapital, periodsPerYear, ws);
}

/* -------------------------------------------------------------------------- */
/*  Buy & Hold benchmark                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Buy & Hold equity over the first `bars` candles: the whole account buys at the
 * first candle's open (paying the same fee and slippage as the strategies) and
 * is marked to market at every close.
 */
export function benchmarkEquity(
  data: { o: Float64Array; c: Float64Array },
  settings: BacktestSettings,
  bars: number = data.c.length,
): Float64Array {
  const n = Math.max(0, Math.min(bars, data.c.length));
  const equity = new Float64Array(n);
  if (n === 0) return equity;
  const entryPrice = data.o[0] * (1 + settings.slippagePct / 100);
  const units = settings.initialCapital / (entryPrice * (1 + settings.feePct / 100));
  for (let i = 0; i < n; i++) equity[i] = units * data.c[i];
  return equity;
}

function decorateBenchmark(m: PerformanceMetrics, first: boolean): PerformanceMetrics {
  return { ...m, trades: first ? 1 : 0, exposure: m.bars > 0 ? 100 : 0 };
}

/** Buy & Hold metrics for the search period (the holdout is not touched). */
export function benchmarkSearch(
  data: { o: Float64Array; c: Float64Array },
  settings: BacktestSettings,
  split: Split,
  periodsPerYear: number,
): SearchMetrics & { equity: Float64Array } {
  const equity = benchmarkEquity(data, settings, split.holdoutStart);
  const m = searchMetrics(equity, split, settings.initialCapital, periodsPerYear, null, null);
  return {
    equity,
    search: decorateBenchmark(m.search, true),
    train: decorateBenchmark(m.train, true),
    validation: decorateBenchmark(m.validation, false),
  };
}

/* -------------------------------------------------------------------------- */
/*  Composite fitness                                                         */
/* -------------------------------------------------------------------------- */

/** Trades (Train + Validation, summed over all assets) below which the low-activity penalty applies. */
export const MIN_TRADES = 100;
/** Validation trades (summed over all assets) below which the low-activity penalty applies. */
export const MIN_VALIDATION_TRADES = 25;
/** Fitness assigned to strategies that never trade. */
export const UNFIT = -1000;

const clampTo = (v: number, min: number, max: number): number => (v < min ? min : v > max ? max : v);

/** Blend of Sortino (60%) and Sharpe (40%) on a Sharpe-like scale, clamped so outliers cannot dominate. */
function riskAdjusted(m: PerformanceMetrics): number {
  return 0.6 * (clampTo(m.sortino, -3, 4.5) / 1.5) + 0.4 * clampTo(m.sharpe, -3, 3);
}

/** Excess return squashed into (−1, 1); the scale adapts to how far the benchmark itself moved. */
function excessReturnScore(strategyRoi: number, benchmarkRoi: number): number {
  const scale = 15 + 0.5 * Math.abs(benchmarkRoi);
  return Math.tanh((strategyRoi - benchmarkRoi) / scale);
}

/**
 * Per-asset score, everything except the deflated-Sharpe term (which changes as
 * more strategies are tried and is therefore added separately – see `deflatedTerm`):
 *
 *   + 0.35 × train risk-adjusted return (0.6 Sortino + 0.4 Sharpe, clamped)
 *   + 1.50 × validation excess return over Buy & Hold (squashed to ±1)
 *   + 0.75 × search-period excess return over Buy & Hold (squashed to ±1)
 *   − drawdown penalty beyond 20%, steeper beyond 45% (intrabar drawdown)
 *   − 0.75 when the strategy lost money on the training segment
 *   − consistency penalty when train and validation behaviour diverge
 */
export function assetBaseScore(m: SearchMetrics, benchmark: SearchMetrics): number {
  const trainRisk = riskAdjusted(m.train);
  const validationRisk = riskAdjusted(m.validation);
  let score =
    0.35 * trainRisk +
    1.5 * excessReturnScore(m.validation.roi, benchmark.validation.roi) +
    0.75 * excessReturnScore(m.search.roi, benchmark.search.roi);
  const dd = m.search.maxDrawdown / 100;
  score -= 4 * Math.max(0, dd - 0.2) + 6 * Math.max(0, dd - 0.45);
  if (m.train.roi <= 0) score -= 0.75;
  score -= 0.4 * Math.max(0, Math.abs(validationRisk - trainRisk) - 1);
  return score;
}

/**
 * Deflated-Sharpe contribution to the fitness: 3 × DSR, plus a small slice of the
 * underlying z-score so the search still has a gradient while the DSR is pinned near 0 or 1.
 */
export function deflatedTerm(d: DeflatedSharpe): number {
  return 3 * d.probability + 0.25 * clampTo(d.z, -4, 4);
}

export interface FitnessInputs {
  /** One score per asset: `assetBaseScore + deflatedTerm`. */
  assetScores: number[];
  /** Train + Validation trades summed over every asset. */
  searchTrades: number;
  /** Validation trades summed over every asset. */
  validationTrades: number;
  /** Share of assets that were profitable on validation (0–1). */
  profitableShare: number;
  /** Number of tunable parameters. */
  params: number;
  /** Reward-to-risk ratio of the exit, or null without a fixed target. */
  rewardToRisk: number | null;
  /** Validation win rate of the primary asset, in percent. */
  validationWinRate: number;
}

/**
 * Composite fitness used to rank strategies (higher is better).
 *
 *   median of the per-asset scores (a single asset when the Robust universe is off)
 *   − up to 6.0 for fewer than 100 trades, up to 2.0 for fewer than 25 validation trades
 *   − 1.0 when fewer than 2/3 of the universe is profitable on validation
 *   − 0.15 for every tunable parameter above 4
 *   − 0.75 for reward:risk below 1:1, unless the validation win rate clears break-even by 10 points
 */
export function combineFitness(f: FitnessInputs): number {
  if (f.searchTrades === 0 || f.assetScores.length === 0) return UNFIT;
  let score = median(f.assetScores);
  if (f.searchTrades < MIN_TRADES) score -= 6 * (1 - f.searchTrades / MIN_TRADES);
  if (f.validationTrades < MIN_VALIDATION_TRADES) score -= 2 * (1 - f.validationTrades / MIN_VALIDATION_TRADES);
  if (f.assetScores.length > 1 && f.profitableShare < 2 / 3 - 1e-9) score -= 1;
  score -= 0.15 * Math.max(0, f.params - FREE_PARAMETERS);
  if (f.rewardToRisk !== null && f.rewardToRisk < 1) {
    const breakeven = 100 / (1 + f.rewardToRisk);
    if (f.validationWinRate < breakeven + 10) score -= 0.75;
  }
  return Number.isFinite(score) ? score : UNFIT;
}

export function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
