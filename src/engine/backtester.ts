/**
 * Event-driven backtester.
 *
 * Execution model
 * ---------------
 * • Signals are evaluated on the CLOSE of candle `t` and executed at the OPEN of
 *   candle `t + 1`. Nothing in a decision can see data it would not have had live.
 * • Every fill pays the configured fee and suffers adverse slippage. Slippage is
 *   volatility-scaled: the larger of the fixed rate and `slippageAtrMult` × ATR(14)
 *   as a % of price, read on the last closed candle so it is known at the fill.
 * • Open positions pay perpetual funding (both sides) and short positions pay a
 *   borrow fee, per candle, when those run settings are non-zero.
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
  allowsShorts,
  REGIME_NAMES,
  SLOPE_LAG,
  TRAIN_FRACTION,
  VALIDATION_FRACTION,
  type RegimeMetrics,
  type BacktestSettings,
  type Condition,
  type PerformanceMetrics,
  type RuleGroup,
  type StrategyGenome,
  type TradeGates,
} from '../types/trading';
import { sharpeConfidenceInterval, type DeflatedSharpe } from '../lib/stats';
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
    case 'DONCHIAN':
      return { left: ind.data.c, right: c.variant === 'upper' ? ind.donchianUpper(c.period) : ind.donchianLower(c.period), level: 0 };
    case 'MA_SLOPE':
      return { left: ind.ma(c.maType, c.period), right: ind.maLagged(c.maType, c.period, SLOPE_LAG), level: 0 };
    case 'KELTNER':
      return { left: ind.data.c, right: c.variant === 'upper' ? ind.kcUpper(c.period, c.mult) : ind.kcLower(c.period, c.mult), level: 0 };
    case 'SQUEEZE':
      return { left: ind.squeeze(c.period, c.mult), right: null, level: c.level };
    case 'RANGE_EXP':
      return { left: ind.rangeExp(c.period, c.mult), right: null, level: c.level };
    case 'PULLBACK':
      return { left: ind.pullback(c.maType, c.period), right: null, level: c.level };
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
  /** Equity at the trade's entry (before the entry fee) – the base its return is measured on. */
  readonly tradeEquity: Float64Array;
  /** Notional at entry as a share of `tradeEquity` (0–1): what a slippage shock is charged on. */
  readonly tradeExposure: Float64Array;
  /** Worst intrabar equity while the trade was open, relative to `tradeEquity` (≤ 0): its adverse excursion. */
  readonly tradeMae: Float64Array;
  tradeCount = 0;
  /** Number of bars simulated by the last run. */
  bars = 0;
  /**
   * The position still open at the final close of the last run, captured before the
   * end-of-data liquidation (a bookkeeping step, not a real exit). Null when flat.
   * `stop` and `target` are the levels that apply to the NEXT candle (trailing stops included).
   */
  openPosition: {
    dir: 1 | -1;
    entryBar: number;
    entryPrice: number;
    stop: number;
    target: number;
    /** 1R: the initial stop distance, in price. */
    risk: number;
    /** True once the breakeven lock has moved the stop to the entry price. */
    breakevenLocked: boolean;
  } | null = null;
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
    this.tradeEquity = new Float64Array(n + 1);
    this.tradeExposure = new Float64Array(n + 1);
    this.tradeMae = new Float64Array(n + 1);
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
/** ATR lookback used to scale slippage with volatility. */
export const SLIPPAGE_ATR_PERIOD = 14;

/**
 * Adverse slippage (as a fraction) of a fill on the candle after `closedIndex`: the larger of the
 * fixed rate and `slippageAtrMult` × ATR% of that last closed candle – the rule `runBacktest`
 * applies to every fill (its `slipAt`). Exposed so forecasts quote fills the same way.
 */
export function fillSlippage(settings: BacktestSettings, ind: IndicatorCache, closedIndex: number): number {
  const slip = settings.slippagePct / 100;
  const mult = settings.slippageAtrMult ?? 0;
  if (!(mult > 0)) return slip;
  const scaled = (mult * ind.atrPercent(SLIPPAGE_ATR_PERIOD)[Math.max(0, closedIndex)]) / 100;
  return scaled > slip ? scaled : slip;
}

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
  const borrow = (settings.borrowPctPerBar ?? 0) / 100;
  const slipAtrMult = settings.slippageAtrMult ?? 0;
  const slipAtrPct = slipAtrMult > 0 ? ind.atrPercent(SLIPPAGE_ATR_PERIOD) : null;
  /**
   * Adverse slippage (as a fraction) of a fill during candle `i`: the larger of the fixed rate and
   * k × ATR% of candle `i − 1`, the last candle that had closed. During the ATR warm-up (NaN) the
   * fixed rate applies.
   */
  const slipAt = (i: number): number => {
    if (slipAtrPct === null) return slip;
    const scaled = (slipAtrMult * slipAtrPct[i > 0 ? i - 1 : 0]) / 100;
    return scaled > slip ? scaled : slip;
  };
  const exit = genome.exit;
  const dynamic = exit.mode === 'dynamic';
  const allowShort = allowsShorts(settings.tradeMode) && genome.shortEntry !== null;

  const { goLong, goShort, exitLong, exitShort, scratch, equity: eq, equityLow: eqLow, inMarket } = ws;
  const { tradeEntryBar, tradeExitBar, tradePnl, tradeDir, tradeAmbiguous, tradeEquity, tradeExposure, tradeMae } = ws;

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
  const breakeven = exit.breakeven === true;
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
  /** 1R of the open trade: its initial stop distance. */
  let risk = 0;
  let locked = false;
  /** Worst intrabar equity of the open trade. */
  let tradeLow = 0;
  let trades = 0;
  let busted = false;
  /** Book a closed trade. */
  const record = (exitBar: number, pnl: number, ambiguous: number): void => {
    tradeEntryBar[trades] = entryBar;
    tradeExitBar[trades] = exitBar;
    tradePnl[trades] = pnl;
    tradeDir[trades] = pos;
    tradeAmbiguous[trades] = ambiguous;
    tradeEquity[trades] = equityBeforeEntry;
    tradeExposure[trades] = equityBeforeEntry > 0 ? Math.min(1, (qty * entryPrice) / equityBeforeEntry) : 0;
    tradeMae[trades] = equityBeforeEntry > 0 ? Math.min(0, tradeLow / equityBeforeEntry - 1) : 0;
    trades++;
  };

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
    let exitedAtOpen = 0;

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
          const sl = slipAt(i);
          const px = pos === 1 ? o * (1 - sl) : o * (1 + sl);
          equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
          if (equity < tradeLow) tradeLow = equity;
          record(i, equity - equityBeforeEntry, 0);
          // A pure time-stop exit does not re-enter the SAME side on the same open; an opposite
          // entry signal still reverses the position.
          exitedAtOpen = timedOut ? pos : 0;
          pos = 0;
          worst = equity;
        }
      }

      /* -- 3b. Entries at the open ---------------------------------------- */
      if (pos === 0 && equity > 0) {
        const dir = goLong[p] === 1 ? 1 : allowShort && goShort[p] === 1 ? -1 : 0;
        if (dir !== 0 && dir !== exitedAtOpen) {
          const a = atr !== null ? atr[p] : 0;
          // Dynamic exits need a valid ATR on the signal candle.
          if (!dynamic || (a === a && a > 0)) {
            const sl = slipAt(i);
            const px = dir === 1 ? o * (1 + sl) : o * (1 - sl);
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
            risk = stopDist;
            locked = false;
            tradeLow = equityAfterEntryFee;
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
        if (hitStop) px = (o < stop ? o : stop) * (1 - slipAt(i));
        else if (hitTarget) px = (o > target ? o : target) * (1 - slipAt(i));
        if (hitStop && hitTarget) ambiguous = 1;
      } else {
        hitStop = h >= stop;
        const hitTarget = l <= target;
        if (hitStop) px = (o > stop ? o : stop) * (1 + slipAt(i));
        else if (hitTarget) px = (o < target ? o : target) * (1 + slipAt(i));
        if (hitStop && hitTarget) ambiguous = 1;
      }

      // Worst mark while the position was open inside this candle.
      const adverse = equityAfterEntryFee + pos * qty * ((pos === 1 ? l : h) - entryPrice);

      if (px === px) {
        equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
        const low = hitStop ? equity : adverse < equity ? adverse : equity;
        if (low < tradeLow) tradeLow = low;
        record(i, equity - equityBeforeEntry, ambiguous);
        // A stop realises the worst case; a target exit may still have dipped first.
        if (!hitStop && adverse < worst) worst = adverse;
        if (equity < worst) worst = equity;
        pos = 0;
      } else {
        if (adverse < worst) worst = adverse;
        if (adverse < tradeLow) tradeLow = adverse;
        /* -- 3d. Breakeven lock and trailing ATR stop, updated after the candle has closed -- */
        if (breakeven && !locked && risk > 0 && (pos === 1 ? h >= entryPrice + risk : l <= entryPrice - risk)) {
          // +1R reached: from the next candle on the trade can no longer turn into a full loss.
          locked = true;
          if (pos === 1 ? entryPrice > stop : entryPrice < stop) stop = entryPrice;
        }
        if (trailing && atr !== null) {
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
        /* -- 3e. Perpetual funding on the open notional, borrow fee on shorts -- */
        if (funding > 0) equityAfterEntryFee -= qty * c * funding;
        if (borrow > 0 && pos === -1) equityAfterEntryFee -= qty * c * borrow;
      }
    }

    /* -- 3f. Mark to market ------------------------------------------------ */
    let mark = pos === 0 ? equity : equityAfterEntryFee + pos * qty * (c - entryPrice);
    if (pos !== 0 && mark <= 0) {
      // Account wiped out (only reachable through an extreme gap).
      tradeLow = 0;
      record(i, -equityBeforeEntry, 0);
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
  ws.openPosition =
    pos !== 0 && n > 0 ? { dir: pos === 1 ? 1 : -1, entryBar, entryPrice, stop, target, risk, breakevenLocked: locked } : null;
  if (pos !== 0 && n > 0) {
    const last = n - 1;
    // Market order on the final close: candle `last` has closed, so its own ATR is known.
    const sl = slipAt(last + 1);
    const px = pos === 1 ? C[last] * (1 - sl) : C[last] * (1 + sl);
    equity = equityAfterEntryFee + pos * qty * (px - entryPrice) - qty * px * fee;
    if (equity < 0) equity = 0;
    if (equity < tradeLow) tradeLow = equity;
    record(last, equity - equityBeforeEntry, 0);
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

/* -------------------------------------------------------------------------- */
/*  Folds and market regimes of the search period                             */
/* -------------------------------------------------------------------------- */

/** Number of equal, sequential folds the search period is cut into for scoring. */
export const FOLDS = 5;

/** Bar boundaries of the folds: `[0, b1, …, holdoutStart]` (FOLDS + 1 entries). */
export function foldBounds(split: Split): number[] {
  const bounds: number[] = [0];
  for (let k = 1; k < FOLDS; k++) bounds.push(Math.floor((split.holdoutStart * k) / FOLDS));
  bounds.push(split.holdoutStart);
  return bounds;
}

/** Bit flags of `Regimes.mask`, in the order of `REGIME_NAMES`. */
export const REGIME_BULL = 1;
export const REGIME_BEAR = 2;
export const REGIME_HIGH_VOL = 4;
const REGIME_BITS = [REGIME_BULL, REGIME_BEAR, REGIME_HIGH_VOL] as const;
/** Bars above this percentile of trailing realised volatility count as "high volatility". */
export const HIGH_VOL_PERCENTILE = 2 / 3;

/** Regime label of every search-period bar. */
export interface Regimes {
  /** Bit mask per bar (REGIME_BULL | REGIME_BEAR | REGIME_HIGH_VOL); 0 during the warm-up and in the holdout. */
  mask: Uint8Array;
  /** Search-period bars carrying each label, in `REGIME_NAMES` order. */
  bars: [number, number, number];
  /** Search-period length the coverage is measured against. */
  searchBars: number;
}

/**
 * Label the search period with market regimes, causally:
 *   bull / bear      – sign of the trailing return over about a quarter of a year;
 *   high volatility  – trailing realised volatility over about a month above the 67th percentile.
 * Each label at bar i uses closes up to bar i only. The volatility threshold is a percentile of
 * the SEARCH period's own volatility, so nothing about the holdout enters the labels. Lookbacks
 * are calendar-based (bars per year), so the same labels mean the same thing on every timeframe,
 * and capped so short histories still get labelled for most of their length.
 */
export function regimeMask(close: Float64Array, split: Split, periodsPerYear: number): Regimes {
  const H = split.holdoutStart;
  const mask = new Uint8Array(close.length);
  const trendLook = Math.max(5, Math.min(Math.round(periodsPerYear / 4), Math.floor(H / 8)));
  const volLook = Math.max(5, Math.min(Math.round(periodsPerYear / 12), Math.floor(H / 16)));

  // Trailing realised volatility (std of log returns over `volLook` bars ending at i).
  const vol = new Float64Array(H).fill(NaN);
  let sum = 0;
  let sumSq = 0;
  for (let i = 1; i < H; i++) {
    const r = Math.log(close[i] / close[i - 1]);
    sum += r;
    sumSq += r * r;
    if (i > volLook) {
      const old = Math.log(close[i - volLook] / close[i - volLook - 1]);
      sum -= old;
      sumSq -= old * old;
    }
    if (i >= volLook) {
      const mean = sum / volLook;
      vol[i] = Math.sqrt(Math.max(0, sumSq / volLook - mean * mean));
    }
  }
  const sorted = Array.from(vol).filter((v) => v === v).sort((a, b) => a - b);
  const threshold = sorted.length > 0 ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * HIGH_VOL_PERCENTILE))] : Infinity;

  const bars: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < H; i++) {
    let m = 0;
    if (i >= trendLook) m |= close[i] > close[i - trendLook] ? REGIME_BULL : REGIME_BEAR;
    if (vol[i] > threshold) m |= REGIME_HIGH_VOL;
    mask[i] = m;
    for (let j = 0; j < 3; j++) if (m & REGIME_BITS[j]) bars[j]++;
  }
  return { mask, bars, searchBars: H };
}

/* -------------------------------------------------------------------------- */
/*  Search-period metrics: segments, folds and regimes in one pass            */
/* -------------------------------------------------------------------------- */

/** Metrics of everything the search may look at. */
export interface SearchMetrics {
  /** Train + Validation combined. */
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
  /** The search period cut into `FOLDS` equal, sequential folds (oldest first). */
  folds: PerformanceMetrics[];
  /** Bull, bear and high-volatility slices of the search period; null when no regime labels were supplied. */
  regimes: RegimeMetrics[] | null;
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

function resetTally(t: Tally, bars: number, start: number, end: number): void {
  t.bars = bars;
  t.start = start;
  t.end = end;
  t.s1 = t.s2 = t.s3 = t.s4 = t.down = t.exposed = t.maxDd = 0;
  t.trades = t.wins = t.grossProfit = t.grossLoss = t.worstStreak = t.ambiguous = t.pnl1 = t.pnl2 = t.pnl3 = 0;
}

/**
 * One way of cutting the search period into consecutive segments, with the walking state
 * the single pass needs: the current segment, its running peak and its losing streak.
 */
interface Partition {
  bounds: number[];
  tallies: Tally[];
  seg: number;
  peak: number;
  tradeSeg: number;
  streak: number;
}

// Scratch state: `searchMetrics` is synchronous and never re-entered.
const SEARCH_PART: Partition = { bounds: [], tallies: [newTally()], seg: 0, peak: 0, tradeSeg: 0, streak: 0 };
const SPLIT_PART: Partition = { bounds: [], tallies: [newTally(), newTally()], seg: 0, peak: 0, tradeSeg: 0, streak: 0 };
const FOLD_PART: Partition = { bounds: [], tallies: Array.from({ length: FOLDS }, newTally), seg: 0, peak: 0, tradeSeg: 0, streak: 0 };
const PARTS = [SEARCH_PART, SPLIT_PART, FOLD_PART];
/** Regime slices: per-bar return sums (s1, s2) and trade totals. */
const REGIME_ACC = Array.from({ length: 3 }, () => ({ s1: 0, s2: 0, trades: 0, pnl: 0 }));

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

/** Reference (slow) regime statistics: walks every search-period bar. Used when there is no trade list. */
function regimeStatsReference(
  equity: Float64Array,
  initialCapital: number,
  periodsPerYear: number,
  regimes: Regimes,
  ws: BacktestWorkspace | null,
): RegimeMetrics[] {
  for (const acc of REGIME_ACC) acc.s1 = acc.s2 = acc.trades = acc.pnl = 0;
  let prev = initialCapital;
  for (let i = 0; i < regimes.searchBars; i++) {
    const value = equity[i];
    const r = prev > 0 ? value / prev - 1 : 0;
    prev = value;
    const m = regimes.mask[i];
    for (let j = 0; j < 3; j++) {
      if (m & REGIME_BITS[j]) {
        REGIME_ACC[j].s1 += r;
        REGIME_ACC[j].s2 += r * r;
      }
    }
  }
  if (ws !== null) {
    for (let k = 0; k < ws.tradeCount; k++) {
      const entry = ws.tradeEntryBar[k];
      if (entry >= regimes.searchBars) break;
      const m = regimes.mask[entry];
      for (let j = 0; j < 3; j++) {
        if (m & REGIME_BITS[j]) {
          REGIME_ACC[j].trades++;
          REGIME_ACC[j].pnl += ws.tradePnl[k];
        }
      }
    }
  }
  return finishRegimes(regimes, periodsPerYear);
}

function finishRegimes(regimes: Regimes, periodsPerYear: number): RegimeMetrics[] {
  const annualiser = Math.sqrt(periodsPerYear);
  return REGIME_NAMES.map((regime, j) => {
    const bars = regimes.bars[j];
    const acc = REGIME_ACC[j];
    const mean = bars > 0 ? acc.s1 / bars : 0;
    const std = bars > 0 ? Math.sqrt(Math.max(0, acc.s2 / bars - mean * mean)) : 0;
    return {
      regime,
      bars,
      coverage: regimes.searchBars > 0 ? bars / regimes.searchBars : 0,
      trades: acc.trades,
      netPnl: acc.pnl,
      sharpe: std > 1e-12 ? (mean / std) * annualiser : 0,
    };
  });
}

/**
 * Train / validation / combined metrics, the `FOLDS` sequential folds and (with `regimes`) the
 * regime slices of a run that simulated bars `[0, split.holdoutStart)` (or more – anything from
 * the holdout onwards is ignored).
 *
 * This is the optimiser's hottest function after the simulation itself, so it does not walk the
 * whole curve. Outside a trade equity cannot move (no position, no funding, the intrabar low
 * equals the close), so such bars add nothing to returns, drawdown or exposure. Only the bars
 * from each trade's entry to its exit are visited – ONCE – and every view of the period is filled
 * from that single pass: each visited bar's return goes into its segment, its fold and its regime
 * slices. Each segment starts from the equity at its first bar, exactly like `segmentMetrics`,
 * whose results this reproduces.
 */
export function searchMetrics(
  equity: Float64Array,
  split: Split,
  initialCapital: number,
  periodsPerYear: number,
  ws: BacktestWorkspace | null,
  equityLow: Float64Array | null = ws ? ws.equityLow : null,
  regimes: Regimes | null = null,
): SearchMetrics {
  const { validationStart, holdoutStart } = split;
  const capitalAt = (bar: number): number => (bar === 0 ? initialCapital : equity[bar - 1]);
  SEARCH_PART.bounds = [0, holdoutStart];
  SPLIT_PART.bounds = [0, validationStart, holdoutStart];
  FOLD_PART.bounds = foldBounds(split);

  let wipedOut = !(initialCapital > 0);
  for (const part of PARTS) for (const b of part.bounds) if (b < holdoutStart && !(capitalAt(b) > 0)) wipedOut = true;
  if (ws === null || equity !== ws.equity || equityLow !== ws.equityLow || wipedOut) {
    // No trade list to walk (the Buy & Hold benchmark), a foreign curve, or a wiped-out
    // account: use the reference implementation, which handles every case.
    const seg = (from: number, to: number): PerformanceMetrics =>
      segmentMetrics(equity, from, to, capitalAt(from), periodsPerYear, ws, equityLow);
    const fb = FOLD_PART.bounds;
    return {
      search: seg(0, holdoutStart),
      train: seg(0, validationStart),
      validation: seg(validationStart, holdoutStart),
      folds: fb.slice(0, -1).map((from, k) => seg(from, fb[k + 1])),
      regimes: regimes ? regimeStatsReference(equity, initialCapital, periodsPerYear, regimes, ws) : null,
    };
  }

  for (const part of PARTS) {
    part.seg = 0;
    part.tradeSeg = 0;
    part.streak = 0;
    part.peak = initialCapital;
    part.tallies.forEach((t, s) => resetTally(t, part.bounds[s + 1] - part.bounds[s], capitalAt(part.bounds[s]), capitalAt(part.bounds[s + 1])));
  }
  for (const acc of REGIME_ACC) acc.s1 = acc.s2 = acc.trades = acc.pnl = 0;
  const mask = regimes ? regimes.mask : null;

  const low = ws.equityLow;
  const { inMarket, tradeEntryBar, tradeExitBar, tradePnl, tradeAmbiguous, tradeCount } = ws;
  /** First bar not yet visited (consecutive trades can share the exit / re-entry bar). */
  let cursor = 0;

  for (let k = 0; k < tradeCount; k++) {
    const entry = tradeEntryBar[k];
    if (entry >= holdoutStart) break; // trades are stored in entry order
    const to = Math.min(tradeExitBar[k], holdoutStart - 1);

    /* -- Equity path of the trade, bar by bar, into every partition -------- */
    for (let i = entry > cursor ? entry : cursor; i <= to; i++) {
      const value = equity[i];
      const prev = capitalAt(i);
      const r = value !== prev && prev > 0 ? value / prev - 1 : 0;
      const r2 = r * r;
      const worst = low[i] < value ? low[i] : value;
      const exposed = inMarket[i];
      for (let p = 0; p < 3; p++) {
        const part = PARTS[p];
        // Entering a later segment: its peak starts from the equity at its first bar.
        while (i >= part.bounds[part.seg + 1]) {
          part.seg++;
          part.peak = capitalAt(part.bounds[part.seg]);
        }
        const t = part.tallies[part.seg];
        if (r !== 0) {
          t.s1 += r;
          t.s2 += r2;
          t.s3 += r2 * r;
          t.s4 += r2 * r2;
          if (r < 0) t.down += r2;
        }
        // Drawdown: from the highest close so far in the segment to the worst intrabar mark.
        if (worst < part.peak) {
          const dd = 1 - worst / part.peak;
          if (dd > t.maxDd) t.maxDd = dd;
        }
        if (value > part.peak) part.peak = value;
        t.exposed += exposed;
      }
      if (mask !== null && r !== 0) {
        const m = mask[i];
        if (m & REGIME_BULL) {
          REGIME_ACC[0].s1 += r;
          REGIME_ACC[0].s2 += r2;
        }
        if (m & REGIME_BEAR) {
          REGIME_ACC[1].s1 += r;
          REGIME_ACC[1].s2 += r2;
        }
        if (m & REGIME_HIGH_VOL) {
          REGIME_ACC[2].s1 += r;
          REGIME_ACC[2].s2 += r2;
        }
      }
    }
    if (to + 1 > cursor) cursor = to + 1;

    /* -- Trade statistics: a trade belongs to the segment it was opened in -- */
    const pnl = tradePnl[k];
    const ambiguous = tradeAmbiguous[k];
    for (let p = 0; p < 3; p++) {
      const part = PARTS[p];
      let s = part.tradeSeg;
      while (entry >= part.bounds[s + 1]) s++;
      if (s !== part.tradeSeg) {
        // Losing streaks are counted per segment.
        part.tradeSeg = s;
        part.streak = 0;
      }
      const t = part.tallies[s];
      t.trades++;
      t.pnl1 += pnl;
      t.pnl2 += pnl * pnl;
      t.pnl3 += pnl * pnl * pnl;
      t.ambiguous += ambiguous;
      if (pnl > 0) {
        t.wins++;
        t.grossProfit += pnl;
        part.streak = 0;
      } else {
        t.grossLoss -= pnl;
        part.streak++;
        if (part.streak > t.worstStreak) t.worstStreak = part.streak;
      }
    }
    if (mask !== null) {
      const m = mask[entry];
      for (let j = 0; j < 3; j++) {
        if (m & REGIME_BITS[j]) {
          REGIME_ACC[j].trades++;
          REGIME_ACC[j].pnl += pnl;
        }
      }
    }
  }

  return {
    search: finishTally(SEARCH_PART.tallies[0], periodsPerYear, true),
    train: finishTally(SPLIT_PART.tallies[0], periodsPerYear, true),
    validation: finishTally(SPLIT_PART.tallies[1], periodsPerYear, true),
    folds: FOLD_PART.tallies.map((t) => finishTally(t, periodsPerYear, true)),
    regimes: regimes ? finishRegimes(regimes, periodsPerYear) : null,
  };
}

/**
 * Fingerprint of the trade log a run produced before bar `bars`: every trade's entry bar, exit bar,
 * direction and PnL to the cent. Two strategies with the same fingerprint traded identically, so
 * the second one is not a new trial – it is the same bet written differently.
 */
export function tradeLogHash(ws: BacktestWorkspace, bars: number): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  let count = 0;
  const mix = (x: number): void => {
    h1 = Math.imul(h1 ^ x, 0x01000193);
    h2 = Math.imul(h2 ^ x, 0x5bd1e995);
    h2 ^= h2 >>> 15;
  };
  for (let k = 0; k < ws.tradeCount; k++) {
    const entry = ws.tradeEntryBar[k];
    if (entry >= bars) break;
    count++;
    const cents = Math.round(ws.tradePnl[k] * 100);
    mix(entry);
    mix(ws.tradeExitBar[k]);
    mix(ws.tradeDir[k] & 0xff);
    mix(cents | 0);
    mix(Math.floor(cents / 4294967296) | 0);
  }
  return `${count}:${(h1 >>> 0).toString(36)}:${(h2 >>> 0).toString(36)}`;
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
  regimes: Regimes | null = null,
): SearchMetrics {
  runBacktest(ind, genome, settings, ws, split.holdoutStart);
  return searchMetrics(ws.equity, split, settings.initialCapital, periodsPerYear, ws, ws.equityLow, regimes);
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
    folds: m.folds.map((fold, k) => decorateBenchmark(fold, k === 0)),
    regimes: null,
  };
}

/* -------------------------------------------------------------------------- */
/*  Composite fitness                                                         */
/* -------------------------------------------------------------------------- */

/*
 * Hard sample-size gates, calibrated to the timeframe and the length of the data. A strategy below
 * ANY of them is disqualified outright (fitness UNFIT). The bar is a number of trades per YEAR of
 * data – 15 a year on daily candles, rising slowly with the number of candles per day (≈ 26 on 4h,
 * ≈ 39 on 1h, ≈ 59 on 15m) – times the years each segment covers, with small absolute floors.
 * (12 a year was tried in the 2026-11 review: on the AUDIT random walks it let 12–14-trade
 * validation flukes pass every robustness gate, so the bar stays at 15. Single-market daily
 * strategies that trade less often belong in a pooled multi-asset run, where the gates apply to
 * the trades summed over the pool.)
 * A flat count would either disqualify every daily swing strategy or wave through intraday noise.
 * With a pooled multi-asset sample the counts are summed over the assets (see `evaluation.ts`).
 */
/** Trades a year required on daily candles. */
export const BASE_TRADES_PER_YEAR = 15;
/** Absolute floors: search period (train + validation), validation segment, holdout. */
export const MIN_SEARCH_TRADES_FLOOR = 30;
export const MIN_VALIDATION_TRADES_FLOOR = 10;
export const MIN_HOLDOUT_TRADES_FLOOR = 10;

/** Minimum trades per year of data for candles of this frequency. */
export function minTradesPerYearFor(periodsPerYear: number): number {
  const barsPerDay = periodsPerYear / 365;
  return BASE_TRADES_PER_YEAR * Math.pow(Math.max(1, barsPerDay), 0.3);
}

export function tradeGatesFor(split: Split, periodsPerYear: number): TradeGates {
  const rate = minTradesPerYearFor(periodsPerYear);
  const years = (bars: number): number => Math.max(0, bars) / periodsPerYear;
  return {
    search: Math.max(MIN_SEARCH_TRADES_FLOOR, Math.ceil(rate * years(split.holdoutStart))),
    validation: Math.max(MIN_VALIDATION_TRADES_FLOOR, Math.ceil(rate * years(split.holdoutStart - split.validationStart))),
    perYear: rate,
    holdout: Math.max(MIN_HOLDOUT_TRADES_FLOOR, Math.ceil(rate * years(split.n - split.holdoutStart))),
  };
}

/**
 * The holdout conditions of a Robust verdict, as a list of the ones a strategy misses (empty =
 * the holdout supports an edge): the holdout Sharpe's 95% confidence interval must lie above zero,
 * the holdout Sharpe must beat Buy & Hold's holdout Sharpe, and there must be at least
 * `minTrades` holdout trades. Raw holdout profit is NOT enough: on a trending market
 * most long strategies make money on the holdout by drift alone.
 */
export function holdoutShortfalls(
  holdout: PerformanceMetrics,
  benchmark: PerformanceMetrics | null,
  periodsPerYear: number,
  minTrades: number,
  trades: number = holdout.trades,
): string[] {
  const out: string[] = [];
  const [ciLow] = sharpeConfidenceInterval(holdout, periodsPerYear);
  if (!(ciLow > 0)) out.push(`The holdout Sharpe's 95% confidence interval reaches ${ciLow.toFixed(2)}: it includes zero.`);
  if (benchmark !== null && !(holdout.sharpe > benchmark.sharpe)) {
    out.push(`Holdout Sharpe ${holdout.sharpe.toFixed(2)} does not beat Buy & Hold's ${benchmark.sharpe.toFixed(2)}.`);
  }
  if (trades < minTrades) out.push(`Only ${trades} holdout trades (at least ${minTrades} needed).`);
  return out;
}

/** Fitness assigned to strategies that never trade or fail a minimum-trade gate. */
export const UNFIT = -1000;

const clampTo = (v: number, min: number, max: number): number => (v < min ? min : v > max ? max : v);

/** Drawdown floor of the Calmar ratio, in percent: a fold that barely dipped is not "infinitely" good. */
const CALMAR_DD_FLOOR = 2;
/** Payoff ratio cap of the R-expectancy (a fold without a single loser is not infinitely good either). */
const PAYOFF_CAP = 5;
/** R per year that scores 1 in the edge term (e.g. 0.35 R × 30 trades a year ≈ 10 R). */
const EDGE_SCALE = 10;

/** Calmar ratio of a segment: CAGR / max drawdown, clamped to ±3. */
export function calmar(m: PerformanceMetrics): number {
  return clampTo(m.cagr / Math.max(CALMAR_DD_FLOOR, m.maxDrawdown), -3, 3);
}

/**
 * Expectancy × frequency of a segment: the average trade in units of the average loss
 * (win rate × payoff − loss rate, Van Tharp's R-expectancy) times trades per year – the R a year the
 * edge earns. 0 without trades or when the bar frequency is unknown.
 */
export function edgeRate(m: PerformanceMetrics, periodsPerYear: number): number {
  if (m.trades === 0 || !(m.bars > 0) || !(periodsPerYear > 0)) return 0;
  const win = m.winRate / 100;
  const payoff = Math.min(PAYOFF_CAP, m.payoffRatio);
  const expectancyR = win * payoff - (1 - win);
  return expectancyR * ((m.trades * periodsPerYear) / m.bars);
}

/**
 * Score of one fold on a Sharpe-like scale, every input clamped so no outlier can dominate:
 * 35% Sortino, 25% Sharpe, 20% Calmar (return over drawdown) and 20% expectancy × trade
 * frequency (R a year). The last two reward an edge that pays for its drawdowns and repeats often
 * enough to matter after costs – not just a smooth curve.
 */
export function foldScore(m: PerformanceMetrics, periodsPerYear = 0): number {
  return (
    0.35 * (clampTo(m.sortino, -3, 4.5) / 1.5) +
    0.25 * clampTo(m.sharpe, -3, 3) +
    0.2 * calmar(m) +
    0.2 * clampTo(edgeRate(m, periodsPerYear) / EDGE_SCALE, -1.5, 3)
  );
}

/** A strategy is consistent when at least this many of the five folds made money (scaled to its active folds). */
export const CONSISTENT_FOLDS = 3;
/** Folds a strategy must be active in (in the market at some point) before flat folds are set aside. */
export const MIN_ACTIVE_FOLDS = 3;
/** Fitness bonus for a consistent strategy, and the cost of each profitable fold it is short of one. */
export const CONSISTENCY_BONUS = 0.5;
/** Intrabar drawdown inside a single fold beyond which the fold counts as a catastrophe. */
export const CATASTROPHIC_FOLD_DRAWDOWN = 0.35;
/** A regime slice is only judged when it covers at least this share of the search period ... */
export const REGIME_MIN_COVERAGE = 0.1;
/** ... and the strategy opened at least this many trades in it. */
export const REGIME_MIN_TRADES = 10;
/** Annualised Sharpe below which a strategy has collapsed in a regime slice. */
export const REGIME_COLLAPSE_SHARPE = -0.5;
/** Fitness cost of each regime slice the strategy collapses in. */
export const REGIME_COLLAPSE_PENALTY = 0.75;

/** Linear-interpolation quantile (`q` in [0, 1]) of a list of numbers. */
export function quantile(values: readonly number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** True when the strategy held a position at some point during the fold. */
export function isActiveFold(fold: PerformanceMetrics): boolean {
  return fold.trades > 0 || fold.exposure > 0;
}

/** Score of every fold of the search period, oldest first (see `foldScore`). */
export function foldScores(m: SearchMetrics, periodsPerYear = 0): number[] {
  return m.folds.map((fold) => foldScore(fold, periodsPerYear));
}

/**
 * The folds a strategy is judged on: when it held a position in at least `MIN_ACTIVE_FOLDS` folds,
 * only those (a regime filter that sat out a bear year did its job); otherwise every fold. Used by
 * BOTH the fitness and the fold gate, so the two never disagree about a flat fold.
 */
export function judgedFolds(folds: readonly PerformanceMetrics[]): number[] {
  const all = folds.map((_, k) => k);
  const active = all.filter((k) => isActiveFold(folds[k]));
  return active.length >= MIN_ACTIVE_FOLDS ? active : all;
}

/** Profitable folds a strategy needs among `judged` folds: 3 of 5, scaled to the folds judged. */
export function foldsNeeded(judged: number): number {
  return Math.ceil((CONSISTENT_FOLDS / FOLDS) * judged - 1e-9);
}

/** Regime slices that carry enough bars and trades to be judged. */
export function testableRegimes(regimes: readonly RegimeMetrics[] | null): RegimeMetrics[] {
  return (regimes ?? []).filter((r) => r.coverage >= REGIME_MIN_COVERAGE && r.trades >= REGIME_MIN_TRADES);
}

/** Testable regime slices in which the strategy collapsed (annualised Sharpe below `REGIME_COLLAPSE_SHARPE`). */
export function collapsedRegimes(regimes: readonly RegimeMetrics[] | null): RegimeMetrics[] {
  return testableRegimes(regimes).filter((r) => r.sharpe < REGIME_COLLAPSE_SHARPE);
}

/**
 * True when ALL of a strategy's profit comes from one regime: at least two regime slices are
 * testable, exactly one of them made money, and the strategy collapsed in every other one.
 * Such a strategy is a bet on that regime continuing, not an edge – it is disqualified.
 */
export function isSingleRegime(regimes: readonly RegimeMetrics[] | null): boolean {
  const testable = testableRegimes(regimes);
  if (testable.length < 2) return false;
  const winners = testable.filter((r) => r.sharpe > 0);
  if (winners.length !== 1) return false;
  return testable.every((r) => r === winners[0] || r.sharpe < REGIME_COLLAPSE_SHARPE);
}

/**
 * Per-asset score, everything except the deflated-Sharpe term (which changes as
 * more strategies are tried and is therefore added separately – see `deflatedTerm`):
 *
 *   + 1.50 × MEDIAN of the fold scores (Sortino, Sharpe, Calmar and expectancy × frequency, clamped –
 *     see `foldScore`)
 *   + 0.50 when at least 3 of 5 folds made money, else − 0.50 per profitable fold short of that
 *
 *   A fold in which the strategy never held a position is FLAT: it neither made nor lost money.
 *   When a strategy was active in at least 3 folds, flat folds are left out of the median and the
 *   consistency count (which then asks for 60% of the active folds) – a regime filter that sat out
 *   a bear year did its job and is not punished for it. Below 3 active folds every fold counts.
 *
 *   − 3 (and 6 more per unit beyond 35%) when any single fold suffers a catastrophic drawdown
 *   − drawdown penalty beyond 20% over the whole search period, steeper beyond 45%
 *   − 0.75 for every regime slice (bull, bear, high volatility) the strategy collapses in
 *
 * The median, not a low quantile: a trend follower that sits out a sideways year loses a little in
 * that fold, and grading it on its second-worst fold punished exactly the strategies that work.
 * Consistency is enforced instead by counting profitable folds, and blow-ups by the per-fold
 * drawdown. Nothing here rewards raw return over Buy & Hold.
 */
export function assetBaseScore(m: SearchMetrics, periodsPerYear = 0): number {
  const scores = foldScores(m, periodsPerYear);
  const judged = judgedFolds(m.folds);
  let score = 1.5 * median(judged.map((k) => scores[k]));
  const profitable = judged.filter((k) => m.folds[k].roi > 0).length;
  const needed = foldsNeeded(judged.length);
  score += profitable >= needed ? CONSISTENCY_BONUS : -CONSISTENCY_BONUS * (needed - profitable);
  let worstFold = 0;
  for (const fold of m.folds) if (fold.maxDrawdown > worstFold) worstFold = fold.maxDrawdown;
  worstFold /= 100;
  if (worstFold > CATASTROPHIC_FOLD_DRAWDOWN) score -= 3 + 6 * (worstFold - CATASTROPHIC_FOLD_DRAWDOWN);
  const dd = m.search.maxDrawdown / 100;
  score -= 4 * Math.max(0, dd - 0.2) + 6 * Math.max(0, dd - 0.45);
  score -= REGIME_COLLAPSE_PENALTY * collapsedRegimes(m.regimes).length;
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
  /** Train + Validation trades: the primary asset's, or the pooled total over a pooled universe. */
  trades: number;
  /** Validation trades, primary or pooled like `trades`. */
  validationTrades: number;
  /** Trades per year of the search period: the lowest rate of any asset, or the pooled rate. */
  tradesPerYear: number;
  /** The minimum-trade gates of the run's data (see `tradeGatesFor`). */
  gates: TradeGates;
  /** True when the primary asset makes all of its money in a single market regime (see `isSingleRegime`). */
  singleRegime: boolean;
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
 *   UNFIT (disqualified) below the minimum-trade gates (see `tradeGatesFor`), or when the primary
 *   asset makes all of its money in one market regime and collapses in the others; otherwise
 *
 *   median of the per-asset scores (a single asset without a universe)
 *   − 1.0 when fewer than 2/3 of the universe is profitable on validation
 *   − 0.15 for every tunable parameter above `FREE_PARAMETERS` (4)
 *   − 0.75 for reward:risk below 1:1, unless the validation win rate clears break-even by 10 points
 */
export function combineFitness(f: FitnessInputs): number {
  if (f.assetScores.length === 0 || !passesTradeGates(f) || f.singleRegime) return UNFIT;
  let score = median(f.assetScores);
  if (f.assetScores.length > 1 && f.profitableShare < 2 / 3 - 1e-9) score -= 1;
  score -= 0.15 * Math.max(0, f.params - FREE_PARAMETERS);
  if (f.rewardToRisk !== null && f.rewardToRisk < 1) {
    const breakeven = 100 / (1 + f.rewardToRisk);
    if (f.validationWinRate < breakeven + 10) score -= 0.75;
  }
  return Number.isFinite(score) ? score : UNFIT;
}

/** Weight of the neighbourhood in the rank fitness of a strategy that sits above its neighbours. */
export const STABILITY_WEIGHT = 0.3;
/** A neighbour counts as at worst this far below the strategy itself (a disqualified neighbour is not −1000). */
export const NEIGHBOUR_SPAN = 3;

/**
 * Rank fitness after the parameter-neighbourhood check: a proportional blend of the strategy's own
 * fitness (70%) and the MEDIAN fitness of its one-grid-step neighbours (30%) when the neighbours
 * are worse; the strategy's own fitness when they are as good or better. A plateau keeps its
 * score, a spike loses 30% of its lead over its neighbourhood – enough to rank plateaus first,
 * not so much that a promising new rule combination is discarded the moment one neighbour slips
 * under a trade gate (neighbours count as at most `NEIGHBOUR_SPAN` below the strategy). The blend
 * never lifts a score, so it never lifts a strategy over the leaderboard floor.
 */
export function plateauFitness(point: number, neighbours: readonly number[]): number {
  if (neighbours.length === 0 || point <= UNFIT) return point;
  const m = median(neighbours.map((value) => Math.max(value, point - NEIGHBOUR_SPAN)));
  return m < point ? (1 - STABILITY_WEIGHT) * point + STABILITY_WEIGHT * m : point;
}

/** True when a strategy clears every hard minimum-trade gate. */
export function passesTradeGates(f: Pick<FitnessInputs, 'trades' | 'validationTrades' | 'tradesPerYear' | 'gates'>): boolean {
  return f.trades >= f.gates.search && f.validationTrades >= f.gates.validation && f.tradesPerYear >= f.gates.perYear - 1e-9;
}

/* -------------------------------------------------------------------------- */
/*  Robustness gates (search period only – see `RobustChecks`)                */
/* -------------------------------------------------------------------------- */

/** Validation Sharpe a robust strategy must exceed. */
export const ROBUST_MIN_VALIDATION_SHARPE = 0.3;
/**
 * Deflated Sharpe a robust strategy must reach: 50% means the validation Sharpe beats the Sharpe the
 * luckiest of that many skill-less (random-walk) strategies is expected to show.
 */
export const ROBUST_MIN_DSR = 0.5;

/** Out-of-sample validation pass: Sharpe above 0.3 and a net profit after every cost. */
export function passesValidationGate(validation: PerformanceMetrics): boolean {
  return validation.sharpe > ROBUST_MIN_VALIDATION_SHARPE && validation.netPnl > 0;
}

/**
 * Multi-fold consistency: 3 of 5 folds profitable – counted over the folds the strategy was active
 * in, exactly as the fitness counts them (see `judgedFolds`) – and no fold beyond the drawdown cap.
 */
export function passesFoldGate(folds: readonly PerformanceMetrics[]): boolean {
  if (folds.length === 0) return false;
  const judged = judgedFolds(folds);
  const profitable = judged.filter((k) => folds[k].roi > 0).length;
  return profitable >= foldsNeeded(judged.length) && folds.every((fold) => fold.maxDrawdown <= 100 * CATASTROPHIC_FOLD_DRAWDOWN);
}

/** Net profit per trade (0 without trades). */
export function expectancy(m: PerformanceMetrics): number {
  return m.trades > 0 ? m.netPnl / m.trades : 0;
}

export function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
