import { describe, expect, it } from 'vitest';
import { buildDataset, normalizeWeights } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  assetBaseScore,
  calmar,
  edgeRate,
  foldScore,
  judgedFolds,
  emptyMetrics,
  designEffects,
  effectiveBets,
  FITNESS_WEIGHTS,
  expectancy,
  medianFoldScore,
  passesFoldGate,
  ROBUST_MIN_DSR,
  foldBounds,
  FOLDS,
  tradeLogHash,
  type SearchMetrics,
  BacktestWorkspace,
  barsPerYear,
  benchmarkEquity,
  benchmarkSearch,
  combineFitness,
  evaluateCondition,
  evaluateGroup,
  evaluateSearch,
  holdoutMetrics,
  minTradesPerYearFor,
  tradeGatesFor,
  runBacktest,
  searchMetrics,
  segmentMetrics,
  splitFor,
  UNFIT,
} from '../src/engine/backtester';
import {
  ARCHETYPE_LABEL,
  describeArchetype,
  describeExit,
  describeLongEntry,
  groupFormula,
  indicatorBadges,
  listIndicators,
} from '../src/engine/describe';
import {
  conditionRole,
  conformsTo,
  filterConforms,
  gaussian,
  HYPOTHESES,
  hypothesisUnsupported,
  indicatorClass,
  orthogonal,
  MAX_CONDITIONS,
  FILTER_RATE,
  FIXED_ATR_PERIOD,
  STATIC_GRIDS,
  staticTier,
  triggerConforms,
  countParameters,
  crossoverGenomes,
  DEFAULT_LIMITS,
  genomeSignature,
  GRIDS,
  isCoherent,
  isEventCondition,
  isShortRegime,
  LOOKBACK_GRID,
  MAX_PARAMETERS,
  mirrorCondition,
  mirrorGroup,
  mulberry32,
  mutateGenome,
  normalizeGenome,
  randomGenome,
  repairGenome,
  STATIC_PCT_GRID,
  stepNeighbours,
  TIME_STOP_GRID,
} from '../src/engine/genome';
import { atr, bollinger, donchian, ema, IndicatorCache, rsi, sma, stochastic, supertrend } from '../src/engine/indicators';
import { generatePineScript } from '../src/engine/pine';
import { createContext, fitnessOf, robustChecks, scoreAssets } from '../src/engine/evaluation';
import { deflatedSharpe, expectedMaxSharpe, inverseNormalCdf, normalCdf, RunningStats, sharpeConfidenceInterval } from '../src/lib/stats';
import {
  clampRange,
  HOLDOUT_FRACTION,
  HYPOTHESIS_IDS,
  rangeAllowed,
  SLOPE_LAG,
  TRAIN_FRACTION,
  VALIDATION_FRACTION,
  type BacktestSettings,
  type Condition,
  type ConditionRole,
  type OHLCV,
  type PerformanceMetrics,
  type StrategyGenome,
  type StrategyResult,
  type TradeMode,
} from '../src/types/trading';

const NOW = Date.UTC(2026, 0, 15);
/** 100% of equity, compounding – the simplest sizing to reason about in fill tests. */
const SETTINGS: BacktestSettings = {
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  tradeMode: 'long-short',
  sizing: { mode: 'full', riskPct: 2, compounding: true },
  fundingPctPerBar: 0,
};
const LONG_ONLY: BacktestSettings = { ...SETTINGS, tradeMode: 'long-only' };
const FREE: BacktestSettings = { ...LONG_ONLY, feePct: 0, slippagePct: 0 };

function dataset(symbol = 'BTC-USD', interval: '1h' | '4h' | '1d' = '1d', range: '1y' | '2y' | '5y' = '2y'): OHLCV {
  const series = generateSyntheticCandles(symbol, interval, range, NOW);
  return buildDataset({ [symbol]: series }, [{ symbol, weight: 1 }]).data;
}

function truncate(data: OHLCV, length: number): OHLCV {
  return {
    t: data.t.slice(0, length),
    o: data.o.slice(0, length),
    h: data.h.slice(0, length),
    l: data.l.slice(0, length),
    c: data.c.slice(0, length),
    v: data.v.slice(0, length),
  };
}

const f64 = (values: number[]): Float64Array => Float64Array.from(values);

/** Hand-built candles on a daily grid. */
function candles(o: number[], h: number[], l: number[], c: number[]): OHLCV {
  const n = c.length;
  return {
    t: Float64Array.from({ length: n }, (_, i) => i * 86_400_000),
    o: f64(o),
    h: f64(h),
    l: f64(l),
    c: f64(c),
    v: new Float64Array(n).fill(1000),
  };
}

/* -------------------------------------------------------------------------- */

describe('indicators', () => {
  it('computes SMA and EMA with a NaN warm-up', () => {
    const src = f64([1, 2, 3, 4, 5, 6]);
    const s = sma(src, 3);
    expect(Number.isNaN(s[1])).toBe(true);
    expect(Array.from(s.slice(2))).toEqual([2, 3, 4, 5]);
    const e = ema(src, 3);
    expect(Number.isNaN(e[1])).toBe(true);
    expect(e[2]).toBeCloseTo(2, 12); // SMA seed
    expect(e[3]).toBeCloseTo(3, 12); // 0.5 * 4 + 0.5 * 2
    expect(e[5]).toBeCloseTo(5, 12);
  });

  it('matches the reference Wilder RSI example', () => {
    // Classic 14-period example from Wilder / StockCharts.
    const closes = [
      44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.1, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.0,
      46.03, 46.41, 46.22, 45.64,
    ];
    const out = rsi(f64(closes), 14);
    expect(Number.isNaN(out[13])).toBe(true);
    expect(out[14]).toBeCloseTo(70.46, 1);
    expect(out[15]).toBeCloseTo(66.25, 1);
    expect(out[19]).toBeCloseTo(57.92, 0);
  });

  it('keeps oscillators and bands inside their theoretical bounds', () => {
    const data = dataset();
    const st = stochastic(data.h, data.l, data.c, 14, 3, 3);
    const bb = bollinger(data.c, 20, 2);
    const r = rsi(data.c, 14);
    const a = atr(data.h, data.l, data.c, 14);
    for (let i = 60; i < data.c.length; i++) {
      expect(st.k[i]).toBeGreaterThanOrEqual(0);
      expect(st.k[i]).toBeLessThanOrEqual(100);
      expect(r[i]).toBeGreaterThanOrEqual(0);
      expect(r[i]).toBeLessThanOrEqual(100);
      expect(bb.upper[i]).toBeGreaterThanOrEqual(bb.middle[i]);
      expect(bb.lower[i]).toBeLessThanOrEqual(bb.middle[i]);
      expect(a[i]).toBeGreaterThan(0);
    }
  });

  it('keeps the Supertrend line on the correct side of price', () => {
    const data = dataset();
    const { line, direction } = supertrend(data.h, data.l, data.c, 10, 3);
    let flips = 0;
    for (let i = 20; i < data.c.length; i++) {
      if (direction[i] === 1) expect(data.c[i]).toBeGreaterThan(line[i]);
      else expect(data.c[i]).toBeLessThan(line[i]);
      if (direction[i] !== direction[i - 1]) flips++;
    }
    expect(flips).toBeGreaterThan(2);
  });

  it('is strictly causal: truncating the future never changes past values', () => {
    const full = dataset();
    const cut = 300;
    const a = new IndicatorCache(full);
    const b = new IndicatorCache(truncate(full, cut));
    const pairs: [Float64Array, Float64Array][] = [
      [a.sma(50), b.sma(50)],
      [a.ema(21), b.ema(21)],
      [a.rsi(14), b.rsi(14)],
      [a.macdLine(12, 26), b.macdLine(12, 26)],
      [a.macdSignal(12, 26, 9), b.macdSignal(12, 26, 9)],
      [a.macdHist(12, 26, 9), b.macdHist(12, 26, 9)],
      [a.macdHistPrev(12, 26, 9), b.macdHistPrev(12, 26, 9)],
      [a.bbUpper(20, 2), b.bbUpper(20, 2)],
      [a.bbLower(20, 2), b.bbLower(20, 2)],
      [a.bbPercentB(20, 2), b.bbPercentB(20, 2)],
      [a.bbBandwidth(20, 2), b.bbBandwidth(20, 2)],
      [a.atr(14), b.atr(14)],
      [a.atrPercent(14), b.atrPercent(14)],
      [a.stochK(14, 3), b.stochK(14, 3)],
      [a.stochD(14, 3, 3), b.stochD(14, 3, 3)],
      [a.adx(14), b.adx(14)],
      [a.plusDI(14), b.plusDI(14)],
      [a.minusDI(14), b.minusDI(14)],
      [a.volumeRatio(20), b.volumeRatio(20)],
      [a.supertrend(10, 3), b.supertrend(10, 3)],
      [a.cci(20), b.cci(20)],
    ];
    for (const [fullSeries, cutSeries] of pairs) {
      expect(cutSeries.length).toBe(cut);
      for (let i = 0; i < cut; i++) {
        if (Number.isNaN(cutSeries[i])) expect(Number.isNaN(fullSeries[i])).toBe(true);
        else expect(fullSeries[i]).toBeCloseTo(cutSeries[i], 9);
      }
    }
  });

  it('bounds the memoisation cache', () => {
    const cache = new IndicatorCache(dataset(), 1); // 1 byte budget → minimum capacity
    for (let p = 5; p < 250; p++) cache.ema(p);
    expect(cache.size).toBeLessThanOrEqual(96);
    expect(cache.ema(249)[400]).toBeGreaterThan(0);
  });
});

/* -------------------------------------------------------------------------- */

const dynamicExit = (stop: number, target: number | null, trailing: boolean, reversalExit = false): StrategyGenome['exit'] => ({
  mode: 'dynamic',
  atrPeriod: 14,
  stopAtrMult: stop,
  trailing,
  takeProfitAtrMult: target,
  reversalExit,
  longExit: null,
  shortExit: null,
});

/** Trend-following: long while price is above the Supertrend, trailing ATR stop. */
const TREND: StrategyGenome = normalizeGenome(
  {
    archetype: 'TREND',
    longEntry: { logic: 'AND', conditions: [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }] },
    shortEntry: null,
    exit: dynamicExit(3, null, true),
    maxBarsInTrade: null,
  },
  'long-short',
  DEFAULT_LIMITS,
);

/** Mean reversion: buy a low RSI inside an uptrend, static bracket. */
const DIP: StrategyGenome = normalizeGenome(
  {
    archetype: 'MEAN_REVERSION',
    longEntry: {
      logic: 'AND',
      conditions: [
        { ind: 'RSI', period: 14, op: 'lt', level: 45 },
        { ind: 'PRICE_MA', maType: 'EMA', period: 112, op: 'gt' },
      ],
    },
    shortEntry: null,
    exit: { mode: 'static', stopLossPct: 4, takeProfitPct: 10 },
    maxBarsInTrade: null,
  },
  'long-only',
  DEFAULT_LIMITS,
);

/** Volume ratio of flat volume is exactly 1 > 0.5: true on every candle once the 10-bar SMA exists (first signal on bar 9). */
const ALWAYS_LONG = (exit: StrategyGenome['exit'], maxBarsInTrade: number | null = null): StrategyGenome => ({
  archetype: 'TREND',
  longEntry: { logic: 'AND', conditions: [{ ind: 'VOLUME', period: 10, op: 'gt', level: 0.5 }] },
  shortEntry: null,
  exit,
  maxBarsInTrade,
});

describe('event triggers, reversals and new primitives', () => {
  it('computes the Donchian channel of the PREVIOUS candles and the EMA slope', () => {
    const data = dataset('ETH-USD', '1d', '2y');
    for (const period of [10, 28, 56]) {
      const upper = donchian(data.h, period, true);
      const lower = donchian(data.l, period, false);
      for (let i = 0; i < data.c.length; i++) {
        if (i < period) {
          expect(Number.isNaN(upper[i])).toBe(true);
          continue;
        }
        let hi = -Infinity;
        let lo = Infinity;
        for (let k = i - period; k < i; k++) {
          hi = Math.max(hi, data.h[k]);
          lo = Math.min(lo, data.l[k]);
        }
        expect(upper[i]).toBe(hi);
        expect(lower[i]).toBe(lo);
      }
    }
    const ind = new IndicatorCache(data);
    const out = new Uint8Array(data.c.length);
    evaluateCondition({ ind: 'MA_SLOPE', maType: 'EMA', period: 56, op: 'gt' }, ind, out);
    const e = ind.ema(56);
    for (let i = 10; i < data.c.length; i++) expect(out[i]).toBe(e[i] > e[i - SLOPE_LAG] ? 1 : 0);
  });

  it('reverses on an opposite entry signal, also on the open a time stop fires', () => {
    const data = dataset('BTC-USD', '1d', '5y');
    const ind = new IndicatorCache(data);
    const n = data.c.length;
    const genome = normalizeGenome(
      {
        archetype: 'TREND',
        hypothesis: 'TREND_FOLLOWING',
        longEntry: { logic: 'AND', conditions: [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 10, slowType: 'EMA', slowPeriod: 28, op: 'cross_above' }] },
        shortEntry: null,
        exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 4, trailing: false, takeProfitAtrMult: 8, reversalExit: false, longExit: null, shortExit: null },
        maxBarsInTrade: 10,
      },
      'long-short',
      DEFAULT_LIMITS,
    );
    const ws = new BacktestWorkspace(n);
    runBacktest(ind, genome, { ...SETTINGS, tradeMode: 'long-short' }, ws);
    const goLong = new Uint8Array(n);
    const goShort = new Uint8Array(n);
    evaluateGroup(genome.longEntry, ind, goLong, new Uint8Array(n));
    evaluateGroup(genome.shortEntry!, ind, goShort, new Uint8Array(n));
    let reversals = 0;
    for (let k = 0; k < ws.tradeCount; k++) {
      const entry = ws.tradeEntryBar[k];
      const exit = ws.tradeExitBar[k];
      const dir = ws.tradeDir[k];
      // While a trade is open, an actionable opposite signal is never ignored: the trade ends on the next open.
      for (let p = entry; p < exit - 1; p++) {
        const opposite = dir === 1 ? goShort[p] === 1 && goLong[p] === 0 : goLong[p] === 1 && goShort[p] === 0;
        expect(opposite).toBe(false);
      }
      if (k + 1 < ws.tradeCount && ws.tradeEntryBar[k + 1] === exit && ws.tradeDir[k + 1] === -dir) reversals++;
    }
    expect(reversals).toBeGreaterThan(5);
  });

  it('does not re-enter the same trend leg after every stop: event triggers churn far less than states', () => {
    const data = dataset('SOL-USD', '1d', '5y');
    const ind = new IndicatorCache(data);
    const n = data.c.length;
    const exit: StrategyGenome['exit'] = { mode: 'static', stopLossPct: 3, takeProfitPct: 5 };
    const make = (op: 'gt' | 'cross_above'): StrategyGenome =>
      normalizeGenome(
        { archetype: 'TREND', longEntry: { logic: 'AND', conditions: [{ ind: 'PRICE_MA', maType: 'EMA', period: 56, op }] }, shortEntry: null, exit, maxBarsInTrade: null },
        'long-only',
        DEFAULT_LIMITS,
      );
    /** Trades that open on the very candle after the previous one closed: re-entries of the same leg. */
    const churn = (genome: StrategyGenome): { trades: number; reentries: number } => {
      const ws = new BacktestWorkspace(n);
      runBacktest(ind, genome, SETTINGS, ws);
      let reentries = 0;
      for (let k = 1; k < ws.tradeCount; k++) if (ws.tradeEntryBar[k] - ws.tradeExitBar[k - 1] <= 1) reentries++;
      return { trades: ws.tradeCount, reentries };
    };
    const state = churn(make('gt'));
    const event = churn(make('cross_above'));
    expect(state.reentries / state.trades).toBeGreaterThan(0.4);
    expect(event.reentries / Math.max(1, event.trades)).toBeLessThan(0.15);
    // The search never builds a state trigger.
    const rng = mulberry32(3);
    for (let i = 0; i < 400; i++) {
      const g = randomGenome(rng, 'long-only', DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      expect(isEventCondition(g.longEntry.conditions[0])).toBe(true);
    }
  });
});

describe('backtester', () => {
  it('has no look-ahead bias: future candles cannot change past equity', () => {
    const full = dataset('ETH-USD');
    const cut = 420;
    for (const genome of [TREND, DIP]) {
      const wsFull = new BacktestWorkspace(full.c.length);
      runBacktest(new IndicatorCache(full), genome, SETTINGS, wsFull);
      const part = truncate(full, cut);
      const wsCut = new BacktestWorkspace(cut);
      runBacktest(new IndicatorCache(part), genome, SETTINGS, wsCut);
      // The final candle of the truncated run differs only by the forced liquidation.
      for (let i = 0; i < cut - 1; i++) {
        expect(wsCut.equity[i]).toBeCloseTo(wsFull.equity[i], 6);
        expect(wsCut.equityLow[i]).toBeCloseTo(wsFull.equityLow[i], 6);
      }
      expect(wsFull.tradeCount).toBeGreaterThan(3);
    }
  });

  it('never reads the holdout when the run is limited to the search period', () => {
    const full = dataset('ETH-USD');
    const split = splitFor(full.c.length);
    const poisoned = truncate(full, full.c.length);
    // Corrupt every holdout candle: a search-period run must be completely unaffected.
    for (let i = split.holdoutStart; i < poisoned.c.length; i++) {
      poisoned.o[i] = poisoned.h[i] = poisoned.l[i] = poisoned.c[i] = 1e-6;
    }
    const ppy = barsPerYear(full.t);
    for (const genome of [TREND, DIP]) {
      const a = evaluateSearch(new IndicatorCache(full), genome, SETTINGS, new BacktestWorkspace(full.c.length), split, ppy);
      const b = evaluateSearch(new IndicatorCache(poisoned), genome, SETTINGS, new BacktestWorkspace(full.c.length), split, ppy);
      expect(b.search.netPnl).toBeCloseTo(a.search.netPnl, 6);
      expect(b.validation.sharpe).toBeCloseTo(a.validation.sharpe, 9);
      expect(b.search.trades).toBe(a.search.trades);
    }
  });

  it('fills a close-of-bar signal at the NEXT bar open, with fee and slippage', () => {
    // Close crosses above SMA(5) on bar 5; the entry must happen at bar 6's open.
    const closes = [100, 100, 100, 100, 100, 110, 111, 112, 113, 114];
    const opens = [100, 100, 100, 100, 100, 100, 120, 111, 112, 113];
    const data = candles(
      opens,
      closes.map((c, i) => Math.max(c, opens[i]) + 1),
      closes.map((c, i) => Math.min(c, opens[i]) - 1),
      closes,
    );
    const n = closes.length;
    const genome: StrategyGenome = {
      archetype: 'TREND',
      longEntry: { logic: 'AND', conditions: [{ ind: 'PRICE_MA', maType: 'SMA', period: 5, op: 'cross_above' }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 15, takeProfitPct: 30 },
      maxBarsInTrade: null,
    };
    const ws = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(data), genome, LONG_ONLY, ws);

    for (let i = 0; i <= 5; i++) expect(ws.equity[i]).toBe(10_000); // still flat on the signal candle
    expect(ws.tradeCount).toBe(1);
    expect(ws.tradeEntryBar[0]).toBe(6);

    const entry = 120 * 1.0005;
    const qty = 10_000 / (entry * 1.001);
    const afterFee = 10_000 - qty * entry * 0.001;
    expect(ws.equity[6]).toBeCloseTo(afterFee + qty * (111 - entry), 6);
    // Liquidated on the last close, paying slippage and the exit fee.
    const exitPx = 114 * (1 - 0.0005);
    expect(ws.equity[n - 1]).toBeCloseTo(afterFee + qty * (exitPx - entry) - qty * exitPx * 0.001, 6);
  });

  it('honours stop-loss and take-profit levels intrabar and counts ambiguous candles', () => {
    const n = 12;
    const flat = new Array<number>(n).fill(100);
    const genome = ALWAYS_LONG({ mode: 'static', stopLossPct: 5, takeProfitPct: 10 });

    // Stop: bar 10 trades down to 90 → exit at 95 (−5%).
    const lows = [...flat];
    lows[10] = 90;
    const wsStop = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(candles(flat, flat, lows, flat)), genome, FREE, wsStop);
    expect(wsStop.tradeEntryBar[0]).toBe(10);
    expect(wsStop.tradePnl[0]).toBeCloseTo(-500, 6);
    expect(wsStop.tradeAmbiguous[0]).toBe(0);

    // Target: bar 10 trades up to 115 → exit at 110 (+10%).
    const highs = [...flat];
    highs[10] = 115;
    const wsTarget = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(candles(flat, highs, flat, flat)), genome, FREE, wsTarget);
    expect(wsTarget.tradePnl[0]).toBeCloseTo(1000, 6);

    // Both touched in the same candle → the stop is assumed to fill first, and the trade is flagged.
    const wsBoth = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(candles(flat, highs, lows, flat)), genome, FREE, wsBoth);
    expect(wsBoth.tradePnl[0]).toBeCloseTo(-500, 6);
    expect(wsBoth.tradeAmbiguous[0]).toBe(1);
    const m = segmentMetrics(wsBoth.equity, 0, n, 10_000, 365, wsBoth);
    expect(m.ambiguousTrades).toBe(1);
  });

  it('sizes by risk: 2% of starting capital at the stop, no compounding, never above 1× equity', () => {
    const n = 14;
    const flat = new Array<number>(n).fill(100);
    const lows = [...flat];
    lows[10] = 90; // stop of trade 1
    lows[11] = 90; // stop of trade 2 (re-entered on bar 11's open)
    const data = candles(flat, flat, lows, flat);
    const genome = ALWAYS_LONG({ mode: 'static', stopLossPct: 3, takeProfitPct: 30 });
    const risk: BacktestSettings = { ...FREE, sizing: { mode: 'risk', riskPct: 2, compounding: false } };
    const ws = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(data), genome, risk, ws);

    // Stopped out twice in a row at a 3% stop while risking 2% → two −$200 trades.
    expect(ws.tradePnl[0]).toBeCloseTo(-200, 6);
    expect(ws.tradePnl[1]).toBeCloseTo(-200, 6); // still $200: risk is based on the starting capital
    const m = segmentMetrics(ws.equity, 0, n, 10_000, 365, ws);
    expect(m.maxDrawdown).toBeGreaterThanOrEqual(3.99);
    expect(m.maxConsecLosses).toBeGreaterThanOrEqual(2);

    // Compounding: the second loss is 2% of the reduced equity.
    runBacktest(new IndicatorCache(data), genome, { ...risk, sizing: { mode: 'risk', riskPct: 2, compounding: true } }, ws);
    expect(ws.tradePnl[1]).toBeCloseTo(-196, 6);

    // A 1% stop with 2% risk would need 2× leverage – exposure is capped at 1× equity instead.
    const tight = ALWAYS_LONG({ mode: 'static', stopLossPct: 1, takeProfitPct: 30 });
    runBacktest(new IndicatorCache(data), tight, risk, ws);
    expect(ws.tradePnl[0]).toBeCloseTo(-100, 6);
  });

  it('measures drawdown from intrabar lows, not only from closes', () => {
    const n = 14;
    const flat = new Array<number>(n).fill(100);
    const lows = [...flat];
    lows[11] = 92; // dips 8% intrabar but closes unchanged and never reaches the 15% stop
    const ws = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(candles(flat, flat, lows, flat)), ALWAYS_LONG({ mode: 'static', stopLossPct: 15, takeProfitPct: 30 }), FREE, ws);
    const m = segmentMetrics(ws.equity, 0, n, 10_000, 365, ws);
    expect(m.maxDrawdown).toBeCloseTo(8, 6);
    // The close-to-close curve alone would report no drawdown at all.
    expect(segmentMetrics(ws.equity, 0, n, 10_000, 365, ws, null).maxDrawdown).toBe(0);
  });

  it('applies the time stop and the funding cost', () => {
    const n = 30;
    const flat = new Array<number>(n).fill(100);
    const data = candles(flat, flat, flat, flat);
    const ws = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(data), ALWAYS_LONG({ mode: 'static', stopLossPct: 15, takeProfitPct: 30 }, 5), FREE, ws);
    // Entry on bar 10, closed at the open of bar 15, no re-entry on that bar, next entry on bar 16.
    expect(ws.tradeEntryBar[0]).toBe(10);
    expect(ws.tradeExitBar[0]).toBe(15);
    expect(ws.inMarket[15]).toBe(0);
    expect(ws.tradeEntryBar[1]).toBe(16);

    // Funding: 0.1% of notional per bar held, for the 5 closes the trade is open.
    runBacktest(new IndicatorCache(data), ALWAYS_LONG({ mode: 'static', stopLossPct: 15, takeProfitPct: 30 }, 5), { ...FREE, fundingPctPerBar: 0.1 }, ws);
    expect(ws.tradePnl[0]).toBeCloseTo(-50, 6);
  });

  it('never opens short trades in Long-Only mode, and mirrors the long rules in Long/Short mode', () => {
    const data = dataset('SOL-USD');
    const ws = new BacktestWorkspace(data.c.length);
    runBacktest(new IndicatorCache(data), normalizeGenome(TREND, 'long-only', DEFAULT_LIMITS), LONG_ONLY, ws);
    expect(ws.tradeCount).toBeGreaterThan(0);
    for (let k = 0; k < ws.tradeCount; k++) expect(ws.tradeDir[k]).toBe(1);

    runBacktest(new IndicatorCache(data), TREND, SETTINGS, ws);
    let shorts = 0;
    for (let k = 0; k < ws.tradeCount; k++) if (ws.tradeDir[k] === -1) shorts++;
    expect(shorts).toBeGreaterThan(0);
  });

  it('charges costs: higher fees can only lower the final equity', () => {
    const data = dataset();
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);
    runBacktest(ind, TREND, { ...SETTINGS, feePct: 0, slippagePct: 0 }, ws);
    const free = ws.equity[data.c.length - 1];
    runBacktest(ind, TREND, SETTINGS, ws);
    expect(ws.equity[data.c.length - 1]).toBeLessThan(free);
  });

  it('splits 60 / 20 / 20 and keeps the three segments consistent', () => {
    expect(TRAIN_FRACTION + VALIDATION_FRACTION + HOLDOUT_FRACTION).toBeCloseTo(1, 12);
    expect([TRAIN_FRACTION, VALIDATION_FRACTION, HOLDOUT_FRACTION]).toEqual([0.6, 0.2, 0.2]);
    const data = dataset();
    const n = data.c.length;
    const split = splitFor(n);
    expect(split).toEqual({ validationStart: Math.floor(n * 0.6), holdoutStart: Math.floor(n * 0.8), n });

    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(n);
    const ppy = barsPerYear(data.t);
    runBacktest(ind, TREND, SETTINGS, ws);
    const m = searchMetrics(ws.equity, split, SETTINGS.initialCapital, ppy, ws);
    const holdout = holdoutMetrics(ws.equity, split, ppy, ws);
    const full = segmentMetrics(ws.equity, 0, n, SETTINGS.initialCapital, ppy, ws);
    expect(m.train.netPnl + m.validation.netPnl).toBeCloseTo(m.search.netPnl, 6);
    expect(m.search.netPnl + holdout.netPnl).toBeCloseTo(full.netPnl, 6);
    expect(m.train.trades + m.validation.trades + holdout.trades).toBe(full.trades);
    expect(m.train.bars + m.validation.bars + holdout.bars).toBe(n);
    expect(full.maxDrawdown).toBeGreaterThanOrEqual(holdout.maxDrawdown - 1e-9);
    expect(full.winRate).toBeGreaterThanOrEqual(0);
    expect(full.winRate).toBeLessThanOrEqual(100);
    expect(full.breakevenWinRate).toBeCloseTo(100 / (1 + full.payoffRatio), 9);
    expect(Number.isFinite(full.skew)).toBe(true);
    expect(full.kurtosis).toBeGreaterThan(1);
    expect(full.sharpe).toBeCloseTo(full.sharpeRaw * Math.sqrt(ppy), 9);
  });

  it('computes the fused search metrics exactly like three separate segment passes', () => {
    const rng = mulberry32(2024);
    const keys = [
      'netPnl', 'roi', 'cagr', 'maxDrawdown', 'sharpe', 'sortino', 'winRate', 'profitFactor', 'trades', 'exposure', 'bars',
      'sharpeRaw', 'skew', 'kurtosis', 'maxConsecLosses', 'payoffRatio', 'breakevenWinRate', 'tradeSkew', 'ambiguousTrades',
    ] as const;
    const close = (a: number, b: number): boolean =>
      a === b || Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(a), Math.abs(b));
    let withTrades = 0;
    let flat = 0;
    for (const [symbol, mode, sizing] of [
      ['BTC-USD', 'long-short', SETTINGS.sizing],
      ['NVDA', 'long-only', { mode: 'risk', riskPct: 2, compounding: false }],
      ['SOL-USD', 'long-short', { mode: 'risk', riskPct: 1, compounding: true }],
    ] as const) {
      const data = dataset(symbol, '1d', '5y');
      const n = data.c.length;
      const split = splitFor(n);
      const ppy = barsPerYear(data.t);
      const settings: BacktestSettings = { ...SETTINGS, tradeMode: mode, sizing };
      const ind = new IndicatorCache(data);
      const ws = new BacktestWorkspace(n);
      const fb = foldBounds(split);
      for (let i = 0; i < 400; i++) {
        const genome = randomGenome(rng, mode, DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
        // Search-period run (what the optimiser does) and a full run (what the holdout reveal does).
        for (const bars of [split.holdoutStart, n]) {
          runBacktest(ind, genome, settings, ws, bars);
          const fused = searchMetrics(ws.equity, split, settings.initialCapital, ppy, ws, ws.equityLow);
          // Folds: the single pass must equal five separate segment passes.
          fb.slice(0, -1).forEach((from, f) => {
            const start = from === 0 ? settings.initialCapital : ws.equity[from - 1];
            const ref = segmentMetrics(ws.equity, from, fb[f + 1], start, ppy, ws);
            for (const key of keys) {
              if (!close(fused.folds[f][key], ref[key])) throw new Error(`${symbol} #${i} fold ${f}.${key}: fused ${fused.folds[f][key]} vs reference ${ref[key]}`);
            }
          });
          const reference = {
            search: segmentMetrics(ws.equity, 0, split.holdoutStart, settings.initialCapital, ppy, ws),
            train: segmentMetrics(ws.equity, 0, split.validationStart, settings.initialCapital, ppy, ws),
            validation: segmentMetrics(ws.equity, split.validationStart, split.holdoutStart, ws.equity[split.validationStart - 1], ppy, ws),
          };
          for (const segment of ['search', 'train', 'validation'] as const) {
            for (const key of keys) {
              const a = fused[segment][key];
              const b = reference[segment][key];
              if (!close(a, b)) throw new Error(`${symbol} #${i} ${segment}.${key}: fused ${a} vs reference ${b}`);
            }
          }
          if (fused.search.trades > 0) withTrades++;
          else flat++;
        }
      }
    }
    expect(withTrades).toBeGreaterThan(1500);
    expect(flat).toBeGreaterThan(5); // the never-traded shortcut is exercised too
  });

  it('evaluates signals only inside the simulated window and matches a plain AND of the conditions', () => {
    const data = dataset('ETH-USD', '1d', '5y');
    const n = data.c.length;
    const ind = new IndicatorCache(data);
    const rng = mulberry32(11);
    const out = new Uint8Array(n);
    const a = new Uint8Array(n);
    const b = new Uint8Array(n);
    let pairs = 0;
    for (let i = 0; i < 600; i++) {
      const genome = randomGenome(rng, 'long-only', DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      const { conditions } = genome.longEntry;
      const limit = 1 + Math.floor(rng() * (n - 1));
      out.fill(7); // sentinel: nothing beyond `limit` may be written
      evaluateGroup(genome.longEntry, ind, out, new Uint8Array(n), limit);
      evaluateCondition(conditions[0], ind, a);
      b.fill(1);
      for (const extra of conditions.slice(1)) {
        const c = new Uint8Array(n);
        evaluateCondition(extra, ind, c);
        for (let k = 0; k < n; k++) b[k] &= c[k];
      }
      if (conditions.length > 1) pairs++;
      for (let k = 0; k < n; k++) {
        if (k < limit) expect(out[k]).toBe(a[k] & b[k]);
        else expect(out[k]).toBe(7);
      }
    }
    expect(pairs).toBeGreaterThan(200);
  });

  it('builds an exact Buy & Hold benchmark', () => {
    const data = dataset('SPY');
    const n = data.c.length;
    const split = splitFor(n);
    const equity = benchmarkEquity(data, SETTINGS);
    const units = 10_000 / (data.o[0] * 1.0005 * 1.001);
    expect(equity[n - 1]).toBeCloseTo(units * data.c[n - 1], 6);
    const bench = benchmarkSearch(data, SETTINGS, split, barsPerYear(data.t));
    expect(bench.equity.length).toBe(split.holdoutStart); // the holdout is not part of the search benchmark
    expect(bench.search.netPnl).toBeCloseTo(units * data.c[split.holdoutStart - 1] - 10_000, 6);
    expect(bench.train.netPnl + bench.validation.netPnl).toBeCloseTo(bench.search.netPnl, 6);
    // A known curve: constant +1% per bar has zero variance → Sharpe 0, no drawdown, positive CAGR.
    const steady = Float64Array.from({ length: 100 }, (_, i) => 100 * 1.01 ** (i + 1));
    const s = segmentMetrics(steady, 0, 100, 100, 252, null, null);
    expect(s.maxDrawdown).toBe(0);
    expect(s.cagr).toBeGreaterThan(0);
    expect(s.roi).toBeCloseTo((1.01 ** 100 - 1) * 100, 6);
  });

  it('ranks by one composite fitness: hard trade gates, the median over assets, and nothing that depends on luck', () => {
    const gates = { search: 60, validation: 15, perYear: 15, holdout: 15 };
    const base = { assetScores: [2], trades: 150, validationTrades: 40, tradesPerYear: 40, gates };
    const score = combineFitness(base);
    expect(score).toBe(2);
    // Hard gates: one trade short of any minimum disqualifies, however good the rest looks.
    expect(combineFitness({ ...base, trades: gates.search })).toBe(score);
    expect(combineFitness({ ...base, trades: gates.search - 1 })).toBe(UNFIT);
    expect(combineFitness({ ...base, validationTrades: gates.validation })).toBe(score);
    expect(combineFitness({ ...base, validationTrades: gates.validation - 1 })).toBe(UNFIT);
    expect(combineFitness({ ...base, tradesPerYear: gates.perYear })).toBe(score);
    expect(combineFitness({ ...base, tradesPerYear: gates.perYear - 0.1 })).toBe(UNFIT);
    for (const trades of [3, 8, 14]) expect(combineFitness({ ...base, assetScores: [50], validationTrades: trades })).toBe(UNFIT);
    expect(combineFitness({ ...base, trades: 0 })).toBe(UNFIT);
    // A pooled run: the median over the assets counts, so one lucky asset cannot carry the pool.
    expect(combineFitness({ ...base, assetScores: [5, 1, 2] })).toBe(2);
    expect(combineFitness({ ...base, assetScores: [9, -1, -2] })).toBe(-1);

    const data = dataset();
    const split = splitFor(data.c.length);
    const ppy = barsPerYear(data.t);
    const m = evaluateSearch(new IndicatorCache(data), TREND, SETTINGS, new BacktestWorkspace(data.c.length), split, ppy);
    const assetScore = assetBaseScore(m, ppy);
    expect(Number.isFinite(assetScore)).toBe(true);
    // No stacked penalties: the search-period drawdown is judged once, by gate 1's per-fold cap, not
    // again by the fitness.
    const withDrawdown = (dd: number): SearchMetrics => ({ ...m, search: { ...m.search, maxDrawdown: dd } });
    expect(Object.keys(FITNESS_WEIGHTS).sort()).toEqual(['edge', 'riskAdjusted']);
    expect(assetBaseScore(withDrawdown(10), ppy)).toBe(assetBaseScore(withDrawdown(60), ppy));
    // Raw validation return over Buy & Hold does not move the score at all.
    const moreValidationRoi = { ...m, validation: { ...m.validation, roi: m.validation.roi + 30 } };
    expect(assetBaseScore(moreValidationRoi, ppy)).toBe(assetScore);
  });

  it('defines three orthogonal robustness gates: sample + folds, deflated Sharpe, stress + stability', () => {
    const m = (over: Partial<PerformanceMetrics>): PerformanceMetrics => ({ ...emptyMetrics(), ...over });
    // Gate 1 (fold half): a positive median-fold Sortino / Calmar and no fold beyond the drawdown cap.
    const fold = (r: number, maxDrawdown = 5): PerformanceMetrics => m({ roi: r, sortino: r, cagr: 5 * r, maxDrawdown, trades: 6, exposure: 30 });
    expect(passesFoldGate([fold(1), fold(2), fold(-1), fold(3), fold(-2)])).toBe(true);
    expect(medianFoldScore([fold(1), fold(2), fold(-1), fold(3), fold(-2)])).toBeGreaterThan(0);
    expect(passesFoldGate([fold(1), fold(2), fold(-1), fold(-3), fold(-2)])).toBe(false);
    expect(passesFoldGate([fold(1), fold(2), fold(1, 40), fold(3), fold(2)])).toBe(false); // one fold beyond 35%
    // Gate 3 reads the neighbours' median expectancy per trade.
    expect(expectancy(m({ netPnl: 50, trades: 10 }))).toBe(5);
    expect(expectancy(m({ netPnl: 50, trades: 0 }))).toBe(0);
    // Gate 2: the deflated Sharpe must beat the random-walk hurdle (50%).
    expect(ROBUST_MIN_DSR).toBe(0.5);
  });

  it('counts correlated, overlapping trades across a basket as fewer independent bets', () => {
    const mask = (bits: number[]): Uint8Array => Uint8Array.from(bits);
    const lockstep = mask([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]);
    // One asset: its own trades.
    expect(effectiveBets([{ trades: 30, inMarket: lockstep }], () => 1)).toBe(30);
    // Two perfectly correlated assets holding positions at the same time: one bet, not two.
    expect(effectiveBets([{ trades: 30, inMarket: lockstep }, { trades: 30, inMarket: lockstep }], () => 1)).toBeCloseTo(30, 12);
    // Uncorrelated (or negatively correlated) assets: every trade counts.
    expect(effectiveBets([{ trades: 30, inMarket: lockstep }, { trades: 30, inMarket: lockstep }], () => 0)).toBe(60);
    expect(effectiveBets([{ trades: 30, inMarket: lockstep }, { trades: 30, inMarket: lockstep }], () => -0.4)).toBe(60);
    // Correlated but never in the market together: independent bets.
    const other = mask([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
    expect(effectiveBets([{ trades: 30, inMarket: lockstep }, { trades: 10, inMarket: other }], () => 0.9)).toBe(40);
    // Half overlap at correlation 0.8: design effect 1 + 0.5 × 0.8 = 1.4 for the first asset.
    const half = mask([1, 1, 0, 0, 0, 0, 1, 1, 0, 0]);
    const design = designEffects([{ trades: 20, inMarket: lockstep }, { trades: 20, inMarket: half }], () => 0.8);
    expect(design[0]).toBeCloseTo(1 + (4 / 8) * 0.8, 12);
    expect(design[1]).toBeCloseTo(1 + 1 * 0.8, 12);
    // Four lockstep crypto majors: 4 × 25 trades count as 25 independent bets.
    const four = Array.from({ length: 4 }, () => ({ trades: 25, inMarket: lockstep }));
    expect(effectiveBets(four, () => 1)).toBeCloseTo(25, 12);
  });

  it('calibrates the minimum-trade gates to the timeframe and the length of the data', () => {
    // 15 trades a year on daily candles, rising slowly with candles per day.
    expect(minTradesPerYearFor(365)).toBeCloseTo(15, 9);
    expect(minTradesPerYearFor(252)).toBeCloseTo(15, 9); // stock daily: never below the daily rate
    expect(minTradesPerYearFor(365 * 6)).toBeCloseTo(15 * 6 ** 0.3, 9); // 4h ≈ 25.7
    expect(minTradesPerYearFor(365 * 96)).toBeGreaterThan(55); // 15m
    // 5 years of daily candles: 4-year search period, 1-year validation and holdout.
    const daily = tradeGatesFor(splitFor(1826), 365);
    expect(daily).toMatchObject({ search: 60, validation: 15, holdout: 16 });
    // 1 year of daily candles: the absolute floors take over.
    expect(tradeGatesFor(splitFor(365), 365)).toMatchObject({ search: 30, validation: 10, holdout: 10 });
    // 2 years of 4h candles need more trades than 5 years of daily ones per year of data.
    const h4 = tradeGatesFor(splitFor(4380), 365 * 6);
    expect(h4.perYear).toBeGreaterThan(daily.perYear);
    expect(h4.search).toBe(Math.ceil(h4.perYear * (splitFor(4380).holdoutStart / (365 * 6))));
  });

  it('scores the median fold of a Sortino / Calmar blend, so one lucky fold cannot carry a strategy', () => {
    const fold = (sharpe: number, trades = 8, maxDrawdown = 5): PerformanceMetrics => ({
      ...emptyMetrics(),
      sharpe,
      sortino: sharpe * 1.5,
      roi: sharpe * 4,
      trades,
      exposure: trades > 0 ? 40 : 0,
      maxDrawdown,
    });
    const asMetrics = (folds: PerformanceMetrics[]): SearchMetrics => ({ search: emptyMetrics(), train: emptyMetrics(), validation: emptyMetrics(), folds });
    const of = (sharpes: number[]): SearchMetrics => asMetrics(sharpes.map((x) => fold(x)));
    const score = (m: SearchMetrics): number => assetBaseScore(m, 365);
    expect(FOLDS).toBe(5);
    expect(FITNESS_WEIGHTS.riskAdjusted).toBe(1.5);
    // Fold score: ½ Sortino / 1.5 + ½ Calmar (CAGR 0 here, so Calmar 0) – i.e. half the Sharpe in these folds.
    expect(foldScore(fold(1))).toBeCloseTo(0.5, 12);
    // One spectacular fold and four losing ones vs five steady, modest folds.
    const lucky = score(of([3, -0.5, -0.4, -0.6, -0.3]));
    const steady = score(of([0.8, 0.7, 0.9, 0.6, 0.75]));
    expect(steady).toBeGreaterThan(lucky);
    expect(lucky).toBeLessThan(0);
    // The fold order does not matter, only the distribution.
    expect(score(of([-0.3, -0.6, -0.4, -0.5, 3]))).toBeCloseTo(lucky, 12);
    // A trend follower that loses a little in two sideways folds keeps its median fold.
    const trend = score(of([1.2, -0.2, 1.0, -0.1, 0.9]));
    expect(trend).toBeCloseTo(1.5 * 0.5 * 0.9, 9);
    // Three losing folds of five pull the median below zero.
    expect(score(of([1.2, -0.2, -0.3, -0.1, 0.9]))).toBeLessThan(0);
    // Flat folds (no position at all – a regime filter sitting out a bear year) are set aside once
    // the strategy was active in at least three folds ...
    const sitsOut = asMetrics([fold(0, 0), fold(0, 0), fold(1.0), fold(0.8), fold(-0.2)]);
    expect(score(sitsOut)).toBeCloseTo(1.5 * 0.5 * 0.8, 9);
    // ... but not before: two active folds out of five are judged on all five.
    const barelyActive = asMetrics([fold(0, 0), fold(0, 0), fold(0, 0), fold(1.0), fold(0.8)]);
    expect(score(barelyActive)).toBeCloseTo(0, 9);
    // The fold boundaries cut the search period into five equal, sequential pieces.
    const bounds = foldBounds(splitFor(1000));
    expect(bounds).toEqual([0, 160, 320, 480, 640, 800]);
  });

  it('rewards Calmar and expectancy x frequency in every fold, and judges flat folds the same way in the gate', () => {
    const m = (over: Partial<PerformanceMetrics>): PerformanceMetrics => ({ ...emptyMetrics(), ...over });
    // Calmar: CAGR over max drawdown (2% floor), clamped to ±3.
    expect(calmar(m({ cagr: 30, maxDrawdown: 15 }))).toBeCloseTo(2, 12);
    expect(calmar(m({ cagr: 30, maxDrawdown: 0.5 }))).toBe(3);
    expect(calmar(m({ cagr: -10, maxDrawdown: 20 }))).toBeCloseTo(-0.5, 12);
    // R-expectancy x trades a year: 45% winners at 2:1 payoff = 0.35 R a trade; 20 trades over 365 daily bars.
    const edge = m({ trades: 20, winRate: 45, payoffRatio: 2, bars: 365 });
    expect(edgeRate(edge, 365)).toBeCloseTo(0.35 * 20, 9);
    expect(edgeRate(edge, 0)).toBe(0); // unknown bar frequency
    expect(edgeRate(m({ trades: 0, bars: 365 }), 365)).toBe(0);
    // Expectancy x frequency is scored over the whole search period: more trades with the same
    // per-trade edge score higher, a losing edge lower; the folds are identical in all three.
    const folds = Array.from({ length: 5 }, () => m({ sharpe: 1, sortino: 1.5, cagr: 20, maxDrawdown: 10, trades: 6, roi: 4, exposure: 30 }));
    const withSearch = (search: PerformanceMetrics): SearchMetrics => ({ search, train: emptyMetrics(), validation: emptyMetrics(), folds });
    const often = m({ trades: 60, winRate: 45, payoffRatio: 2, bars: 365 });
    const rarely = { ...often, trades: 10 };
    const losing = { ...often, winRate: 25 };
    expect(assetBaseScore(withSearch(often), 365)).toBeGreaterThan(assetBaseScore(withSearch(rarely), 365));
    expect(assetBaseScore(withSearch(losing), 365)).toBeLessThan(assetBaseScore(withSearch(rarely), 365));
    expect(assetBaseScore(withSearch(often), 365) - assetBaseScore(withSearch(rarely), 365)).toBeCloseTo(FITNESS_WEIGHTS.edge * (0.35 * 50) / 10, 9);
    // The fold gate sets flat folds aside exactly like the fitness: a regime filter that sat out two
    // folds and made money in two of the three it traded passes.
    const fold = (roi: number, trades = 8, maxDrawdown = 5): PerformanceMetrics =>
      m({ roi, sortino: roi, cagr: 5 * roi, trades, exposure: trades > 0 ? 30 : 0, maxDrawdown });
    const sitsOut = [fold(0, 0), fold(0, 0), fold(4), fold(-1), fold(3)];
    expect(passesFoldGate(sitsOut)).toBe(true);
    expect(judgedFolds(sitsOut)).toEqual([2, 3, 4]);
    // Below three active folds every fold counts, flat ones as not profitable.
    expect(passesFoldGate([fold(0, 0), fold(0, 0), fold(0, 0), fold(4), fold(3)])).toBe(false);
    // One losing active fold of three is fine; two are not.
    expect(passesFoldGate([fold(0, 0), fold(0, 0), fold(-2), fold(-1), fold(3)])).toBe(false);
  });

  it('judges parameter stability once, as half of gate 3 – never stacked onto the fitness', () => {
    const data = dataset('ETH-USD', '4h', '1y');
    const ctx = createContext([{ symbol: 'ETH-USD', data }], LONG_ONLY, 64 * 1024 * 1024);
    const { scores } = scoreAssets(ctx, TREND);
    const { fitness, dsr } = fitnessOf(ctx, scores);
    const mc = { simulations: 500, trades: 40, probProfit: 0.9, drawdownP95: 20, medianReturn: 5, p5Return: 1, baseDrawdown: 10, passed: true };
    // Not through the second tier yet: gate 3 is open and the strategy cannot pass.
    expect(robustChecks(ctx, scores, dsr, null).stress).toBeNull();
    expect(robustChecks(ctx, scores, 1, null).passed).toBe(false);
    // Stress passes only with a passed Monte Carlo AND a positive neighbour expectancy.
    expect(robustChecks(ctx, scores, dsr, { monteCarlo: mc, neighbourExpectancy: 3 }).stress).toBe(true);
    expect(robustChecks(ctx, scores, dsr, { monteCarlo: mc, neighbourExpectancy: -1 }).stress).toBe(false);
    expect(robustChecks(ctx, scores, dsr, { monteCarlo: { ...mc, passed: false }, neighbourExpectancy: 3 }).stress).toBe(false);
    // The fitness is the same number whatever the neighbourhood or the hurdle says.
    ctx.hurdle = 0.5;
    expect(fitnessOf(ctx, scores).fitness).toBe(fitness);
  });
});

/* -------------------------------------------------------------------------- */

describe('selection-bias statistics', () => {
  it('implements the normal distribution helpers', () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 6);
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 4);
    expect(inverseNormalCdf(0.975)).toBeCloseTo(1.96, 3);
    expect(normalCdf(inverseNormalCdf(0.123))).toBeCloseTo(0.123, 5);
  });

  it('tracks a running mean and variance (Welford)', () => {
    const stats = new RunningStats();
    const values = [0.1, -0.05, 0.02, 0.3, -0.2, 0.07];
    values.forEach((v) => stats.push(v));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
    expect(stats.count).toBe(6);
    expect(stats.mean).toBeCloseTo(mean, 12);
    expect(stats.variance).toBeCloseTo(variance, 12);
  });

  it('deflates the Sharpe ratio as more strategies are tried', () => {
    const m = { sharpeRaw: 0.12, bars: 365, skew: 0, kurtosis: 3 };
    // A single trial: plain probabilistic Sharpe against zero.
    expect(deflatedSharpe(m, 1, 0).probability).toBeGreaterThan(0.98);
    const few = deflatedSharpe(m, 10, 0.003);
    const many = deflatedSharpe(m, 10_000, 0.003);
    expect(many.hurdle).toBeGreaterThan(few.hurdle);
    expect(many.probability).toBeLessThan(few.probability);
    expect(many.probability).toBeLessThan(0.5);
    expect(expectedMaxSharpe(10_000, 0.003)).toBeCloseTo(Math.sqrt(0.003) * 3.85, 1);
    // Negative skew and fat tails widen the uncertainty.
    expect(deflatedSharpe({ ...m, skew: -2, kurtosis: 12 }, 10, 0.003).probability).toBeLessThan(few.probability);
  });

  it('reports a 95% confidence interval for the Sharpe ratio', () => {
    // 23 trades over ~1.5 years of daily bars with an annualised Sharpe of 4.2: the interval is huge.
    const perBar = 4.2 / Math.sqrt(365);
    const [lo, hi] = sharpeConfidenceInterval({ sharpeRaw: perBar, bars: 550, skew: 0, kurtosis: 3 }, 365);
    expect(lo).toBeLessThan(4.2);
    expect(hi).toBeGreaterThan(4.2);
    expect(hi - lo).toBeGreaterThan(3);
    const [lo2, hi2] = sharpeConfidenceInterval({ sharpeRaw: perBar, bars: 5500, skew: 0, kurtosis: 3 }, 365);
    expect(hi2 - lo2).toBeLessThan(hi - lo);
  });
});

/* -------------------------------------------------------------------------- */

function conditionsOf(genome: StrategyGenome): Condition[] {
  const all = [...genome.longEntry.conditions, ...(genome.shortEntry?.conditions ?? [])];
  if (genome.exit.mode === 'dynamic') {
    all.push(...(genome.exit.longExit?.conditions ?? []), ...(genome.exit.shortExit?.conditions ?? []));
  }
  return all;
}

function numericParams(c: Condition): number[] {
  const out: number[] = [];
  for (const [key, value] of Object.entries(c)) {
    if (typeof value === 'number' && /period|fast|slow|kPeriod/i.test(key) && key !== 'dPeriod') out.push(value);
  }
  return out;
}

function assertValid(genome: StrategyGenome, mode: TradeMode): void {
  // Structure.
  expect(genome.longEntry.logic).toBe('AND');
  expect(genome.longEntry.conditions.length).toBeGreaterThanOrEqual(1);
  expect(genome.longEntry.conditions.length).toBeLessThanOrEqual(3);
  expect(isCoherent(genome)).toBe(true);
  expect('sizing' in genome).toBe(false);

  // Hypothesis (search-built genomes) or archetype (hand-built ones).
  const [trigger, ...rest] = genome.longEntry.conditions;
  const role = conditionRole(trigger);
  if (genome.hypothesis !== undefined) {
    expect(conformsTo(genome)).toBe(true);
    expect(genome.archetype).toBe(HYPOTHESES[genome.hypothesis].archetype);
    // The entry trigger is always a discrete event, never a state that holds bar after bar.
    expect(isEventCondition(trigger)).toBe(true);
    for (const filter of rest) expect(['gt', 'lt']).toContain(filter.op);
    for (const filter of rest) expect(isEventCondition(filter)).toBe(false);
  } else if (genome.archetype === 'TREND') {
    expect(role).toBe('TREND_UP');
    for (const second of rest) expect(conditionRole(second)).toBe('FILTER');
  } else {
    expect(['OVERSOLD', 'TREND_DOWN']).toContain(role);
    for (const second of rest) expect(['FILTER', 'TREND_UP']).toContain(conditionRole(second));
  }

  // Shorts: none, the exact mirror of the longs, or the mirrored trigger inside the short's own bear regime.
  if (mode === 'long-only') expect(genome.shortEntry).toBeNull();
  else if (mode === 'long-short') expect(genome.shortEntry).toEqual(mirrorGroup(genome.longEntry));
  else {
    expect(genome.shortRegime).toBeTruthy();
    expect(isShortRegime(genome.shortRegime as Condition)).toBe(true);
    expect(genome.shortEntry?.conditions).toEqual([mirrorCondition(trigger), genome.shortRegime]);
  }
  if (mode !== 'asymmetric') expect(genome.shortRegime).toBeUndefined();
  expect(countParameters(genome)).toBeLessThanOrEqual(MAX_PARAMETERS + (mode === 'asymmetric' ? 1 : 0));

  // Grids.
  for (const c of conditionsOf(genome)) {
    for (const lookback of numericParams(c)) expect(LOOKBACK_GRID).toContain(lookback);
    if (c.ind === 'MACD') expect(c.slow).toBeGreaterThan(c.fast);
    if (c.ind === 'MA_CROSS') expect(c.slowPeriod).toBeGreaterThan(c.fastPeriod);
    if (c.ind === 'RSI' || (c.ind === 'STOCH' && c.variant === 'level') || c.ind === 'CCI') expect(Math.abs(c.level % 5)).toBe(0);
    if (c.ind === 'SUPERTREND' || c.ind === 'BB') expect((c.mult * 2) % 1).toBe(0);
    if (c.ind === 'VOLUME' || c.ind === 'ATR') expect(['gt', 'lt']).toContain(c.op);
  }

  // Exits.
  if (genome.exit.mode === 'static') {
    expect(STATIC_PCT_GRID).toContain(genome.exit.stopLossPct);
    expect(STATIC_PCT_GRID).toContain(genome.exit.takeProfitPct);
    expect(genome.exit.stopLossPct).toBeLessThanOrEqual(15);
  } else {
    const exit = genome.exit;
    expect(GRIDS.stopAtrMult).toContain(exit.stopAtrMult);
    expect(GRIDS.atrPeriod).toContain(exit.atrPeriod);
    if (exit.takeProfitAtrMult !== null) {
      expect(GRIDS.takeProfitAtrMult).toContain(exit.takeProfitAtrMult);
      // A trail is only kept when the target sits at least 1 ATR beyond the stop distance.
      if (exit.trailing) expect(exit.takeProfitAtrMult).toBeGreaterThanOrEqual(exit.stopAtrMult + 1);
    } else {
      expect(exit.trailing || exit.reversalExit || genome.maxBarsInTrade !== null).toBe(true);
    }
    // Opposite-signal exits: any hypothesis (hand-built genomes: TREND only), a single condition, the mirror of the entry trigger.
    if (genome.archetype === 'MEAN_REVERSION' && genome.hypothesis === undefined) expect(exit.reversalExit).toBe(false);
    if (exit.reversalExit) {
      expect(exit.longExit?.conditions).toEqual([mirrorCondition(trigger)]);
      if (mode !== 'long-only') expect(exit.shortExit?.conditions).toEqual([trigger]);
      else expect(exit.shortExit).toBeNull();
    } else {
      expect(exit.longExit).toBeNull();
      expect(exit.shortExit).toBeNull();
    }
  }
  if (genome.maxBarsInTrade !== null) expect(TIME_STOP_GRID).toContain(genome.maxBarsInTrade);
}

describe('genome', () => {
  it('classifies every condition variant', () => {
    const table: [Condition, ConditionRole][] = [
      // Filters.
      [{ ind: 'ATR', period: 14, op: 'gt', level: 2 }, 'FILTER'],
      [{ ind: 'VOLUME', period: 20, op: 'gt', level: 1.5 }, 'FILTER'],
      [{ ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 25 }, 'FILTER'],
      [{ ind: 'BB', period: 20, mult: 2, variant: 'bandwidth', op: 'lt', level: 5 }, 'FILTER'],
      // Oscillators: the zone decides.
      [{ ind: 'RSI', period: 14, op: 'lt', level: 30 }, 'OVERSOLD'],
      [{ ind: 'RSI', period: 14, op: 'cross_above', level: 30 }, 'OVERSOLD'],
      [{ ind: 'RSI', period: 14, op: 'gt', level: 70 }, 'OVERBOUGHT'],
      [{ ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'lt', level: 20 }, 'OVERSOLD'],
      [{ ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'gt', level: 80 }, 'OVERBOUGHT'],
      [{ ind: 'CCI', period: 20, op: 'lt', level: -100 }, 'OVERSOLD'],
      [{ ind: 'CCI', period: 20, op: 'gt', level: 100 }, 'OVERBOUGHT'],
      [{ ind: 'BB', period: 20, mult: 2, variant: 'pctb', op: 'lt', level: 0.1 }, 'OVERSOLD'],
      [{ ind: 'BB', period: 20, mult: 2, variant: 'pctb', op: 'gt', level: 0.9 }, 'OVERBOUGHT'],
      [{ ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'lt', level: 0 }, 'OVERSOLD'],
      [{ ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'gt', level: 0 }, 'OVERBOUGHT'],
      // Trend: the operator decides.
      [{ ind: 'PRICE_MA', maType: 'SMA', period: 200, op: 'gt' }, 'TREND_UP'],
      [{ ind: 'PRICE_MA', maType: 'EMA', period: 20, op: 'cross_below' }, 'TREND_DOWN'],
      [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'SMA', slowPeriod: 50, op: 'cross_above' }, 'TREND_UP'],
      [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'SMA', slowPeriod: 50, op: 'lt' }, 'TREND_DOWN'],
      [{ ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'signal', op: 'cross_above' }, 'TREND_UP'],
      [{ ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'zero', op: 'lt' }, 'TREND_DOWN'],
      [{ ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'hist', op: 'gt' }, 'TREND_UP'],
      [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }, 'TREND_UP'],
      [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'cross_below' }, 'TREND_DOWN'],
      [{ ind: 'ADX', period: 14, variant: 'di', op: 'gt', level: 25 }, 'TREND_UP'],
      [{ ind: 'ADX', period: 14, variant: 'di', op: 'cross_below', level: 25 }, 'TREND_DOWN'],
      [{ ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'kd', op: 'cross_above', level: 50 }, 'TREND_UP'],
      [{ ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'kd', op: 'lt', level: 50 }, 'TREND_DOWN'],
    ];
    const mirrored: Record<ConditionRole, ConditionRole> = {
      FILTER: 'FILTER',
      OVERSOLD: 'OVERBOUGHT',
      OVERBOUGHT: 'OVERSOLD',
      TREND_UP: 'TREND_DOWN',
      TREND_DOWN: 'TREND_UP',
    };
    const variants = new Set<string>();
    for (const [condition, role] of table) {
      expect(conditionRole(condition), JSON.stringify(condition)).toBe(role);
      expect(conditionRole(mirrorCondition(condition)), JSON.stringify(condition)).toBe(mirrored[role]);
      expect(mirrorCondition(mirrorCondition(condition))).toEqual(condition);
      variants.add(`${condition.ind}:${'variant' in condition ? condition.variant : ''}`);
    }
    // Every indicator family and every variant is covered.
    expect([...variants].sort()).toEqual(
      [
        'ADX:di', 'ADX:strength', 'ATR:', 'BB:bandwidth', 'BB:lower', 'BB:pctb', 'BB:upper', 'CCI:',
        'MACD:hist', 'MACD:signal', 'MACD:zero', 'MA_CROSS:', 'PRICE_MA:', 'RSI:', 'STOCH:kd', 'STOCH:level',
        'SUPERTREND:', 'VOLUME:',
      ].sort(),
    );
  });

  it('generates, mutates and recombines only genomes built from the run\'s hypothesis', () => {
    const rng = mulberry32(42);
    const families = new Set<string>();
    for (const mode of ['long-only', 'long-short', 'asymmetric'] as const) {
      for (const h of HYPOTHESIS_IDS) {
        let genome = randomGenome(rng, mode, DEFAULT_LIMITS, h);
        let reversals = 0;
        let timeStops = 0;
        let filters = 0;
        let full = 0;
        const triggers = new Set<string>();
        for (let i = 0; i < 1000; i++) {
          const r = rng();
          if (r < 0.2) genome = randomGenome(rng, mode, DEFAULT_LIMITS, h);
          else if (r < 0.3) genome = crossoverGenomes(rng, genome, randomGenome(rng, mode, DEFAULT_LIMITS, h), mode, DEFAULT_LIMITS);
          else genome = mutateGenome(rng, genome, mode, DEFAULT_LIMITS, 1 + Math.floor(rng() * 3));
          assertValid(genome, mode);
          // Nothing leaves the hypothesis: its triggers, and filters that never change direction.
          expect(genome.hypothesis).toBe(h);
          expect(triggerConforms(genome.longEntry.conditions[0], h)).toBe(true);
          triggers.add(genome.longEntry.conditions[0].ind);
          const rest = genome.longEntry.conditions.slice(1);
          if (rest.length > 0) filters++;
          else full++; // single-indicator strategies
          for (const filter of rest) {
            expect(filterConforms(filter, h, DEFAULT_LIMITS)).toBe(true);
            // Orthogonality: the filter never comes from the trigger's indicator family.
            expect(indicatorClass(filter)).not.toBe(indicatorClass(genome.longEntry.conditions[0]));
          }
          if (HYPOTHESES[h].filterRequired) expect(rest.length).toBeGreaterThan(0);
          for (const c of genome.longEntry.conditions) families.add(c.ind);
          if (genome.exit.mode === 'dynamic' && genome.exit.reversalExit) reversals++;
          if (genome.maxBarsInTrade !== null) timeStops++;
          // Repair and normalisation are idempotent; signatures are stable.
          expect(genomeSignature(repairGenome(genome, mode, DEFAULT_LIMITS, rng))).toBe(genomeSignature(genome));
          expect(normalizeGenome(genome, mode, DEFAULT_LIMITS)).toEqual(genome);
        }
        // Every hypothesis may close on the opposite signal.
        expect(reversals).toBeGreaterThan(20);
        expect(timeStops).toBeGreaterThan(20);
        // Parsimony: most strategies are a single indicator, a large minority trigger + one filter
        // (dip-buying always carries its required trend filter).
        if (HYPOTHESES[h].filterRequired) expect(full).toBe(0);
        else {
          expect(full / 1000).toBeGreaterThan(0.5);
          expect(filters / 1000).toBeGreaterThan(0.2);
        }
        // Every trigger primitive of the hypothesis is used.
        expect(triggers.size).toBeGreaterThanOrEqual(2);
      }
    }
    // The building blocks: event triggers, slow regime states and confirmation states – no oscillator soup.
    expect([...families].sort()).toEqual([
      'ADX', 'BB', 'DONCHIAN', 'KELTNER', 'MACD', 'MA_CROSS', 'MA_SLOPE', 'PRICE_MA', 'PULLBACK', 'RANGE_EXP', 'RSI', 'SQUEEZE', 'VOLUME',
    ]);
  });

  it('starts about 60% of random strategies as a single indicator, the rest with one orthogonal filter', () => {
    const rng = mulberry32(8);
    for (const h of HYPOTHESIS_IDS) {
      let single = 0;
      for (let i = 0; i < 2000; i++) {
        const g = randomGenome(rng, 'long-only', DEFAULT_LIMITS, h);
        expect(g.longEntry.conditions.length).toBeLessThanOrEqual(MAX_CONDITIONS);
        expect(countParameters(g)).toBeLessThanOrEqual(MAX_PARAMETERS);
        if (g.longEntry.conditions.length === 1) single++;
      }
      if (HYPOTHESES[h].filterRequired) expect(single).toBe(0);
      else expect(Math.abs(single / 2000 - (1 - FILTER_RATE))).toBeLessThan(0.08);
    }
    expect(MAX_CONDITIONS).toBe(2);
    expect(MAX_PARAMETERS).toBe(5);
    // Two moving-average rules or two oscillators never pair; different families do.
    const cross: Condition = { ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 20, slowType: 'EMA', slowPeriod: 56, op: 'cross_above' };
    expect(orthogonal(cross, { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' })).toBe(false);
    expect(orthogonal(cross, { ind: 'MA_SLOPE', maType: 'EMA', period: 112, op: 'gt' })).toBe(false);
    expect(orthogonal({ ind: 'RSI', period: 14, op: 'cross_above', level: 30 }, { ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'lt', level: 20 })).toBe(false);
    expect(orthogonal(cross, { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 })).toBe(true);
    const ema200: Condition = { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' };
    const stacked = repairGenome(
      { archetype: 'TREND', hypothesis: 'TREND_FOLLOWING', longEntry: { logic: 'AND', conditions: [cross, ema200] }, shortEntry: null, exit: { mode: 'dynamic', atrPeriod: 20, stopAtrMult: 2, trailing: false, takeProfitAtrMult: 4, reversalExit: false, longExit: null, shortExit: null }, maxBarsInTrade: null },
      'long-only',
      DEFAULT_LIMITS,
      rng,
    );
    expect(stacked.longEntry.conditions).toHaveLength(1); // the redundant EMA filter is dropped
    expect(stacked.exit).toMatchObject({ atrPeriod: FIXED_ATR_PERIOD }); // ATR period is not tuned
  });

  it('scales static stops and targets to the candle timeframe', () => {
    expect(staticTier(15 * 60_000)).toBe('intraday');
    expect(staticTier(3_600_000)).toBe('intraday');
    expect(staticTier(4 * 3_600_000)).toBe('swing');
    expect(staticTier(86_400_000)).toBe('daily');
    const intraday = STATIC_GRIDS.intraday;
    expect(Math.min(...intraday.stopLossPct)).toBe(0.5);
    expect(Math.max(...intraday.stopLossPct)).toBe(3.5);
    expect(Math.min(...intraday.takeProfitPct)).toBe(1);
    expect(Math.max(...intraday.takeProfitPct)).toBe(10);
    // A 15m strategy never draws a macro 10%+ stop; random strategies mostly use ATR exits.
    const rng = mulberry32(12);
    const limits = { ...DEFAULT_LIMITS, barMs: 15 * 60_000 };
    let atrExits = 0;
    for (let i = 0; i < 1000; i++) {
      const g = randomGenome(rng, 'long-only', limits, HYPOTHESIS_IDS[i % 4]);
      if (g.exit.mode === 'static') {
        expect(intraday.stopLossPct).toContain(g.exit.stopLossPct);
        expect(intraday.takeProfitPct).toContain(g.exit.takeProfitPct);
      } else atrExits++;
    }
    expect(atrExits / 1000).toBeGreaterThan(0.6);
    // A daily stop snaps onto the intraday grid when the strategy moves to a 15m chart.
    const daily = normalizeGenome({ ...TREND, exit: { mode: 'static', stopLossPct: 10, takeProfitPct: 20 } }, 'long-only', limits);
    expect(daily.exit).toMatchObject({ stopLossPct: 3.5, takeProfitPct: 10 });
  });

  it('never produces a mutation identical to its parent', () => {
    const rng = mulberry32(99);
    let total = 0;
    for (const mode of ['long-only', 'long-short'] as const) {
      for (const h of HYPOTHESIS_IDS) {
        for (let i = 0; i < 1500; i++) {
          const parent = randomGenome(rng, mode, DEFAULT_LIMITS, h);
          const child = mutateGenome(rng, parent, mode, DEFAULT_LIMITS, 1);
          expect(genomeSignature(child)).not.toBe(genomeSignature(parent));
          expect(conformsTo(child)).toBe(true);
          total++;
        }
      }
    }
    expect(total).toBe(12_000);
    // Hand-built genomes have no hypothesis to stay inside, so they cannot be mutated.
    expect(() => mutateGenome(rng, TREND, 'long-only', DEFAULT_LIMITS)).toThrow(/hypothesis/);
  });

  it('describes every hypothesis and rejects the ones the data cannot support', () => {
    for (const h of HYPOTHESIS_IDS) {
      const info = HYPOTHESES[h];
      expect(info.id).toBe(h);
      expect(info.rationale.length).toBeGreaterThan(40);
      expect(info.replaces.length).toBeGreaterThan(10);
      expect(hypothesisUnsupported(h, DEFAULT_LIMITS)).toBeNull();
    }
    // A slow trend filter (a 56+ candle EMA slope) needs history.
    expect(hypothesisUnsupported('DIP_IN_UPTREND', { maxLookback: 60, hasVolume: true })).toBeNull();
    expect(hypothesisUnsupported('DIP_IN_UPTREND', { maxLookback: 40, hasVolume: true })).toMatch(/280 candles/);
    expect(hypothesisUnsupported('TREND_FOLLOWING', { maxLookback: 40, hasVolume: true })).toBeNull();
    // The lookback grid is geometric: roughly √2 per step and no near-duplicates (the old grid had 20 and 21).
    for (let k = 1; k < LOOKBACK_GRID.length; k++) {
      const ratio = LOOKBACK_GRID[k] / LOOKBACK_GRID[k - 1];
      expect(ratio).toBeGreaterThan(1.2);
      expect(ratio).toBeLessThan(1.5);
    }
  });

  it('repairs incoherent and over-parameterised genomes', () => {
    const rng = mulberry32(5);
    // The reviewed strategy's pattern: a "trend" long that fires on bearish signals, four conditions, OR logic.
    const mess: StrategyGenome = {
      archetype: 'TREND',
      longEntry: {
        logic: 'OR',
        conditions: [
          { ind: 'SUPERTREND', period: 19, mult: 3.6, op: 'cross_below' },
          { ind: 'RSI', period: 9, op: 'cross_below', level: 12 },
          { ind: 'MA_CROSS', fastType: 'SMA', fastPeriod: 162, slowType: 'SMA', slowPeriod: 228, op: 'cross_above' },
          { ind: 'CCI', period: 33, op: 'gt', level: 87 },
        ],
      },
      shortEntry: { logic: 'AND', conditions: [{ ind: 'RSI', period: 9, op: 'gt', level: 71 }] },
      exit: dynamicExit(2.4, 1.6, true),
      maxBarsInTrade: 13,
    };
    const fixed = repairGenome(mess, 'long-short', DEFAULT_LIMITS, rng);
    assertValid(fixed, 'long-short');
    // Indicator soup is migrated onto a hypothesis: a trend genome becomes trend following with one
    // of its event triggers (the bearish Supertrend cross is not one, so it is redrawn).
    expect(fixed.hypothesis).toBe('TREND_FOLLOWING');
    expect(triggerConforms(fixed.longEntry.conditions[0], 'TREND_FOLLOWING')).toBe(true);
    expect(fixed.longEntry.conditions.length).toBeLessThanOrEqual(2);
    // 2.4 / 1.6 ATR snaps to 2.5 / 1.5: the target is not 1 ATR beyond the stop, so the trail is dropped.
    expect(fixed.exit).toMatchObject({ stopAtrMult: 2.5, takeProfitAtrMult: 1.5, trailing: false });
    // At most 5 tunable numbers: the time stop is shed before the entry loses anything.
    expect(countParameters(fixed)).toBeLessThanOrEqual(MAX_PARAMETERS);

    // A Supertrend fade inside a slow uptrend is the dip-buy idea: its trigger becomes a pullback rejection,
    // and its slow-uptrend filter is kept (on the EMA primitive, snapped to the geometric grid).
    const fade = repairGenome(
      {
        archetype: 'MEAN_REVERSION',
        longEntry: {
          logic: 'AND',
          conditions: [
            { ind: 'SUPERTREND', period: 10, mult: 3, op: 'cross_below' },
            { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' },
          ],
        },
        shortEntry: null,
        exit: dynamicExit(2.5, 3, false),
        maxBarsInTrade: 20,
      },
      'long-short',
      DEFAULT_LIMITS,
      rng,
    );
    assertValid(fade, 'long-short');
    expect(fade.hypothesis).toBe('DIP_IN_UPTREND');
    expect(triggerConforms(fade.longEntry.conditions[0], 'DIP_IN_UPTREND')).toBe(true);
    expect(groupFormula({ logic: 'AND', conditions: [fade.longEntry.conditions[1]] })).toBe('Close > EMA(200)');

    // A genome already built from a hypothesis's primitives survives repair intact.
    const dip: StrategyGenome = {
      archetype: 'MEAN_REVERSION',
      hypothesis: 'DIP_IN_UPTREND',
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'RSI', period: 14, op: 'cross_above', level: 30 },
          { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' },
        ],
      },
      shortEntry: null,
      exit: dynamicExit(2.5, 3, false),
      maxBarsInTrade: 20,
    };
    const kept = repairGenome(dip, 'long-short', DEFAULT_LIMITS, rng);
    expect(groupFormula(kept.longEntry)).toBe('RSI(14) crosses above 30 AND Close > EMA(200)');
    expect(groupFormula(kept.shortEntry!)).toBe('RSI(14) crosses below 70 AND Close < EMA(200)');
    // A state trigger ("RSI below 30" on every candle it holds) is not an event: repair replaces it.
    const state = repairGenome({ ...dip, longEntry: { logic: 'AND', conditions: [{ ind: 'RSI', period: 14, op: 'lt', level: 30 }, dip.longEntry.conditions[1]] } }, 'long-only', DEFAULT_LIMITS, rng);
    expect(['cross_above']).toContain(state.longEntry.conditions[0].op);
    expect(countParameters(kept)).toBe(5); // RSI 2 + EMA 1 + ATR stop and target; the time stop is shed at the 5-number cap

    // Parameter cap on a hand-built genome: MACD (3) + Bollinger bandwidth (3) + ADX (2) + ATR exit (3) + time stop (1) = 12 → ≤ 10.
    const heavy = normalizeGenome(
      {
        archetype: 'TREND',
        longEntry: {
          logic: 'AND',
          conditions: [
            { ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'signal', op: 'gt' },
            { ind: 'BB', period: 20, mult: 2, variant: 'bandwidth', op: 'gt', level: 10 },
            { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 20 },
          ],
        },
        shortEntry: null,
        exit: dynamicExit(2, 4, false),
        maxBarsInTrade: 20,
      },
      'long-only',
      DEFAULT_LIMITS,
    );
    expect(countParameters(heavy)).toBeLessThanOrEqual(MAX_PARAMETERS);
    expect(heavy.maxBarsInTrade).toBeNull();
    // Repairing it puts it on a hypothesis, which can never exceed the cap.
    expect(countParameters(repairGenome(heavy, 'long-only', DEFAULT_LIMITS, rng))).toBeLessThanOrEqual(MAX_PARAMETERS);
  });

  it('respects dataset limits (short histories, missing volume)', () => {
    const rng = mulberry32(7);
    const limits = { maxLookback: 50, hasVolume: false };
    for (const h of ['TREND_FOLLOWING', 'VOL_BREAKOUT', 'RANGE_REVERSION'] as const) {
      let genome = randomGenome(rng, 'long-short', limits, h);
      for (let i = 0; i < 700; i++) {
        genome = rng() < 0.3 ? randomGenome(rng, 'long-short', limits, h) : mutateGenome(rng, genome, 'long-short', limits, 2);
        expect(isCoherent(genome, limits)).toBe(true);
        for (const c of conditionsOf(genome)) {
          expect(c.ind).not.toBe('VOLUME'); // no usable volume: the breakout runs without its confirmation filter
          for (const lookback of numericParams(c)) expect(lookback).toBeLessThanOrEqual(50);
        }
      }
    }
    // No 100-bar trend filter fits, so the dip-buy hypothesis is refused rather than run without its filter.
    expect(hypothesisUnsupported('DIP_IN_UPTREND', limits)).not.toBeNull();
  });

  it('enumerates one-step single-parameter neighbours on the grids', () => {
    const base = normalizeGenome(DIP, 'long-only', DEFAULT_LIMITS);
    const variants = stepNeighbours(DIP, 'long-only', DEFAULT_LIMITS);
    // RSI period, RSI level, stop %, target % → two neighbours each; the 100-bar regime filter can only move up.
    expect(variants.length).toBeGreaterThanOrEqual(8);
    const signatures = new Set(variants.map(genomeSignature));
    expect(signatures.size).toBe(variants.length);
    expect(signatures.has(genomeSignature(base))).toBe(false);
    for (const variant of variants) {
      assertValid(variant, 'long-only');
      expect(variant.archetype).toBe(DIP.archetype);
      // Exactly one number differs from the original.
      const a = JSON.stringify(variant).match(/-?\d+(\.\d+)?/g) ?? [];
      const b = JSON.stringify(base).match(/-?\d+(\.\d+)?/g) ?? [];
      expect(a.length).toBe(b.length);
      expect(a.filter((value, index) => value !== b[index]).length).toBe(1);
    }
    // RSI(14) on the lookback grid: one step down is 10, one step up is 20.
    const periods = variants.map((v) => (v.longEntry.conditions[0] as { period: number }).period);
    expect(periods).toContain(10);
    expect(periods).toContain(20);
  });
});

/* -------------------------------------------------------------------------- */

describe('descriptions and Pine export', () => {
  const data = dataset();
  const ppy = barsPerYear(data.t);
  const split = splitFor(data.c.length);
  const RISK: BacktestSettings = { ...SETTINGS, sizing: { mode: 'risk', riskPct: 2, compounding: false } };

  function resultFor(genome: StrategyGenome, settings: BacktestSettings): StrategyResult {
    const m = evaluateSearch(new IndicatorCache(data), genome, settings, new BacktestWorkspace(data.c.length), split, ppy);
    return {
      id: 'S-1',
      genome,
      fitness: 1,
      dsr: 0.42,
      params: countParameters(genome),
      ...m,
      holdout: null,
      alpha: 0,
      equity: new Float32Array(0),
      foundAt: 1,
      assets: [],
      robustness: null,
      verdict: null,
    };
  }

  const ctx = (settings: BacktestSettings) => ({
    label: 'XRP-USD',
    assets: ['XRP-USD'],
    interval: '1d' as const,
    settings,
    components: [{ symbol: 'XRP-USD', weight: 1 }],
    units: [],
    windowStart: data.t[0],
    windowEnd: data.t[data.t.length - 1],
    bars: data.t.length,
    benchmarkRoi: 12.3,
    venue: 'XRPUSDT on Binance',
    simulatedData: true,
  });

  it('renders formulas and coherent plain-English rules', () => {
    expect(groupFormula(DIP.longEntry)).toBe('RSI(14) < 45 AND Close > EMA(112)');
    expect(indicatorBadges(DIP)).toEqual(['RSI', 'EMA']);
    expect(indicatorBadges(TREND)).toEqual(['Supertrend', 'ATR']);
    const entry = describeLongEntry(DIP);
    expect(entry.intro).toContain('BOTH of the following');
    expect(entry.bullets[0].text.startsWith('Entry trigger: ')).toBe(true);
    expect(entry.bullets[1].text.startsWith('Regime filter: ')).toBe(true);
    expect(ARCHETYPE_LABEL[DIP.archetype]).toBe('Mean reversion');
    expect(describeArchetype(DIP)).toContain('buys weakness');
    expect(describeArchetype(DIP)).toContain('slow trend is still up');
    expect(describeArchetype(TREND)).toContain('buys strength');
    expect(describeArchetype(TREND)).toContain('mirror image');

    const exit = describeExit(DIP, RISK);
    expect(exit.mode).toBe('Static %');
    expect(exit.bullets.map((b) => b.title)).toEqual(['Stop loss', 'Take profit', 'Position size (run setting)', 'Execution']);
    expect(exit.bullets[2].text).toContain('risks 2.00% of the starting capital (no compounding)');
    const dynamic = describeExit({ ...TREND, maxBarsInTrade: 20 }, { ...SETTINGS, fundingPctPerBar: 0.03 });
    expect(dynamic.mode).toBe('Dynamic ATR');
    expect(dynamic.bullets.map((b) => b.title)).toEqual([
      'Trailing ATR stop', 'Take profit', 'Time stop', 'Stop-and-reverse', 'Position size (run setting)', 'Funding', 'Execution',
    ]);
    expect(listIndicators(TREND).map((u) => u.name)).toEqual(['Supertrend', 'ATR']);
  });

  it('exports syntactically well-formed Pine Script v5 for every kind of genome', () => {
    const rng = mulberry32(99);
    for (const mode of ['long-only', 'long-short', 'asymmetric'] as const) {
      for (const sizing of [RISK.sizing, SETTINGS.sizing, { mode: 'risk', riskPct: 1, compounding: true }] as const) {
        const settings: BacktestSettings = { ...SETTINGS, tradeMode: mode, sizing };
        for (let i = 0; i < 150; i++) {
          const genome = randomGenome(rng, mode, DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
          const script = generatePineScript(resultFor(genome, settings), ctx(settings));
          expect(script.startsWith('//@version=5\n')).toBe(true);
          expect(script).toContain('strategy("StrategyFinder S-1 | XRP-USD 1d"');
          expect(script).toContain('use_bar_magnifier=false');
          expect(script).not.toContain('use_bar_magnifier=true');
          expect(script).toContain('commission_value=0.15');
          expect(script).toContain('Tested on    : XRPUSDT on Binance');
          expect(script).toContain(`Archetype    : ${ARCHETYPE_LABEL[genome.archetype]}`);
          expect(script).toContain(`Parameters   : ${countParameters(genome)} tunable values`);
          expect(script).toContain('strategy.entry("Long", strategy.long, qty=orderQty)');
          expect(script.includes('strategy.entry("Short"')).toBe(mode !== 'long-only');
          expect(script).toContain(
            mode === 'long-only' ? 'Mode         : Long-Only' : mode === 'asymmetric' ? 'Mode         : Asymmetric Long/Short' : 'Mode         : Mirrored Long/Short',
          );
          // Managed exits: the stop is re-sent after every close; breakeven and trailing only when the genome has them.
          expect(script).toContain('strategy.exit("Long exit", from_entry="Long", stop=longStop');
          expect(script.includes('// +1R reached: breakeven lock')).toBe(genome.exit.breakeven === true);
          expect(script.includes('longStop := math.max(longStop, longPeak - stopAtrMult * atrValue)')).toBe(genome.exit.mode === 'dynamic' && genome.exit.trailing);
          // Never more than the account can buy.
          expect(script).toContain('maxQty = math.max(strategy.equity, 0) / (close * (1 + costPct / 100))');
          expect(script).toMatch(sizing.mode === 'full' ? /orderQty = maxQty/ : /math\.min\(riskCapital \* riskPct \/ 100 \/ \(stopDistance \* \(1 \+ feePct \/ 100\)\), maxQty\)/);
          expect(script.includes('timedOut')).toBe(genome.maxBarsInTrade !== null);
          expect(script).not.toMatch(/(Entry|Signal) = .* or /); // rules are AND-only now
          // ASCII only, balanced brackets, no leaked JS values.
          expect(/^[\x09\x0a\x20-\x7e]*$/.test(script)).toBe(true);
          expect(script).not.toMatch(/undefined|NaN|\[object/);
          const code = script
            .split('\n')
            .filter((line) => !line.trimStart().startsWith('//'))
            .map((line) => line.replace(/\/\/.*$/, '').replace(/"[^"]*"/g, '""'))
            .join('\n');
          for (const [open, close] of [['(', ')'], ['[', ']']] as const) {
            expect(code.split(open).length).toBe(code.split(close).length);
          }
          // Every identifier is declared before it is used.
          const declared = new Set<string>();
          for (const line of code.split('\n')) {
            const tuple = /^\[([^\]]+)\]\s*=/.exec(line);
            if (tuple) tuple[1].split(',').forEach((name) => declared.add(name.trim()));
            const single = /^(?:var float )?([A-Za-z_][A-Za-z0-9_]*)\s*=[^=]/.exec(line);
            if (single) declared.add(single[1]);
          }
          const used =
            code.match(
              /\b(?:rsi|ema|sma|macdLine|macdSignal|macdHist|bbUpper|bbLower|bbBasis|bbPctB|bbWidth|atr|atrPct|stochK|stochD|diPlus|diMinus|adx|volRatio|supertrend|cci|kcUpper|kcLower|squeezeOn|squeeze|rangeExp|pullback_ema|pullback_sma|donchianHigh|donchianLow)_[0-9_]+\b/g,
            ) ?? [];
          for (const name of used) expect(declared.has(name)).toBe(true);
          for (const name of ['orderQty', 'maxQty', 'goLong', 'canEnter', 'stopTicks']) expect(declared.has(name)).toBe(true);
        }
      }
    }
  });

  it('writes the exact exit construction, sizing and time stop into the script', () => {
    const longOnly = { ...RISK, tradeMode: 'long-only' as const };
    const staticScript = generatePineScript(resultFor(DIP, longOnly), ctx(longOnly));
    expect(staticScript).toContain('stopLossPct   = input.float(4.0, "Stop loss %"');
    expect(staticScript).toContain('takeProfitPct = input.float(10.0, "Take profit %"');
    expect(staticScript).toContain('longCond1 = rsi_14 < 45');
    expect(staticScript).toContain('longCond2 = close > ema_112');
    expect(staticScript).toContain('riskPct       = input.float(2.0, "Risk per trade (%)"');
    expect(staticScript).toContain('compound      = input.bool(false,');
    expect(staticScript).toContain('riskCapital  = compound ? math.max(strategy.equity, 0) : strategy.initial_capital');
    expect(staticScript).toContain('stopDistance = close * stopLossPct / 100');
    expect(staticScript).toContain('if goLong and strategy.position_size == 0');
    expect(staticScript).not.toContain('goShort');
    expect(staticScript).toContain('Holdout      : not revealed yet');

    const timed = { ...TREND, maxBarsInTrade: 20 };
    const trailScript = generatePineScript({ ...resultFor(timed, RISK), holdout: resultFor(timed, RISK).validation }, ctx(RISK));
    expect(trailScript).toContain('[supertrend_10_3, supertrendDir_10_3] = ta.supertrend(3.0, 10)');
    expect(trailScript).toContain('stopDistance = stopAtrMult * atrValue');
    expect(trailScript).toContain('longStop := math.max(longStop, longPeak - stopAtrMult * atrValue)');
    expect(trailScript).toContain('shortStop := math.min(shortStop, shortTrough + stopAtrMult * atrValue)');
    expect(trailScript).toContain('        longRisk   := stopAtrMult * atrValue[1]');
    expect(trailScript).toContain('        longStop   := strategy.position_avg_price - longRisk');
    // Static brackets are pinned to the fill price after the fill candle, a breakeven lock moves the stop to the entry.
    expect(staticScript).toContain('        longRisk   := strategy.position_avg_price * stopLossPct / 100');
    expect(staticScript).toContain('        longTarget := strategy.position_avg_price * (1 + takeProfitPct / 100)');
    const locked = generatePineScript(resultFor({ ...DIP, exit: { ...DIP.exit, breakeven: true } } as StrategyGenome, longOnly), ctx(longOnly));
    expect(locked).toContain('    if longPeak >= strategy.position_avg_price + longRisk // +1R reached: breakeven lock\n        longStop := math.max(longStop, strategy.position_avg_price)');
    expect(trailScript).toContain('maxBarsInTrade = input.int(20, "Time stop (bars in trade)"');
    expect(trailScript).toContain('timedOut = strategy.position_size != 0 and bar_index - strategy.opentrades.entry_bar_index(0) >= maxBarsInTrade - 1');
    // The time stop never cancels a same-open reversal: the reversal entry closes the position itself.
    expect(trailScript).toContain(
      'if timedOut and not (strategy.position_size > 0 and goShort) and not (strategy.position_size < 0 and goLong)\n    strategy.close_all(comment="Time stop")',
    );
    // Risk sizing budgets the entry fee inside the size, as the backtester does.
    expect(trailScript).toContain('orderQty = na(stopDistance) or stopDistance <= 0 ? 0.0 : math.min(riskCapital * riskPct / 100 / (stopDistance * (1 + feePct / 100)), maxQty)');
    expect(trailScript).toContain('if useWindow and time >= windowEnd and strategy.position_size != 0');
    // A time stop blocks re-entering the same side only; the opposite side may still reverse.
    expect(trailScript).toContain('and not (timedOut and strategy.position_size > 0)');
    expect(trailScript).toContain('and not (timedOut and strategy.position_size < 0)');
    expect(trailScript).toContain('if goShort and strategy.position_size >= 0');
    expect(trailScript).not.toContain('not revealed yet');
  });
});

/* -------------------------------------------------------------------------- */

describe('fallback data and baskets', () => {
  it('generates deterministic, self-consistent candles', () => {
    const a = generateSyntheticCandles('BTC-USD', '1d', '5y', NOW);
    const b = generateSyntheticCandles('BTC-USD', '1d', '5y', NOW);
    expect(a).toEqual(b);
    expect(a.t.length).toBeGreaterThan(1800);
    for (let i = 0; i < a.t.length; i++) {
      expect(a.l[i]).toBeGreaterThan(0);
      expect(a.h[i]).toBeGreaterThanOrEqual(Math.max(a.o[i], a.c[i]));
      expect(a.l[i]).toBeLessThanOrEqual(Math.min(a.o[i], a.c[i]));
      expect(a.v[i]).toBeGreaterThan(0);
      if (i > 0) expect(a.t[i]).toBeGreaterThan(a.t[i - 1]);
    }
    expect(a.c[a.c.length - 1]).toBeCloseTo(96_000, -2);
  });

  it('uses a weekday session calendar for stocks and 24/7 for crypto', () => {
    const spy = generateSyntheticCandles('SPY', '1h', '1y', NOW);
    for (const t of spy.t) {
      const d = new Date(t);
      expect([1, 2, 3, 4, 5]).toContain(d.getUTCDay());
      expect(d.getUTCHours()).toBeGreaterThanOrEqual(13);
      expect(d.getUTCHours()).toBeLessThanOrEqual(19);
    }
    const btc = generateSyntheticCandles('BTC-USD', '1h', '1y', NOW);
    expect(btc.t.length).toBeGreaterThan(8700);
    expect(generateSyntheticCandles('SPY', '1d', '1y', NOW).t.length).toBeGreaterThan(250);
  });

  it('generates 15m and 30m candles: 24/7 for crypto, the US session over the 60-day Yahoo window for stocks', () => {
    const btc = generateSyntheticCandles('BTC-USD', '15m', '1y', NOW);
    expect(btc.t.length).toBeGreaterThan(35_000);
    expect(btc.t[1] - btc.t[0]).toBe(15 * 60_000);

    for (const [interval, perDay] of [['15m', 26], ['30m', 13]] as const) {
      const spy = generateSyntheticCandles('SPY', interval, '1y', NOW);
      const days = new Set<number>();
      for (const t of spy.t) {
        const d = new Date(t);
        expect([1, 2, 3, 4, 5]).toContain(d.getUTCDay());
        const minute = d.getUTCHours() * 60 + d.getUTCMinutes();
        expect(minute).toBeGreaterThanOrEqual(13 * 60 + 30);
        expect(minute).toBeLessThan(20 * 60);
        days.add(Math.floor(t / 86_400_000));
      }
      expect(spy.t.length).toBe(days.size * perDay);
      expect(NOW - spy.t[0]).toBeLessThanOrEqual(60 * 86_400_000);
      expect(days.size).toBeGreaterThan(35);
      for (let i = 0; i < spy.t.length; i++) expect(spy.h[i]).toBeGreaterThanOrEqual(Math.max(spy.o[i], spy.c[i]));
    }
  });

  it('limits how much history short timeframes may load', () => {
    expect(rangeAllowed('15m', '1y')).toBe(true);
    expect(rangeAllowed('15m', '2y')).toBe(false);
    expect(rangeAllowed('30m', '2y')).toBe(true);
    expect(rangeAllowed('30m', '5y')).toBe(false);
    for (const interval of ['1h', '4h', '1d'] as const) expect(rangeAllowed(interval, '5y')).toBe(true);
    expect(clampRange('15m', '5y')).toBe('1y');
    expect(clampRange('30m', '5y')).toBe('2y');
    expect(clampRange('30m', '1y')).toBe('1y');
    expect(clampRange('1d', '5y')).toBe('5y');
  });

  it('normalises basket weights to 100%', () => {
    const weights = normalizeWeights([
      { symbol: 'btc-usd', weight: 5 },
      { symbol: 'ETH-USD', weight: 3 },
      { symbol: 'NVDA', weight: 2 },
      { symbol: '', weight: 10 },
      { symbol: 'AAPL', weight: 0 },
      { symbol: 'BTC-USD', weight: 5 },
    ]);
    expect(weights.map((w) => w.symbol)).toEqual(['BTC-USD', 'ETH-USD', 'NVDA']);
    expect(weights.reduce((sum, w) => sum + w.weight, 0)).toBeCloseTo(1, 12);
    expect(weights[0].weight).toBeCloseTo(10 / 15, 12);
  });

  it('builds a weighted basket whose Buy & Hold return is the weighted component return', () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'NVDA'];
    const series = Object.fromEntries(symbols.map((s) => [s, generateSyntheticCandles(s, '1d', '2y', NOW)]));
    const components = normalizeWeights([
      { symbol: 'BTC-USD', weight: 50 },
      { symbol: 'ETH-USD', weight: 30 },
      { symbol: 'NVDA', weight: 20 },
    ]);
    const basket = buildDataset(series, components);
    expect(basket.isBasket).toBe(true);
    expect(basket.label).toBe('50% BTC-USD · 30% ETH-USD · 20% NVDA');
    // Only timestamps shared by every component survive (stock trading days).
    expect(basket.data.t.length).toBe(series.NVDA.t.length);
    expect(basket.data.o[0]).toBeCloseTo(100, 9);

    const n = basket.data.t.length;
    let expected = 0;
    for (const component of components) {
      const s = series[component.symbol];
      const first = s.t.indexOf(basket.data.t[0]);
      const last = s.t.indexOf(basket.data.t[n - 1]);
      expected += component.weight * (s.c[last] / s.o[first]);
    }
    expect(basket.data.c[n - 1] / basket.data.o[0]).toBeCloseTo(expected, 9);
    for (let i = 0; i < n; i++) {
      expect(basket.data.h[i]).toBeGreaterThanOrEqual(Math.max(basket.data.o[i], basket.data.c[i]) - 1e-9);
      expect(basket.data.l[i]).toBeLessThanOrEqual(Math.min(basket.data.o[i], basket.data.c[i]) + 1e-9);
    }
  });
});

/* -------------------------------------------------------------------------- */

describe('stage 1: execution friction and basket hygiene', () => {
  /** Shorts every bar (the long rule can never fire), so funding / borrow can be checked on shorts alone. */
  const ALWAYS_SHORT = (maxBarsInTrade: number | null): StrategyGenome => ({
    archetype: 'TREND',
    longEntry: { logic: 'AND', conditions: [{ ind: 'RSI', period: 14, op: 'lt', level: -1 }] },
    shortEntry: { logic: 'AND', conditions: [{ ind: 'VOLUME', period: 10, op: 'gt', level: 0.5 }] },
    exit: { mode: 'static', stopLossPct: 15, takeProfitPct: 30 },
    maxBarsInTrade,
  });

  it('scales slippage with the ATR of the last closed candle, never below the fixed floor', () => {
    // 40 calm candles with a ±2 range, a breakout close on bar 30 and the entry at bar 31's open.
    const n = 40;
    const closes = Array.from({ length: n }, (_, i) => (i < 30 ? 100 : 110 + i - 30));
    const opens = closes.map((_, i) => (i === 31 ? 111 : i > 0 ? closes[i - 1] : 100));
    const data = candles(
      opens,
      closes.map((c, i) => Math.max(c, opens[i]) + 2),
      closes.map((c, i) => Math.min(c, opens[i]) - 2),
      closes,
    );
    const genome: StrategyGenome = {
      archetype: 'TREND',
      longEntry: { logic: 'AND', conditions: [{ ind: 'PRICE_MA', maType: 'SMA', period: 5, op: 'cross_above' }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 15, takeProfitPct: 30 },
      maxBarsInTrade: null,
    };
    const ind = new IndicatorCache(data);
    const atrPct = ind.atrPercent(14);
    const k = 0.5;
    const ws = new BacktestWorkspace(n);
    runBacktest(ind, genome, { ...FREE, slippagePct: 0.05, slippageAtrMult: k }, ws);
    expect(ws.tradeEntryBar[0]).toBe(31);
    // Full sizing and no fee: equity on the entry bar = capital × close / entry price.
    const entryPx = (10_000 * closes[31]) / ws.equity[31];
    const expectedSlip = Math.max(0.0005, (k * atrPct[30]) / 100);
    expect(expectedSlip).toBeGreaterThan(0.0005); // the volatility term is the binding one here
    expect(entryPx).toBeCloseTo(opens[31] * (1 + expectedSlip), 6);

    // A tiny multiple falls back to the fixed floor; k = 0 is exactly the old fixed-slippage model.
    runBacktest(ind, genome, { ...FREE, slippagePct: 0.05, slippageAtrMult: 1e-6 }, ws);
    expect((10_000 * closes[31]) / ws.equity[31]).toBeCloseTo(opens[31] * 1.0005, 6);
    const real = dataset('ETH-USD');
    const a = new BacktestWorkspace(real.c.length);
    const b = new BacktestWorkspace(real.c.length);
    runBacktest(new IndicatorCache(real), TREND, SETTINGS, a);
    runBacktest(new IndicatorCache(real), TREND, { ...SETTINGS, slippageAtrMult: 0 }, b);
    expect(Array.from(b.equity)).toEqual(Array.from(a.equity));
    // More volatility-scaled slippage can only cost money.
    runBacktest(new IndicatorCache(real), TREND, { ...SETTINGS, slippageAtrMult: 0.05 }, b);
    expect(b.equity[real.c.length - 1]).toBeLessThan(a.equity[real.c.length - 1]);
  });

  it('charges the short borrow fee on shorts only', () => {
    const n = 30;
    const flat = new Array<number>(n).fill(100);
    const data = candles(flat, flat, flat, flat);
    const ws = new BacktestWorkspace(n);
    const shortSettings: BacktestSettings = { ...FREE, tradeMode: 'long-short', borrowPctPerBar: 0.1 };
    runBacktest(new IndicatorCache(data), ALWAYS_SHORT(5), shortSettings, ws);
    expect(ws.tradeDir[0]).toBe(-1);
    // 0.1% of notional for each of the 5 closes the short is open.
    expect(ws.tradePnl[0]).toBeCloseTo(-50, 6);
    runBacktest(new IndicatorCache(data), ALWAYS_LONG({ mode: 'static', stopLossPct: 15, takeProfitPct: 30 }, 5), { ...FREE, borrowPctPerBar: 0.1 }, ws);
    expect(ws.tradeDir[0]).toBe(1);
    expect(ws.tradePnl[0]).toBeCloseTo(0, 9);
  });

  it('builds basket highs and lows that stay inside the summed envelope and contain the body', () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
    const series = Object.fromEntries(symbols.map((s) => [s, generateSyntheticCandles(s, '1d', '1y', NOW)]));
    const components = normalizeWeights(symbols.map((symbol) => ({ symbol, weight: 1 })));
    const basket = buildDataset(series, components);
    const { o, h, l, c, t } = basket.data;
    let tighter = 0;
    for (let i = 0; i < t.length; i++) {
      let sumHigh = 0;
      let sumLow = 0;
      components.forEach((component, k) => {
        const row = series[component.symbol].t.indexOf(t[i]);
        sumHigh += basket.units[k] * series[component.symbol].h[row];
        sumLow += basket.units[k] * series[component.symbol].l[row];
      });
      expect(h[i]).toBeGreaterThanOrEqual(Math.max(o[i], c[i]) - 1e-9);
      expect(l[i]).toBeLessThanOrEqual(Math.min(o[i], c[i]) + 1e-9);
      expect(h[i]).toBeLessThanOrEqual(sumHigh + 1e-9);
      expect(l[i]).toBeGreaterThanOrEqual(sumLow - 1e-9);
      expect(l[i]).toBeGreaterThan(0);
      if (h[i] < sumHigh - 1e-9 || l[i] > sumLow + 1e-9) tighter++;
    }
    // The impossible "every leg peaks at once" envelope is no longer used.
    expect(tighter).toBeGreaterThan(t.length * 0.9);
  });

  it('normalises basket volume with trailing data only, so later volume cannot leak backwards', () => {
    const symbols = ['BTC-USD', 'ETH-USD'];
    const series = Object.fromEntries(symbols.map((s) => [s, generateSyntheticCandles(s, '1d', '1y', NOW)]));
    const components = normalizeWeights(symbols.map((symbol) => ({ symbol, weight: 1 })));
    const before = buildDataset(series, components).data;
    const n = before.t.length;
    const cut = Math.floor(n * 0.8); // the holdout boundary
    // Multiply every holdout volume by 1,000: nothing before the boundary may change.
    const spiked = { ...series, 'ETH-USD': { ...series['ETH-USD'], v: series['ETH-USD'].v.map((v, i) => (i >= cut ? v * 1000 : v)) } };
    const after = buildDataset(spiked, components).data;
    for (let i = 0; i < cut; i++) expect(after.v[i]).toBe(before.v[i]);
    expect(after.v[n - 1]).not.toBe(before.v[n - 1]);
    // Prices are untouched by volume.
    expect(Array.from(after.c)).toEqual(Array.from(before.c));
  });
});

/* -------------------------------------------------------------------------- */

describe('multiple testing and deduplication', () => {
  it('seeds and merges trial statistics exactly as if every trial had been pushed', () => {
    const rng = mulberry32(3);
    const values = Array.from({ length: 500 }, () => gaussian(rng) * 0.7 + 0.1);
    const all = new RunningStats();
    for (const v of values) all.push(v);
    const a = new RunningStats();
    const b = new RunningStats();
    values.slice(0, 180).forEach((v) => a.push(v));
    values.slice(180).forEach((v) => b.push(v));
    const merged = RunningStats.from(a);
    merged.merge(b);
    expect(merged.count).toBe(all.count);
    expect(merged.mean).toBeCloseTo(all.mean, 12);
    expect(merged.variance).toBeCloseTo(all.variance, 12);
    // Scaling (per-bar ↔ annualised) scales the mean linearly and the variance quadratically.
    const scaled = RunningStats.from(all.scaled(Math.sqrt(365)));
    expect(scaled.mean).toBeCloseTo(all.mean * Math.sqrt(365), 9);
    expect(scaled.variance).toBeCloseTo(all.variance * 365, 9);
    // Seeding with earlier trials raises the selection-bias hurdle.
    const fresh = expectedMaxSharpe(all.count, all.variance);
    const seeded = RunningStats.from({ count: 9000, mean: all.mean, m2: all.variance * 8999 });
    for (const v of values) seeded.push(v);
    expect(expectedMaxSharpe(seeded.count, seeded.variance)).toBeGreaterThan(fresh);
  });

  it('fingerprints trade logs: identical trades match, different trades do not', () => {
    const data = dataset('ETH-USD');
    const n = data.c.length;
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(n);
    runBacktest(ind, TREND, SETTINGS, ws);
    const a = tradeLogHash(ws, n);
    runBacktest(ind, TREND, SETTINGS, ws);
    expect(tradeLogHash(ws, n)).toBe(a);
    // Only trades opened before `bars` count.
    expect(tradeLogHash(ws, 0)).toMatch(/^0:/);
    runBacktest(ind, DIP, SETTINGS, ws);
    expect(tradeLogHash(ws, n)).not.toBe(a);
    // Higher fees change only the PnL – still a different trade log.
    runBacktest(ind, TREND, { ...SETTINGS, feePct: 0.3 }, ws);
    expect(tradeLogHash(ws, n)).not.toBe(a);
  });
});
