import { describe, expect, it } from 'vitest';
import { buildDataset, normalizeWeights } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  assetBaseScore,
  collapsedRegimes,
  emptyMetrics,
  FOLD_QUANTILE,
  foldBounds,
  FOLDS,
  isSingleRegime,
  plateauFitness,
  quantile,
  regimeMask,
  REGIME_COLLAPSE_PENALTY,
  REGIME_HIGH_VOL,
  testableRegimes,
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
  MIN_TRADES,
  MIN_TRADES_PER_YEAR,
  MIN_VALIDATION_TRADES,
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
  countParameters,
  crossoverGenomes,
  DEFAULT_LIMITS,
  genomeSignature,
  GRIDS,
  isCoherent,
  LOOKBACK_GRID,
  MAX_PARAMETERS,
  mirrorCondition,
  mirrorGroup,
  mulberry32,
  mutateGenome,
  normalizeGenome,
  parameterVariants,
  randomGenome,
  repairGenome,
  STATIC_PCT_GRID,
  TIME_STOP_GRID,
} from '../src/engine/genome';
import { atr, bollinger, ema, IndicatorCache, rsi, sma, stochastic, supertrend } from '../src/engine/indicators';
import { generatePineScript } from '../src/engine/pine';
import { deflatedSharpe, expectedMaxSharpe, inverseNormalCdf, normalCdf, RunningStats, sharpeConfidenceInterval } from '../src/lib/stats';
import {
  clampRange,
  HOLDOUT_FRACTION,
  rangeAllowed,
  TRAIN_FRACTION,
  VALIDATION_FRACTION,
  type BacktestSettings,
  type Condition,
  type ConditionRole,
  type OHLCV,
  type PerformanceMetrics,
  type RegimeMetrics,
  type StrategyGenome,
  type StrategyResult,
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
        { ind: 'PRICE_MA', maType: 'EMA', period: 100, op: 'gt' },
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
      const regimes = regimeMask(data.c, split, ppy);
      const fb = foldBounds(split);
      for (let i = 0; i < 400; i++) {
        const genome = randomGenome(rng, mode, DEFAULT_LIMITS);
        // Search-period run (what the optimiser does) and a full run (what the holdout reveal does).
        for (const bars of [split.holdoutStart, n]) {
          runBacktest(ind, genome, settings, ws, bars);
          const fused = searchMetrics(ws.equity, split, settings.initialCapital, ppy, ws, ws.equityLow, regimes);
          // Folds: the single pass must equal five separate segment passes.
          fb.slice(0, -1).forEach((from, f) => {
            const start = from === 0 ? settings.initialCapital : ws.equity[from - 1];
            const ref = segmentMetrics(ws.equity, from, fb[f + 1], start, ppy, ws);
            for (const key of keys) {
              if (!close(fused.folds[f][key], ref[key])) throw new Error(`${symbol} #${i} fold ${f}.${key}: fused ${fused.folds[f][key]} vs reference ${ref[key]}`);
            }
          });
          // Regimes: a copy of the curve forces the bar-by-bar reference implementation.
          const slow = searchMetrics(Float64Array.from(ws.equity), split, settings.initialCapital, ppy, ws, ws.equityLow, regimes);
          fused.regimes?.forEach((r, j) => {
            const ref = slow.regimes?.[j];
            for (const key of ['bars', 'coverage', 'trades', 'netPnl', 'sharpe'] as const) {
              if (!ref || !close(r[key], ref[key])) throw new Error(`${symbol} #${i} regime ${r.regime}.${key}: fused ${r[key]} vs reference ${ref?.[key]}`);
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

  it('labels bull, bear and high-volatility regimes causally, without looking at the holdout', () => {
    const data = dataset('BTC-USD', '1d', '5y');
    const n = data.c.length;
    const split = splitFor(n);
    const ppy = barsPerYear(data.t);
    const r = regimeMask(data.c, split, ppy);
    expect(r.searchBars).toBe(split.holdoutStart);
    // Bull and bear partition the labelled bars; every slice is populated on five years of data.
    for (let i = 0; i < split.holdoutStart; i++) expect((r.mask[i] & 3) !== 3).toBe(true);
    expect(r.bars[0] + r.bars[1]).toBeGreaterThan(0.8 * split.holdoutStart);
    expect(r.bars[0]).toBeGreaterThan(0.1 * split.holdoutStart);
    expect(r.bars[1]).toBeGreaterThan(0.1 * split.holdoutStart);
    // High volatility is the top third of the search period's own trailing volatility.
    expect(r.bars[2] / split.holdoutStart).toBeGreaterThan(0.25);
    expect(r.bars[2] / split.holdoutStart).toBeLessThan(0.4);
    for (let i = split.holdoutStart; i < n; i++) expect(r.mask[i]).toBe(0);
    // Wrecking every holdout price changes nothing in the search period.
    const poisoned = Float64Array.from(data.c);
    for (let i = split.holdoutStart; i < n; i++) poisoned[i] = poisoned[i] * (i % 2 === 0 ? 10 : 0.1);
    expect(Array.from(regimeMask(poisoned, split, ppy).mask)).toEqual(Array.from(r.mask));
    // Truncating the future cannot relabel the past trend (only the volatility threshold may move).
    const cut = truncate(data, Math.floor(n * 0.9));
    const rCut = regimeMask(cut.c, splitFor(cut.c.length), ppy);
    for (let i = 0; i < rCut.searchBars; i++) expect(rCut.mask[i] & 3).toBe(r.mask[i] & 3);
    expect(REGIME_HIGH_VOL).toBe(4);
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
      const genome = randomGenome(rng, 'long-only', DEFAULT_LIMITS);
      const { conditions } = genome.longEntry;
      const limit = 1 + Math.floor(rng() * (n - 1));
      out.fill(7); // sentinel: nothing beyond `limit` may be written
      evaluateGroup(genome.longEntry, ind, out, new Uint8Array(n), limit);
      evaluateCondition(conditions[0], ind, a);
      if (conditions.length > 1) {
        evaluateCondition(conditions[1], ind, b);
        pairs++;
      } else b.fill(1);
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

  it('ranks by a fitness that disqualifies small samples and penalises complexity, poor payoff and deep drawdowns', () => {
    const base = {
      assetScores: [2],
      primaryTrades: 150,
      primaryValidationTrades: 70,
      minTradesPerYear: 40,
      singleRegime: false,
      profitableShare: 1,
      params: 4,
      rewardToRisk: 2,
      validationWinRate: 50,
    };
    const score = combineFitness(base);
    expect(score).toBe(2);
    // Hard gates: one trade short of any minimum disqualifies, however good the rest looks.
    expect(MIN_TRADES).toBe(100);
    expect(MIN_VALIDATION_TRADES).toBe(60);
    expect(MIN_TRADES_PER_YEAR).toBe(25);
    expect(combineFitness({ ...base, primaryTrades: MIN_TRADES })).toBe(score);
    expect(combineFitness({ ...base, primaryTrades: MIN_TRADES - 1 })).toBe(UNFIT);
    expect(combineFitness({ ...base, primaryValidationTrades: MIN_VALIDATION_TRADES })).toBe(score);
    expect(combineFitness({ ...base, primaryValidationTrades: MIN_VALIDATION_TRADES - 1 })).toBe(UNFIT);
    expect(combineFitness({ ...base, minTradesPerYear: MIN_TRADES_PER_YEAR })).toBe(score);
    expect(combineFitness({ ...base, minTradesPerYear: MIN_TRADES_PER_YEAR - 0.1 })).toBe(UNFIT);
    // The audit's noise champions (17–55 validation trades) can never rank, even with a perfect score.
    for (const trades of [17, 32, 55]) expect(combineFitness({ ...base, assetScores: [50], primaryValidationTrades: trades })).toBe(UNFIT);
    expect(combineFitness({ ...base, primaryTrades: 0 })).toBe(UNFIT);
    // A strategy that makes all of its money in one market regime is disqualified.
    expect(combineFitness({ ...base, assetScores: [50], singleRegime: true })).toBe(UNFIT);
    expect(combineFitness({ ...base, params: 8 })).toBeCloseTo(score - 0.6, 9);
    // Reward:risk below 1 needs a win rate 10 points above break-even (66.7% at 0.5:1).
    expect(combineFitness({ ...base, rewardToRisk: 0.5, validationWinRate: 70 })).toBeCloseTo(score - 0.75, 9);
    expect(combineFitness({ ...base, rewardToRisk: 0.5, validationWinRate: 80 })).toBe(score);
    // Robust universe: the median counts, and fewer than 2/3 profitable assets costs a point.
    expect(combineFitness({ ...base, assetScores: [5, 1, 2], profitableShare: 1 })).toBe(2);
    expect(combineFitness({ ...base, assetScores: [5, 1, 2], profitableShare: 1 / 3 })).toBe(1);

    const data = dataset();
    const split = splitFor(data.c.length);
    const ppy = barsPerYear(data.t);
    const m = evaluateSearch(new IndicatorCache(data), TREND, SETTINGS, new BacktestWorkspace(data.c.length), split, ppy);
    const assetScore = assetBaseScore(m);
    expect(Number.isFinite(assetScore)).toBe(true);
    const deeper = { ...m, search: { ...m.search, maxDrawdown: m.search.maxDrawdown + 40 } };
    expect(assetBaseScore(deeper)).toBeLessThan(assetScore);
    // Stage 2: raw validation return over Buy & Hold no longer moves the score at all.
    const moreValidationRoi = { ...m, validation: { ...m.validation, roi: m.validation.roi + 30 } };
    expect(assetBaseScore(moreValidationRoi)).toBe(assetScore);
  });

  it('scores the 25th-percentile fold, so one lucky fold cannot carry a strategy', () => {
    const fold = (sharpe: number): PerformanceMetrics => ({ ...emptyMetrics(), sharpe, sortino: sharpe * 1.5 });
    const asMetrics = (sharpes: number[]): SearchMetrics => ({
      search: emptyMetrics(),
      train: emptyMetrics(),
      validation: emptyMetrics(),
      folds: sharpes.map(fold),
      regimes: null,
    });
    expect(FOLDS).toBe(5);
    // One spectacular fold and four losing ones vs five steady, modest folds.
    const lucky = assetBaseScore(asMetrics([3, -0.5, -0.4, -0.6, -0.3]));
    const steady = assetBaseScore(asMetrics([0.8, 0.7, 0.9, 0.6, 0.75]));
    expect(steady).toBeGreaterThan(lucky);
    expect(lucky).toBeLessThan(0);
    // The fold order does not matter, only the distribution.
    expect(assetBaseScore(asMetrics([-0.3, -0.6, -0.4, -0.5, 3]))).toBeCloseTo(lucky, 12);
    // 25th percentile of five values = the second-worst.
    expect(quantile([5, 1, 4, 2, 3], FOLD_QUANTILE)).toBe(2);
    expect(quantile([1, 2], 0.5)).toBe(1.5);
    // The fold boundaries cut the search period into five equal, sequential pieces.
    const bounds = foldBounds(splitFor(1000));
    expect(bounds).toEqual([0, 160, 320, 480, 640, 800]);
  });

  it('penalises regime collapse and disqualifies one-regime strategies', () => {
    const slice = (regime: RegimeMetrics['regime'], sharpe: number, trades = 30, coverage = 0.4): RegimeMetrics => ({
      regime, bars: 400, coverage, trades, netPnl: sharpe * 100, sharpe,
    });
    const steadyFolds = [0.8, 0.7, 0.9, 0.6, 0.75].map((sharpe) => ({ ...emptyMetrics(), sharpe, sortino: sharpe * 1.5 }));
    const withRegimes = (regimes: RegimeMetrics[]): SearchMetrics => ({
      search: emptyMetrics(), train: emptyMetrics(), validation: emptyMetrics(), folds: steadyFolds, regimes,
    });
    const healthy = [slice('Bull', 1.2), slice('Bear', 0.3), slice('High volatility', 0.5)];
    const bearCollapse = [slice('Bull', 1.2), slice('Bear', -1.4), slice('High volatility', 0.5)];
    expect(collapsedRegimes(healthy)).toHaveLength(0);
    expect(collapsedRegimes(bearCollapse).map((r) => r.regime)).toEqual(['Bear']);
    expect(assetBaseScore(withRegimes(bearCollapse))).toBeCloseTo(assetBaseScore(withRegimes(healthy)) - REGIME_COLLAPSE_PENALTY, 12);
    // Profitable in the bull slice only, collapsing everywhere else: a bet on one regime.
    expect(isSingleRegime([slice('Bull', 2), slice('Bear', -1.4), slice('High volatility', -0.9)])).toBe(true);
    expect(isSingleRegime(bearCollapse)).toBe(false); // two regimes made money
    expect(isSingleRegime([slice('Bull', 2), slice('Bear', -0.2), slice('High volatility', -0.9)])).toBe(false); // bear only mildly negative
    // Slices with too little coverage or too few trades are not judged at all.
    const untestable = [slice('Bull', 2), slice('Bear', -3, 4), slice('High volatility', -3, 30, 0.05)];
    expect(testableRegimes(untestable)).toHaveLength(1);
    expect(isSingleRegime(untestable)).toBe(false);
    expect(collapsedRegimes(untestable)).toHaveLength(0);
  });

  it('caps a strategy at the median of its one-step neighbours, so spikes rank below plateaus', () => {
    // An isolated spike: brilliant at its exact parameters, poor one grid step away.
    const spike = plateauFitness(3, [0.1, -1, 0.2, 0, 0.3]);
    // A broad plateau: slightly lower peak, but the neighbourhood holds up.
    const plateau = plateauFitness(2, [1.9, 2.1, 2, 1.8, 2.05]);
    expect(spike).toBeCloseTo(0.1, 12);
    expect(plateau).toBeCloseTo(2, 12);
    expect(plateau).toBeGreaterThan(spike);
    // Better neighbours never lift a strategy above its own fitness; no neighbours leaves it unchanged.
    expect(plateauFitness(1, [5, 6, 7])).toBe(1);
    expect(plateauFitness(1.5, [])).toBe(1.5);
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

function assertValid(genome: StrategyGenome, mode: 'long-only' | 'long-short'): void {
  // Structure.
  expect(genome.longEntry.logic).toBe('AND');
  expect(genome.longEntry.conditions.length).toBeGreaterThanOrEqual(1);
  expect(genome.longEntry.conditions.length).toBeLessThanOrEqual(2);
  expect(isCoherent(genome)).toBe(true);
  expect(countParameters(genome)).toBeLessThanOrEqual(MAX_PARAMETERS);
  expect('sizing' in genome).toBe(false);

  // Archetype.
  const [trigger, second] = genome.longEntry.conditions;
  const role = conditionRole(trigger);
  if (genome.archetype === 'TREND') {
    expect(role).toBe('TREND_UP');
    if (second) expect(conditionRole(second)).toBe('FILTER');
  } else {
    expect(['OVERSOLD', 'TREND_DOWN']).toContain(role);
    if (second) expect(['FILTER', 'TREND_UP']).toContain(conditionRole(second));
  }

  // Shorts are the exact mirror of the longs.
  if (mode === 'long-only') expect(genome.shortEntry).toBeNull();
  else expect(genome.shortEntry).toEqual(mirrorGroup(genome.longEntry));

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
    // Reversal exits: TREND only, a single condition, the mirror of the entry trigger.
    if (genome.archetype === 'MEAN_REVERSION') expect(exit.reversalExit).toBe(false);
    if (exit.reversalExit) {
      expect(exit.longExit?.conditions).toEqual([mirrorCondition(trigger)]);
      if (mode === 'long-short') expect(exit.shortExit?.conditions).toEqual([trigger]);
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

  it('generates, mutates and recombines only coherent, capped, on-grid genomes', () => {
    const rng = mulberry32(42);
    for (const mode of ['long-only', 'long-short'] as const) {
      let genome = randomGenome(rng, mode, DEFAULT_LIMITS);
      const kinds = new Set<string>();
      const archetypes = new Set<string>();
      let reversals = 0;
      let timeStops = 0;
      for (let i = 0; i < 4000; i++) {
        const r = rng();
        if (r < 0.2) genome = randomGenome(rng, mode, DEFAULT_LIMITS);
        else if (r < 0.3) genome = crossoverGenomes(rng, genome, randomGenome(rng, mode, DEFAULT_LIMITS), mode, DEFAULT_LIMITS);
        else genome = mutateGenome(rng, genome, mode, DEFAULT_LIMITS, 1 + Math.floor(rng() * 3));
        assertValid(genome, mode);
        for (const c of genome.longEntry.conditions) kinds.add(c.ind);
        archetypes.add(genome.archetype);
        if (genome.exit.mode === 'dynamic' && genome.exit.reversalExit) reversals++;
        if (genome.maxBarsInTrade !== null) timeStops++;
        // Repair and normalisation are idempotent; signatures are stable.
        expect(genomeSignature(repairGenome(genome, mode, DEFAULT_LIMITS, rng))).toBe(genomeSignature(genome));
        expect(normalizeGenome(genome, mode, DEFAULT_LIMITS)).toEqual(genome);
      }
      expect(kinds.size).toBe(11); // every indicator family is reachable
      expect(archetypes.size).toBe(2);
      expect(reversals).toBeGreaterThan(50);
      expect(timeStops).toBeGreaterThan(50);
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
    expect(fixed.longEntry.conditions.length).toBeLessThanOrEqual(2);
    // 2.4 / 1.6 ATR snaps to 2.5 / 1.5: the target is not 1 ATR beyond the stop, so the trail is dropped.
    expect(fixed.exit).toMatchObject({ stopAtrMult: 2.5, takeProfitAtrMult: 1.5, trailing: false });
    expect(fixed.maxBarsInTrade).toBe(10);

    // A valid mean-reversion version of the same idea survives intact.
    const fade = repairGenome(
      {
        archetype: 'MEAN_REVERSION',
        longEntry: {
          logic: 'AND',
          conditions: [
            { ind: 'SUPERTREND', period: 10, mult: 3, op: 'cross_below' },
            { ind: 'PRICE_MA', maType: 'SMA', period: 200, op: 'gt' },
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
    expect(groupFormula(fade.longEntry)).toBe('Close crosses below Supertrend(10,3.0) AND Close > SMA(200)');
    expect(groupFormula(fade.shortEntry!)).toBe('Close crosses above Supertrend(10,3.0) AND Close < SMA(200)');
    expect(countParameters(fade)).toBe(7); // 2 + 1 + ATR period, stop, target + time stop

    // Parameter cap: MACD (3) + Bollinger bandwidth (3) + ATR exit (3) + time stop (1) = 10 → trimmed to ≤ 8.
    const heavy = repairGenome(
      {
        archetype: 'TREND',
        longEntry: {
          logic: 'AND',
          conditions: [
            { ind: 'MACD', fast: 10, slow: 26, signal: 10, variant: 'signal', op: 'gt' },
            { ind: 'BB', period: 20, mult: 2, variant: 'bandwidth', op: 'gt', level: 10 },
          ],
        },
        shortEntry: null,
        exit: dynamicExit(2, 4, false),
        maxBarsInTrade: 20,
      },
      'long-only',
      DEFAULT_LIMITS,
      rng,
    );
    expect(countParameters(heavy)).toBeLessThanOrEqual(MAX_PARAMETERS);
    expect(heavy.maxBarsInTrade).toBeNull();
  });

  it('respects dataset limits (short histories, missing volume)', () => {
    const rng = mulberry32(7);
    const limits = { maxLookback: 50, hasVolume: false };
    let genome = randomGenome(rng, 'long-short', limits);
    for (let i = 0; i < 2000; i++) {
      genome = rng() < 0.3 ? randomGenome(rng, 'long-short', limits) : mutateGenome(rng, genome, 'long-short', limits, 2);
      expect(isCoherent(genome, limits)).toBe(true);
      for (const c of conditionsOf(genome)) {
        expect(c.ind).not.toBe('VOLUME');
        for (const lookback of numericParams(c)) expect(lookback).toBeLessThanOrEqual(50);
      }
      // No 100-bar regime filter is possible, so the optional second condition must be a FILTER.
      const second = genome.longEntry.conditions[1];
      if (second) expect(conditionRole(second)).toBe('FILTER');
    }
  });

  it('enumerates ±20% single-parameter neighbours on the grids', () => {
    const variants = parameterVariants(DIP, 'long-only', DEFAULT_LIMITS);
    // RSI period, RSI level, stop %, target % → two neighbours each; the 100-bar regime filter can only move up.
    expect(variants.length).toBeGreaterThanOrEqual(8);
    const base = genomeSignature(DIP);
    const signatures = new Set(variants.map(genomeSignature));
    expect(signatures.size).toBe(variants.length);
    expect(signatures.has(base)).toBe(false);
    for (const variant of variants) {
      assertValid(variant, 'long-only');
      expect(variant.archetype).toBe(DIP.archetype);
      // Exactly one number differs from the original.
      const a = JSON.stringify(variant).match(/-?\d+(\.\d+)?/g) ?? [];
      const b = JSON.stringify(DIP).match(/-?\d+(\.\d+)?/g) ?? [];
      expect(a.length).toBe(b.length);
      expect(a.filter((value, index) => value !== b[index]).length).toBe(1);
    }
    // RSI(14): −20% = 11.2 → 10, +20% = 16.8 → snaps back to 14, so the next grid step (20) is used.
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
    expect(groupFormula(DIP.longEntry)).toBe('RSI(14) < 45 AND Close > EMA(100)');
    expect(indicatorBadges(DIP)).toEqual(['RSI', 'EMA']);
    expect(indicatorBadges(TREND)).toEqual(['Supertrend', 'ATR']);
    const entry = describeLongEntry(DIP);
    expect(entry.intro).toContain('BOTH of the following');
    expect(entry.bullets[0].text.startsWith('Trigger: ')).toBe(true);
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
    for (const mode of ['long-only', 'long-short'] as const) {
      for (const sizing of [RISK.sizing, SETTINGS.sizing, { mode: 'risk', riskPct: 1, compounding: true }] as const) {
        const settings: BacktestSettings = { ...SETTINGS, tradeMode: mode, sizing };
        for (let i = 0; i < 150; i++) {
          const genome = randomGenome(rng, mode, DEFAULT_LIMITS);
          const script = generatePineScript(resultFor(genome, settings), ctx(settings));
          expect(script.startsWith('//@version=5\n')).toBe(true);
          expect(script).toContain('strategy("StrategyFinder S-1 | XRP-USD 1d"');
          expect(script).toContain('use_bar_magnifier=true');
          expect(script).toContain('commission_value=0.15');
          expect(script).toContain('Tested on    : XRPUSDT on Binance');
          expect(script).toContain(`Archetype    : ${ARCHETYPE_LABEL[genome.archetype]}`);
          expect(script).toContain(`Parameters   : ${countParameters(genome)} tunable values`);
          expect(script).toContain('strategy.entry("Long", strategy.long, qty=orderQty)');
          expect(script.includes('strategy.entry("Short"')).toBe(mode === 'long-short');
          // Never more than the account can buy.
          expect(script).toContain('maxQty = math.max(strategy.equity, 0) / (close * (1 + costPct / 100))');
          expect(script).toMatch(sizing.mode === 'full' ? /orderQty = maxQty/ : /math\.min\(riskCapital \* riskPct \/ 100 \/ stopDistance, maxQty\)/);
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
          const used = code.match(/\b(?:rsi|ema|sma|macdLine|macdSignal|macdHist|bbUpper|bbLower|bbBasis|bbPctB|bbWidth|atr|atrPct|stochK|stochD|diPlus|diMinus|adx|volRatio|supertrend|cci)_[0-9_]+\b/g) ?? [];
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
    expect(staticScript).toContain('longCond2 = close > ema_100');
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
    expect(trailScript).toContain('longTrail := math.max(longTrail, longPeak - stopAtrMult * atrValue)');
    expect(trailScript).toContain('shortTrail  := math.min(shortTrail, shortTrough + stopAtrMult * atrValue)');
    expect(trailScript).toContain('maxBarsInTrade = input.int(20, "Time stop (bars in trade)"');
    expect(trailScript).toContain('timedOut = strategy.position_size != 0 and bar_index - strategy.opentrades.entry_bar_index(0) >= maxBarsInTrade - 1');
    expect(trailScript).toContain('if timedOut\n    strategy.close_all(comment="Time stop")');
    expect(trailScript).toContain('and not timedOut');
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
