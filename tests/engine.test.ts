import { describe, expect, it } from 'vitest';
import { buildDataset, normalizeWeights } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  BacktestWorkspace,
  barsPerYear,
  buildBenchmark,
  computeFitness,
  evaluateGenome,
  runBacktest,
  splitMetrics,
  UNFIT,
} from '../src/engine/backtester';
import { describeExit, describeLongEntry, groupFormula, indicatorBadges, listIndicators } from '../src/engine/describe';
import {
  crossoverGenomes,
  DEFAULT_LIMITS,
  genomeSignature,
  mirrorCondition,
  mulberry32,
  mutateGenome,
  randomGenome,
  RANGES,
  repairGenome,
} from '../src/engine/genome';
import { atr, bollinger, ema, IndicatorCache, rsi, sma, stochastic, supertrend } from '../src/engine/indicators';
import { generatePineScript } from '../src/engine/pine';
import type { BacktestSettings, Condition, OHLCV, StrategyGenome, StrategyResult } from '../src/types/trading';

const NOW = Date.UTC(2026, 0, 15);
const SETTINGS: BacktestSettings = { initialCapital: 10_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-short' };

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

const TREND: StrategyGenome = {
  longEntry: { logic: 'AND', conditions: [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }] },
  shortEntry: { logic: 'AND', conditions: [{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'lt' }] },
  exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 3, trailing: true, takeProfitAtrMult: null, longExit: null, shortExit: null },
  sizing: { mode: 'full' },
};

const DIP: StrategyGenome = {
  longEntry: {
    logic: 'AND',
    conditions: [
      { ind: 'RSI', period: 14, op: 'lt', level: 48 },
      { ind: 'PRICE_MA', maType: 'EMA', period: 50, op: 'gt' },
    ],
  },
  shortEntry: null,
  exit: { mode: 'static', stopLossPct: 4, takeProfitPct: 9 },
  sizing: { mode: 'full' },
};

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
      for (let i = 0; i < cut - 1; i++) expect(wsCut.equity[i]).toBeCloseTo(wsFull.equity[i], 6);
      expect(wsFull.tradeCount).toBeGreaterThan(3);
    }
  });

  it('fills a close-of-bar signal at the NEXT bar open, with fee and slippage', () => {
    // Close crosses above SMA(3) on bar 5; the entry must happen at bar 6's open.
    const closes = [100, 100, 100, 100, 100, 110, 111, 112, 113, 114];
    const opens = [100, 100, 100, 100, 100, 100, 120, 111, 112, 113];
    const n = closes.length;
    const data: OHLCV = {
      t: Float64Array.from({ length: n }, (_, i) => i * 86_400_000),
      o: f64(opens),
      h: f64(closes.map((c, i) => Math.max(c, opens[i]) + 1)),
      l: f64(closes.map((c, i) => Math.min(c, opens[i]) - 1)),
      c: f64(closes),
      v: new Float64Array(n).fill(1000),
    };
    const genome: StrategyGenome = {
      longEntry: { logic: 'AND', conditions: [{ ind: 'PRICE_MA', maType: 'SMA', period: 3, op: 'cross_above' }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 15, takeProfitPct: 60 },
      sizing: { mode: 'full' },
    };
    const settings: BacktestSettings = { initialCapital: 10_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-only' };
    const ws = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(data), genome, settings, ws);

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

  it('honours stop-loss and take-profit levels intrabar', () => {
    const n = 12;
    const flat = new Array<number>(n).fill(100);
    const make = (lows: number[], highs: number[], closes: number[]): OHLCV => ({
      t: Float64Array.from({ length: n }, (_, i) => i * 86_400_000),
      o: f64(flat),
      h: f64(highs),
      l: f64(lows),
      c: f64(closes),
      v: new Float64Array(n).fill(1000),
    });
    const genome: StrategyGenome = {
      // Volume ratio is exactly 1 > 0.5 → always true once the SMA exists (first signal on bar 9).
      longEntry: { logic: 'AND', conditions: [{ ind: 'VOLUME', period: 10, op: 'gt', level: 0.5 }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 5, takeProfitPct: 10 },
      sizing: { mode: 'full' },
    };
    const settings: BacktestSettings = { initialCapital: 10_000, feePct: 0, slippagePct: 0, tradeMode: 'long-only' };

    // Stop: bar 10 trades down to 90 → exit at 95 (−5%).
    const lows = [...flat];
    lows[10] = 90;
    const wsStop = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(make(lows, flat, flat)), genome, settings, wsStop);
    expect(wsStop.tradeEntryBar[0]).toBe(10);
    expect(wsStop.tradePnl[0]).toBeCloseTo(-500, 6);

    // Target: bar 10 trades up to 115 → exit at 110 (+10%).
    const highs = [...flat];
    highs[10] = 115;
    const wsTarget = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(make(flat, highs, flat)), genome, settings, wsTarget);
    expect(wsTarget.tradePnl[0]).toBeCloseTo(1000, 6);

    // Both touched in the same candle → the stop is assumed to fill first.
    const wsBoth = new BacktestWorkspace(n);
    runBacktest(new IndicatorCache(make(lows, highs, flat)), genome, settings, wsBoth);
    expect(wsBoth.tradePnl[0]).toBeCloseTo(-500, 6);
  });

  it('never opens short trades in Long-Only mode', () => {
    const data = dataset('SOL-USD');
    const ws = new BacktestWorkspace(data.c.length);
    runBacktest(new IndicatorCache(data), TREND, { ...SETTINGS, tradeMode: 'long-only' }, ws);
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
    const costly = ws.equity[data.c.length - 1];
    expect(costly).toBeLessThan(free);
  });

  it('volatility sizing reduces exposure and drawdown', () => {
    const data = dataset();
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);
    const ppy = barsPerYear(data.t);
    const split = Math.floor(data.c.length * 0.7);
    const full = evaluateGenome(ind, TREND, SETTINGS, ws, split, ppy).full;
    const sized = evaluateGenome(ind, { ...TREND, sizing: { mode: 'risk', riskPct: 1 } }, SETTINGS, ws, split, ppy).full;
    expect(sized.trades).toBe(full.trades);
    expect(sized.maxDrawdown).toBeLessThan(full.maxDrawdown);
  });

  it('splits metrics so in-sample + out-of-sample PnL equals the full-period PnL', () => {
    const data = dataset();
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);
    const split = Math.floor(data.c.length * 0.7);
    const m = evaluateGenome(ind, TREND, SETTINGS, ws, split, barsPerYear(data.t));
    expect(m.inSample.netPnl + m.outOfSample.netPnl).toBeCloseTo(m.full.netPnl, 6);
    expect(m.inSample.trades + m.outOfSample.trades).toBe(m.full.trades);
    expect(m.full.roi).toBeCloseTo((m.full.netPnl / SETTINGS.initialCapital) * 100, 9);
    expect(m.full.maxDrawdown).toBeGreaterThanOrEqual(m.outOfSample.maxDrawdown - 1e-9);
    expect(m.full.winRate).toBeGreaterThanOrEqual(0);
    expect(m.full.winRate).toBeLessThanOrEqual(100);
  });

  it('builds an exact Buy & Hold benchmark', () => {
    const data = dataset('SPY');
    const n = data.c.length;
    const split = Math.floor(n * 0.7);
    const bench = buildBenchmark(data, SETTINGS, split, barsPerYear(data.t));
    const units = 10_000 / (data.o[0] * 1.0005 * 1.001);
    expect(bench.equity[n - 1]).toBeCloseTo(units * data.c[n - 1], 6);
    expect(bench.full.netPnl).toBeCloseTo(units * data.c[n - 1] - 10_000, 6);
    expect(bench.inSample.netPnl + bench.outOfSample.netPnl).toBeCloseTo(bench.full.netPnl, 6);
    // Sharpe of a known curve: constant +1% per bar has zero variance → Sharpe 0, positive CAGR.
    const steady = Float64Array.from({ length: 100 }, (_, i) => 100 * 1.01 ** (i + 1));
    const s = splitMetrics(steady, 70, 100, 252, null);
    expect(s.full.maxDrawdown).toBe(0);
    expect(s.full.cagr).toBeGreaterThan(0);
    expect(s.full.roi).toBeCloseTo((1.01 ** 100 - 1) * 100, 6);
  });

  it('ranks by a fitness that penalises inactivity and deep drawdowns', () => {
    const data = dataset();
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);
    const ppy = barsPerYear(data.t);
    const split = Math.floor(data.c.length * 0.7);
    const bench = buildBenchmark(data, SETTINGS, split, ppy);
    const m = evaluateGenome(ind, TREND, SETTINGS, ws, split, ppy);
    const base = computeFitness(m, bench);
    expect(Number.isFinite(base)).toBe(true);

    const fewTrades = { ...m, full: { ...m.full, trades: 5 }, outOfSample: { ...m.outOfSample, trades: 1 } };
    expect(computeFitness(fewTrades, bench)).toBeLessThan(base);
    const deepDd = { ...m, full: { ...m.full, maxDrawdown: m.full.maxDrawdown + 40 } };
    expect(computeFitness(deepDd, bench)).toBeLessThan(base);
    const betterOos = { ...m, outOfSample: { ...m.outOfSample, sharpe: m.outOfSample.sharpe + 1, sortino: m.outOfSample.sortino + 1 } };
    expect(computeFitness(betterOos, bench)).toBeGreaterThan(base);
    const moreAlpha = { ...m, outOfSample: { ...m.outOfSample, roi: m.outOfSample.roi + 30 } };
    expect(computeFitness(moreAlpha, bench)).toBeGreaterThan(base);
    expect(computeFitness({ ...m, full: { ...m.full, trades: 0 } }, bench)).toBe(UNFIT);
  });
});

/* -------------------------------------------------------------------------- */

function assertValid(genome: StrategyGenome, mode: 'long-only' | 'long-short'): void {
  expect(genome.longEntry.conditions.length).toBeGreaterThanOrEqual(1);
  expect(genome.longEntry.conditions.length).toBeLessThanOrEqual(4);
  if (mode === 'long-only') expect(genome.shortEntry).toBeNull();
  else {
    expect(genome.shortEntry).not.toBeNull();
    expect(genome.shortEntry?.conditions.length).toBeGreaterThanOrEqual(1);
    expect(genome.shortEntry?.conditions.length).toBeLessThanOrEqual(4);
  }
  if (genome.exit.mode === 'static') {
    expect(genome.exit.stopLossPct).toBeGreaterThanOrEqual(RANGES.stopLossPct[0]);
    expect(genome.exit.stopLossPct).toBeLessThanOrEqual(RANGES.stopLossPct[1]);
    expect(genome.exit.takeProfitPct).toBeGreaterThanOrEqual(RANGES.takeProfitPct[0]);
    expect(genome.exit.takeProfitPct).toBeLessThanOrEqual(RANGES.takeProfitPct[1]);
  } else {
    expect(genome.exit.stopAtrMult).toBeGreaterThanOrEqual(RANGES.stopAtrMult[0]);
    expect(genome.exit.stopAtrMult).toBeLessThanOrEqual(RANGES.stopAtrMult[1]);
    if (genome.exit.takeProfitAtrMult !== null) {
      expect(genome.exit.takeProfitAtrMult).toBeGreaterThanOrEqual(RANGES.takeProfitAtrMult[0]);
      expect(genome.exit.takeProfitAtrMult).toBeLessThanOrEqual(RANGES.takeProfitAtrMult[1]);
    } else {
      // Without a target there must be another profit-taking path.
      expect(genome.exit.trailing || genome.exit.longExit !== null).toBe(true);
    }
    if (mode === 'long-only') expect(genome.exit.shortExit).toBeNull();
    if (mode === 'long-short' && genome.exit.longExit) expect(genome.exit.shortExit).not.toBeNull();
  }
  const groups = [genome.longEntry, genome.shortEntry];
  if (genome.exit.mode === 'dynamic') groups.push(genome.exit.longExit, genome.exit.shortExit);
  for (const group of groups) {
    if (!group) continue;
    for (const c of group.conditions) {
      if (c.ind === 'RSI') {
        expect(c.period).toBeGreaterThanOrEqual(7);
        expect(c.period).toBeLessThanOrEqual(28);
      }
      if (c.ind === 'MACD') expect(c.slow).toBeGreaterThan(c.fast);
      if (c.ind === 'MA_CROSS') expect(c.slowPeriod).toBeGreaterThan(c.fastPeriod);
      if (c.ind === 'VOLUME' || c.ind === 'ATR') expect(['gt', 'lt']).toContain(c.op);
    }
  }
}

describe('genome', () => {
  it('generates, mutates and recombines only valid genomes', () => {
    const rng = mulberry32(42);
    for (const mode of ['long-only', 'long-short'] as const) {
      let genome = randomGenome(rng, mode, DEFAULT_LIMITS);
      const kinds = new Set<string>();
      for (let i = 0; i < 3000; i++) {
        const r = rng();
        if (r < 0.2) genome = randomGenome(rng, mode, DEFAULT_LIMITS);
        else if (r < 0.3) genome = crossoverGenomes(rng, genome, randomGenome(rng, mode, DEFAULT_LIMITS), mode, DEFAULT_LIMITS);
        else genome = mutateGenome(rng, genome, mode, DEFAULT_LIMITS, 1 + Math.floor(rng() * 3));
        assertValid(genome, mode);
        for (const c of genome.longEntry.conditions) kinds.add(c.ind);
        // Repair is idempotent and signatures are stable.
        const repaired = repairGenome(genome, mode, DEFAULT_LIMITS, rng);
        expect(genomeSignature(repaired)).toBe(genomeSignature(genome));
      }
      // Every indicator family is reachable.
      expect(kinds.size).toBe(11);
    }
  });

  it('respects dataset limits (short histories, missing volume)', () => {
    const rng = mulberry32(7);
    const limits = { ...DEFAULT_LIMITS, maxLookback: 40, hasVolume: false };
    let genome = randomGenome(rng, 'long-short', limits);
    for (let i = 0; i < 2000; i++) {
      genome = rng() < 0.3 ? randomGenome(rng, 'long-short', limits) : mutateGenome(rng, genome, 'long-short', limits, 2);
      for (const group of [genome.longEntry, genome.shortEntry]) {
        for (const c of group?.conditions ?? []) {
          expect(c.ind).not.toBe('VOLUME');
          if (c.ind === 'PRICE_MA') expect(c.period).toBeLessThanOrEqual(40);
          if (c.ind === 'MA_CROSS') expect(c.slowPeriod).toBeLessThanOrEqual(40);
          if (c.ind === 'MACD') expect(c.slow).toBeLessThanOrEqual(40);
        }
      }
    }
  });

  it('mirrors directional conditions symmetrically', () => {
    const samples: Condition[] = [
      { ind: 'RSI', period: 14, op: 'lt', level: 30 },
      { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'cross_above' },
      { ind: 'BB', period: 20, mult: 2, variant: 'lower', op: 'cross_above', level: 0 },
      { ind: 'BB', period: 20, mult: 2, variant: 'pctb', op: 'lt', level: 0.1 },
      { ind: 'CCI', period: 20, op: 'lt', level: -100 },
      { ind: 'STOCH', kPeriod: 14, smooth: 3, dPeriod: 3, variant: 'level', op: 'lt', level: 20 },
      { ind: 'VOLUME', period: 20, op: 'gt', level: 1.3 },
    ];
    for (const c of samples) expect(mirrorCondition(mirrorCondition(c))).toEqual(c);
    expect(mirrorCondition(samples[0])).toEqual({ ind: 'RSI', period: 14, op: 'gt', level: 70 });
    expect(mirrorCondition(samples[2])).toMatchObject({ variant: 'upper', op: 'cross_below' });
    expect(mirrorCondition(samples[6])).toEqual(samples[6]); // filters are direction-agnostic
  });

  it('treats condition order as irrelevant for identity', () => {
    const a: StrategyGenome = { ...DIP, longEntry: { logic: 'AND', conditions: [...DIP.longEntry.conditions] } };
    const b: StrategyGenome = { ...DIP, longEntry: { logic: 'AND', conditions: [...DIP.longEntry.conditions].reverse() } };
    expect(genomeSignature(a)).toBe(genomeSignature(b));
    expect(genomeSignature(a)).not.toBe(genomeSignature(TREND));
  });
});

/* -------------------------------------------------------------------------- */

describe('descriptions and Pine export', () => {
  const data = dataset();
  const ppy = barsPerYear(data.t);
  const split = Math.floor(data.c.length * 0.7);

  function resultFor(genome: StrategyGenome, settings: BacktestSettings): StrategyResult {
    const ws = new BacktestWorkspace(data.c.length);
    const m = evaluateGenome(new IndicatorCache(data), genome, settings, ws, split, ppy);
    return {
      id: 'S-1',
      genome,
      fitness: 1,
      ...m,
      alpha: 0,
      alphaOos: 0,
      equity: new Float32Array(0),
      foundAt: 1,
    };
  }

  const ctx = (settings: BacktestSettings) => ({
    label: 'BTC-USD',
    interval: '1d' as const,
    settings,
    components: [{ symbol: 'BTC-USD', weight: 1 }],
    units: [],
    windowStart: data.t[0],
    windowEnd: data.t[data.t.length - 1],
    bars: data.t.length,
    benchmarkRoi: 12.3,
    simulatedData: true,
  });

  it('renders formulas and plain-English rules', () => {
    const genome: StrategyGenome = {
      ...DIP,
      longEntry: {
        logic: 'AND',
        conditions: [
          { ind: 'RSI', period: 14, op: 'lt', level: 34 },
          { ind: 'PRICE_MA', maType: 'EMA', period: 200, op: 'gt' },
          { ind: 'VOLUME', period: 20, op: 'gt', level: 1.3 },
        ],
      },
    };
    expect(groupFormula(genome.longEntry)).toBe('RSI(14) < 34 AND Close > EMA(200) AND Volume_Ratio(20) > 1.30');
    expect(indicatorBadges(genome)).toEqual(['RSI', 'EMA', 'Volume']);
    expect(indicatorBadges(TREND)).toEqual(['Supertrend', 'ATR']);
    const entry = describeLongEntry(genome);
    expect(entry.intro).toContain('ALL of the following');
    expect(entry.bullets).toHaveLength(3);
    const exit = describeExit(genome, SETTINGS);
    expect(exit.mode).toBe('Static %');
    expect(exit.bullets.map((b) => b.title)).toEqual(['Stop loss', 'Take profit', 'Position size', 'Execution']);
    const dynamic = describeExit(TREND, SETTINGS);
    expect(dynamic.mode).toBe('Dynamic ATR');
    expect(dynamic.bullets[0].title).toBe('Trailing ATR stop');
    expect(listIndicators(TREND).map((u) => u.name)).toEqual(['Supertrend', 'ATR']);
  });

  it('exports syntactically well-formed Pine Script v5 for every kind of genome', () => {
    const rng = mulberry32(99);
    for (const mode of ['long-only', 'long-short'] as const) {
      const settings = { ...SETTINGS, tradeMode: mode };
      for (let i = 0; i < 400; i++) {
        const genome = randomGenome(rng, mode, DEFAULT_LIMITS);
        const script = generatePineScript(resultFor(genome, settings), ctx(settings));
        expect(script.startsWith('//@version=5\n')).toBe(true);
        expect(script).toContain('strategy("StrategyFinder S-1 | BTC-USD 1d"');
        expect(script).toContain('commission_value=0.15');
        expect(script).toContain('strategy.entry("Long", strategy.long, qty=orderQty)');
        expect(script.includes('strategy.entry("Short"')).toBe(mode === 'long-short');
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
        // Every identifier used in a rule is declared before the rules section.
        const declared = new Set<string>();
        for (const line of code.split('\n')) {
          const tuple = /^\[([^\]]+)\]\s*=/.exec(line);
          if (tuple) tuple[1].split(',').forEach((name) => declared.add(name.trim()));
          const single = /^(?:var float )?([A-Za-z_][A-Za-z0-9_]*)\s*=[^=]/.exec(line);
          if (single) declared.add(single[1]);
        }
        const used = code.match(/\b(?:rsi|ema|sma|macdLine|macdSignal|macdHist|bbUpper|bbLower|bbBasis|bbPctB|bbWidth|atr|atrPct|stochK|stochD|diPlus|diMinus|adx|volRatio|supertrend|cci)_[0-9_]+\b/g) ?? [];
        for (const name of used) expect(declared.has(name)).toBe(true);
      }
    }
  });

  it('writes the exact exit construction into the script', () => {
    const staticScript = generatePineScript(resultFor(DIP, { ...SETTINGS, tradeMode: 'long-only' }), ctx({ ...SETTINGS, tradeMode: 'long-only' }));
    expect(staticScript).toContain('stopLossPct   = input.float(4.0, "Stop loss %"');
    expect(staticScript).toContain('takeProfitPct = input.float(9.0, "Take profit %"');
    expect(staticScript).toContain('longCond1 = rsi_14 < 48');
    expect(staticScript).toContain('longCond2 = close > ema_50');
    expect(staticScript).toContain('if goLong and strategy.position_size == 0');
    expect(staticScript).not.toContain('goShort');

    const trailScript = generatePineScript(resultFor(TREND, SETTINGS), ctx(SETTINGS));
    expect(trailScript).toContain('[supertrend_10_3, supertrendDir_10_3] = ta.supertrend(3.0, 10)');
    expect(trailScript).toContain('longTrail := math.max(longTrail, longPeak - stopAtrMult * atrValue)');
    expect(trailScript).toContain('shortTrail  := math.min(shortTrail, shortTrough + stopAtrMult * atrValue)');
    expect(trailScript).toContain('strategy.exit("Long exit", from_entry="Long", loss=stopTicks)');
    expect(trailScript).toContain('if goShort and strategy.position_size >= 0');
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
