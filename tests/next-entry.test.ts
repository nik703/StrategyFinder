import { describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { BacktestWorkspace, evaluateGroup, runBacktest } from '../src/engine/backtester';
import { DEFAULT_LIMITS, mirrorGroup, mulberry32, randomGenome } from '../src/engine/genome';
import { HYPOTHESIS_IDS } from '../src/types/trading';
import { ema, IndicatorCache } from '../src/engine/indicators';
import {
  BISECTION_STEPS,
  proximityRank,
  readinessScore,
  solveNextEntry,
  verdictFor,
  type NextEntryForecast,
  type SideForecast,
} from '../src/engine/nextEntrySolver';
import type { BacktestSettings, Condition, OHLCV, RuleGroup, StrategyGenome } from '../src/types/trading';

const DAY = 86_400_000;

const settingsFor = (tradeMode: BacktestSettings['tradeMode']): BacktestSettings => ({
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  tradeMode,
  sizing: { mode: 'full', riskPct: 2, compounding: true },
  fundingPctPerBar: 0,
});

/** Candles from a list of closes: each opens at the previous close, with a small wick. */
function fromCloses(closes: number[], volume = 1000): OHLCV {
  const n = closes.length;
  const data: OHLCV = {
    t: Float64Array.from({ length: n }, (_, i) => i * DAY),
    o: new Float64Array(n),
    h: new Float64Array(n),
    l: new Float64Array(n),
    c: Float64Array.from(closes),
    v: new Float64Array(n).fill(volume),
  };
  for (let i = 0; i < n; i++) {
    const open = i === 0 ? closes[0] : closes[i - 1];
    data.o[i] = open;
    data.h[i] = Math.max(open, closes[i]) * 1.002;
    data.l[i] = Math.min(open, closes[i]) * 0.998;
  }
  return data;
}

/** The candles plus the synthetic next candle the solver assumes (open = last close, close = price). */
function withNextBar(data: OHLCV, price: number, extra?: { open: number }): OHLCV {
  const n = data.c.length;
  const last = data.c[n - 1];
  const step = data.t[n - 1] - data.t[n - 2];
  const bars = extra ? 2 : 1;
  const grow = (src: Float64Array, ...values: number[]): Float64Array => {
    const out = new Float64Array(n + bars);
    out.set(src);
    values.forEach((value, i) => (out[n + i] = value));
    return out;
  };
  let avgVolume = 0;
  for (let i = Math.max(0, n - 20); i < n; i++) avgVolume += data.v[i];
  avgVolume /= Math.min(20, n);
  const next = [data.t[n - 1] + step, last, Math.max(last, price), Math.min(last, price), price, avgVolume];
  const after = extra ? [data.t[n - 1] + 2 * step, extra.open, extra.open * 1.001, extra.open * 0.999, extra.open, avgVolume] : null;
  const pick = (k: number): number[] => (after ? [next[k], after[k]] : [next[k]]);
  return {
    t: grow(data.t, ...pick(0)),
    o: grow(data.o, ...pick(1)),
    h: grow(data.h, ...pick(2)),
    l: grow(data.l, ...pick(3)),
    c: grow(data.c, ...pick(4)),
    v: grow(data.v, ...pick(5)),
  };
}

/** Is `group` true on the last candle of `data`, according to the backtester's own evaluator? */
function firesOnLastBar(group: RuleGroup, data: OHLCV): boolean {
  const n = data.c.length;
  const out = new Uint8Array(n);
  evaluateGroup(group, new IndicatorCache(data), out, new Uint8Array(n), n);
  return out[n - 1] === 1;
}

/**
 * Proves a solved trigger price with backtester.ts: the rule fires when the next candle
 * closes at the trigger, not when it stops just short of it, and runBacktest really
 * opens the trade at the following open.
 */
function verifyTrigger(data: OHLCV, genome: StrategyGenome, settings: BacktestSettings, side: SideForecast): void {
  const price = side.triggerPrice as number;
  const group = (side.side === 'long' ? genome.longEntry : genome.shortEntry) as RuleGroup;
  const last = data.c[data.c.length - 1];
  const towards = Math.sign(last - price); // direction back towards the current price
  expect(firesOnLastBar(group, withNextBar(data, price * (1 + 1e-9 * -towards)))).toBe(true);
  if (side.status === 'trigger') expect(firesOnLastBar(group, withNextBar(data, price * (1 + 1e-6 * towards)))).toBe(false);

  if (side.conflict === null) {
    const extended = withNextBar(data, price * (1 + 1e-9 * -towards), { open: price });
    const ws = new BacktestWorkspace(extended.c.length);
    runBacktest(new IndicatorCache(extended), genome, settings, ws);
    const k = ws.tradeCount - 1;
    expect(ws.tradeEntryBar[k]).toBe(extended.c.length - 1);
    expect(ws.tradeDir[k]).toBe(side.side === 'long' ? 1 : -1);
    // The quoted fill is exactly the entry price the backtester records (same slippage rule).
    const recorded = ws.openPosition?.entryPrice as number;
    expect(Math.abs(recorded / (side.fillPrice as number) - 1)).toBeLessThan(1e-9);
    // The quoted initial stop is the backtester's (1R = its initial stop distance), and so is the
    // breakeven level; the stop mode says whether the backtester will trail it.
    const open = ws.openPosition;
    if (open && side.stopLoss !== null) {
      const dir = side.side === 'long' ? 1 : -1;
      expect(Math.abs((open.entryPrice - dir * open.risk) / side.stopLoss - 1)).toBeLessThan(1e-6);
      if (genome.exit.breakeven) expect(Math.abs((side.breakevenAt as number) / (open.entryPrice + dir * open.risk) - 1)).toBeLessThan(1e-6);
      else expect(side.breakevenAt).toBeNull();
      const trailing = genome.exit.mode === 'dynamic' && genome.exit.trailing;
      expect(side.stopMode).toBe(trailing ? 'trailing' : 'fixed');
      expect(side.trailAtrMult).toBe(trailing && genome.exit.mode === 'dynamic' ? genome.exit.stopAtrMult : null);
    }
  }
}

/** Closes that alternate ±0.5 around 100: a tight, trendless range. */
const RANGE_BOUND = Array.from({ length: 120 }, (_, i) => 100 + (i % 2 === 0 ? 0.5 : -0.5));

const genome = (longConditions: Condition[], options: Partial<StrategyGenome> & { shorts?: boolean } = {}): StrategyGenome => {
  const longEntry: RuleGroup = { logic: 'AND', conditions: longConditions };
  return {
    archetype: options.archetype ?? 'TREND',
    longEntry,
    shortEntry: options.shorts ? mirrorGroup(longEntry) : null,
    exit: options.exit ?? { mode: 'static', stopLossPct: 5, takeProfitPct: 10 },
    maxBarsInTrade: options.maxBarsInTrade ?? null,
  };
};

/* -------------------------------------------------------------------------- */

describe('next-entry solver: trigger prices', () => {
  it('solves an EMA crossover exactly (matches the closed-form answer)', () => {
    // A long flat stretch, then a gentle decline: EMA(5) sits just below EMA(20).
    const closes = [...Array.from({ length: 80 }, () => 100), ...Array.from({ length: 15 }, (_, i) => 99.9 - 0.1 * i)];
    const data = fromCloses(closes);
    const g = genome([{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 5, slowType: 'EMA', slowPeriod: 20, op: 'cross_above' }]);
    const settings = settingsFor('long-only');
    const f = solveNextEntry({ data, genome: g, settings });

    expect(f.long.status).toBe('trigger');
    // EMA_new = a·P + (1 − a)·EMA_prev, so the cross happens at P* = ((1−a20)E20 − (1−a5)E5) / (a5 − a20).
    const n = closes.length;
    const e5 = ema(data.c, 5)[n - 1];
    const e20 = ema(data.c, 20)[n - 1];
    const a5 = 2 / 6;
    const a20 = 2 / 21;
    const analytic = ((1 - a20) * e20 - (1 - a5) * e5) / (a5 - a20);
    expect(f.long.triggerPrice).toBeCloseTo(analytic, 6);
    expect(f.long.movePct).toBeCloseTo(((analytic - closes[n - 1]) / closes[n - 1]) * 100, 4);
    expect(f.short.status).toBe('disabled');
    verifyTrigger(data, g, settings, f.long);

    // Fill and exits at that entry: next-open fill plus slippage, static 5% / 10% bracket.
    const fill = analytic * 1.0005;
    expect(f.long.fillPrice).toBeCloseTo(fill, 6);
    expect(f.long.stopLoss).toBeCloseTo(fill * 0.95, 6);
    expect(f.long.takeProfit).toBeCloseTo(fill * 1.1, 6);
    expect(f.long.conditions[0].text).toMatch(/^EMA\(5\) vs EMA\(20\): \d+\.\d\d ATR away from bullish cross \[Pending\]$/);
    expect(BISECTION_STEPS).toBeGreaterThanOrEqual(40);
  });

  it('solves breakouts above the upper Bollinger band and the Supertrend (and the mirrored breakdowns)', () => {
    // A tight range: price never left its bands, so no trade is open and a breakout is needed.
    const data = fromCloses(RANGE_BOUND);
    const settings = settingsFor('long-short');
    const bands = genome([{ ind: 'BB', period: 20, mult: 2, variant: 'upper', op: 'cross_above', level: 0 }], { shorts: true });
    const f = solveNextEntry({ data, genome: bands, settings });
    expect(f.position).toBeNull();
    expect(f.long.status).toBe('trigger');
    expect(f.short.status).toBe('trigger');
    expect(f.long.movePct).toBeGreaterThan(0); // a breakout needs a higher close
    expect(f.short.movePct).toBeLessThan(0); // a breakdown (mirror: close crosses below the lower band) needs a lower close
    verifyTrigger(data, bands, settings, f.long);
    verifyTrigger(data, bands, settings, f.short);

    // Supertrend: in a range the line is the upper band, so the long breakout is solvable, while the
    // mirrored short ("close crosses below the line") first needs price to get above it.
    const trend = genome([{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'cross_above' }], { shorts: true });
    const st = solveNextEntry({ data, genome: trend, settings });
    expect(st.long.status).toBe('trigger');
    expect(st.long.movePct).toBeGreaterThan(0);
    verifyTrigger(data, trend, settings, st.long);
    expect(st.short.status).toBe('unreachable');
    expect(st.short.conditions[0].state).toBe('reset');
    expect(st.short.conditions[0].text).toContain('[Needs reset]');
    expect(st.short.reason).toContain('already below Supertrend(10,3.0)');
  });

  it('solves oscillator thresholds on both sides (RSI below 30 / mirrored above 70)', () => {
    // A range-bound market: RSI hovers near 50, never reached 30 or 70, so no trade is open.
    const data = fromCloses(RANGE_BOUND);
    const settings = settingsFor('long-short');
    const g = genome([{ ind: 'RSI', period: 14, op: 'lt', level: 30 }], { archetype: 'MEAN_REVERSION', shorts: true });
    const f = solveNextEntry({ data, genome: g, settings });

    expect(f.position).toBeNull();
    expect(f.long.status).toBe('trigger');
    expect(f.short.status).toBe('trigger');
    expect(f.long.movePct).toBeLessThan(0); // oversold needs a lower close
    expect(f.short.movePct).toBeGreaterThan(0); // overbought needs a higher close
    verifyTrigger(data, g, settings, f.long);
    verifyTrigger(data, g, settings, f.short);
    // Short fill: sell at the open minus slippage; the stop sits above the fill.
    expect(f.short.fillPrice).toBeCloseTo((f.short.triggerPrice as number) * (1 - 0.0005), 6);
    expect(f.short.stopLoss).toBeGreaterThan(f.short.fillPrice as number);
    expect(f.long.conditions[0].text).toMatch(/^RSI\(14\) = [\d.]+, needs < 30 \([\d.]+ pts away\) \[Pending\]$/);
  });

  it('reports a conditional price when a non-price filter blocks the entry', () => {
    // A quiet market: ADX is low, so "ADX > 45" cannot be met by one more candle.
    const closes = Array.from({ length: 150 }, (_, i) => 100 + Math.sin(i / 2));
    const data = fromCloses(closes);
    const settings = settingsFor('long-only');
    const g = genome([
      { ind: 'PRICE_MA', maType: 'SMA', period: 20, op: 'cross_above' },
      { ind: 'ADX', period: 14, variant: 'strength', op: 'gt', level: 45 },
    ]);
    const f = solveNextEntry({ data, genome: g, settings });

    expect(f.long.status).toBe('blocked');
    expect(f.long.triggerPrice).not.toBeNull();
    expect(f.long.blockers).toHaveLength(1);
    expect(f.long.blockers[0].detail).toMatch(/^ADX\(14\) [\d.]+ < 45$/);
    expect(f.long.conditions[1].state).toBe('blocked');
    expect(f.long.conditions[1].text).toContain('[Blocked]');
    // The conditional price fires the price rule alone – the full rule stays blocked.
    const priceOnly: RuleGroup = { logic: 'AND', conditions: [g.longEntry.conditions[0]] };
    expect(firesOnLastBar(priceOnly, withNextBar(data, (f.long.triggerPrice as number) * (1 + 1e-9)))).toBe(true);
    expect(firesOnLastBar(g.longEntry, withNextBar(data, (f.long.triggerPrice as number) * (1 + 1e-9)))).toBe(false);
    expect(f.long.readiness).toBeLessThan(60);
  });

  it('treats volume spikes as a non-price blocker under an average-volume assumption', () => {
    // A slow decline: price sits below EMA(10), so the cross is reachable but the volume spike is not.
    const closes = Array.from({ length: 100 }, (_, i) => 110 - 0.1 * i + (i % 2 === 0 ? 0.05 : -0.05));
    const data = fromCloses(closes);
    const g = genome([
      { ind: 'PRICE_MA', maType: 'EMA', period: 10, op: 'cross_above' },
      { ind: 'VOLUME', period: 20, op: 'gt', level: 2 },
    ]);
    const f = solveNextEntry({ data, genome: g, settings: settingsFor('long-only') });
    expect(f.long.status).toBe('blocked');
    expect(f.long.blockers[0].detail).toContain('Volume_Ratio(20)');
    expect(f.notes.join(' ')).toContain('average volume');
  });

  it('reports an entry pending at the next open when the rule fired on the last candle', () => {
    // Flat, then a jump on the very last candle: Close crosses above SMA(20) right now.
    const closes = [...Array.from({ length: 60 }, () => 100), 104];
    const data = fromCloses(closes);
    const g = genome([{ ind: 'PRICE_MA', maType: 'SMA', period: 20, op: 'cross_above' }]);
    const f = solveNextEntry({ data, genome: g, settings: settingsFor('long-only') });
    expect(f.long.status).toBe('pending');
    expect(f.long.readiness).toBe(100);
    expect(f.long.fillPrice).toBeCloseTo(104 * 1.0005, 9);
    expect(f.verdict).toBe('Strong Long');
    expect(f.long.conditions[0].state).toBe('met');
    // Same answer as the backtester: one more candle opening at 104 enters long.
    const extended = withNextBar(data, 104);
    const ws = new BacktestWorkspace(extended.c.length);
    runBacktest(new IndicatorCache(extended), g, settingsFor('long-only'), ws);
    expect(ws.tradeEntryBar[ws.tradeCount - 1]).toBe(extended.c.length - 1);
  });

  it('switches to In Position with the live stop, target and time stop of the open trade', () => {
    // A steady uptrend: a Supertrend long is open at the end.
    const closes = Array.from({ length: 120 }, (_, i) => 100 + i * 0.5 + Math.sin(i));
    const data = fromCloses(closes);
    const g = genome([{ ind: 'SUPERTREND', period: 10, mult: 3, op: 'gt' }], {
      exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 3, trailing: true, takeProfitAtrMult: null, reversalExit: false, longExit: null, shortExit: null },
      maxBarsInTrade: 40,
    });
    const settings = settingsFor('long-only');
    const f = solveNextEntry({ data, genome: g, settings });
    expect(f.verdict).toBe('In Position');
    expect(f.long.status).toBe('holding');
    const p = f.position;
    expect(p?.dir).toBe('long');
    expect(p?.stopLoss).toBeLessThan(f.lastClose);
    expect(p?.takeProfit).toBeNull();
    expect(p?.timeStopIn).toBeGreaterThanOrEqual(0);
    expect(p?.timeStopIn).toBeLessThanOrEqual(40);
    // The stop is exactly the backtester's trailing stop.
    const ws = new BacktestWorkspace(data.c.length);
    runBacktest(new IndicatorCache(data), g, settings, ws);
    expect(p?.stopLoss).toBe(ws.openPosition?.stop);
    expect(p?.entryPrice).toBe(ws.openPosition?.entryPrice);
  });
});

describe('next-entry solver: parity with the backtester on random strategies', () => {
  const settings = settingsFor('long-short');
  const series = (symbol: string, interval: '1d' | '1h', range: '5y' | '2y'): OHLCV =>
    buildDataset({ [symbol]: generateSyntheticCandles(symbol, interval, range, Date.UTC(2026, 0, 15)) }, [{ symbol, weight: 1 }]).data;

  it('every solved trigger fires in backtester.ts, opens the trade and fills at the quoted price', () => {
    const data = series('BTC-USD', '1d', '5y');
    const rng = mulberry32(2026);
    // Volatility-scaled slippage on, so the fill check exercises the ATR-based model too.
    const atrSlip: BacktestSettings = { ...settings, slippageAtrMult: 0.1 };
    let verified = 0;
    let asymmetricShorts = 0;
    for (let i = 0; i < 160; i++) {
      // Mirrored and Asymmetric L/S alternate, so both short constructions are proven against the backtester.
      const mode = i % 4 < 2 ? 'long-short' : 'asymmetric';
      const g = randomGenome(rng, mode, DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      const base = i % 2 === 0 ? settings : atrSlip;
      const run: BacktestSettings = { ...base, tradeMode: mode };
      const f = solveNextEntry({ data, genome: g, settings: run });
      for (const side of [f.long, f.short]) {
        if (side.status === 'trigger' || side.status === 'now') {
          verifyTrigger(data, g, run, side);
          verified++;
          if (mode === 'asymmetric' && side.side === 'short') asymmetricShorts++;
        }
        expect(side.readiness).toBeGreaterThanOrEqual(0);
        expect(side.readiness).toBeLessThanOrEqual(100);
      }
    }
    expect(verified).toBeGreaterThan(40);
    // On a mostly rising market the bear-regime gate blocks most Asymmetric shorts; the ones that can fire are verified exactly.
    expect(asymmetricShorts).toBeGreaterThan(1);
  }, 180_000);

  it('the 2,000-candle window gives the same answers as the full 17,520-candle history', () => {
    const data = series('ETH-USD', '1h', '2y');
    const rng = mulberry32(77);
    let compared = 0;
    for (let i = 0; i < 25; i++) {
      const g = randomGenome(rng, 'long-short', DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      const windowed = solveNextEntry({ data, genome: g, settings });
      const full = solveNextEntry({ data, genome: g, settings, window: data.c.length });
      for (const side of ['long', 'short'] as const) {
        expect(windowed[side].status).toBe(full[side].status);
        if (full[side].triggerPrice !== null) {
          expect(windowed[side].triggerPrice).toBeCloseTo(full[side].triggerPrice as number, 4);
          compared++;
        }
      }
      expect(windowed.verdict).toBe(full.verdict);
    }
    expect(compared).toBeGreaterThan(10);
  }, 120_000);
});

describe('next-entry solver: readiness and verdicts', () => {
  const check = (state: 'met' | 'pending' | 'blocked' | 'reset', proximity: number) =>
    ({ state, proximity }) as unknown as SideForecast['conditions'][number];

  it('scores readiness from the checklist, normalised distance and price distance', () => {
    expect(readinessScore({ status: 'pending', conditions: [], triggerPrice: 100 }, 100, 2)).toBe(100);
    expect(readinessScore({ status: 'disabled', conditions: [check('met', 1)], triggerPrice: null }, 100, 2)).toBe(0);
    const near = readinessScore({ status: 'trigger', conditions: [check('pending', 0.8), check('met', 1)], triggerPrice: 101 }, 100, 2);
    const far = readinessScore({ status: 'trigger', conditions: [check('pending', 0.1), check('met', 1)], triggerPrice: 110 }, 100, 2);
    const blocked = readinessScore({ status: 'blocked', conditions: [check('pending', 0.8), check('blocked', 0.2)], triggerPrice: 101 }, 100, 2);
    expect(near).toBeGreaterThan(far);
    expect(near).toBeGreaterThan(blocked);
    expect(near).toBeLessThan(100);
  });

  it('derives the directional verdict', () => {
    expect(verdictFor(90, 10, true, false)).toBe('Strong Long');
    expect(verdictFor(10, 90, true, false)).toBe('Strong Short');
    expect(verdictFor(55, 40, true, false)).toBe('Leaning Long');
    expect(verdictFor(40, 55, true, false)).toBe('Leaning Short');
    expect(verdictFor(50, 48, true, false)).toBe('Neutral / Waiting');
    expect(verdictFor(20, 15, true, false)).toBe('Neutral / Waiting');
    expect(verdictFor(80, 0, false, false)).toBe('Strong Long');
    expect(verdictFor(50, 0, false, false)).toBe('Leaning Long');
    expect(verdictFor(99, 0, true, true)).toBe('In Position');
  });

  it('ranks assets by how close they are to an entry', () => {
    const side = (status: SideForecast['status'], movePct: number | null): SideForecast =>
      ({ status, movePct }) as SideForecast;
    const f = (long: SideForecast, short: SideForecast): NextEntryForecast => ({ long, short }) as NextEntryForecast;
    const pending = f(side('pending', 0), side('disabled', null));
    const near = f(side('trigger', 1.2), side('trigger', -3));
    const far = f(side('trigger', 6), side('unreachable', null));
    const blocked = f(side('blocked', 0.5), side('disabled', null));
    const order = [far, blocked, near, pending].sort((a, b) => proximityRank(a) - proximityRank(b));
    expect(order).toEqual([pending, near, far, blocked]);
  });
});
