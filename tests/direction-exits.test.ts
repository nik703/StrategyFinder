import { describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { BacktestWorkspace, evaluateCondition, runBacktest } from '../src/engine/backtester';
import { describeArchetype, describeExit, shortRuleLabel } from '../src/engine/describe';
import {
  conformsTo,
  countParameters,
  DEFAULT_LIMITS,
  isShortRegime,
  MAX_PARAMETERS,
  mirrorCondition,
  mulberry32,
  mutateGenome,
  normalizeGenome,
  parameterAxes,
  randomGenome,
  stepNeighbours,
} from '../src/engine/genome';
import { atr, ema, IndicatorCache, keltner, pullbackTouch, rangeExpansion, squeezeRelease } from '../src/engine/indicators';
import { solveNextEntry } from '../src/engine/nextEntrySolver';
import { generatePineScript } from '../src/engine/pine';
import { emptyMetrics } from '../src/engine/backtester';
import {
  allowsShorts,
  HYPOTHESIS_IDS,
  TRADE_MODES,
  type BacktestSettings,
  type Condition,
  type OHLCV,
  type StrategyGenome,
  type TradeMode,
} from '../src/types/trading';

const DAY = 86_400_000;

const settingsFor = (tradeMode: TradeMode, costs = true): BacktestSettings => ({
  initialCapital: 10_000,
  feePct: costs ? 0.1 : 0,
  slippagePct: costs ? 0.05 : 0,
  tradeMode,
  sizing: { mode: 'full', riskPct: 2, compounding: true },
  fundingPctPerBar: 0,
});

const synthetic = (symbol: string): OHLCV =>
  buildDataset({ [symbol]: generateSyntheticCandles(symbol, '1d', '5y', Date.UTC(2026, 0, 15)) }, [{ symbol, weight: 1 }]).data;

/** Candles from explicit [open, high, low, close] rows. */
function candles(rows: [number, number, number, number][]): OHLCV {
  const n = rows.length;
  return {
    t: Float64Array.from({ length: n }, (_, i) => i * DAY),
    o: Float64Array.from(rows.map((r) => r[0])),
    h: Float64Array.from(rows.map((r) => r[1])),
    l: Float64Array.from(rows.map((r) => r[2])),
    c: Float64Array.from(rows.map((r) => r[3])),
    v: new Float64Array(n).fill(1000),
  };
}

/** A flat range at 100, then one breakout candle that closes at 101 (bar 20). */
function breakoutThen(after: [number, number, number, number][]): OHLCV {
  const flat: [number, number, number, number][] = Array.from({ length: 20 }, () => [100, 100.2, 99.8, 100]);
  return candles([...flat, [100, 101.1, 99.9, 101], ...after]);
}

const BREAKOUT: Condition = { ind: 'DONCHIAN', period: 5, variant: 'upper', op: 'cross_above' };

const hand = (exit: StrategyGenome['exit']): StrategyGenome => ({
  archetype: 'TREND',
  longEntry: { logic: 'AND', conditions: [BREAKOUT] },
  shortEntry: null,
  exit,
  maxBarsInTrade: null,
});

describe('direction modes', () => {
  it('lists Long-Only, Asymmetric and Mirrored modes; only Long-Only never shorts', () => {
    expect(TRADE_MODES).toEqual(['long-only', 'asymmetric', 'long-short']);
    expect(TRADE_MODES.map(allowsShorts)).toEqual([false, true, true]);
  });

  it('builds Asymmetric shorts from the mirrored trigger AND an independent, tunable bear-regime filter', () => {
    const rng = mulberry32(7);
    let independent = 0;
    for (let i = 0; i < 400; i++) {
      const h = HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length];
      const g = randomGenome(rng, 'asymmetric', DEFAULT_LIMITS, h);
      expect(conformsTo(g)).toBe(true);
      const regime = g.shortRegime as Condition;
      expect(isShortRegime(regime)).toBe(true);
      expect(['lt']).toContain(regime.op);
      const [trigger, longFilter] = g.longEntry.conditions;
      expect(g.shortEntry?.conditions).toEqual([mirrorCondition(trigger), regime]);
      // The short regime is its own gene: it is not simply the mirror of the long filter.
      if (!longFilter || JSON.stringify(mirrorCondition(longFilter)) !== JSON.stringify(regime)) independent++;
      // It is a tunable parameter of its own, with one extra slot on top of the cap.
      expect(countParameters(g)).toBeLessThanOrEqual(MAX_PARAMETERS + 1);
      expect(parameterAxes(g, DEFAULT_LIMITS).some((axis) => axis.label.startsWith('Short regime'))).toBe(true);
      // Normalisation is idempotent and keeps the regime; Long-Only and Mirrored drop it.
      expect(normalizeGenome(g, 'asymmetric', DEFAULT_LIMITS)).toEqual(g);
      expect(normalizeGenome(g, 'long-only', DEFAULT_LIMITS).shortRegime).toBeUndefined();
      expect(normalizeGenome(g, 'long-only', DEFAULT_LIMITS).shortEntry).toBeNull();
      expect(normalizeGenome(g, 'long-short', DEFAULT_LIMITS).shortRegime).toBeUndefined();
    }
    expect(independent).toBeGreaterThan(200);
  });

  it('mutates the short regime, and the plateau check steps it like any other parameter', () => {
    const rng = mulberry32(11);
    const g = randomGenome(rng, 'asymmetric', DEFAULT_LIMITS, 'VOL_BREAKOUT');
    const regimes = new Set<string>();
    for (let i = 0; i < 300; i++) regimes.add(JSON.stringify(mutateGenome(rng, g, 'asymmetric', DEFAULT_LIMITS).shortRegime));
    expect(regimes.size).toBeGreaterThan(3);
    const moved = stepNeighbours(g, 'asymmetric', DEFAULT_LIMITS).filter((n) => JSON.stringify(n.shortRegime) !== JSON.stringify(g.shortRegime));
    expect(moved.length).toBeGreaterThan(0);
  });

  it('only opens Asymmetric shorts inside the bear regime, and far fewer of them than Mirrored mode on a rising market', () => {
    const data = synthetic('BTC-USD');
    const ind = new IndicatorCache(data);
    const n = data.c.length;
    const ws = new BacktestWorkspace(n);
    const rng = mulberry32(2026);
    let asymmetricShorts = 0;
    let mirroredShorts = 0;
    for (let i = 0; i < 60; i++) {
      const g = randomGenome(rng, 'asymmetric', DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      runBacktest(ind, g, settingsFor('asymmetric'), ws);
      const regime = new Uint8Array(n);
      evaluateCondition(g.shortRegime as Condition, ind, regime, n);
      for (let k = 0; k < ws.tradeCount; k++) {
        if (ws.tradeDir[k] !== -1) continue;
        asymmetricShorts++;
        const entry = ws.tradeEntryBar[k];
        // The signal candle (one before the fill) was inside the bear regime – unless the backtest
        // closed the position at its last candle, which is the liquidation of a trade opened earlier.
        expect(regime[entry - 1]).toBe(1);
      }
      const mirrored = normalizeGenome(g, 'long-short', DEFAULT_LIMITS);
      runBacktest(ind, mirrored, settingsFor('long-short'), ws);
      for (let k = 0; k < ws.tradeCount; k++) if (ws.tradeDir[k] === -1) mirroredShorts++;
    }
    expect(asymmetricShorts).toBeGreaterThan(0);
    expect(asymmetricShorts).toBeLessThan(mirroredShorts * 0.75);
  });

  it('exports, describes and forecasts all three modes', () => {
    const rng = mulberry32(5);
    const g = randomGenome(rng, 'asymmetric', DEFAULT_LIMITS, 'TREND_FOLLOWING');
    const settings = settingsFor('asymmetric');
    expect(shortRuleLabel(g)).toBe('Asymmetric L/S');
    expect(shortRuleLabel(normalizeGenome(g, 'long-short', DEFAULT_LIMITS))).toBe('Mirrored L/S');
    expect(shortRuleLabel(normalizeGenome(g, 'long-only', DEFAULT_LIMITS))).toBe('Long-Only');
    expect(describeArchetype(g)).toContain('bear-market filter');
    const script = generatePineScript(
      { id: 'S-1', genome: g, dsr: 0, holdout: null, train: emptyMetrics(), validation: emptyMetrics(), search: emptyMetrics() } as never,
      {
        label: 'BTC-USD',
        assets: ['BTC-USD'],
        interval: '1d',
        settings,
        components: [{ symbol: 'BTC-USD', weight: 1 }],
        units: [],
        windowStart: 0,
        windowEnd: DAY * 400,
        bars: 400,
        benchmarkRoi: 0,
        venue: 'BTCUSDT on Binance',
        simulatedData: true,
      } as never,
    );
    expect(script).toContain('Mode         : Asymmetric Long/Short');
    expect(script).toContain('// Short entry (mirrored trigger, only inside its own bear-regime filter)');
    expect(script).toMatch(/shortEntry = shortCond1 and shortCond2/);
    expect(script).toContain('if goShort and strategy.position_size >= 0');
    // The solver forecasts the short side from the full Asymmetric rule, regime included.
    const data = synthetic('ETH-USD');
    const f = solveNextEntry({ data, genome: g, settings });
    expect(f.short.status).not.toBe('disabled');
    expect(f.short.conditions).toHaveLength(2);
    expect(f.short.conditions[1].job).toBe('Regime filter');
    const longOnly = solveNextEntry({ data, genome: normalizeGenome(g, 'long-only', DEFAULT_LIMITS), settings: settingsFor('long-only') });
    expect(longOnly.short.status).toBe('disabled');
  });
});

describe('dynamic trade management', () => {
  it('moves the stop to the entry price once the trade has been 1R in profit (breakeven lock)', () => {
    // Entry at 101 (open of bar 21); 2% stop = 98.98 (1R = 2.02); bar 21 reaches 103.5 > 101 + 1R.
    const data = breakoutThen([
      [101, 103.5, 100.8, 103],
      [103, 103.1, 100.5, 100.7], // dips to 100.5: below the breakeven stop (101), above the original stop
      [100.7, 100.9, 99.5, 100],
    ]);
    const exit = { mode: 'static', stopLossPct: 2, takeProfitPct: 10 } as const;
    const settings = settingsFor('long-only', false);
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);

    runBacktest(ind, hand({ ...exit, breakeven: true }), settings, ws);
    expect(ws.tradeCount).toBe(1);
    expect(ws.tradeEntryBar[0]).toBe(21);
    expect(ws.tradeExitBar[0]).toBe(22);
    expect(ws.tradePnl[0]).toBeCloseTo(0, 9); // out at the entry price: neither won nor lost
    // Without the lock the same trade rides out the dip and is still open at the end.
    runBacktest(ind, hand({ ...exit, breakeven: false }), settings, ws);
    expect(ws.tradeExitBar[0]).toBe(data.c.length - 1);

    // While the trade is open the workspace reports 1R and the lock state.
    runBacktest(ind, hand({ ...exit, breakeven: true }), settings, ws, 22);
    expect(ws.openPosition?.risk).toBeCloseTo(2.02, 9);
    expect(ws.openPosition?.breakevenLocked).toBe(true);
    expect(ws.openPosition?.stop).toBeCloseTo(101, 9);
  });

  it('does not lock below +1R, and never lowers a trailing stop that is already above the entry', () => {
    const data = breakoutThen([
      [101, 102.9, 100.8, 102.5], // +1.9: short of 1R (2.02)
      [102.5, 102.6, 99.5, 99.8], // dips under the entry: no lock, so no exit at 101
      [99.8, 100, 99.4, 99.6],
    ]);
    const settings = settingsFor('long-only', false);
    const ws = new BacktestWorkspace(data.c.length);
    runBacktest(new IndicatorCache(data), hand({ mode: 'static', stopLossPct: 2, takeProfitPct: 10, breakeven: true }), settings, ws, 23);
    expect(ws.openPosition?.breakevenLocked).toBe(false);
    expect(ws.openPosition?.stop).toBeCloseTo(98.98, 9);

    // Trailing + breakeven: the stop only ever ratchets up, and after the lock it is at least the entry.
    const rising = Array.from({ length: 40 }, (_, i): [number, number, number, number] => {
      const c = 101 + i * 1.2;
      return [c - 1, c + 0.6, c - 1.4, c];
    });
    const trend = breakoutThen(rising);
    const trailExit = {
      mode: 'dynamic',
      atrPeriod: 14,
      stopAtrMult: 2,
      trailing: true,
      takeProfitAtrMult: null,
      reversalExit: false,
      longExit: null,
      shortExit: null,
      breakeven: true,
    } as const;
    const ind = new IndicatorCache(trend);
    const a = atr(trend.h, trend.l, trend.c, 14);
    let previous = -Infinity;
    for (let bars = 23; bars <= trend.c.length; bars++) {
      runBacktest(ind, hand(trailExit), settings, ws, bars);
      const open = ws.openPosition;
      expect(open).not.toBeNull();
      if (!open) break;
      expect(open.stop).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = open.stop;
      if (open.breakevenLocked) expect(open.stop).toBeGreaterThanOrEqual(open.entryPrice);
      // The trail is the highest high since entry minus 2 ATR of the last closed candle (or higher).
      let peak = open.entryPrice;
      for (let i = open.entryBar; i < bars; i++) peak = Math.max(peak, trend.h[i]);
      expect(open.stop).toBeGreaterThanOrEqual(peak - 2 * a[bars - 1] - 1e-9);
    }
  });

  it('shows the breakeven level of an open trade in the Next Entry forecast, and when it has locked', () => {
    const exit = { mode: 'static', stopLossPct: 2, takeProfitPct: 10, breakeven: true } as const;
    const settings = settingsFor('long-only', false);
    const slice = (data: OHLCV, bars: number): OHLCV => ({
      t: data.t.slice(0, bars), o: data.o.slice(0, bars), h: data.h.slice(0, bars), l: data.l.slice(0, bars), c: data.c.slice(0, bars), v: data.v.slice(0, bars),
    });
    // Short of +1R: the forecast quotes the price that will lock the stop at the entry.
    const waiting = solveNextEntry({ data: slice(breakoutThen([[101, 102.9, 100.8, 102.5]]), 22), genome: hand(exit), settings });
    expect(waiting.verdict).toBe('In Position');
    expect(waiting.position?.breakevenLocked).toBe(false);
    expect(waiting.position?.breakevenAt).toBeCloseTo(101 + 2.02, 9);
    expect(waiting.position?.stopLoss).toBeCloseTo(98.98, 9);
    // +1R reached: the stop now sits at the entry.
    const locked = solveNextEntry({ data: slice(breakoutThen([[101, 103.5, 100.8, 103]]), 22), genome: hand(exit), settings });
    expect(locked.position?.breakevenLocked).toBe(true);
    expect(locked.position?.breakevenAt).toBeNull();
    expect(locked.position?.stopLoss).toBeCloseTo(101, 9);
  });

  it('describes the breakeven lock and drops it when the target is inside 1R', () => {
    const g = normalizeGenome(
      { ...hand({ mode: 'static', stopLossPct: 4, takeProfitPct: 3, breakeven: true }), hypothesis: 'VOL_BREAKOUT' },
      'long-only',
      DEFAULT_LIMITS,
    );
    expect(g.exit.breakeven).toBe(false);
    const kept = normalizeGenome(
      { ...hand({ mode: 'static', stopLossPct: 2, takeProfitPct: 10, breakeven: true }), hypothesis: 'VOL_BREAKOUT' },
      'long-only',
      DEFAULT_LIMITS,
    );
    expect(kept.exit.breakeven).toBe(true);
    const text = describeExit(kept, settingsFor('long-only'));
    expect(text.bullets.map((b) => b.title)).toContain('Breakeven lock');
  });
});

describe('volatility and pullback primitives', () => {
  it('computes the Keltner channel as EMA ± mult × ATR', () => {
    const data = synthetic('SOL-USD');
    const kc = keltner(data.h, data.l, data.c, 20, 1.5);
    const mid = ema(data.c, 20);
    const a = atr(data.h, data.l, data.c, 20);
    for (const i of [30, 400, 1500]) {
      expect(kc.middle[i]).toBeCloseTo(mid[i], 9);
      expect(kc.upper[i] - kc.middle[i]).toBeCloseTo(1.5 * a[i], 9);
      expect(kc.middle[i] - kc.lower[i]).toBeCloseTo(1.5 * a[i], 9);
    }
  });

  it('fires a squeeze release when the Bollinger bands leave a Keltner channel they were inside', () => {
    // 40 candles of a quiet trend (bands inside the channel), then a violent breakout.
    const rows: [number, number, number, number][] = [];
    for (let i = 0; i < 40; i++) {
      const c = 100 + i * 0.05;
      rows.push([c - 0.05, c + 0.6, c - 0.6, c]);
    }
    rows.push([102, 108, 101.9, 107.5]);
    const data = candles(rows);
    const s = squeezeRelease(data.h, data.l, data.c, 20, 1.5);
    const fired = [...s.keys()].filter((i) => s[i] !== 0 && s[i] === s[i]);
    expect(fired).toEqual([40]);
    expect(s[40]).toBe(1); // released upwards, above the basis
    // The search's condition fires on exactly that candle, and its mirror on none.
    const ind = new IndicatorCache(data);
    const out = new Uint8Array(rows.length);
    evaluateCondition({ ind: 'SQUEEZE', period: 20, mult: 1.5, op: 'gt', level: 0.5 }, ind, out);
    expect([...out.keys()].filter((i) => out[i] === 1)).toEqual([40]);
    evaluateCondition(mirrorCondition({ ind: 'SQUEEZE', period: 20, mult: 1.5, op: 'gt', level: 0.5 }), ind, out);
    expect(out.every((v) => v === 0)).toBe(true);
  });

  it('flags range-expansion candles and pullbacks that hold their average', () => {
    const rows: [number, number, number, number][] = Array.from({ length: 30 }, () => [100, 100.5, 99.5, 100]);
    rows.push([100, 104, 99.8, 103.8]); // true range 4.2 vs ATR ≈ 1: an up expansion
    rows.push([103.8, 104, 99, 99.2]); // true range 5 vs ATR ≈ 1.1: a down expansion
    const data = candles(rows);
    const rx = rangeExpansion(data.h, data.l, data.c, 14, 2);
    expect(rx[30]).toBe(1);
    expect(rx[31]).toBe(-1);
    expect(rx[20]).toBe(0);

    const ma = Float64Array.from({ length: 4 }, () => 100);
    const touch = pullbackTouch(ma, Float64Array.from([101, 101, 100.5, 99.8]), Float64Array.from([99.5, 100.4, 99.9, 99]), Float64Array.from([100.6, 100.9, 99.7, 99.5]));
    // Dipped to the average and closed above it; stayed above it; rallied to it and closed below; below all candle.
    expect([...touch]).toEqual([1, 0, -1, 0]);
  });
});
