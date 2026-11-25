/**
 * Stateful Next Entry solver: the solver carries the engine's full state vector (open position with
 * its trailing anchor, breakeven lock and age; squeeze run; time-stop cooldown) into the hypothetical
 * next candle, and reports the blockers no next-candle close can remove.
 */

import { describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { BacktestWorkspace, runBacktest } from '../src/engine/backtester';
import { DEFAULT_LIMITS, normalizeGenome } from '../src/engine/genome';
import { IndicatorCache } from '../src/engine/indicators';
import { solveNextEntry, solverState } from '../src/engine/nextEntrySolver';
import type { BacktestSettings, OHLCV, StrategyGenome, TradeMode } from '../src/types/trading';

const NOW = Date.UTC(2026, 0, 15);
const full: OHLCV = buildDataset({ BTC: generateSyntheticCandles('BTC-USD', '1d', '2y', NOW) }, [{ symbol: 'BTC', weight: 1 }]).data;

const settingsFor = (tradeMode: TradeMode): BacktestSettings => ({
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  slippageAtrMult: 0.02,
  tradeMode,
  sizing: { mode: 'risk', riskPct: 1, compounding: false },
  fundingPctPerBar: 0,
});

/** The first `n` candles. */
function head(data: OHLCV, n: number): OHLCV {
  return { t: data.t.slice(0, n), o: data.o.slice(0, n), h: data.h.slice(0, n), l: data.l.slice(0, n), c: data.c.slice(0, n), v: data.v.slice(0, n) };
}

/** The candles plus one flat candle at the last close (the solver's "closes where it is" candle). */
function withFlatCandle(data: OHLCV): OHLCV {
  const n = data.c.length;
  const grow = (src: Float64Array, value: number): Float64Array => {
    const out = new Float64Array(n + 1);
    out.set(src);
    out[n] = value;
    return out;
  };
  const last = data.c[n - 1];
  return { t: grow(data.t, 2 * data.t[n - 1] - data.t[n - 2]), o: grow(data.o, last), h: grow(data.h, last), l: grow(data.l, last), c: grow(data.c, last), v: grow(data.v, data.v[n - 1]) };
}

const trigger: StrategyGenome['longEntry'] = {
  logic: 'AND',
  conditions: [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 10, slowType: 'EMA', slowPeriod: 30, op: 'cross_above' }],
};

function genomeWith(exit: StrategyGenome['exit'], mode: TradeMode, extra: Partial<StrategyGenome> = {}): StrategyGenome {
  return normalizeGenome({ archetype: 'TREND', hypothesis: 'TREND_FOLLOWING', longEntry: trigger, shortEntry: null, exit, maxBarsInTrade: null, ...extra }, mode, DEFAULT_LIMITS);
}

const dynamicExit = (trailing: boolean, breakeven: boolean): StrategyGenome['exit'] => ({
  mode: 'dynamic',
  atrPeriod: 14,
  stopAtrMult: 2,
  trailing,
  takeProfitAtrMult: null,
  reversalExit: false,
  longExit: null,
  shortExit: null,
  breakeven,
});

describe('stateful Next Entry solver', () => {
  it('projects the stop the backtester actually sets after the next close – trailing anchors and breakeven locks carried over', () => {
    let compared = 0;
    let moved = 0;
    for (const mode of ['long-only', 'long-short'] as TradeMode[]) {
      for (const [trailing, breakeven] of [
        [true, false],
        [true, true],
        [false, true],
      ] as const) {
        const genome = genomeWith(dynamicExit(trailing, breakeven), mode);
        const settings = settingsFor(mode);
        // Cut the history at many points and compare whenever a position is open at the cut.
        for (let n = 120; n < full.c.length; n += 7) {
          const data = head(full, n);
          const forecast = solveNextEntry({ data, genome, settings });
          if (!forecast.position) continue;
          const extended = withFlatCandle(data);
          const ws = new BacktestWorkspace(n + 1).recordLevels();
          runBacktest(new IndicatorCache(extended), genome, settings, ws);
          const levels = ws.levels as NonNullable<typeof ws.levels>;
          // Closed or reversed at that open (time stop, signal, reversal – the solver says so): nothing to project.
          if (forecast.position.exitAtNextOpen !== null) {
            expect(levels.dir[n] === 0 || levels.entry[n] !== forecast.position.entryPrice).toBe(true);
            continue;
          }
          expect(levels.entry[n]).toBe(forecast.position.entryPrice);
          expect(forecast.position.projectedStop).not.toBeNull();
          expect(forecast.position.projectedStop as number).toBeCloseTo(levels.stop[n], 9);
          // The stop shown for the next candle is the one in force now.
          expect(forecast.position.stopLoss as number).toBeCloseTo(levels.stop[n - 1], 9);
          if (Math.abs(levels.stop[n] - levels.stop[n - 1]) > 1e-9) moved++;
          compared++;
        }
      }
    }
    expect(compared).toBeGreaterThan(40);
    expect(moved).toBeGreaterThan(3); // the carried trailing anchor really moved the projected stop
  });

  it('carries the position, its trailing anchor and its age in the state vector', () => {
    const genome = genomeWith(dynamicExit(true, true), 'long-only');
    const settings = settingsFor('long-only');
    for (let n = 150; n < full.c.length; n += 11) {
      const data = head(full, n);
      const ind = new IndicatorCache(data);
      const ws = new BacktestWorkspace(n);
      runBacktest(ind, genome, settings, ws);
      const state = solverState(ind, ws, genome);
      if (!state.position) continue;
      expect(state.position.barsHeld).toBe(n - state.position.entryBar);
      // The trailing anchor is the best high since entry.
      let best = -Infinity;
      for (let i = state.position.entryBar; i < n; i++) best = Math.max(best, data.h[i]);
      expect(state.position.extreme).toBeLessThanOrEqual(best + 1e-9);
      expect(state.position.extreme).toBeGreaterThanOrEqual(state.position.entryPrice * 0.9);
      return;
    }
    throw new Error('no open position found at any cut');
  });

  it('reports filters no next-candle close can change next to the trigger price', () => {
    // A volume filter: the solver's next candle carries the average volume, so "volume > 3× its
    // average" stays false whatever the close – a non-price blocker.
    const genome = genomeWith(dynamicExit(false, false), 'long-only', {
      longEntry: { logic: 'AND', conditions: [trigger.conditions[0], { ind: 'VOLUME', period: 20, op: 'gt', level: 3 }] },
    });
    let reported = 0;
    for (let n = 120; n < full.c.length; n += 13) {
      const forecast = solveNextEntry({ data: head(full, n), genome, settings: settingsFor('long-only') });
      if (forecast.position) continue;
      const volume = forecast.long.nonPriceBlockers.find((b) => /vol/i.test(b.formula));
      expect(volume, JSON.stringify(forecast.long.nonPriceBlockers)).toBeDefined();
      expect(volume?.detail).toMatch(/no next-candle close can change it/);
      reported++;
    }
    expect(reported).toBeGreaterThan(10);
  });

  it('reports the time-stop cooldown when the position closes at the next open', () => {
    const genome = genomeWith(dynamicExit(false, false), 'long-only', { maxBarsInTrade: 5 });
    const settings = settingsFor('long-only');
    let found = 0;
    for (let n = 100; n < full.c.length; n++) {
      const data = head(full, n);
      const ind = new IndicatorCache(data);
      const ws = new BacktestWorkspace(n);
      runBacktest(ind, genome, settings, ws);
      const state = solverState(ind, ws, genome);
      if (!state.timeStopAtNextOpen) continue;
      const forecast = solveNextEntry({ data, genome, settings });
      expect(forecast.long.nonPriceBlockers.map((b) => b.formula)).toContain('Time-stop cooldown');
      if (++found >= 3) break;
    }
    expect(found).toBeGreaterThan(0);
  });
});
