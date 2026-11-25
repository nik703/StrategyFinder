/**
 * The consolidated engine and app state: one composite fitness with the deflated Sharpe as a gate,
 * pooled multi-asset runs, the parsimony mix of the search, the one-step neighbourhood, the
 * next-entry exit levels and the "Reset Runs & Clear Cache" wipe.
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { createContext, fitnessOf, sampleOf, scoreAssets } from '../src/engine/evaluation';
import { countParameters, DEFAULT_LIMITS, FILTER_RATE, HYPOTHESES, MAX_CONDITIONS, MAX_PARAMETERS, mulberry32, randomGenome } from '../src/engine/genome';
import { exitLevels } from '../src/engine/nextEntrySolver';
import { holdoutWindowFor, priorReveals, recordHoldoutReveal, HOLDOUT_LOG_KEYS } from '../src/lib/holdoutLog';
import { clearRunStorage, RUN_STATE_KEYS, STORAGE_PREFIX } from '../src/lib/resetRuns';
import { addSavedTickers, loadSavedTickers } from '../src/lib/savedTickers';
import { savedRuns, saveRun } from '../src/lib/storage';
import { budgetStatus, extendWindowBudget, priorTrials, recordRunTrials, searchWindowFor, TRIAL_LEDGER_KEYS, windowBudget } from '../src/lib/trialLedger';
import { HYPOTHESIS_IDS, WINDOW_TRIAL_BUDGET, type BacktestSettings, type OHLCV, type StrategyGenome, type TradeMode } from '../src/types/trading';

/** A full in-memory Web Storage stand-in (with `length` and `key`, like the browser's). */
class MemoryStorage {
  private data = new Map<string, string>();
  get length(): number {
    return this.data.size;
  }
  key(index: number): string | null {
    return [...this.data.keys()][index] ?? null;
  }
  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.data.set(key, value);
  }
  removeItem(key: string): void {
    this.data.delete(key);
  }
  clear(): void {
    this.data.clear();
  }
}

const storage = new MemoryStorage();
(globalThis as unknown as { window: { localStorage: MemoryStorage } }).window = { localStorage: storage };
beforeEach(() => storage.clear());

const NOW = Date.UTC(2026, 0, 15);
const series = (symbol: string, interval: '1d' | '4h' = '1d', range: '2y' | '5y' = '5y'): OHLCV =>
  buildDataset({ [symbol]: generateSyntheticCandles(symbol, interval, range, NOW) }, [{ symbol, weight: 1 }]).data;

const SETTINGS: BacktestSettings = {
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  slippageAtrMult: 0.03,
  tradeMode: 'long-only',
  sizing: { mode: 'risk', riskPct: 1, compounding: false },
  fundingPctPerBar: 0,
};

describe('Reset Runs & Clear Cache', () => {
  const dailyTimes = (n: number): Float64Array => Float64Array.from({ length: n }, (_, i) => NOW - (n - 1 - i) * 86_400_000);

  it('wipes the trial ledger, budget extensions, holdout log and saved runs, and keeps the saved tickers', () => {
    const t = dailyTimes(1826);
    const trialWindow = searchWindowFor(['BTC-USD', 'ETH-USD'], '1d', t);
    const holdoutWindow = holdoutWindowFor(['BTC-USD', 'ETH-USD'], '1d', t);
    recordRunTrials(trialWindow, 'run-1', { count: 25_000, mean: 0.2, m2: 400 }, 'TREND_FOLLOWING', NOW);
    recordRunTrials(trialWindow, 'run-2', { count: 25_000, mean: 0.1, m2: 300 }, 'VOL_BREAKOUT', NOW);
    extendWindowBudget(trialWindow, NOW);
    recordHoldoutReveal(holdoutWindow, 'S-42', NOW);
    saveRun({ id: 'run-1', savedAt: NOW, label: 'x', interval: '1d', range: '5y', hypothesis: 'TREND_FOLLOWING', tradeMode: 'long-only', lockedId: null, strategies: [] });
    addSavedTickers([], ['PLTR']);
    storage.setItem('strategyfinder.someOldCache.v0', '{"stale":true}');
    storage.setItem('unrelated.app.key', 'keep me');

    expect(priorTrials(trialWindow).count).toBe(50_000);
    expect(windowBudget(trialWindow)).toBeGreaterThan(WINDOW_TRIAL_BUDGET);
    expect(budgetStatus(trialWindow, 50_000).level).not.toBe('fresh');
    expect(priorReveals(holdoutWindow)).toHaveLength(1);
    expect(savedRuns()).toHaveLength(1);

    const cleared = clearRunStorage();
    // Leftover localStorage keys of older versions are swept too; the saved tickers are kept.
    expect(cleared).toEqual(['strategyfinder.someOldCache.v0']);
    expect(storage.getItem('strategyfinder.someOldCache.v0')).toBeNull();
    // A clean 0-trial slate: no prior trials, the default budget, no recorded reveal, no saved run.
    expect(priorTrials(trialWindow)).toMatchObject({ count: 0, runs: 0 });
    expect(windowBudget(trialWindow)).toBe(WINDOW_TRIAL_BUDGET);
    expect(budgetStatus(trialWindow, 0)).toMatchObject({ level: 'fresh', used: 0, extensions: 0 });
    expect(priorReveals(holdoutWindow)).toHaveLength(0);
    expect(savedRuns()).toHaveLength(0);
    // Preferences and other apps' data survive.
    expect(loadSavedTickers()).toContain('PLTR');
    expect(storage.getItem('unrelated.app.key')).toBe('keep me');
    // A second reset has nothing left to clear.
    expect(clearRunStorage()).toEqual([]);
    expect(STORAGE_PREFIX).toBe('strategyfinder.');
    expect(HOLDOUT_LOG_KEYS.length + TRIAL_LEDGER_KEYS.length).toBe(RUN_STATE_KEYS.length);
  });

  it('clears the known keys from minimal or broken storage without throwing', () => {
    const minimal = new Map<string, string>([[RUN_STATE_KEYS[0], '[]']]);
    const stub = { getItem: (k: string) => minimal.get(k) ?? null, removeItem: (k: string) => void minimal.delete(k) };
    expect(clearRunStorage(stub)).toEqual([RUN_STATE_KEYS[0]]);
    expect(minimal.size).toBe(0);
    const broken = {
      getItem: (): string | null => {
        throw new Error('SecurityError');
      },
      removeItem: (): void => {
        throw new Error('SecurityError');
      },
      get length(): number {
        throw new Error('SecurityError');
      },
    };
    expect(clearRunStorage(broken)).toEqual([]);
    expect(clearRunStorage(null)).toEqual([]);
  });
});

describe('composite fitness with the deflated Sharpe as a gate', () => {
  it('scores a strategy the same whatever the luck hurdle; only its deflated Sharpe and gates move', () => {
    const ctx = createContext([{ symbol: 'BTC-USD', data: series('BTC-USD') }], SETTINGS, 64 * 1024 * 1024);
    const rng = mulberry32(8);
    let compared = 0;
    for (let i = 0; i < 60 && compared < 12; i++) {
      const g = randomGenome(rng, 'long-only', ctx.limits, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      const { scores, complete } = scoreAssets(ctx, g);
      if (!complete) continue;
      ctx.hurdle = 0;
      const easy = fitnessOf(ctx, scores);
      ctx.hurdle = 0.2; // a per-bar Sharpe hurdle far above anything daily data shows
      const hard = fitnessOf(ctx, scores);
      expect(hard.fitness).toBe(easy.fitness);
      expect(hard.dsr).toBeLessThanOrEqual(easy.dsr);
      if (easy.fitness > -1000) compared++;
    }
    expect(compared).toBeGreaterThan(3);
  }, 60_000);

  it('treats several assets as one pooled sample: effective independent bets, median score, median deflated Sharpe', () => {
    const datasets = ['BTC-USD', 'ETH-USD', 'SOL-USD'].map((symbol) => ({ symbol, data: series(symbol, '4h', '2y') }));
    const ctx = createContext(datasets, SETTINGS, 128 * 1024 * 1024);
    expect(ctx.pooled).toBe(true);
    expect(createContext(datasets.slice(0, 1), SETTINGS, 64 * 1024 * 1024).pooled).toBe(false);
    const rng = mulberry32(5);
    const g = randomGenome(rng, 'long-only', ctx.limits, 'VOL_BREAKOUT');
    const { scores } = scoreAssets(ctx, g);
    const sample = sampleOf(ctx, scores);
    const raw = scores.reduce((sum, score) => sum + score.metrics.search.trades, 0);
    expect(sample.rawTrades).toBe(raw);
    expect(sample.rawValidationTrades).toBe(scores.reduce((sum, score) => sum + score.metrics.validation.trades, 0));
    // Correlated assets trading at the same time are discounted: never more bets than trades, never
    // fewer than the busiest single asset contributes on its own.
    expect(sample.trades).toBeLessThanOrEqual(raw + 1e-9);
    expect(sample.trades).toBeGreaterThan(0);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) expect(ctx.correlation[a][b]).toBeCloseTo(ctx.correlation[b][a], 12);
  }, 60_000);
});

describe('parsimony of the search', () => {
  it('draws ~60% single-trigger and ~40% trigger + filter strategies, never more than 2 conditions or 5 parameters', () => {
    const rng = mulberry32(2026);
    for (const mode of ['long-only', 'asymmetric', 'long-short'] as TradeMode[]) {
      for (const h of HYPOTHESIS_IDS) {
        let single = 0;
        const n = 600;
        const params = new Map<number, number>();
        for (let i = 0; i < n; i++) {
          const g: StrategyGenome = randomGenome(rng, mode, DEFAULT_LIMITS, h);
          const conditions = g.longEntry.conditions.length;
          expect(conditions).toBeGreaterThanOrEqual(1);
          expect(conditions).toBeLessThanOrEqual(MAX_CONDITIONS);
          if (conditions === 1) single++;
          const p = countParameters(g, DEFAULT_LIMITS);
          // An Asymmetric short regime filter gets one slot of its own on top of the cap.
          expect(p).toBeLessThanOrEqual(MAX_PARAMETERS + (g.shortRegime ? 1 : 0));
          expect(p).toBeGreaterThanOrEqual(2);
          params.set(p, (params.get(p) ?? 0) + 1);
        }
        if (HYPOTHESES[h].filterRequired) {
          // A trend pullback IS a slow trend filter plus a short-term pullback trigger.
          expect(single).toBe(0);
        } else {
          // The parameter cap occasionally sheds a filter, so the single share sits a little above 1 − FILTER_RATE.
          expect(single / n).toBeGreaterThan(1 - FILTER_RATE - 0.08);
          expect(single / n).toBeLessThan(1 - FILTER_RATE + 0.15);
        }
        // Most strategies carry 3–5 tunable numbers.
        let mid = 0;
        for (const [p, count] of params) if (p >= 3 && p <= MAX_PARAMETERS + 1) mid += count;
        expect(mid / n).toBeGreaterThan(0.8);
      }
    }
  });
});

describe('next-entry exit levels', () => {
  const dynamic = (over: Partial<Extract<StrategyGenome['exit'], { mode: 'dynamic' }>>): StrategyGenome => ({
    archetype: 'TREND',
    longEntry: { logic: 'AND', conditions: [{ ind: 'DONCHIAN', period: 20, variant: 'upper', op: 'cross_above' }] },
    shortEntry: null,
    exit: { mode: 'dynamic', atrPeriod: 14, stopAtrMult: 2, trailing: false, takeProfitAtrMult: 4, reversalExit: false, longExit: null, shortExit: null, ...over },
    maxBarsInTrade: null,
  });

  it('quotes the initial stop, the target, the stop mode and the breakeven level of a new entry', () => {
    // Fixed ATR bracket: stop 2 ATR below, target 4 ATR above, no lock.
    expect(exitLevels(dynamic({}), 'long', 100, 2)).toEqual({ stopLoss: 96, takeProfit: 108, stopMode: 'fixed', trailAtrMult: null, breakevenAt: null });
    // Trailing stop without a target, breakeven lock at +1R; mirrored for a short.
    expect(exitLevels(dynamic({ trailing: true, takeProfitAtrMult: null, breakeven: true }), 'long', 100, 2)).toEqual({
      stopLoss: 96,
      takeProfit: null,
      stopMode: 'trailing',
      trailAtrMult: 2,
      breakevenAt: 104,
    });
    expect(exitLevels(dynamic({ trailing: true, takeProfitAtrMult: null, breakeven: true }), 'short', 100, 2)).toMatchObject({
      stopLoss: 104,
      breakevenAt: 96,
      stopMode: 'trailing',
    });
    // Static percentage bracket with a lock.
    const fixed: StrategyGenome = { ...dynamic({}), exit: { mode: 'static', stopLossPct: 5, takeProfitPct: 12, breakeven: true } };
    const levels = exitLevels(fixed, 'long', 200, 0);
    expect(levels.stopLoss).toBeCloseTo(190, 9);
    expect(levels.takeProfit).toBeCloseTo(224, 9);
    expect(levels.breakevenAt).toBeCloseTo(210, 9);
    expect(levels.stopMode).toBe('fixed');
    // Without an ATR on the signal candle an ATR exit has no levels yet.
    expect(exitLevels(dynamic({}), 'long', 100, Number.NaN)).toMatchObject({ stopLoss: null, takeProfit: null, stopMode: null, breakevenAt: null });
  });
});
