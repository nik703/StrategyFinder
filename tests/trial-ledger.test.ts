import { beforeEach, describe, expect, it } from 'vitest';
import { RunningStats } from '../src/lib/stats';
import {
  budgetStatus,
  estimateSearchWindow,
  MIN_WINDOW_OVERLAP,
  priorTrials,
  recordRunTrials,
  searchWindowFor,
  windowsMatch,
  type TrialWindow,
} from '../src/lib/trialLedger';
import { WINDOW_TRIAL_BUDGET } from '../src/types/trading';

/** In-memory stand-in for window.localStorage – it survives "page reloads" because the module re-reads it every call. */
class MemoryStorage {
  private data = new Map<string, string>();
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

const DAY = 86_400_000;
const END = Date.UTC(2026, 10, 15);

function dailyTimes(days: number, end: number = END): Float64Array {
  return Float64Array.from({ length: days }, (_, i) => end - (days - 1 - i) * DAY);
}

const stats = (values: number[]): { count: number; mean: number; m2: number } => {
  const s = new RunningStats();
  values.forEach((v) => s.push(v));
  return { count: s.count, mean: s.mean, m2: s.m2 };
};

describe('trial ledger', () => {
  it('uses the search period (train + validation) of the engine split as the window', () => {
    const t = dailyTimes(1826);
    const w = searchWindowFor(['btc-usd', 'ETH-USD'], '1d', t);
    expect(w.symbols).toEqual(['BTC-USD', 'ETH-USD']);
    expect(w.start).toBe(t[0]);
    expect(w.end).toBe(t[Math.floor(1826 * 0.8)]);
    // The pre-download estimate lands on (nearly) the same window.
    const estimate = estimateSearchWindow(['BTC-USD', 'ETH-USD'], '1d', '5y', END + DAY);
    expect(windowsMatch(estimate, w)).toBe(true);
    expect(Math.abs(estimate.end - w.end)).toBeLessThan(3 * DAY);
  });

  it('matches overlapping windows: next day, nested ranges, other timeframes, shared symbols', () => {
    const base = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    expect(windowsMatch(base, searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826, END + DAY)))).toBe(true); // the next day
    expect(windowsMatch(base, searchWindowFor(['BTC-USD'], '1d', dailyTimes(730)))).toBe(true); // 2y nested in 5y
    expect(windowsMatch(base, searchWindowFor(['BTC-USD'], '4h', Float64Array.from({ length: 6 * 1826 }, (_, i) => END - (6 * 1826 - 1 - i) * DAY / 6)))).toBe(true);
    expect(windowsMatch(base, searchWindowFor(['ETH-USD', 'BTC-USD'], '1d', dailyTimes(1826)))).toBe(true); // a universe containing it
    expect(windowsMatch(base, searchWindowFor(['SPY'], '1d', dailyTimes(1826)))).toBe(false); // other symbols
    // Less than half of the shorter window in common: different data.
    const a: TrialWindow = { symbols: ['X'], interval: '1d', start: 0, end: 100 };
    expect(windowsMatch(a, { ...a, start: 49, end: 149 })).toBe(true);
    expect(windowsMatch(a, { ...a, start: 51, end: 151 })).toBe(false);
    expect(MIN_WINDOW_OVERLAP).toBe(0.5);
  });

  it('persists trials across reloads and pools them over every matching run', () => {
    const window = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    expect(priorTrials(window)).toMatchObject({ count: 0, runs: 0 });
    const first = stats([0.5, -0.2, 1.1, 0.3]);
    recordRunTrials(window, 'run-1', first, 'TREND_FOLLOWING', END);
    // A "reload": the module keeps nothing in memory, so this is read back from storage.
    expect(priorTrials(window)).toMatchObject({ count: 4, runs: 1 });

    // The same run reporting progress again replaces its record – it is never double-counted.
    const firstLater = stats([0.5, -0.2, 1.1, 0.3, 0.9, -0.4]);
    recordRunTrials(window, 'run-1', firstLater, 'TREND_FOLLOWING', END + 1);
    expect(priorTrials(window).count).toBe(6);

    // A second run on an overlapping window (one day later, 2y range) adds to the same pool.
    const second = stats([0.1, 0.2, -0.7]);
    recordRunTrials(searchWindowFor(['BTC-USD'], '1d', dailyTimes(730, END + DAY)), 'run-2', second, 'RANGE_REVERSION', END + DAY);
    const pooled = priorTrials(window);
    expect(pooled.count).toBe(9);
    expect(pooled.runs).toBe(2);
    const all = stats([0.5, -0.2, 1.1, 0.3, 0.9, -0.4, 0.1, 0.2, -0.7]);
    expect(pooled.mean).toBeCloseTo(all.mean, 12);
    expect(pooled.m2).toBeCloseTo(all.m2, 12);
    // A run can exclude its own record (the worker adds its live trials itself).
    expect(priorTrials(window, 'run-2').count).toBe(6);
    // Other assets are untouched.
    expect(priorTrials(searchWindowFor(['SPY'], '1d', dailyTimes(1826))).count).toBe(0);
  });

  it('reports the budget level and the luck hurdle, and ignores corrupt storage', () => {
    const window = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    expect(WINDOW_TRIAL_BUDGET).toBe(10_000);
    expect(budgetStatus(window, 0).level).toBe('fresh');
    expect(budgetStatus(window, 5_000).level).toBe('degrading');
    expect(budgetStatus(window, 9_000).level).toBe('critical');
    const spent = budgetStatus(window, 10_000);
    expect(spent).toMatchObject({ level: 'exhausted', remaining: 0, share: 1 });
    // The luck hurdle rises with every trial.
    expect(budgetStatus(window, 2_000).luckSharpe).toBeGreaterThan(budgetStatus(window, 200).luckSharpe);
    expect(budgetStatus(window, 2_000).luckSharpe).toBeGreaterThan(1);
    storage.setItem('strategyfinder.trialLedger.v1', '{not json');
    expect(priorTrials(window).count).toBe(0);
    storage.setItem('strategyfinder.trialLedger.v1', JSON.stringify([{ runId: 'x', symbols: ['BTC-USD'], start: 1, end: 0, count: 5 }, null]));
    expect(priorTrials(window).count).toBe(0);
  });
});
