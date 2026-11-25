import { beforeEach, describe, expect, it } from 'vitest';
import {
  holdoutWindowFor,
  priorRevealCount,
  priorReveals,
  recordHoldoutReveal,
  windowsOverlap,
  type HoldoutWindow,
} from '../src/lib/holdoutLog';
import { initStorage, resetStorageForTests } from '../src/lib/storage';
import { INTERVAL_MS, RANGE_DAYS } from '../src/types/trading';

/** In-memory stand-in for window.localStorage (persists across "page loads" within a test). */
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
beforeEach(() => {
  storage.clear();
  resetStorageForTests();
});

const DAY = 86_400_000;
const END = Date.UTC(2026, 10, 15);

/** Daily candle open times covering `range` and ending on `end` (inclusive). */
function dailyTimes(range: keyof typeof RANGE_DAYS, end: number = END): Float64Array {
  const n = RANGE_DAYS[range];
  return Float64Array.from({ length: n }, (_, i) => end - (n - 1 - i) * DAY);
}

describe('holdout log', () => {
  it('derives the holdout window from the engine split', () => {
    const t = dailyTimes('5y');
    const w = holdoutWindowFor(['btc-usd', 'ETH-USD', 'BTC-USD'], '1d', t);
    expect(w.symbols).toEqual(['BTC-USD', 'ETH-USD']);
    expect(w.start).toBe(t[Math.floor(t.length * 0.8)]);
    expect(w.end).toBe(t[t.length - 1] + INTERVAL_MS['1d']);
  });

  it('persists reveals and does not reset the next day, on another range or on another timeframe', () => {
    const window5y = holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('5y'));
    expect(priorRevealCount(window5y)).toBe(0);
    recordHoldoutReveal(window5y, 'S-42', END);

    // A new end date (the next day): still the same holdout, almost entirely.
    expect(priorRevealCount(holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('5y', END + DAY)))).toBe(1);
    // A shorter range: its holdout is nested inside the 5y one.
    expect(priorRevealCount(holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('1y')))).toBe(1);
    expect(priorRevealCount(holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('2y')))).toBe(1);
    // Another timeframe over the same dates.
    const hourly = Float64Array.from({ length: 24 * 365 }, (_, i) => END - (24 * 365 - 1 - i) * 3_600_000);
    expect(priorRevealCount(holdoutWindowFor(['BTC-USD'], '1h', hourly))).toBe(1);
    // A universe that contains the revealed symbol.
    expect(priorRevealCount(holdoutWindowFor(['ETH-USD', 'BTC-USD'], '1d', dailyTimes('5y')))).toBe(1);

    const [entry] = priorReveals(window5y);
    expect(entry.strategyId).toBe('S-42');
    expect(entry.revealedAt).toBe(END);
  });

  it('does not count reveals of other symbols or of periods that do not overlap', () => {
    recordHoldoutReveal(holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('5y')), 'S-1', END);
    expect(priorRevealCount(holdoutWindowFor(['SPY'], '1d', dailyTimes('5y')))).toBe(0);
    // Two years later the old holdout lies entirely inside the new train segment.
    expect(priorRevealCount(holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('1y', END + 2 * 365 * DAY)))).toBe(0);
    const a: HoldoutWindow = { symbols: ['A'], interval: '1d', start: 0, end: 10 };
    expect(windowsOverlap(a, { ...a, start: 10, end: 20 })).toBe(false); // touching is not overlapping
    expect(windowsOverlap(a, { ...a, start: 9, end: 20 })).toBe(true);
  });

  it('counts every reveal and migrates the old end-date-keyed log into IndexedDB on first load', async () => {
    storage.setItem('strategyfinder.holdoutViews.v1', JSON.stringify({ 'BTC-USD+ETH-USD|1d|5y|2026-11-14': 3, garbage: 'x' }));
    // The first load (no IndexedDB here: the in-memory store) migrates the legacy localStorage key.
    expect((await initStorage(null)).migrated).toBe(1);
    const window5y = holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('5y'));
    expect(priorRevealCount(window5y)).toBe(3);
    expect(storage.getItem('strategyfinder.holdoutViews.v1')).toBeNull();
    recordHoldoutReveal(window5y, 'S-7', END);
    expect(priorRevealCount(window5y)).toBe(4);
  });

  it('drops corrupt legacy logs during the migration', async () => {
    const w = holdoutWindowFor(['BTC-USD'], '1d', dailyTimes('1y'));
    storage.setItem('strategyfinder.holdoutReveals.v2', '{not json');
    expect((await initStorage(null)).migrated).toBe(0);
    expect(priorRevealCount(w)).toBe(0);
    expect(storage.getItem('strategyfinder.holdoutReveals.v2')).toBeNull();
    resetStorageForTests();
    storage.setItem('strategyfinder.holdoutReveals.v2', JSON.stringify([{ symbols: 'BTC-USD', start: 1, end: 0 }, null]));
    expect((await initStorage(null)).migrated).toBe(0);
    expect(priorRevealCount(w)).toBe(0);
  });
});
