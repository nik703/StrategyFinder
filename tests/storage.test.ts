/**
 * IndexedDB persistence: the ledger / runs / candles stores survive a reload, legacy localStorage keys
 * are migrated once, the reset empties every store, and snapshots round-trip between browsers
 * without ever erasing multiple-testing history.
 */

import { IDBFactory } from 'fake-indexeddb';
import { beforeEach, describe, expect, it } from 'vitest';
import { loadWithCache, mergeCandles, rangeStart, type CandleFetcher } from '../src/data/candleCache';
import { holdoutWindowFor, priorRevealCount, recordHoldoutReveal } from '../src/lib/holdoutLog';
import {
  clearAllRunData,
  exportSnapshot,
  flushWrites,
  importSnapshot,
  initStorage,
  readCandles,
  resetStorageForTests,
  savedRuns,
  saveRun,
  SNAPSHOT_FORMAT,
} from '../src/lib/storage';
import { extendWindowBudget, priorTrials, recordRunTrials, searchWindowFor, windowBudget } from '../src/lib/trialLedger';
import { INTERVAL_MS, WINDOW_TRIAL_BUDGET, type CandleSeries, type HistoryRange, type Interval } from '../src/types/trading';

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
const local = new MemoryStorage();
(globalThis as unknown as { window: { localStorage: MemoryStorage } }).window = { localStorage: local };

const DAY = 86_400_000;
const END = Date.UTC(2026, 10, 15);
const dailyTimes = (days: number): Float64Array => Float64Array.from({ length: days }, (_, i) => END - (days - 1 - i) * DAY);

let factory: IDBFactory;
/** Simulate a page reload: drop the in-memory mirror and open the same database again. */
async function reload(): Promise<void> {
  await flushWrites();
  resetStorageForTests();
  await initStorage(factory);
}

beforeEach(async () => {
  local.clear();
  resetStorageForTests();
  factory = new IDBFactory();
  const status = await initStorage(factory);
  expect(status.persistent).toBe(true);
});

describe('IndexedDB storage', () => {
  it('persists trial counts, budget extensions, holdout reveals and saved runs across reloads', async () => {
    const trialWindow = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    const holdoutWindow = holdoutWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    recordRunTrials(trialWindow, 'run-1', { count: 1200, mean: 0.1, m2: 3 }, 'TREND_FOLLOWING', END);
    extendWindowBudget(trialWindow, END);
    recordHoldoutReveal(holdoutWindow, 'S-9', END);
    saveRun({ id: 'run-1', savedAt: END, label: 'BTC 1d', interval: '1d', range: '5y', hypothesis: 'TREND_FOLLOWING', tradeMode: 'long-only', lockedId: 'S-9', strategies: [{ id: 'S-9', equity: new Float32Array([1, 2]) }] });

    await reload();
    expect(priorTrials(trialWindow).count).toBe(1200);
    expect(windowBudget(trialWindow)).toBeGreaterThan(WINDOW_TRIAL_BUDGET);
    expect(priorRevealCount(holdoutWindow)).toBe(1);
    const [run] = savedRuns();
    expect(run.lockedId).toBe('S-9');
    // Typed arrays are stored as plain JSON arrays.
    expect((run.strategies[0] as { equity: number[] }).equity).toEqual([1, 2]);
  });

  it('migrates the legacy localStorage keys on first load, once', async () => {
    const trialWindow = searchWindowFor(['ETH-USD'], '1d', dailyTimes(730));
    local.setItem('strategyfinder.trialLedger.v1', JSON.stringify([{ runId: 'legacy', ...trialWindow, count: 900, mean: 0, m2: 0, updatedAt: END }]));
    resetStorageForTests();
    const status = await initStorage(factory);
    expect(status.migrated).toBe(1);
    expect(local.getItem('strategyfinder.trialLedger.v1')).toBeNull();
    expect(priorTrials(trialWindow).count).toBe(900);
    // Persisted to IndexedDB: still there after another reload, with nothing left to migrate.
    await reload();
    expect(priorTrials(trialWindow).count).toBe(900);
    resetStorageForTests();
    expect((await initStorage(factory)).migrated).toBe(0);
  });

  it('empties every store on reset, in memory and in IndexedDB', async () => {
    const trialWindow = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    recordRunTrials(trialWindow, 'run-1', { count: 500, mean: 0, m2: 0 }, 'TREND_FOLLOWING', END);
    saveRun({ id: 'r', savedAt: END, label: '', interval: '1d', range: '5y', hypothesis: '', tradeMode: 'long-only', lockedId: null, strategies: [] });
    const fetcher: CandleFetcher = async (symbol) => series(symbol, 0, 10, '1d', 'binance');
    await loadWithCache('BTC-USD', '1d', '1y', fetcher, { now: 10 * DAY + 1 });
    expect(await readCandles('BTC-USD', '1d')).toHaveLength(1);

    clearAllRunData();
    expect(priorTrials(trialWindow).count).toBe(0);
    expect(savedRuns()).toHaveLength(0);
    await reload();
    expect(priorTrials(trialWindow).count).toBe(0);
    expect(savedRuns()).toHaveLength(0);
    expect(await readCandles('BTC-USD', '1d')).toHaveLength(0);
  });
});

describe('snapshots', () => {
  it('round-trips the ledger and the saved runs through JSON into another browser', async () => {
    const trialWindow = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    const holdoutWindow = holdoutWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    recordRunTrials(trialWindow, 'run-1', { count: 3000, mean: 0.2, m2: 5 }, 'TREND_FOLLOWING', END);
    recordHoldoutReveal(holdoutWindow, 'S-1', END);
    saveRun({ id: 'run-1', savedAt: END, label: 'x', interval: '1d', range: '5y', hypothesis: 'TREND_FOLLOWING', tradeMode: 'long-only', lockedId: 'S-1', strategies: [] });
    const text = JSON.stringify(exportSnapshot(END));
    expect(JSON.parse(text).format).toBe(SNAPSHOT_FORMAT);

    // Another browser: a fresh database.
    resetStorageForTests();
    factory = new IDBFactory();
    await initStorage(factory);
    expect(priorTrials(trialWindow).count).toBe(0);
    const summary = importSnapshot(text);
    expect(summary).toMatchObject({ trials: 1, holdouts: 1, runs: 1, skipped: 0 });
    await reload();
    expect(priorTrials(trialWindow).count).toBe(3000);
    expect(priorRevealCount(holdoutWindow)).toBe(1);
    expect(savedRuns().map((r) => r.id)).toEqual(['run-1']);
    // Importing the same file twice changes nothing.
    importSnapshot(text);
    expect(priorTrials(trialWindow).count).toBe(3000);
    expect(priorRevealCount(holdoutWindow)).toBe(1);
  });

  it('never lowers a trial count or erases a reveal, skips malformed records and refuses foreign files', () => {
    const trialWindow = searchWindowFor(['BTC-USD'], '1d', dailyTimes(1826));
    recordRunTrials(trialWindow, 'run-1', { count: 3000, mean: 0, m2: 0 }, 'TREND_FOLLOWING', END);
    const older = exportSnapshot(END);
    recordRunTrials(trialWindow, 'run-1', { count: 8000, mean: 0, m2: 0 }, 'TREND_FOLLOWING', END);
    const summary = importSnapshot({ ...older, ledger: [...older.ledger, { id: 'bad' }, null], runs: [{ nope: true }] });
    expect(priorTrials(trialWindow).count).toBe(8000);
    expect(summary.skipped).toBe(3);
    expect(() => importSnapshot({ hello: 'world' })).toThrow(/Not a StrategyFinder snapshot/);
    expect(() => importSnapshot({ ...older, version: 99 })).toThrow(/Unsupported snapshot version/);
    expect(() => importSnapshot('{oops')).toThrow();
  });
});

/* -------------------------------------------------------------------------- */
/*  Incremental candle cache                                                   */
/* -------------------------------------------------------------------------- */

/** `count` candles of `interval` opening from bar index `first`. */
function series(symbol: string, first: number, count: number, interval: Interval, source: 'binance' | 'yahoo'): CandleSeries {
  const step = INTERVAL_MS[interval];
  const t = Array.from({ length: count }, (_, i) => (first + i) * step);
  return { symbol, source, venueSymbol: symbol.replace('-', ''), t, o: t.map((x) => x / step), h: t.map((x) => x / step + 1), l: t.map((x) => x / step - 1), c: t.map((x) => x / step + 0.5), v: t.map(() => 10) } as CandleSeries;
}

describe('incremental candle cache', () => {
  const interval: Interval = '1d';
  const range: HistoryRange = '1y';
  const step = INTERVAL_MS[interval];

  /** A fake exchange with `available` closed daily candles, recording what was asked for. */
  function exchange() {
    const calls: (number | null)[] = [];
    let available = 0;
    const fetcher: CandleFetcher = async (symbol, iv, _range, since) => {
      calls.push(since);
      const first = since === null ? Math.max(0, available - 400) : since / step;
      return series(symbol, first, Math.max(0, available - first), iv, 'binance');
    };
    return { calls, fetcher, setAvailable: (n: number) => (available = n) };
  }

  it('downloads the whole range once, then only the candles that closed since', async () => {
    const ex = exchange();
    ex.setAvailable(500);
    // "now" is just after candle 499 closed (open 499·step, close 500·step).
    const first = await loadWithCache('BTC-USD', interval, range, ex.fetcher, { now: 500 * step + 1 });
    expect(first.mode).toBe('full');
    expect(ex.calls).toEqual([null]);
    expect(first.series.t[0]).toBeGreaterThanOrEqual(rangeStart(interval, range, 500 * step + 1));
    expect(first.series.t[first.series.t.length - 1]).toBe(499 * step);

    // Three more candles close: only they are downloaded.
    ex.setAvailable(503);
    const second = await loadWithCache('BTC-USD', interval, range, ex.fetcher, { now: 503 * step + 1 });
    expect(second.mode).toBe('delta');
    expect(ex.calls[1]).toBe(500 * step);
    expect(second.fetchedBars).toBe(3);
    expect(second.cachedBars).toBe(second.series.t.length - 3);
    expect(second.series.t[second.series.t.length - 1]).toBe(502 * step);
    // Contiguous, sorted, no duplicates.
    for (let i = 1; i < second.series.t.length; i++) expect(second.series.t[i] - second.series.t[i - 1]).toBe(step);

    // Nothing new has closed: no download at all.
    const third = await loadWithCache('BTC-USD', interval, range, ex.fetcher, { now: 503 * step + 2 });
    expect(third.mode).toBe('none');
    expect(ex.calls).toHaveLength(2);
    expect(third.series.t).toEqual(second.series.t);
  });

  it('never stores a candle that has not closed, and Force Refresh downloads the whole range again', async () => {
    const ex = exchange();
    ex.setAvailable(501); // candle 500 is still forming at `now`
    const out = await loadWithCache('ETH-USD', interval, range, ex.fetcher, { now: 500 * step + step / 2 });
    expect(out.series.t[out.series.t.length - 1]).toBe(499 * step);
    const [stored] = await readCandles('ETH-USD', interval);
    expect(stored.t[stored.t.length - 1]).toBe(499 * step);

    const forced = await loadWithCache('ETH-USD', interval, range, ex.fetcher, { now: 500 * step + step / 2, force: true });
    expect(forced.mode).toBe('full');
    expect(ex.calls).toEqual([null, null]);
  });

  it('merges by open time with the newer values winning', () => {
    const a = { t: [0, 1, 2], o: [1, 1, 1], h: [1, 1, 1], l: [1, 1, 1], c: [1, 1, 1], v: [1, 1, 1] };
    const b = { t: [2, 3, 4], o: [9, 9, 9], h: [9, 9, 9], l: [9, 9, 9], c: [9, 9, 9], v: [9, 9, 9] };
    const merged = mergeCandles(a, b, 4);
    expect(merged.t).toEqual([0, 1, 2, 3]);
    expect(merged.c).toEqual([1, 1, 9, 9]);
  });
});
