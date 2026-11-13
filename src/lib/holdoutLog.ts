/**
 * Holdout log.
 *
 * A holdout is only unseen once. Every time a run ends, its holdout is revealed
 * and the dataset's fingerprint is counted here, so later runs on the same data
 * can warn that the "unseen" segment has in fact been looked at before.
 */

import type { HistoryRange, Interval } from '../types/trading';

const STORAGE_KEY = 'strategyfinder.holdoutViews.v1';

/** Symbols, interval, range and end date identify a dataset. */
export function datasetFingerprint(symbols: readonly string[], interval: Interval, range: HistoryRange, endTime: number): string {
  const endDate = Number.isFinite(endTime) ? new Date(endTime).toISOString().slice(0, 10) : 'unknown';
  return `${[...symbols].sort().join('+')}|${interval}|${range}|${endDate}`;
}

function readAll(): Record<string, number> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, number>) : {};
  } catch {
    return {}; // private mode, blocked storage or corrupt data: behave as if nothing was recorded
  }
}

/** Number of times the holdout of this dataset has been revealed so far. */
export function holdoutViews(fingerprint: string): number {
  const count = readAll()[fingerprint];
  return typeof count === 'number' && count > 0 ? Math.floor(count) : 0;
}

/** Record one more reveal and return the new total. */
export function recordHoldoutView(fingerprint: string): number {
  const all = readAll();
  const next = (typeof all[fingerprint] === 'number' ? all[fingerprint] : 0) + 1;
  all[fingerprint] = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    // Storage unavailable – the warning simply will not persist.
  }
  return next;
}
