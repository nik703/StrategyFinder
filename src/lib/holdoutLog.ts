/**
 * Holdout log.
 *
 * A holdout is only unseen once. Every time the user locks a strategy and
 * reveals its holdout, the holdout's symbols and DATE WINDOW are recorded in the
 * IndexedDB ledger store (see `storage.ts`), so the log survives page reloads and
 * browser restarts, and travels with an exported snapshot.
 *
 * A later run counts as having seen its holdout before when an earlier reveal
 * shares at least one symbol AND its date window overlaps. The log is therefore
 * not fooled by the things that used to reset it: a new end date the next day,
 * a different range (the 1y / 2y / 5y holdouts are nested), another timeframe
 * over the same dates, or adding a symbol to the pool.
 */

import { splitFor } from '../engine/backtester';
import { LEGACY_KEYS, ledgerRecords, putLedgerRecords, registerLegacyMigration, type LedgerRecord } from './storage';
import { INTERVAL_MS, RANGE_DAYS, type HistoryRange, type Interval } from '../types/trading';

/** Legacy localStorage keys of the holdout log (migrated into IndexedDB on first load). */
export const HOLDOUT_LOG_KEYS: readonly string[] = [LEGACY_KEYS.holdouts, LEGACY_KEYS.holdoutsV1];
const DAY_MS = 86_400_000;

/** The market period a run's holdout covers. */
export interface HoldoutWindow {
  /** Every symbol whose holdout the run can show (every traded asset), upper case, sorted. */
  symbols: string[];
  interval: Interval;
  /** Open time of the first holdout candle (ms). */
  start: number;
  /** End of the last holdout candle (ms, exclusive). */
  end: number;
}

export interface HoldoutReveal extends HoldoutWindow {
  /** When the holdout was revealed (ms). */
  revealedAt: number;
  /** Strategy that was locked, when known. */
  strategyId: string | null;
  /** How many reveals this record stands for (legacy records can hold several). */
  views: number;
}

/** Holdout window of a dataset with candle open times `t`, using the engine's own split. */
export function holdoutWindowFor(symbols: readonly string[], interval: Interval, t: ArrayLike<number>): HoldoutWindow {
  const n = t.length;
  const { holdoutStart } = splitFor(n);
  return {
    symbols: normaliseSymbols(symbols),
    interval,
    start: n > 0 ? t[Math.min(holdoutStart, n - 1)] : 0,
    end: n > 0 ? t[n - 1] + INTERVAL_MS[interval] : 0,
  };
}

/** True when two windows share a symbol and their date ranges overlap. */
export function windowsOverlap(a: Pick<HoldoutWindow, 'symbols' | 'start' | 'end'>, b: Pick<HoldoutWindow, 'symbols' | 'start' | 'end'>): boolean {
  if (!(a.start < b.end && b.start < a.end)) return false;
  return a.symbols.some((symbol) => b.symbols.includes(symbol));
}

/** Every earlier reveal whose holdout overlaps `window`, oldest first. */
export function priorReveals(window: HoldoutWindow): HoldoutReveal[] {
  return readAll()
    .filter((entry) => windowsOverlap(entry, window))
    .sort((a, b) => a.revealedAt - b.revealedAt);
}

/** Total number of earlier reveals of an overlapping holdout. */
export function priorRevealCount(window: HoldoutWindow): number {
  return priorReveals(window).reduce((sum, entry) => sum + entry.views, 0);
}

/** Record one reveal of `window` and return the updated log. */
export function recordHoldoutReveal(window: HoldoutWindow, strategyId: string | null, now: number = Date.now()): HoldoutReveal[] {
  putLedgerRecords([revealEntry({ ...window, symbols: normaliseSymbols(window.symbols), revealedAt: now, strategyId, views: 1 })]);
  return readAll();
}

let revealSerial = 0;
const revealEntry = (reveal: HoldoutReveal): LedgerRecord<HoldoutReveal> => ({
  id: `holdout:${reveal.revealedAt}:${reveal.strategyId ?? '-'}:${reveal.symbols.join('+')}:${reveal.start}:${revealSerial++}`,
  kind: 'holdout',
  data: reveal,
});

/* -------------------------------------------------------------------------- */

function normaliseSymbols(symbols: readonly string[]): string[] {
  return [...new Set(symbols.map((symbol) => symbol.trim().toUpperCase()).filter(Boolean))].sort();
}

function isReveal(value: unknown): value is HoldoutReveal {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Record<string, unknown>;
  return (
    Array.isArray(r.symbols) &&
    r.symbols.every((symbol) => typeof symbol === 'string') &&
    typeof r.start === 'number' &&
    typeof r.end === 'number' &&
    typeof r.revealedAt === 'number' &&
    r.end > r.start
  );
}

function readAll(): HoldoutReveal[] {
  return normaliseReveals(ledgerRecords<unknown>('holdout').map((record) => record.data));
}

function normaliseReveals(parsed: unknown[]): HoldoutReveal[] {
  return parsed.filter(isReveal).map((entry) => ({
    ...entry,
    interval: entry.interval in INTERVAL_MS ? entry.interval : '1d',
    strategyId: typeof entry.strategyId === 'string' ? entry.strategyId : null,
    views: typeof entry.views === 'number' && entry.views > 0 ? Math.floor(entry.views) : 1,
  }));
}

/**
 * Legacy localStorage → ledger records (run once by `initStorage`): the v2 log as is, and the v1
 * log, which only kept `symbols|interval|range|endDate → count`. The holdout was the last 20% of the
 * range, so its window can be reconstructed closely enough for overlap checks.
 */
registerLegacyMigration((storage) => {
  const out: HoldoutReveal[] = [];
  try {
    const parsed: unknown = JSON.parse(storage.getItem(LEGACY_KEYS.holdouts) ?? '[]');
    if (Array.isArray(parsed)) out.push(...normaliseReveals(parsed));
  } catch {
    // corrupt v2 log: dropped
  }
  try {
    const raw = storage.getItem(LEGACY_KEYS.holdoutsV1);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (typeof parsed === 'object' && parsed !== null) {
      for (const [key, count] of Object.entries(parsed as Record<string, unknown>)) {
        const [symbols, interval, range, endDate] = key.split('|');
        const end = Date.parse(`${endDate}T00:00:00Z`);
        if (!symbols || !(interval in INTERVAL_MS) || !(range in RANGE_DAYS) || !Number.isFinite(end) || typeof count !== 'number' || count < 1) continue;
        const span = RANGE_DAYS[range as HistoryRange] * DAY_MS;
        out.push({
          symbols: normaliseSymbols(symbols.split('+')),
          interval: interval as Interval,
          start: end + DAY_MS - 0.2 * span,
          end: end + DAY_MS,
          revealedAt: end,
          strategyId: null,
          views: Math.floor(count),
        });
      }
    }
  } catch {
    // corrupt v1 log: dropped
  }
  return out.map(revealEntry);
});
