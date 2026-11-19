/**
 * Holdout log.
 *
 * A holdout is only unseen once. Every time the user locks a strategy and
 * reveals its holdout, the holdout's symbols and DATE WINDOW are recorded in
 * localStorage, so the log survives page reloads and browser restarts.
 *
 * A later run counts as having seen its holdout before when an earlier reveal
 * shares at least one symbol AND its date window overlaps. The log is therefore
 * not fooled by the things that used to reset it: a new end date the next day,
 * a different range (the 1y / 2y / 5y holdouts are nested), another timeframe
 * over the same dates, or adding a symbol to the pool.
 */

import { splitFor } from '../engine/backtester';
import { INTERVAL_MS, RANGE_DAYS, type HistoryRange, type Interval } from '../types/trading';

const STORAGE_KEY = 'strategyfinder.holdoutReveals.v2';
/** Pre-v2 log: reveal counts keyed by `symbols|interval|range|endDate`. Migrated on first read. */
const LEGACY_KEY = 'strategyfinder.holdoutViews.v1';
/** Every localStorage key of the holdout log (cleared by "Reset Runs & Clear Cache"). */
export const HOLDOUT_LOG_KEYS: readonly string[] = [STORAGE_KEY, LEGACY_KEY];
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
  const all = readAll();
  all.push({ ...window, symbols: normaliseSymbols(window.symbols), revealedAt: now, strategyId, views: 1 });
  persist(all);
  return all;
}

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
  let entries: HoldoutReveal[] = [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) {
      entries = parsed.filter(isReveal).map((entry) => ({
        ...entry,
        interval: entry.interval in INTERVAL_MS ? entry.interval : '1d',
        strategyId: typeof entry.strategyId === 'string' ? entry.strategyId : null,
        views: typeof entry.views === 'number' && entry.views > 0 ? Math.floor(entry.views) : 1,
      }));
    }
  } catch {
    return []; // private mode, blocked storage or corrupt data: behave as if nothing was recorded
  }
  const migrated = migrateLegacy();
  if (migrated.length > 0) {
    entries = [...entries, ...migrated];
    persist(entries);
  }
  return entries;
}

function persist(entries: HoldoutReveal[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage unavailable – the warning simply will not persist.
  }
}

/**
 * The v1 log only kept `symbols|interval|range|endDate → count`. The holdout was the last 20% of
 * the range, so its window can be reconstructed closely enough for overlap checks.
 */
function migrateLegacy(): HoldoutReveal[] {
  try {
    const raw = window.localStorage.getItem(LEGACY_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    const out: HoldoutReveal[] = [];
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
    window.localStorage.removeItem(LEGACY_KEY);
    return out;
  } catch {
    return [];
  }
}
