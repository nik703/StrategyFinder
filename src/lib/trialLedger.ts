/**
 * Trial ledger: every strategy evaluated on a data window, across runs and sessions.
 *
 * The deflated Sharpe ratio and the haircut Sharpe only protect against luck if they know how
 * many strategies were really tried on the data. A run that started from zero every time would let
 * a user run the optimiser ten times and pick the luckiest run. So every run records its trials
 * here – count, mean and spread of their annualised Sharpe ratios – in localStorage, keyed by the
 * symbols and the dates of the SEARCH period (train + validation). A later run whose search period
 * overlaps an earlier one (on a shared symbol, by at least half of the shorter window) inherits
 * those trials: they raise its selection-bias hurdle and count against the window's budget of
 * `WINDOW_TRIAL_BUDGET` evaluations. Refreshing the page, a new end date, another range or another
 * timeframe over the same dates does not reset the count.
 */

import { splitFor } from '../engine/backtester';
import { expectedMaxSharpe, RunningStats } from './stats';
import {
  INTERVAL_MS,
  RANGE_DAYS,
  TRAIN_FRACTION,
  VALIDATION_FRACTION,
  WINDOW_TRIAL_BUDGET,
  type HistoryRange,
  type Interval,
  type TrialStats,
} from '../types/trading';

const STORAGE_KEY = 'strategyfinder.trialLedger.v1';
const DAY_MS = 86_400_000;
const YEAR_MS = 365.25 * DAY_MS;
/** Oldest records are dropped beyond this many. */
const MAX_RECORDS = 500;
/** Two search windows are the same data when they overlap by at least this share of the shorter one. */
export const MIN_WINDOW_OVERLAP = 0.5;

/** The search period (train + validation) of a run. */
export interface TrialWindow {
  /** Every symbol the run scored strategies on (traded assets and universe), upper case, sorted. */
  symbols: string[];
  interval: Interval;
  /** Open time of the first candle (ms). */
  start: number;
  /** Open time of the first holdout candle (ms, exclusive). */
  end: number;
}

export interface TrialRecord extends TrialWindow, TrialStats {
  runId: string;
  hypothesis: string | null;
  updatedAt: number;
}

/** Search window of a dataset with candle open times `t`, using the engine's own split. */
export function searchWindowFor(symbols: readonly string[], interval: Interval, t: ArrayLike<number>): TrialWindow {
  const n = t.length;
  const { holdoutStart } = splitFor(n);
  return {
    symbols: normaliseSymbols(symbols),
    interval,
    start: n > 0 ? t[0] : 0,
    end: n > 0 ? (holdoutStart < n ? t[holdoutStart] : t[n - 1] + INTERVAL_MS[interval]) : 0,
  };
}

/** Best guess of the search window before the data is downloaded: the last `range` up to `now`. */
export function estimateSearchWindow(symbols: readonly string[], interval: Interval, range: HistoryRange, now: number): TrialWindow {
  const span = RANGE_DAYS[range] * DAY_MS;
  const start = now - span;
  return { symbols: normaliseSymbols(symbols), interval, start, end: start + span * (TRAIN_FRACTION + VALIDATION_FRACTION) };
}

/** True when two windows share a symbol and overlap by at least `MIN_WINDOW_OVERLAP` of the shorter one. */
export function windowsMatch(a: TrialWindow, b: TrialWindow): boolean {
  if (!a.symbols.some((symbol) => b.symbols.includes(symbol))) return false;
  const overlap = Math.min(a.end, b.end) - Math.max(a.start, b.start);
  const shorter = Math.min(a.end - a.start, b.end - b.start);
  return shorter > 0 && overlap >= MIN_WINDOW_OVERLAP * shorter;
}

/** Every recorded run whose window matches `window`, oldest first. */
export function matchingTrials(window: TrialWindow, excludeRunId: string | null = null): TrialRecord[] {
  return readAll()
    .filter((record) => record.runId !== excludeRunId && windowsMatch(record, window))
    .sort((a, b) => a.updatedAt - b.updatedAt);
}

/** Pooled trial statistics (annualised Sharpe) of every earlier run on this window. */
export function priorTrials(window: TrialWindow, excludeRunId: string | null = null): TrialStats & { runs: number } {
  const records = matchingTrials(window, excludeRunId);
  const pooled = new RunningStats();
  for (const record of records) pooled.merge(record);
  return { count: pooled.count, mean: pooled.mean, m2: pooled.m2, runs: records.length };
}

/** Create or update this run's record. Called as the run progresses, so a reload mid-run keeps its trials. */
export function recordRunTrials(window: TrialWindow, runId: string, stats: TrialStats, hypothesis: string | null, now: number = Date.now()): void {
  if (!(stats.count > 0)) return;
  const all = readAll().filter((record) => record.runId !== runId);
  all.push({ ...window, symbols: normaliseSymbols(window.symbols), runId, hypothesis, updatedAt: now, count: stats.count, mean: stats.mean, m2: stats.m2 });
  persist(all.slice(-MAX_RECORDS));
}

export type BudgetLevel = 'fresh' | 'degrading' | 'critical' | 'exhausted';

export interface BudgetStatus {
  used: number;
  limit: number;
  remaining: number;
  /** used / limit, capped at 1. */
  share: number;
  /**
   * fresh      – under half the budget;
   * degrading  – half or more: the luck hurdle is high, significance is eroding;
   * critical   – 90% or more;
   * exhausted  – no more search on this window.
   */
  level: BudgetLevel;
  /** Annualised Sharpe the luckiest of `used` skill-less strategies is expected to reach on this window. */
  luckSharpe: number;
}

/** Budget status of a window after `used` trials. */
export function budgetStatus(window: TrialWindow, used: number, limit: number = WINDOW_TRIAL_BUDGET): BudgetStatus {
  const share = limit > 0 ? Math.min(1, used / limit) : 1;
  const years = Math.max(1 / 365, (window.end - window.start) / YEAR_MS);
  // Under the null the annualised Sharpe estimate over `years` has a variance of about 1 / years.
  const luckSharpe = used > 1 ? expectedMaxSharpe(used, 1 / years) : 0;
  const level: BudgetLevel = used >= limit ? 'exhausted' : share >= 0.9 ? 'critical' : share >= 0.5 ? 'degrading' : 'fresh';
  return { used, limit, remaining: Math.max(0, limit - used), share, level, luckSharpe };
}

/* -------------------------------------------------------------------------- */

function normaliseSymbols(symbols: readonly string[]): string[] {
  return [...new Set(symbols.map((symbol) => symbol.trim().toUpperCase()).filter(Boolean))].sort();
}

function isRecord(value: unknown): value is TrialRecord {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Record<string, unknown>;
  return (
    typeof r.runId === 'string' &&
    Array.isArray(r.symbols) &&
    r.symbols.every((symbol) => typeof symbol === 'string') &&
    typeof r.start === 'number' &&
    typeof r.end === 'number' &&
    r.end > r.start &&
    typeof r.count === 'number' &&
    r.count > 0 &&
    typeof r.mean === 'number' &&
    Number.isFinite(r.mean) &&
    typeof r.m2 === 'number' &&
    Number.isFinite(r.m2)
  );
}

function readAll(): TrialRecord[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecord).map((record) => ({
      ...record,
      interval: record.interval in INTERVAL_MS ? record.interval : '1d',
      hypothesis: typeof record.hypothesis === 'string' ? record.hypothesis : null,
      updatedAt: typeof record.updatedAt === 'number' ? record.updatedAt : 0,
      count: Math.floor(record.count),
      m2: Math.max(0, record.m2),
    }));
  } catch {
    return []; // private mode, blocked storage or corrupt data: behave as if nothing was recorded
  }
}

function persist(records: TrialRecord[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Storage unavailable – the budget simply will not persist.
  }
}
