/**
 * "Reset Runs & Clear Cache": wipe every piece of run state StrategyFinder keeps in the browser, so
 * the next run starts from a clean slate with 0 trials.
 *
 * Persistent state removed here:
 *   • the IndexedDB ledger store: the trial ledger – every trial counted against each data window –
 *     its budget extensions and the holdout log of every recorded reveal;
 *   • the IndexedDB saved runs and the candle cache;
 *   • any `strategyfinder.*` localStorage key except the saved tickers, so state written by older
 *     versions goes too.
 * The saved tickers are kept: they are a preference, not run data. The App resets the in-memory
 * state (leaderboard, robust archive, elite, restart and trial counters) and terminates the workers.
 *
 * Clearing the ledger and the log resets multiple-testing tracking: earlier trials stop raising the
 * luck hurdle of the deflated Sharpe, and holdouts revealed before look untouched again. The UI asks
 * for confirmation and says so before calling this.
 */

import { HOLDOUT_LOG_KEYS } from './holdoutLog';
import { clearAllRunData } from './storage';
import { SAVED_TICKERS_KEY } from './savedTickers';
import { TRIAL_LEDGER_KEYS } from './trialLedger';

/** Prefix of every localStorage key StrategyFinder writes (today only the saved tickers). */
export const STORAGE_PREFIX = 'strategyfinder.';

/** The run-state localStorage keys of earlier versions (migrated to IndexedDB on first load). */
export const RUN_STATE_KEYS: readonly string[] = [...TRIAL_LEDGER_KEYS, ...HOLDOUT_LOG_KEYS];

/** The part of the Web Storage API the reset needs (`length` / `key` are optional for minimal stubs). */
export interface ResettableStorage {
  removeItem(key: string): void;
  getItem(key: string): string | null;
  readonly length?: number;
  key?(index: number): string | null;
}

function browserStorage(): ResettableStorage | null {
  try {
    return typeof window !== 'undefined' && window.localStorage ? window.localStorage : null;
  } catch {
    return null; // storage blocked
  }
}

/**
 * Remove every run-state key from `storage` (default: the browser's localStorage) and return the
 * keys that held data. Never throws: blocked storage simply has nothing to clear.
 */
export function clearRunStorage(storage: ResettableStorage | null = browserStorage()): string[] {
  clearAllRunData();
  if (!storage) return [];
  const keys = new Set(RUN_STATE_KEYS);
  try {
    const count = typeof storage.length === 'number' && typeof storage.key === 'function' ? storage.length : 0;
    for (let i = 0; i < count; i++) {
      const key = storage.key?.(i);
      if (key && key.startsWith(STORAGE_PREFIX) && key !== SAVED_TICKERS_KEY) keys.add(key);
    }
  } catch {
    // Enumeration unavailable: the known keys are still removed below.
  }
  const cleared: string[] = [];
  for (const key of keys) {
    try {
      if (storage.getItem(key) !== null) cleared.push(key);
      storage.removeItem(key);
    } catch {
      // A key that cannot be removed is left alone.
    }
  }
  return cleared.sort();
}
