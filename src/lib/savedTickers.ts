/**
 * Saved tickers.
 *
 * Custom tickers the user has run (or confirmed with Enter) are remembered in
 * localStorage and offered as one-click chips and autocomplete suggestions, so
 * they never have to be typed twice. Storage failures (private mode, blocked
 * storage) are ignored: the list then simply lasts for the session.
 */

import { TICKER_PATTERN } from '../data/basket';
import { PRESET_TICKERS } from './config';

const STORAGE_KEY = 'strategyfinder.savedTickers.v1';
/** Oldest entries are dropped beyond this many. */
export const MAX_SAVED_TICKERS = 12;

const isPreset = (symbol: string): boolean => (PRESET_TICKERS as readonly string[]).includes(symbol);

function normalise(symbol: string): string | null {
  const ticker = symbol.trim().toUpperCase();
  return TICKER_PATTERN.test(ticker) && !isPreset(ticker) ? ticker : null;
}

/** Saved tickers in the order they were first saved. */
export function loadSavedTickers(): string[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    const out: string[] = [];
    for (const entry of parsed) {
      const ticker = typeof entry === 'string' ? normalise(entry) : null;
      if (ticker && !out.includes(ticker)) out.push(ticker);
    }
    return out.slice(-MAX_SAVED_TICKERS);
  } catch {
    return [];
  }
}

function persist(list: string[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    // Storage unavailable – keep the list in memory only.
  }
}

/**
 * Add tickers to the saved list. Presets, invalid symbols and duplicates are
 * ignored; existing entries keep their place so the chips do not jump around.
 */
export function addSavedTickers(current: readonly string[], symbols: readonly string[]): string[] {
  const next = [...current];
  for (const symbol of symbols) {
    const ticker = normalise(symbol);
    if (ticker && !next.includes(ticker)) next.push(ticker);
  }
  if (next.length === current.length) return current as string[];
  const capped = next.slice(-MAX_SAVED_TICKERS);
  persist(capped);
  return capped;
}

export function removeSavedTicker(current: readonly string[], symbol: string): string[] {
  const next = current.filter((ticker) => ticker !== symbol);
  persist(next);
  return next;
}
