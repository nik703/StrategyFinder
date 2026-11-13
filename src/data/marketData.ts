/**
 * Client-side market data loader.
 *
 * Tries the Netlify Function first and transparently falls back to the built-in
 * synthetic generator when the function is unreachable (plain `vite preview`,
 * offline work) or an upstream API fails or rate-limits.
 */

import type { CandleSeries, HistoryRange, Interval, MarketDataResponse } from '../types/trading';
import { generateSyntheticCandles } from './synthetic';

/** Override with VITE_MARKET_DATA_URL to point a local build at a deployed function. */
const ENDPOINT: string =
  (import.meta.env.VITE_MARKET_DATA_URL as string | undefined) ?? '/.netlify/functions/market-data';

const REQUEST_TIMEOUT_MS = 30_000;
/** Fewer candles than this is treated as a failed download. */
const MIN_CANDLES = 60;

export interface LoadedMarketData {
  series: Record<string, CandleSeries>;
  /** True when the synthetic fallback generator supplied the candles. */
  usedFallback: boolean;
  /** Why the fallback was used (null when live data loaded). */
  fallbackReason: string | null;
  /** Notes passed through from the data provider (e.g. truncated intraday history). */
  notes: string[];
}

function isCandleSeries(value: unknown): value is CandleSeries {
  if (typeof value !== 'object' || value === null) return false;
  const s = value as Record<string, unknown>;
  const columns = [s.t, s.o, s.h, s.l, s.c, s.v];
  if (!columns.every(Array.isArray)) return false;
  const length = (s.t as unknown[]).length;
  return columns.every((column) => (column as unknown[]).length === length);
}

async function fetchSymbol(symbol: string, interval: Interval, range: HistoryRange): Promise<CandleSeries> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const query = new URLSearchParams({ symbols: symbol, interval, range });
    const response = await fetch(`${ENDPOINT}?${query.toString()}`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) {
      // e.g. the SPA fallback page when no function runtime is available.
      throw new Error('market-data function is not available');
    }
    const body = (await response.json()) as Partial<MarketDataResponse> & { error?: string };
    const series = body.series?.[symbol];
    if (!response.ok || !isCandleSeries(series)) {
      throw new Error(body.errors?.[symbol] ?? body.error ?? `HTTP ${response.status}`);
    }
    if (series.t.length < MIN_CANDLES) throw new Error(`only ${series.t.length} candles returned`);
    return series;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('request timed out');
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Load candles for every symbol. One request per symbol keeps each response
 * small and lets the downloads run in parallel.
 *
 * Live and synthetic data are never mixed: if any symbol fails, the whole
 * request is served by the fallback generator so a basket stays coherent.
 */
export async function loadMarketData(
  symbols: readonly string[],
  interval: Interval,
  range: HistoryRange,
): Promise<LoadedMarketData> {
  const unique = [...new Set(symbols)];
  const settled = await Promise.allSettled(unique.map((symbol) => fetchSymbol(symbol, interval, range)));

  const series: Record<string, CandleSeries> = {};
  const failures: string[] = [];
  const notes: string[] = [];
  settled.forEach((outcome, index) => {
    const symbol = unique[index];
    if (outcome.status === 'fulfilled') {
      series[symbol] = outcome.value;
      if (outcome.value.note) notes.push(`${symbol}: ${outcome.value.note}`);
    } else {
      const reason = outcome.reason instanceof Error ? outcome.reason.message : String(outcome.reason);
      failures.push(`${symbol}: ${reason}`);
    }
  });

  if (failures.length === 0) return { series, usedFallback: false, fallbackReason: null, notes };

  const now = Date.now();
  const synthetic: Record<string, CandleSeries> = {};
  for (const symbol of unique) synthetic[symbol] = generateSyntheticCandles(symbol, interval, range, now);
  return { series: synthetic, usedFallback: true, fallbackReason: failures.join(' · '), notes: [] };
}
