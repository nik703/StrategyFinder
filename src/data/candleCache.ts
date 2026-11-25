/**
 * Incremental historical candle cache.
 *
 * Closed candles are stored in IndexedDB keyed by (exchange, symbol, timeframe) (see `storage.ts`).
 * A later run on the same symbol and timeframe downloads only the candles that closed after the last
 * stored one and merges them in, instead of the whole history:
 *
 *   • the cache is used when it covers the start of the requested range (it remembers how far back
 *     it was downloaded, so a symbol listed after the range start still counts as covered);
 *   • a response that itself reaches back to the cache's first candle replaces the cache (Yahoo
 *     always returns the full range, and its dividend-adjusted prices can change retroactively);
 *     a shorter one is a delta and is merged in, newer values winning;
 *   • only CLOSED candles (open time + bar length ≤ now) are ever stored;
 *   • "Force Refresh Data" (Advanced settings) ignores the cache and downloads the whole range.
 */

import { candleKey, readCandles, writeCandles, type CandleRecord } from '../lib/storage';
import { INTERVAL_MS, RANGE_DAYS, type CandleSeries, type HistoryRange, type Interval } from '../types/trading';

const DAY_MS = 86_400_000;

/** Downloads candles: the whole range, or (with `since`) the candles opening at or after it. */
export type CandleFetcher = (symbol: string, interval: Interval, range: HistoryRange, since: number | null) => Promise<CandleSeries>;

export interface CacheOutcome {
  series: CandleSeries;
  /** Candles that came from the cache. */
  cachedBars: number;
  /** Candles downloaded by this call. */
  fetchedBars: number;
  /** 'delta' – only new candles were downloaded; 'full' – the whole range; 'none' – nothing new had closed. */
  mode: 'delta' | 'full' | 'none';
}

/** Cache record extended with how far back it was fully downloaded. */
interface StoredRecord extends CandleRecord {
  coveredFrom?: number;
}

/** First open time of `range` ending at `now`, on the bar grid. */
export function rangeStart(interval: Interval, range: HistoryRange, now: number): number {
  const step = INTERVAL_MS[interval];
  return Math.floor((now - RANGE_DAYS[range] * DAY_MS) / step) * step;
}

/** Merge candle columns by open time (later values win), drop unclosed candles, sort. */
export function mergeCandles(base: Pick<CandleSeries, 't' | 'o' | 'h' | 'l' | 'c' | 'v'>, extra: Pick<CandleSeries, 't' | 'o' | 'h' | 'l' | 'c' | 'v'>, closedBefore: number): Pick<CandleSeries, 't' | 'o' | 'h' | 'l' | 'c' | 'v'> {
  const byTime = new Map<number, [number, number, number, number, number]>();
  for (const s of [base, extra]) {
    for (let i = 0; i < s.t.length; i++) byTime.set(s.t[i], [s.o[i], s.h[i], s.l[i], s.c[i], s.v[i]]);
  }
  const times = [...byTime.keys()].filter((t) => t < closedBefore).sort((a, b) => a - b);
  const out = { t: [] as number[], o: [] as number[], h: [] as number[], l: [] as number[], c: [] as number[], v: [] as number[] };
  for (const t of times) {
    const [o, h, l, c, v] = byTime.get(t) as [number, number, number, number, number];
    out.t.push(t);
    out.o.push(o);
    out.h.push(h);
    out.l.push(l);
    out.c.push(c);
    out.v.push(v);
  }
  return out;
}

/** The candles of `columns` opening at or after `from`. */
function sliceFrom<T extends Pick<CandleSeries, 't' | 'o' | 'h' | 'l' | 'c' | 'v'>>(columns: T, from: number): T {
  let k = 0;
  while (k < columns.t.length && columns.t[k] < from) k++;
  return { ...columns, t: columns.t.slice(k), o: columns.o.slice(k), h: columns.h.slice(k), l: columns.l.slice(k), c: columns.c.slice(k), v: columns.v.slice(k) };
}

/**
 * Candles of `symbol` for `range` up to `now`, from the cache plus whatever closed since (or the
 * whole range on a cache miss or with `force`). Writes the merged closed candles back.
 */
export async function loadWithCache(
  symbol: string,
  interval: Interval,
  range: HistoryRange,
  fetcher: CandleFetcher,
  options: { now?: number; force?: boolean } = {},
): Promise<CacheOutcome> {
  const now = options.now ?? Date.now();
  const step = INTERVAL_MS[interval];
  const from = rangeStart(interval, range, now);
  // A candle has closed once its full bar length lies in the past.
  const closedBefore = now - step + 1;

  const cached = options.force ? [] : ((await readCandles(symbol, interval)) as StoredRecord[]);
  const usable = cached
    .filter((record) => (record.coveredFrom ?? record.t[0] ?? Infinity) <= from + step && record.t.length > 0)
    .sort((a, b) => b.t[b.t.length - 1] - a.t[a.t.length - 1])[0];

  let mode: CacheOutcome['mode'];
  let columns: Pick<CandleSeries, 't' | 'o' | 'h' | 'l' | 'c' | 'v'>;
  let meta: Pick<CandleSeries, 'source' | 'venueSymbol' | 'note'>;
  let fetchedBars = 0;
  let coveredFrom: number;

  if (usable) {
    const last = usable.t[usable.t.length - 1];
    const since = last + step;
    meta = { source: usable.exchange as CandleSeries['source'], venueSymbol: usable.venueSymbol };
    coveredFrom = Math.min(usable.coveredFrom ?? usable.t[0], from);
    if (since >= closedBefore) {
      mode = 'none';
      columns = usable;
    } else {
      const delta = await fetcher(symbol, interval, range, since);
      fetchedBars = delta.t.length;
      if (delta.source !== usable.exchange && delta.t.length > 0) {
        // The provider changed (e.g. Binance failed over to Yahoo): never splice two feeds together.
        mode = 'full';
        columns = delta;
        meta = { source: delta.source, venueSymbol: delta.venueSymbol, note: delta.note };
        coveredFrom = from;
      } else if (delta.t.length > 0 && delta.t[0] <= usable.t[0]) {
        // The response covers everything the cache had: it replaces it (adjusted prices may have moved).
        mode = 'full';
        columns = mergeCandles({ t: [], o: [], h: [], l: [], c: [], v: [] }, delta, closedBefore);
        meta = { ...meta, note: delta.note };
      } else {
        mode = 'delta';
        columns = mergeCandles(usable, delta, closedBefore);
      }
    }
  } else {
    const full = await fetcher(symbol, interval, range, null);
    fetchedBars = full.t.length;
    mode = 'full';
    columns = mergeCandles({ t: [], o: [], h: [], l: [], c: [], v: [] }, full, closedBefore);
    meta = { source: full.source, venueSymbol: full.venueSymbol, note: full.note };
    coveredFrom = from;
  }

  const record: StoredRecord = {
    key: candleKey(meta.source, symbol, interval),
    exchange: meta.source,
    symbol: symbol.toUpperCase(),
    interval,
    venueSymbol: meta.venueSymbol,
    ...columns,
    coveredFrom,
    updatedAt: now,
  };
  if (mode !== 'none' && meta.source !== 'synthetic') await writeCandles(record);

  const inRange = sliceFrom(columns, from);
  const series: CandleSeries = { symbol, source: meta.source, venueSymbol: meta.venueSymbol, ...inRange };
  if (meta.note) series.note = meta.note;
  return { series, cachedBars: Math.max(0, inRange.t.length - fetchedBars), fetchedBars, mode };
}
