/**
 * Netlify Function: market-data proxy
 *
 *   GET /.netlify/functions/market-data?symbols=BTC-USD,NVDA&interval=1d&range=2y
 *
 * Fetches historical OHLCV candles and returns clean, CORS-enabled JSON.
 *   • Crypto pairs (BASE-USD / BASE-USDT) → Binance public market-data API
 *     (data-api.binance.vision), falling back to Yahoo Finance.
 *   • Stocks, ETFs and indices → Yahoo Finance chart API.
 *
 * Timestamps are normalised to the bar OPEN time in UTC milliseconds:
 *   • 1d → 00:00 UTC of the exchange-local trading date
 *   • 1h → top of the hour
 *   • 4h → UTC-aligned 4-hour buckets (00, 04, 08, 12, 16, 20)
 * so series from different venues line up when they are combined into a basket.
 *
 * The function only proxies and normalises data. All backtesting happens in the
 * browser, so the serverless execution limit is never a concern.
 */

import type { Context } from '@netlify/functions';
import type { CandleSeries, HistoryRange, Interval, MarketDataResponse } from '../../src/types/trading';

/* -------------------------------------------------------------------------- */
/*  Configuration                                                             */
/* -------------------------------------------------------------------------- */

const INTERVAL_MS: Record<Interval, number> = { '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000 };
const RANGE_DAYS: Record<HistoryRange, number> = { '1y': 365, '2y': 730, '5y': 1826 };
const DAY_MS = 86_400_000;

const BINANCE_KLINES = 'https://data-api.binance.vision/api/v3/klines';
const BINANCE_PAGE = 1000;
const BINANCE_CONCURRENCY = 8;
const YAHOO_HOSTS = ['query1.finance.yahoo.com', 'query2.finance.yahoo.com'];
/** Yahoo only serves hourly candles for roughly the last 730 days. */
const YAHOO_INTRADAY_MAX_DAYS = 729;

const MAX_SYMBOLS = 10;
const UPSTREAM_TIMEOUT_MS = 8000;
const SYMBOL_PATTERN = /^[A-Z0-9][A-Z0-9.\-^=]{0,19}$/;
const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

const CORS_HEADERS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};

/* -------------------------------------------------------------------------- */
/*  HTTP helpers                                                              */
/* -------------------------------------------------------------------------- */

function json(status: number, body: unknown, cacheSeconds = 0): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...CORS_HEADERS,
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': cacheSeconds > 0 ? `public, max-age=${cacheSeconds}` : 'no-store',
    },
  });
}

class UpstreamError extends Error {
  constructor(
    message: string,
    readonly status = 0,
  ) {
    super(message);
  }
}

async function fetchJson(url: string, headers: Record<string, string> = {}): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const response = await fetch(url, { headers: { Accept: 'application/json', ...headers }, signal: controller.signal });
    if (!response.ok) throw new UpstreamError(`HTTP ${response.status}`, response.status);
    return (await response.json()) as unknown;
  } catch (error) {
    if (error instanceof UpstreamError) throw error;
    if (error instanceof Error && error.name === 'AbortError') throw new UpstreamError('upstream timed out');
    throw new UpstreamError(error instanceof Error ? error.message : 'network error');
  } finally {
    clearTimeout(timer);
  }
}

/** Run `tasks` with at most `limit` in flight, preserving result order. */
async function mapLimit<T, R>(items: readonly T[], limit: number, task: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await task(items[index]);
    }
  });
  await Promise.all(workers);
  return results;
}

/* -------------------------------------------------------------------------- */
/*  Candle normalisation                                                      */
/* -------------------------------------------------------------------------- */

interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

const isPrice = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x) && x > 0;

/** Trim float noise without destroying precision on very small prices. */
const tidy = (x: number): number => Number(x.toPrecision(10));

/** Sort by time and keep the most recent candle for each timestamp. */
function sortAndDedupe(candles: Candle[]): Candle[] {
  const byTime = new Map<number, Candle>();
  for (const candle of candles) byTime.set(candle.t, candle);
  return [...byTime.values()].sort((a, b) => a.t - b.t);
}

/** Aggregate candles into UTC-aligned buckets of `bucketMs`. */
function aggregate(candles: Candle[], bucketMs: number): Candle[] {
  const out: Candle[] = [];
  let current: Candle | null = null;
  for (const candle of candles) {
    const bucket = Math.floor(candle.t / bucketMs) * bucketMs;
    if (current === null || current.t !== bucket) {
      current = { t: bucket, o: candle.o, h: candle.h, l: candle.l, c: candle.c, v: candle.v };
      out.push(current);
    } else {
      if (candle.h > current.h) current.h = candle.h;
      if (candle.l < current.l) current.l = candle.l;
      current.c = candle.c;
      current.v += candle.v;
    }
  }
  return out;
}

function toSeries(symbol: string, source: CandleSeries['source'], candles: Candle[], note?: string): CandleSeries {
  const series: CandleSeries = { symbol, source, t: [], o: [], h: [], l: [], c: [], v: [] };
  for (const candle of candles) {
    // Guarantee a self-consistent candle even if the upstream feed is slightly off.
    const high = Math.max(candle.h, candle.o, candle.c);
    const low = Math.min(candle.l, candle.o, candle.c);
    series.t.push(candle.t);
    series.o.push(tidy(candle.o));
    series.h.push(tidy(high));
    series.l.push(tidy(low));
    series.c.push(tidy(candle.c));
    series.v.push(Number.isFinite(candle.v) && candle.v > 0 ? tidy(candle.v) : 0);
  }
  if (note) series.note = note;
  return series;
}

/* -------------------------------------------------------------------------- */
/*  Binance (crypto)                                                          */
/* -------------------------------------------------------------------------- */

/** "BTC-USD" / "BTC-USDT" / "BTCUSDT" → "BTCUSDT"; anything else → null. */
function toBinancePair(symbol: string): string | null {
  const dashed = /^([A-Z0-9]{2,10})-(USD|USDT|USDC)$/.exec(symbol);
  if (dashed) return `${dashed[1]}USDT`;
  if (/^[A-Z0-9]{2,10}USDT$/.test(symbol)) return symbol;
  return null;
}

async function fetchBinance(
  symbol: string,
  pair: string,
  interval: Interval,
  range: HistoryRange,
  now: number,
): Promise<CandleSeries> {
  const step = INTERVAL_MS[interval];
  const start = Math.floor((now - RANGE_DAYS[range] * DAY_MS) / step) * step;
  const pageSpan = BINANCE_PAGE * step;
  const pageStarts: number[] = [];
  for (let from = start; from < now; from += pageSpan) pageStarts.push(from);

  const pages = await mapLimit(pageStarts, BINANCE_CONCURRENCY, async (from) => {
    const url =
      `${BINANCE_KLINES}?symbol=${encodeURIComponent(pair)}&interval=${interval}` +
      `&startTime=${from}&endTime=${Math.min(now, from + pageSpan - 1)}&limit=${BINANCE_PAGE}`;
    const body = await fetchJson(url);
    if (!Array.isArray(body)) throw new UpstreamError('unexpected Binance response');
    return body as unknown[];
  });

  const candles: Candle[] = [];
  for (const page of pages) {
    for (const row of page) {
      if (!Array.isArray(row) || row.length < 7) continue;
      const openTime = Number(row[0]);
      const closeTime = Number(row[6]);
      // Skip the candle that is still forming.
      if (!Number.isFinite(openTime) || closeTime >= now) continue;
      const o = Number(row[1]);
      const h = Number(row[2]);
      const l = Number(row[3]);
      const c = Number(row[4]);
      const v = Number(row[5]);
      if (!isPrice(o) || !isPrice(h) || !isPrice(l) || !isPrice(c)) continue;
      candles.push({ t: openTime, o, h, l, c, v: Number.isFinite(v) ? v : 0 });
    }
  }
  if (candles.length === 0) throw new UpstreamError(`Binance returned no candles for ${pair}`);
  return toSeries(symbol, 'binance', sortAndDedupe(candles));
}

/* -------------------------------------------------------------------------- */
/*  Yahoo Finance (stocks, ETFs, indices, crypto fallback)                    */
/* -------------------------------------------------------------------------- */

interface YahooChart {
  chart?: {
    error?: { code?: string; description?: string } | null;
    result?: Array<{
      meta?: { gmtoffset?: number };
      timestamp?: Array<number | null>;
      indicators?: {
        quote?: Array<{
          open?: Array<number | null>;
          high?: Array<number | null>;
          low?: Array<number | null>;
          close?: Array<number | null>;
          volume?: Array<number | null>;
        }>;
      };
    }> | null;
  };
}

async function fetchYahooChart(symbol: string, query: string): Promise<YahooChart> {
  let lastError: unknown = new UpstreamError('Yahoo Finance unavailable');
  for (const host of YAHOO_HOSTS) {
    try {
      const url = `https://${host}/v8/finance/chart/${encodeURIComponent(symbol)}?${query}`;
      return (await fetchJson(url, { 'User-Agent': USER_AGENT })) as YahooChart;
    } catch (error) {
      lastError = error;
      // A 404 means the symbol does not exist – the second host will not help.
      if (error instanceof UpstreamError && error.status === 404) break;
    }
  }
  throw lastError;
}

async function fetchYahoo(symbol: string, interval: Interval, range: HistoryRange, now: number): Promise<CandleSeries> {
  const intraday = interval !== '1d';
  const requestedDays = RANGE_DAYS[range];
  const days = intraday ? Math.min(requestedDays, YAHOO_INTRADAY_MAX_DAYS) : requestedDays;
  const period1 = Math.floor((now - days * DAY_MS) / 1000);
  const period2 = Math.floor(now / 1000);
  // 4h candles are built from hourly candles so they align with Binance's UTC buckets.
  const query = `period1=${period1}&period2=${period2}&interval=${intraday ? '1h' : '1d'}&includePrePost=false&events=div%2Csplits`;

  const body = await fetchYahooChart(symbol, query);
  const chartError = body.chart?.error;
  if (chartError) throw new UpstreamError(chartError.description ?? chartError.code ?? 'Yahoo Finance error');
  const result = body.chart?.result?.[0];
  const timestamps = result?.timestamp;
  const quote = result?.indicators?.quote?.[0];
  if (!result || !timestamps || !quote) throw new UpstreamError(`Yahoo Finance returned no data for ${symbol}`);

  const offsetMs = (result.meta?.gmtoffset ?? 0) * 1000;
  const candles: Candle[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    const ts = timestamps[i];
    const o = quote.open?.[i];
    const h = quote.high?.[i];
    const l = quote.low?.[i];
    const c = quote.close?.[i];
    const v = quote.volume?.[i];
    if (typeof ts !== 'number' || !isPrice(o) || !isPrice(h) || !isPrice(l) || !isPrice(c)) continue;
    const ms = ts * 1000;
    const t = intraday
      ? // Hourly: floor to the top of the hour (US sessions start at :30).
        Math.floor(ms / INTERVAL_MS['1h']) * INTERVAL_MS['1h']
      : // Daily: 00:00 UTC of the exchange-local trading date.
        Math.floor((ms + offsetMs) / DAY_MS) * DAY_MS;
    candles.push({ t, o, h, l, c, v: typeof v === 'number' && Number.isFinite(v) ? v : 0 });
  }
  if (candles.length === 0) throw new UpstreamError(`Yahoo Finance returned no candles for ${symbol}`);

  let clean = sortAndDedupe(candles);
  if (interval === '4h') clean = aggregate(clean, INTERVAL_MS['4h']);
  const note =
    intraday && requestedDays > YAHOO_INTRADAY_MAX_DAYS
      ? `Yahoo Finance limits intraday history to ~${YAHOO_INTRADAY_MAX_DAYS} days.`
      : undefined;
  return toSeries(symbol, 'yahoo', clean, note);
}

/* -------------------------------------------------------------------------- */
/*  Routing                                                                   */
/* -------------------------------------------------------------------------- */

async function fetchSymbol(symbol: string, interval: Interval, range: HistoryRange, now: number): Promise<CandleSeries> {
  const pair = toBinancePair(symbol);
  if (pair === null) return fetchYahoo(symbol, interval, range, now);
  try {
    return await fetchBinance(symbol, pair, interval, range, now);
  } catch (binanceError) {
    // Yahoo quotes crypto as BASE-USD.
    const yahooSymbol = symbol.includes('-') ? symbol.replace(/-(USDT|USDC)$/, '-USD') : `${symbol.slice(0, -4)}-USD`;
    try {
      const series = await fetchYahoo(yahooSymbol, interval, range, now);
      return { ...series, symbol };
    } catch (yahooError) {
      const a = binanceError instanceof Error ? binanceError.message : String(binanceError);
      const b = yahooError instanceof Error ? yahooError.message : String(yahooError);
      throw new UpstreamError(`Binance: ${a}; Yahoo: ${b}`);
    }
  }
}

function parseSymbols(raw: string): string[] {
  const unique = new Set<string>();
  for (const part of raw.split(',')) {
    const symbol = part.trim().toUpperCase();
    if (symbol) unique.add(symbol);
  }
  return [...unique];
}

const isInterval = (x: string): x is Interval => x === '1h' || x === '4h' || x === '1d';
const isRange = (x: string): x is HistoryRange => x === '1y' || x === '2y' || x === '5y';

export default async function handler(request: Request, _context?: Context): Promise<Response> {
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS_HEADERS });
  if (request.method !== 'GET') return json(405, { error: 'Method not allowed. Use GET.' });

  const params = new URL(request.url).searchParams;
  const symbols = parseSymbols(params.get('symbols') ?? params.get('symbol') ?? '');
  const interval = params.get('interval') ?? '1d';
  const range = params.get('range') ?? '1y';

  if (symbols.length === 0) return json(400, { error: 'Missing required query parameter "symbols".' });
  if (symbols.length > MAX_SYMBOLS) return json(400, { error: `At most ${MAX_SYMBOLS} symbols per request.` });
  const invalid = symbols.filter((symbol) => !SYMBOL_PATTERN.test(symbol));
  if (invalid.length > 0) return json(400, { error: `Invalid symbol(s): ${invalid.join(', ')}` });
  if (!isInterval(interval)) return json(400, { error: 'Invalid "interval". Use one of: 1h, 4h, 1d.' });
  if (!isRange(range)) return json(400, { error: 'Invalid "range". Use one of: 1y, 2y, 5y.' });

  const now = Date.now();
  const body: MarketDataResponse = { interval, range, generatedAt: now, series: {}, errors: {} };

  await Promise.all(
    symbols.map(async (symbol) => {
      try {
        body.series[symbol] = await fetchSymbol(symbol, interval, range, now);
      } catch (error) {
        body.errors[symbol] = error instanceof Error ? error.message : 'Unknown upstream error';
      }
    }),
  );

  const ok = Object.keys(body.series).length > 0;
  // 502 when every upstream failed, so the client can switch to its offline fallback.
  return json(ok ? 200 : 502, body, ok ? 300 : 0);
}
