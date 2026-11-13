import { afterEach, describe, expect, it, vi } from 'vitest';
import handler from '../netlify/functions/market-data';
import type { MarketDataResponse } from '../src/types/trading';

const HOUR = 3_600_000;
const DAY = 86_400_000;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** Binance kline rows covering [startTime, endTime] for the requested interval. */
function binanceRows(url: URL): unknown[] {
  const step = { '1h': HOUR, '4h': 4 * HOUR, '1d': DAY }[url.searchParams.get('interval') as '1h' | '4h' | '1d'];
  const start = Number(url.searchParams.get('startTime'));
  const end = Number(url.searchParams.get('endTime'));
  const limit = Number(url.searchParams.get('limit'));
  const rows: unknown[] = [];
  for (let t = Math.ceil(start / step) * step; t <= end && rows.length < limit; t += step) {
    const base = 100 + (t / DAY) * 0.01;
    rows.push([t, String(base), String(base + 2), String(base - 2), String(base + 1), '12.5', t + step - 1, '0', 10, '0', '0', '0']);
  }
  return rows;
}

function yahooBody(timestamps: number[], gmtoffset = -14_400): unknown {
  return {
    chart: {
      error: null,
      result: [
        {
          meta: { gmtoffset },
          timestamp: timestamps,
          indicators: {
            quote: [
              {
                open: timestamps.map((_, i) => 100 + i),
                high: timestamps.map((_, i) => 102 + i),
                low: timestamps.map((_, i) => 99 + i),
                close: timestamps.map((_, i) => 101 + i),
                volume: timestamps.map(() => 1000),
              },
            ],
          },
        },
      ],
    },
  };
}

function mockFetch(route: (url: URL) => Response | Promise<Response>): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async (input: string | URL | Request) => route(new URL(String(input instanceof Request ? input.url : input))));
  vi.stubGlobal('fetch', mock);
  return mock;
}

const call = (query: string, method = 'GET'): Promise<Response> =>
  handler(new Request(`https://example.netlify.app/.netlify/functions/market-data${query}`, { method }));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('market-data function', () => {
  it('answers CORS preflight and rejects bad input', async () => {
    const preflight = await call('?symbols=SPY', 'OPTIONS');
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get('access-control-allow-origin')).toBe('*');

    expect((await call('')).status).toBe(400);
    expect((await call('?symbols=SPY&interval=5m')).status).toBe(400);
    expect((await call('?symbols=SPY&range=10y')).status).toBe(400);
    expect((await call('?symbols=SPY;rm')).status).toBe(400);
    expect((await call('?symbols=SPY', 'POST')).status).toBe(405);
    const bad = await call('?symbols=SPY&interval=5m');
    expect(bad.headers.get('access-control-allow-origin')).toBe('*');
    expect(bad.headers.get('content-type')).toContain('application/json');
  });

  it('pages through Binance klines for crypto pairs', async () => {
    const fetchMock = mockFetch((url) => {
      expect(url.hostname).toBe('data-api.binance.vision');
      expect(url.searchParams.get('symbol')).toBe('BTCUSDT');
      return jsonResponse(binanceRows(url));
    });
    const response = await call('?symbols=btc-usd&interval=4h&range=1y');
    expect(response.status).toBe(200);
    expect(response.headers.get('access-control-allow-origin')).toBe('*');
    const body = (await response.json()) as MarketDataResponse;
    const series = body.series['BTC-USD'];
    expect(series.source).toBe('binance');
    expect(series.venueSymbol).toBe('BTCUSDT'); // BTC-USD is routed to the USDT pair on Binance
    // 365 days of 4h candles = 2190 → three pages of 1000.
    expect(fetchMock.mock.calls.length).toBe(3);
    expect(series.t.length).toBeGreaterThan(2180);
    expect(series.t.length).toBeLessThanOrEqual(2190);
    for (let i = 1; i < series.t.length; i++) expect(series.t[i] - series.t[i - 1]).toBe(4 * HOUR);
    expect(series.t[0] % (4 * HOUR)).toBe(0);
    expect(series.h[0]).toBeGreaterThanOrEqual(series.c[0]);
    expect(series.v[0]).toBe(12.5);
    // The still-forming candle is dropped.
    expect(series.t[series.t.length - 1] + 4 * HOUR).toBeLessThanOrEqual(body.generatedAt);
  });

  it('normalises Yahoo daily timestamps to the exchange-local date and drops bad rows', async () => {
    // 13:30 UTC = 09:30 New York (EDT). Includes a null row and a duplicated "live" row.
    const d1 = Date.UTC(2025, 5, 2, 13, 30) / 1000;
    const d2 = Date.UTC(2025, 5, 3, 13, 30) / 1000;
    const d3 = Date.UTC(2025, 5, 4, 13, 30) / 1000;
    const live = Date.UTC(2025, 5, 4, 19, 59) / 1000;
    const payload = yahooBody([d1, d2, d3, live]) as { chart: { result: { indicators: { quote: { close: (number | null)[] }[] } }[] } };
    payload.chart.result[0].indicators.quote[0].close[1] = null;
    mockFetch((url) => {
      expect(url.hostname).toBe('query1.finance.yahoo.com');
      expect(url.pathname).toBe('/v8/finance/chart/SPY');
      expect(url.searchParams.get('interval')).toBe('1d');
      return jsonResponse(payload);
    });
    const body = (await (await call('?symbols=SPY&interval=1d&range=1y')).json()) as MarketDataResponse;
    const series = body.series.SPY;
    expect(series.source).toBe('yahoo');
    expect(series.t).toEqual([Date.UTC(2025, 5, 2), Date.UTC(2025, 5, 4)]);
    // The later "live" row for 4 June replaces the earlier one.
    expect(series.c[1]).toBe(104);
  });

  it('builds UTC-aligned 4h candles from Yahoo hourly data', async () => {
    const day = Date.UTC(2025, 5, 2);
    const hours = [13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5].map((h) => (day + h * HOUR) / 1000);
    mockFetch((url) => {
      expect(url.searchParams.get('interval')).toBe('1h');
      return jsonResponse(yahooBody(hours));
    });
    const body = (await (await call('?symbols=NVDA&interval=4h&range=5y')).json()) as MarketDataResponse;
    const series = body.series.NVDA;
    expect(series.t).toEqual([day + 12 * HOUR, day + 16 * HOUR]);
    expect(series.o).toEqual([100, 103]); // first hourly open of each bucket
    expect(series.c).toEqual([103, 107]); // last hourly close of each bucket
    expect(series.h).toEqual([104, 108]);
    expect(series.l).toEqual([99, 102]);
    expect(series.v).toEqual([3000, 4000]);
    expect(series.note).toContain('intraday history');
  });

  it('falls back from Binance to Yahoo for crypto, and reports total failure as 502', async () => {
    const ts = Date.UTC(2025, 5, 2) / 1000;
    mockFetch((url) => {
      if (url.hostname === 'data-api.binance.vision') return jsonResponse({ code: -1121, msg: 'Invalid symbol.' }, 400);
      expect(url.pathname).toBe('/v8/finance/chart/DOGE-USD');
      return jsonResponse(yahooBody([ts, ts + 86_400], 0));
    });
    const ok = (await (await call('?symbols=DOGE-USDT&interval=1d&range=1y')).json()) as MarketDataResponse;
    expect(ok.series['DOGE-USDT'].source).toBe('yahoo');
    expect(ok.series['DOGE-USDT'].symbol).toBe('DOGE-USDT');
    expect(ok.series['DOGE-USDT'].venueSymbol).toBe('DOGE-USD');
    expect(ok.series['DOGE-USDT'].t).toHaveLength(2);

    mockFetch(() => jsonResponse({ chart: { error: { code: 'Not Found', description: 'No data found' }, result: null } }, 404));
    const failed = await call('?symbols=NOPE,SPY&interval=1d&range=1y');
    expect(failed.status).toBe(502);
    const body = (await failed.json()) as MarketDataResponse;
    expect(Object.keys(body.series)).toHaveLength(0);
    expect(Object.keys(body.errors).sort()).toEqual(['NOPE', 'SPY']);
  });

  it('returns partial results when only some symbols fail', async () => {
    mockFetch((url) => {
      if (url.hostname === 'data-api.binance.vision') return jsonResponse(binanceRows(url));
      return jsonResponse({}, 429);
    });
    const response = await call('?symbols=ETH-USD,AAPL&interval=1d&range=1y');
    expect(response.status).toBe(200);
    const body = (await response.json()) as MarketDataResponse;
    expect(body.series['ETH-USD'].t.length).toBeGreaterThan(360);
    expect(body.errors.AAPL).toContain('429');
  });
});
