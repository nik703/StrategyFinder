/**
 * Offline OHLCV fallback generator.
 *
 * Produces realistic, deterministic candles so the app is fully usable without
 * network access, without `netlify dev`, or when an upstream API rate-limits.
 *
 * The price process is a regime-switching geometric Brownian motion with
 * GARCH-style volatility clustering, fat-tailed shocks and a shared market
 * factor (so basket components are correlated like real assets). Stocks follow
 * a weekday session calendar with overnight gaps; crypto trades 24/7.
 * Timestamps use the same normalisation as the market-data function.
 */

import { INTERVAL_MS, RANGE_DAYS, type CandleSeries, type HistoryRange, type Interval } from '../types/trading';

interface AssetProfile {
  /** Approximate price of the most recent candle. */
  price: number;
  /** Long-run annual log return. */
  drift: number;
  /** Long-run annualised volatility. */
  vol: number;
  crypto: boolean;
  /** Average units traded per day. */
  dailyVolume: number;
  /** Correlation with the shared market factor of its asset class. */
  correlation: number;
}

const PROFILES: Record<string, AssetProfile> = {
  'BTC-USD': { price: 96_000, drift: 0.3, vol: 0.55, crypto: true, dailyVolume: 28_000, correlation: 0.85 },
  'ETH-USD': { price: 3_400, drift: 0.24, vol: 0.7, crypto: true, dailyVolume: 420_000, correlation: 0.8 },
  'SOL-USD': { price: 185, drift: 0.34, vol: 0.95, crypto: true, dailyVolume: 3_800_000, correlation: 0.72 },
  SPY: { price: 590, drift: 0.11, vol: 0.17, crypto: false, dailyVolume: 62_000_000, correlation: 0.92 },
  NVDA: { price: 138, drift: 0.48, vol: 0.5, crypto: false, dailyVolume: 240_000_000, correlation: 0.62 },
  AAPL: { price: 232, drift: 0.14, vol: 0.28, crypto: false, dailyVolume: 55_000_000, correlation: 0.7 },
};

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

/* -------------------------------------------------------------------------- */
/*  Deterministic randomness                                                  */
/* -------------------------------------------------------------------------- */

function hashString(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng: () => number): number {
  let u = rng();
  if (u < 1e-12) u = 1e-12;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** Standard normal that depends only on (seed, timestamp) – shared by every asset of a class. */
function factorShock(seed: number, timestamp: number): number {
  return normal(mulberry32((seed ^ Math.imul(Math.floor(timestamp / HOUR_MS), 2654435761)) >>> 0));
}

function profileFor(symbol: string): AssetProfile {
  const known = PROFILES[symbol];
  if (known) return known;
  // Unknown ticker: derive a stable, plausible profile from its name.
  const rng = mulberry32(hashString(symbol));
  const crypto = /-(USD|USDT|USDC)$/.test(symbol) || /USDT$/.test(symbol);
  return {
    price: crypto ? Math.exp(Math.log(0.5) + rng() * Math.log(4000)) : 20 + rng() * 480,
    drift: crypto ? 0.1 + rng() * 0.3 : 0.04 + rng() * 0.16,
    vol: crypto ? 0.6 + rng() * 0.5 : 0.2 + rng() * 0.3,
    crypto,
    dailyVolume: crypto ? 1_000_000 + rng() * 9_000_000 : 5_000_000 + rng() * 45_000_000,
    correlation: 0.5 + rng() * 0.3,
  };
}

/* -------------------------------------------------------------------------- */
/*  Calendars                                                                 */
/* -------------------------------------------------------------------------- */

/** Bar open times for the requested window, oldest first. */
function buildCalendar(crypto: boolean, interval: Interval, range: HistoryRange, now: number): number[] {
  const step = INTERVAL_MS[interval];
  const lastOpen = Math.floor(now / step) * step - step; // last fully closed bar
  const start = Math.floor((now - RANGE_DAYS[range] * DAY_MS) / step) * step;
  const times: number[] = [];
  for (let t = start; t <= lastOpen; t += step) {
    if (!crypto) {
      const weekday = new Date(t).getUTCDay();
      if (weekday === 0 || weekday === 6) continue;
      if (interval !== '1d') {
        // US cash session expressed in the function's normalised UTC buckets.
        const hour = new Date(t).getUTCHours();
        if (interval === '1h' && (hour < 13 || hour > 19)) continue;
        if (interval === '4h' && hour !== 12 && hour !== 16) continue;
      }
    }
    times.push(t);
  }
  return times;
}

/* -------------------------------------------------------------------------- */
/*  Generator                                                                 */
/* -------------------------------------------------------------------------- */

const REGIMES = [
  // annual log-drift offset (in units of the asset's volatility), volatility multiplier, mean duration in days
  { drift: 0.7, vol: 0.85, days: 90 }, // bull
  { drift: -0.9, vol: 1.3, days: 50 }, // bear
  { drift: 0, vol: 0.75, days: 70 }, // range
] as const;

/**
 * Generate deterministic synthetic candles for `symbol`.
 * @param now Reference "current time" (injectable for tests).
 */
export function generateSyntheticCandles(
  symbol: string,
  interval: Interval,
  range: HistoryRange,
  now: number = Date.now(),
): CandleSeries {
  const profile = profileFor(symbol);
  const times = buildCalendar(profile.crypto, interval, range, now);
  const n = times.length;
  const series: CandleSeries = { symbol, source: 'synthetic', t: times, o: [], h: [], l: [], c: [], v: [] };
  if (n === 0) return series;

  const rng = mulberry32(hashString(`${symbol}|${interval}`));
  const factorSeed = hashString(profile.crypto ? 'crypto-factor' : 'equity-factor');
  const barsPerDay = profile.crypto ? DAY_MS / INTERVAL_MS[interval] : interval === '1d' ? 1 : interval === '1h' ? 7 : 2;
  const barsPerYear = barsPerDay * (profile.crypto ? 365 : 252);
  const dt = 1 / barsPerYear;
  const baseSigma = profile.vol * Math.sqrt(dt);
  const loading = Math.sqrt(profile.correlation);
  const idio = Math.sqrt(1 - profile.correlation);

  // GARCH(1,1) normalised to unit long-run variance.
  const alpha = 0.09;
  const beta = 0.88;
  const omega = 1 - alpha - beta;
  let variance = 1;
  let lastShock = 0;

  let regime = Math.floor(rng() * REGIMES.length);
  const logReturns = new Float64Array(n);
  const sigmas = new Float64Array(n);
  const gaps = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const spec = REGIMES[regime];
    if (rng() < 1 / (spec.days * barsPerDay)) {
      // Switch to one of the other two regimes.
      regime = (regime + 1 + Math.floor(rng() * (REGIMES.length - 1))) % REGIMES.length;
    }
    variance = omega + alpha * lastShock * lastShock + beta * variance;
    const sigma = baseSigma * Math.sqrt(variance) * REGIMES[regime].vol;
    // Fat tails: an occasional 3-sigma-scaled shock.
    const tail = rng() < 0.02 ? 2.8 : 1;
    const z = (loading * factorShock(factorSeed, times[i]) + idio * normal(rng)) * tail;
    lastShock = Math.max(-4, Math.min(4, z));
    logReturns[i] = REGIMES[regime].drift * profile.vol * dt + sigma * z;
    sigmas[i] = sigma;
    // Overnight / weekend gap for session-based assets.
    const sessionBreak = !profile.crypto && (i === 0 || times[i] - times[i - 1] > INTERVAL_MS[interval]);
    gaps[i] = sessionBreak ? normal(rng) * sigma * (interval === '1d' ? 0.35 : 1.1) : normal(rng) * sigma * 0.03;
  }

  // Bridge the path: keep its regimes and volatility structure, but pin the total
  // return to the asset's long-run drift so multi-year histories stay plausible.
  let total = 0;
  for (let i = 0; i < n; i++) total += logReturns[i];
  const targetTotal = profile.drift * (n / barsPerYear) * (0.75 + 0.5 * rng());
  const correction = (targetTotal - total) / n;
  for (let i = 0; i < n; i++) logReturns[i] += correction;
  // Anchor the final close on the profile's reference price.
  let close = profile.price * Math.exp(-targetTotal);
  const volumePerBar = profile.dailyVolume / barsPerDay;

  for (let i = 0; i < n; i++) {
    const prevClose = close;
    close = prevClose * Math.exp(logReturns[i]);
    const open = prevClose * Math.exp(gaps[i]);
    const top = Math.max(open, close);
    const bottom = Math.min(open, close);
    const wick = sigmas[i] * 0.55;
    const high = top * (1 + Math.abs(normal(rng)) * wick);
    const low = bottom * (1 - Math.min(0.5, Math.abs(normal(rng)) * wick));
    const surprise = Math.abs(logReturns[i]) / (sigmas[i] || 1);
    const volume = volumePerBar * (0.55 + 0.45 * surprise) * Math.exp(normal(rng) * 0.35) * Math.sqrt(sigmas[i] / baseSigma);
    series.o.push(Number(open.toPrecision(8)));
    series.h.push(Number(Math.max(high, open, close).toPrecision(8)));
    series.l.push(Number(Math.min(low, open, close).toPrecision(8)));
    series.c.push(Number(close.toPrecision(8)));
    series.v.push(Math.round(volume * 100) / 100);
  }

  // Rounding the close may leave high/low a hair inside the body – re-assert the envelope.
  for (let i = 0; i < n; i++) {
    if (series.h[i] < series.o[i]) series.h[i] = series.o[i];
    if (series.h[i] < series.c[i]) series.h[i] = series.c[i];
    if (series.l[i] > series.o[i]) series.l[i] = series.o[i];
    if (series.l[i] > series.c[i]) series.l[i] = series.c[i];
  }
  return series;
}
