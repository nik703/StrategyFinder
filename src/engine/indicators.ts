/**
 * Vectorised technical-indicator library.
 *
 * Every function takes typed arrays and returns a `Float64Array` of the same
 * length, with `NaN` during the warm-up window. All calculations are strictly
 * causal: the value at index `i` depends only on bars `0..i`, so nothing here
 * can leak future information into a signal.
 *
 * The formulas intentionally mirror TradingView's built-ins (`ta.rsi`, `ta.ema`,
 * `ta.atr`, `ta.dmi`, `ta.supertrend`, ...) so that exported Pine Script
 * reproduces the same series.
 */

import type { MaType, OHLCV } from '../types/trading';

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

function nanArray(n: number): Float64Array {
  return new Float64Array(n).fill(NaN);
}

/** Index of the first non-NaN value (or `src.length` when there is none). */
function firstValid(src: Float64Array): number {
  for (let i = 0; i < src.length; i++) {
    if (src[i] === src[i]) return i;
  }
  return src.length;
}

/* -------------------------------------------------------------------------- */
/*  Moving averages                                                           */
/* -------------------------------------------------------------------------- */

/** Simple moving average. Leading NaNs in `src` are skipped. */
export function sma(src: Float64Array, period: number): Float64Array {
  const n = src.length;
  const out = nanArray(n);
  const start = firstValid(src);
  if (period < 1 || start + period > n) return out;
  let sum = 0;
  for (let i = start; i < n; i++) {
    sum += src[i];
    if (i >= start + period) sum -= src[i - period];
    if (i >= start + period - 1) out[i] = sum / period;
  }
  return out;
}

/** Recursive smoothing seeded with an SMA of the first `period` valid values. */
function smooth(src: Float64Array, period: number, alpha: number): Float64Array {
  const n = src.length;
  const out = nanArray(n);
  const start = firstValid(src);
  if (period < 1 || start + period > n) return out;
  let seed = 0;
  for (let i = start; i < start + period; i++) seed += src[i];
  let prev = seed / period;
  out[start + period - 1] = prev;
  for (let i = start + period; i < n; i++) {
    prev = alpha * src[i] + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

/** Exponential moving average (alpha = 2 / (period + 1)), SMA-seeded like `ta.ema`. */
export function ema(src: Float64Array, period: number): Float64Array {
  return smooth(src, period, 2 / (period + 1));
}

/** Wilder's moving average (alpha = 1 / period), as used by RSI, ATR and ADX. */
export function rma(src: Float64Array, period: number): Float64Array {
  return smooth(src, period, 1 / period);
}

/* -------------------------------------------------------------------------- */
/*  RSI                                                                       */
/* -------------------------------------------------------------------------- */

/** Wilder's Relative Strength Index. */
export function rsi(close: Float64Array, period: number): Float64Array {
  const n = close.length;
  const out = nanArray(n);
  if (period < 1 || n <= period) return out;
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const change = close[i] - close[i - 1];
    if (change > 0) avgGain += change;
    else avgLoss -= change;
  }
  avgGain /= period;
  avgLoss /= period;
  const toRsi = (gain: number, loss: number): number => {
    if (loss === 0) return gain === 0 ? 50 : 100;
    return 100 - 100 / (1 + gain / loss);
  };
  out[period] = toRsi(avgGain, avgLoss);
  for (let i = period + 1; i < n; i++) {
    const change = close[i] - close[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out[i] = toRsi(avgGain, avgLoss);
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/*  MACD                                                                      */
/* -------------------------------------------------------------------------- */

export interface MacdResult {
  line: Float64Array;
  signal: Float64Array;
  hist: Float64Array;
}

/** MACD line (fast EMA − slow EMA), signal line (EMA of the line) and histogram. */
export function macd(close: Float64Array, fast: number, slow: number, signalPeriod: number): MacdResult {
  const n = close.length;
  const fastEma = ema(close, fast);
  const slowEma = ema(close, slow);
  const line = nanArray(n);
  for (let i = 0; i < n; i++) line[i] = fastEma[i] - slowEma[i];
  const signal = ema(line, signalPeriod);
  const hist = nanArray(n);
  for (let i = 0; i < n; i++) hist[i] = line[i] - signal[i];
  return { line, signal, hist };
}

/* -------------------------------------------------------------------------- */
/*  Bollinger Bands                                                           */
/* -------------------------------------------------------------------------- */

export interface BollingerResult {
  upper: Float64Array;
  middle: Float64Array;
  lower: Float64Array;
  /** (Close − Lower) / (Upper − Lower). 0 = on the lower band, 1 = on the upper band. */
  percentB: Float64Array;
  /** (Upper − Lower) / Middle × 100. */
  bandwidth: Float64Array;
}

/** Bollinger Bands using the population standard deviation (same as `ta.bb`). */
export function bollinger(close: Float64Array, period: number, mult: number): BollingerResult {
  const n = close.length;
  const middle = sma(close, period);
  const upper = nanArray(n);
  const lower = nanArray(n);
  const percentB = nanArray(n);
  const bandwidth = nanArray(n);
  for (let i = period - 1; i < n; i++) {
    const mean = middle[i];
    if (mean !== mean) continue;
    let variance = 0;
    for (let j = i - period + 1; j <= i; j++) {
      const d = close[j] - mean;
      variance += d * d;
    }
    const dev = mult * Math.sqrt(variance / period);
    const up = mean + dev;
    const lo = mean - dev;
    upper[i] = up;
    lower[i] = lo;
    const width = up - lo;
    percentB[i] = width > 0 ? (close[i] - lo) / width : 0.5;
    bandwidth[i] = mean !== 0 ? (width / mean) * 100 : 0;
  }
  return { upper, middle, lower, percentB, bandwidth };
}

/* -------------------------------------------------------------------------- */
/*  ATR                                                                       */
/* -------------------------------------------------------------------------- */

/** True range. The first bar falls back to High − Low (no previous close). */
export function trueRange(high: Float64Array, low: Float64Array, close: Float64Array): Float64Array {
  const n = close.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  out[0] = high[0] - low[0];
  for (let i = 1; i < n; i++) {
    const prevClose = close[i - 1];
    const a = high[i] - low[i];
    const b = Math.abs(high[i] - prevClose);
    const c = Math.abs(low[i] - prevClose);
    out[i] = a > b ? (a > c ? a : c) : b > c ? b : c;
  }
  return out;
}

/** Average True Range (Wilder smoothing of the true range). */
export function atr(high: Float64Array, low: Float64Array, close: Float64Array, period: number): Float64Array {
  return rma(trueRange(high, low, close), period);
}

/* -------------------------------------------------------------------------- */
/*  Stochastic Oscillator                                                     */
/* -------------------------------------------------------------------------- */

export interface StochasticResult {
  k: Float64Array;
  d: Float64Array;
}

/**
 * Stochastic Oscillator.
 * raw %K = 100 × (Close − LowestLow(kPeriod)) / (HighestHigh(kPeriod) − LowestLow(kPeriod))
 * %K     = SMA(raw %K, smoothK)
 * %D     = SMA(%K, dPeriod)
 */
export function stochastic(
  high: Float64Array,
  low: Float64Array,
  close: Float64Array,
  kPeriod: number,
  smoothK: number,
  dPeriod: number,
): StochasticResult {
  const n = close.length;
  const raw = nanArray(n);
  let last = 50;
  for (let i = kPeriod - 1; i < n; i++) {
    let hh = -Infinity;
    let ll = Infinity;
    for (let j = i - kPeriod + 1; j <= i; j++) {
      if (high[j] > hh) hh = high[j];
      if (low[j] < ll) ll = low[j];
    }
    const span = hh - ll;
    last = span > 0 ? (100 * (close[i] - ll)) / span : last;
    raw[i] = last;
  }
  const k = smoothK > 1 ? sma(raw, smoothK) : raw;
  const d = sma(k, dPeriod);
  return { k, d };
}

/* -------------------------------------------------------------------------- */
/*  ADX / DMI                                                                 */
/* -------------------------------------------------------------------------- */

export interface AdxResult {
  adx: Float64Array;
  plusDI: Float64Array;
  minusDI: Float64Array;
}

/** Average Directional Index with +DI / −DI (same construction as `ta.dmi(len, len)`). */
export function adx(high: Float64Array, low: Float64Array, close: Float64Array, period: number): AdxResult {
  const n = close.length;
  const plusDM = nanArray(n);
  const minusDM = nanArray(n);
  const tr = nanArray(n);
  for (let i = 1; i < n; i++) {
    const up = high[i] - high[i - 1];
    const down = low[i - 1] - low[i];
    plusDM[i] = up > down && up > 0 ? up : 0;
    minusDM[i] = down > up && down > 0 ? down : 0;
    const prevClose = close[i - 1];
    tr[i] = Math.max(high[i] - low[i], Math.abs(high[i] - prevClose), Math.abs(low[i] - prevClose));
  }
  const trSmoothed = rma(tr, period);
  const plusSmoothed = rma(plusDM, period);
  const minusSmoothed = rma(minusDM, period);
  const plusDI = nanArray(n);
  const minusDI = nanArray(n);
  const dx = nanArray(n);
  let lastPlus = NaN;
  let lastMinus = NaN;
  for (let i = 0; i < n; i++) {
    const range = trSmoothed[i];
    if (range !== range) continue;
    if (range > 0) {
      lastPlus = (100 * plusSmoothed[i]) / range;
      lastMinus = (100 * minusSmoothed[i]) / range;
    } else if (lastPlus !== lastPlus) {
      lastPlus = 0;
      lastMinus = 0;
    }
    plusDI[i] = lastPlus;
    minusDI[i] = lastMinus;
    const sum = lastPlus + lastMinus;
    dx[i] = Math.abs(lastPlus - lastMinus) / (sum === 0 ? 1 : sum);
  }
  const adxRaw = rma(dx, period);
  const adxOut = nanArray(n);
  for (let i = 0; i < n; i++) adxOut[i] = 100 * adxRaw[i];
  return { adx: adxOut, plusDI, minusDI };
}

/* -------------------------------------------------------------------------- */
/*  Volume                                                                    */
/* -------------------------------------------------------------------------- */

/** Volume spike ratio: Volume / SMA(Volume, period). */
export function volumeRatio(volume: Float64Array, period: number): Float64Array {
  const n = volume.length;
  const avg = sma(volume, period);
  const out = nanArray(n);
  for (let i = 0; i < n; i++) {
    const a = avg[i];
    out[i] = a > 0 ? volume[i] / a : NaN;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/*  Supertrend                                                                */
/* -------------------------------------------------------------------------- */

export interface SupertrendResult {
  /** The Supertrend line: the lower band in an uptrend, the upper band in a downtrend. */
  line: Float64Array;
  /** +1 = uptrend (Close above the line), −1 = downtrend, NaN during warm-up. */
  direction: Float64Array;
}

/** Supertrend (ATR period + multiplier), matching `ta.supertrend(mult, period)`. */
export function supertrend(
  high: Float64Array,
  low: Float64Array,
  close: Float64Array,
  period: number,
  mult: number,
): SupertrendResult {
  const n = close.length;
  const atrValues = atr(high, low, close, period);
  const line = nanArray(n);
  const direction = nanArray(n);
  let prevUpper = NaN;
  let prevLower = NaN;
  let prevLine = NaN;
  let prevAtr = NaN;
  for (let i = 0; i < n; i++) {
    const a = atrValues[i];
    if (a !== a) continue;
    const mid = (high[i] + low[i]) / 2;
    let upper = mid + mult * a;
    let lower = mid - mult * a;
    let dir: number;
    if (prevAtr !== prevAtr) {
      // First bar with a valid ATR: TradingView starts in a downtrend.
      dir = -1;
    } else {
      const prevClose = close[i - 1];
      lower = lower > prevLower || prevClose < prevLower ? lower : prevLower;
      upper = upper < prevUpper || prevClose > prevUpper ? upper : prevUpper;
      if (prevLine === prevUpper) dir = close[i] > upper ? 1 : -1;
      else dir = close[i] < lower ? -1 : 1;
    }
    const value = dir === 1 ? lower : upper;
    line[i] = value;
    direction[i] = dir;
    prevUpper = upper;
    prevLower = lower;
    prevLine = value;
    prevAtr = a;
  }
  return { line, direction };
}

/* -------------------------------------------------------------------------- */
/*  CCI                                                                       */
/* -------------------------------------------------------------------------- */

/** Commodity Channel Index on the typical price (HLC3). */
export function cci(high: Float64Array, low: Float64Array, close: Float64Array, period: number): Float64Array {
  const n = close.length;
  const typical = new Float64Array(n);
  for (let i = 0; i < n; i++) typical[i] = (high[i] + low[i] + close[i]) / 3;
  const mean = sma(typical, period);
  const out = nanArray(n);
  for (let i = period - 1; i < n; i++) {
    const m = mean[i];
    if (m !== m) continue;
    let deviation = 0;
    for (let j = i - period + 1; j <= i; j++) deviation += Math.abs(typical[j] - m);
    deviation /= period;
    out[i] = deviation > 0 ? (typical[i] - m) / (0.015 * deviation) : 0;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/*  Memoising indicator cache                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Donchian channel of the PRIOR `period` candles: the highest high (upper) or lowest low (lower)
 * of bars `i - period .. i - 1`. The current candle is excluded so "close above the channel"
 * is a genuine breakout of the earlier range – Pine: `ta.highest(high, period)[1]`.
 */
export function donchian(src: Float64Array, period: number, upper: boolean): Float64Array {
  const n = src.length;
  const out = nanArray(n);
  // Monotonic deque of indices over the window [i - period, i - 1].
  const dq = new Int32Array(n);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < n; i++) {
    // The deque holds bars up to i − 1; drop those older than the window [i − period, i − 1].
    while (head < tail && dq[head] < i - period) head++;
    if (i >= period) out[i] = src[dq[head]];
    const v = src[i];
    while (tail > head && (upper ? src[dq[tail - 1]] <= v : src[dq[tail - 1]] >= v)) tail--;
    dq[tail++] = i;
  }
  return out;
}

/** Keltner channel: EMA(close, period) ± mult × ATR(period) (Wilder ATR, like `ta.atr`). */
export function keltner(
  high: Float64Array,
  low: Float64Array,
  close: Float64Array,
  period: number,
  mult: number,
): { middle: Float64Array; upper: Float64Array; lower: Float64Array } {
  const n = close.length;
  const middle = ema(close, period);
  const range = atr(high, low, close, period);
  const upper = nanArray(n);
  const lower = nanArray(n);
  for (let i = 0; i < n; i++) {
    const m = middle[i];
    const r = range[i];
    if (m === m && r === r) {
      upper[i] = m + mult * r;
      lower[i] = m - mult * r;
    }
  }
  return { middle, upper, lower };
}

/**
 * Volatility-squeeze release (TTM-style): the squeeze is ON while Bollinger(period, 2σ) lies inside
 * Keltner(period, kcMult). The output is +1 on the candle the squeeze turns OFF with the close above
 * the Bollinger basis, −1 when it turns off with the close below it, 0 otherwise (NaN in warm-up).
 */
export function squeezeRelease(
  high: Float64Array,
  low: Float64Array,
  close: Float64Array,
  period: number,
  kcMult: number,
): Float64Array {
  const n = close.length;
  const bb = bollinger(close, period, 2);
  const kc = keltner(high, low, close, period, kcMult);
  const out = nanArray(n);
  let prevOn = -1; // -1 = unknown (warm-up)
  for (let i = 0; i < n; i++) {
    const bu = bb.upper[i];
    const bl = bb.lower[i];
    const ku = kc.upper[i];
    const kl = kc.lower[i];
    if (!(bu === bu && bl === bl && ku === ku && kl === kl)) {
      prevOn = -1;
      continue;
    }
    const on = bu < ku && bl > kl ? 1 : 0;
    if (prevOn !== -1) {
      const basis = bb.middle[i];
      out[i] = prevOn === 1 && on === 0 ? (close[i] > basis ? 1 : close[i] < basis ? -1 : 0) : 0;
    }
    prevOn = on;
  }
  return out;
}

/**
 * ATR range expansion: +1 when the candle's true range exceeds `mult` × ATR(period) of the previous
 * candle and it closed up, −1 when it closed down, 0 otherwise (NaN while the ATR warms up).
 */
export function rangeExpansion(
  high: Float64Array,
  low: Float64Array,
  close: Float64Array,
  period: number,
  mult: number,
): Float64Array {
  const n = close.length;
  const tr = trueRange(high, low, close);
  const a = atr(high, low, close, period);
  const out = nanArray(n);
  for (let i = 1; i < n; i++) {
    const prev = a[i - 1];
    if (prev !== prev) continue;
    out[i] = tr[i] > mult * prev ? (close[i] > close[i - 1] ? 1 : close[i] < close[i - 1] ? -1 : 0) : 0;
  }
  return out;
}

/**
 * Pullback to a moving average: +1 when the candle dipped to the average and closed back above it
 * (Low ≤ MA < Close), −1 when it rallied to it and closed back below (High ≥ MA > Close), else 0.
 */
export function pullbackTouch(ma: Float64Array, high: Float64Array, low: Float64Array, close: Float64Array): Float64Array {
  const n = close.length;
  const out = nanArray(n);
  for (let i = 0; i < n; i++) {
    const m = ma[i];
    if (m !== m) continue;
    out[i] = low[i] <= m && close[i] > m ? 1 : high[i] >= m && close[i] < m ? -1 : 0;
  }
  return out;
}

/** `src` shifted `lag` bars back (NaN for the first `lag` bars) – Pine: `src[lag]`. */
export function lagged(src: Float64Array, lag: number): Float64Array {
  const out = nanArray(src.length);
  for (let i = lag; i < src.length; i++) out[i] = src[i - lag];
  return out;
}

/**
 * Lazily computes and memoises indicator series for one dataset.
 *
 * The optimiser evaluates tens of thousands of strategies that mostly share
 * indicator settings, so each series is computed once and reused. A small LRU
 * bound keeps memory in check on long intraday histories (44k+ bars).
 */
export class IndicatorCache {
  readonly n: number;
  private readonly store = new Map<string, Float64Array>();
  private readonly maxEntries: number;

  constructor(
    readonly data: OHLCV,
    memoryBudgetBytes = 192 * 1024 * 1024,
  ) {
    this.n = data.c.length;
    const bytesPerSeries = Math.max(1, this.n) * 8;
    this.maxEntries = Math.max(96, Math.min(8000, Math.floor(memoryBudgetBytes / bytesPerSeries)));
  }

  /** Number of memoised series currently held (exposed for tests / diagnostics). */
  get size(): number {
    return this.store.size;
  }

  private read(key: string): Float64Array | undefined {
    const hit = this.store.get(key);
    // Recency only matters once eviction is in sight; below half capacity a plain lookup is enough.
    if (hit !== undefined && this.store.size * 2 > this.maxEntries) {
      // Refresh recency: Map iteration order is insertion order.
      this.store.delete(key);
      this.store.set(key, hit);
    }
    return hit;
  }

  private write(key: string, value: Float64Array): Float64Array {
    this.store.set(key, value);
    while (this.store.size > this.maxEntries) {
      const oldest = this.store.keys().next().value;
      if (oldest === undefined) break;
      this.store.delete(oldest);
    }
    return value;
  }

  private memo(key: string, compute: () => Float64Array): Float64Array {
    return this.read(key) ?? this.write(key, compute());
  }

  sma(period: number): Float64Array {
    return this.memo(`sma:${period}`, () => sma(this.data.c, period));
  }

  ema(period: number): Float64Array {
    return this.memo(`ema:${period}`, () => ema(this.data.c, period));
  }

  ma(type: MaType, period: number): Float64Array {
    return type === 'EMA' ? this.ema(period) : this.sma(period);
  }

  rsi(period: number): Float64Array {
    return this.memo(`rsi:${period}`, () => rsi(this.data.c, period));
  }

  macdLine(fast: number, slow: number): Float64Array {
    return this.memo(`macdL:${fast}:${slow}`, () => {
      const f = this.ema(fast);
      const s = this.ema(slow);
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) out[i] = f[i] - s[i];
      return out;
    });
  }

  macdSignal(fast: number, slow: number, signal: number): Float64Array {
    return this.memo(`macdS:${fast}:${slow}:${signal}`, () => ema(this.macdLine(fast, slow), signal));
  }

  macdHist(fast: number, slow: number, signal: number): Float64Array {
    return this.memo(`macdH:${fast}:${slow}:${signal}`, () => {
      const line = this.macdLine(fast, slow);
      const sig = this.macdSignal(fast, slow, signal);
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) out[i] = line[i] - sig[i];
      return out;
    });
  }

  /** Histogram shifted one bar back, i.e. `hist[1]` in Pine. */
  macdHistPrev(fast: number, slow: number, signal: number): Float64Array {
    return this.memo(`macdHP:${fast}:${slow}:${signal}`, () => {
      const hist = this.macdHist(fast, slow, signal);
      const out = nanArray(this.n);
      for (let i = 1; i < this.n; i++) out[i] = hist[i - 1];
      return out;
    });
  }

  private bb(period: number, mult: number, part: 'U' | 'L' | 'P' | 'W'): Float64Array {
    const base = `bb:${period}:${mult}:`;
    const hit = this.read(base + part);
    if (hit !== undefined) return hit;
    const res = bollinger(this.data.c, period, mult);
    this.write(base + 'U', res.upper);
    this.write(base + 'L', res.lower);
    this.write(base + 'P', res.percentB);
    this.write(base + 'W', res.bandwidth);
    return part === 'U' ? res.upper : part === 'L' ? res.lower : part === 'P' ? res.percentB : res.bandwidth;
  }

  bbUpper(period: number, mult: number): Float64Array {
    return this.bb(period, mult, 'U');
  }

  bbLower(period: number, mult: number): Float64Array {
    return this.bb(period, mult, 'L');
  }

  bbPercentB(period: number, mult: number): Float64Array {
    return this.bb(period, mult, 'P');
  }

  bbBandwidth(period: number, mult: number): Float64Array {
    return this.bb(period, mult, 'W');
  }

  atr(period: number): Float64Array {
    return this.memo(`atr:${period}`, () => atr(this.data.h, this.data.l, this.data.c, period));
  }

  /** ATR expressed as a percentage of the close. */
  atrPercent(period: number): Float64Array {
    return this.memo(`atrP:${period}`, () => {
      const a = this.atr(period);
      const c = this.data.c;
      const out = nanArray(this.n);
      for (let i = 0; i < this.n; i++) out[i] = c[i] > 0 ? (a[i] / c[i]) * 100 : NaN;
      return out;
    });
  }

  stochK(kPeriod: number, smoothK: number): Float64Array {
    return this.memo(
      `stK:${kPeriod}:${smoothK}`,
      () => stochastic(this.data.h, this.data.l, this.data.c, kPeriod, smoothK, 1).k,
    );
  }

  stochD(kPeriod: number, smoothK: number, dPeriod: number): Float64Array {
    return this.memo(`stD:${kPeriod}:${smoothK}:${dPeriod}`, () => sma(this.stochK(kPeriod, smoothK), dPeriod));
  }

  private dmi(period: number, part: 'A' | 'P' | 'M'): Float64Array {
    const base = `dmi:${period}:`;
    const hit = this.read(base + part);
    if (hit !== undefined) return hit;
    const res = adx(this.data.h, this.data.l, this.data.c, period);
    this.write(base + 'A', res.adx);
    this.write(base + 'P', res.plusDI);
    this.write(base + 'M', res.minusDI);
    return part === 'A' ? res.adx : part === 'P' ? res.plusDI : res.minusDI;
  }

  adx(period: number): Float64Array {
    return this.dmi(period, 'A');
  }

  plusDI(period: number): Float64Array {
    return this.dmi(period, 'P');
  }

  minusDI(period: number): Float64Array {
    return this.dmi(period, 'M');
  }

  volumeRatio(period: number): Float64Array {
    return this.memo(`vol:${period}`, () => volumeRatio(this.data.v, period));
  }

  supertrend(period: number, mult: number): Float64Array {
    return this.memo(
      `st:${period}:${mult}`,
      () => supertrend(this.data.h, this.data.l, this.data.c, period, mult).line,
    );
  }

  cci(period: number): Float64Array {
    return this.memo(`cci:${period}`, () => cci(this.data.h, this.data.l, this.data.c, period));
  }

  /** Highest high of the previous `period` candles. */
  donchianUpper(period: number): Float64Array {
    return this.memo(`dcU:${period}`, () => donchian(this.data.h, period, true));
  }

  /** Lowest low of the previous `period` candles. */
  donchianLower(period: number): Float64Array {
    return this.memo(`dcL:${period}`, () => donchian(this.data.l, period, false));
  }

  private kc(period: number, mult: number, part: 'U' | 'L'): Float64Array {
    const base = `kc:${period}:${mult}:`;
    const hit = this.read(base + part);
    if (hit !== undefined) return hit;
    const res = keltner(this.data.h, this.data.l, this.data.c, period, mult);
    this.write(base + 'U', res.upper);
    this.write(base + 'L', res.lower);
    return part === 'U' ? res.upper : res.lower;
  }

  /** Upper Keltner channel: EMA(period) + mult × ATR(period). */
  kcUpper(period: number, mult: number): Float64Array {
    return this.kc(period, mult, 'U');
  }

  /** Lower Keltner channel: EMA(period) − mult × ATR(period). */
  kcLower(period: number, mult: number): Float64Array {
    return this.kc(period, mult, 'L');
  }

  /** Squeeze release series (+1 / −1 / 0), see `squeezeRelease`. */
  squeeze(period: number, kcMult: number): Float64Array {
    return this.memo(`sq:${period}:${kcMult}`, () => squeezeRelease(this.data.h, this.data.l, this.data.c, period, kcMult));
  }

  /** Range-expansion series (+1 / −1 / 0), see `rangeExpansion`. */
  rangeExp(period: number, mult: number): Float64Array {
    return this.memo(`rx:${period}:${mult}`, () => rangeExpansion(this.data.h, this.data.l, this.data.c, period, mult));
  }

  /** Pullback-to-MA series (+1 / −1 / 0), see `pullbackTouch`. */
  pullback(type: MaType, period: number): Float64Array {
    return this.memo(`pb:${type}:${period}`, () => pullbackTouch(this.ma(type, period), this.data.h, this.data.l, this.data.c));
  }

  /** Moving average `lag` bars ago (to compare with its current value: the slope's sign). */
  maLagged(type: MaType, period: number, lag: number): Float64Array {
    return this.memo(`maLag:${type}:${period}:${lag}`, () => lagged(this.ma(type, period), lag));
  }
}
