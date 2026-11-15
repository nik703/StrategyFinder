/**
 * Asset selection helpers: weight normalisation and construction of the
 * dataset the optimiser trades – either a single ticker or a synthetic,
 * weighted multi-asset basket.
 */

import type { BasketComponent, CandleSeries, DataSource, OHLCV } from '../types/trading';

export const TICKER_PATTERN = /^[A-Z0-9][A-Z0-9.\-^=]{0,19}$/;

/** A user-entered allocation before normalisation. */
export interface RawAllocation {
  symbol: string;
  weight: number;
}

/**
 * Normalise arbitrary positive weights so they sum to exactly 1.
 * Blank symbols and non-positive weights are dropped; duplicate symbols are merged.
 */
export function normalizeWeights(allocations: readonly RawAllocation[]): BasketComponent[] {
  const merged = new Map<string, number>();
  for (const { symbol, weight } of allocations) {
    const ticker = symbol.trim().toUpperCase();
    if (!ticker || !Number.isFinite(weight) || weight <= 0) continue;
    merged.set(ticker, (merged.get(ticker) ?? 0) + weight);
  }
  let total = 0;
  for (const weight of merged.values()) total += weight;
  if (total <= 0) return [];
  return [...merged.entries()].map(([symbol, weight]) => ({ symbol, weight: weight / total }));
}

export function describeAllocation(components: readonly BasketComponent[]): string {
  if (components.length === 1) return components[0].symbol;
  return components.map((c) => `${formatWeight(c.weight)} ${c.symbol}`).join(' · ');
}

export function formatWeight(weight: number): string {
  const pct = weight * 100;
  return `${Number.isInteger(Math.round(pct * 10) / 10) ? Math.round(pct) : pct.toFixed(1)}%`;
}

export interface Dataset {
  /** Human-readable name, e.g. "BTC-USD" or "50% BTC-USD · 30% ETH-USD · 20% NVDA". */
  label: string;
  components: BasketComponent[];
  /**
   * Units of each component held per 100 units of basket value at inception
   * (basket = Σ unitsᵢ × priceᵢ). Empty for a single asset.
   */
  units: number[];
  data: OHLCV;
  sources: DataSource[];
  isBasket: boolean;
}

function toTyped(series: CandleSeries): OHLCV {
  return {
    t: Float64Array.from(series.t),
    o: Float64Array.from(series.o),
    h: Float64Array.from(series.h),
    l: Float64Array.from(series.l),
    c: Float64Array.from(series.c),
    v: Float64Array.from(series.v),
  };
}

/** Trailing window, in bars, over which each basket leg's volume is normalised. */
export const VOLUME_NORM_WINDOW = 100;

/**
 * Build the tradable dataset.
 *
 * Single asset: the candles are used as-is.
 *
 * Basket: components are aligned on the timestamps they ALL share (a basket
 * that holds a stock can only be traded while the stock market is open), then
 * combined into a buy-and-hold index that starts at 100:
 *
 *     unitsᵢ   = weightᵢ × 100 / openᵢ[0]
 *     basket X = Σ unitsᵢ × Xᵢ        for X ∈ {open, close}
 *
 * Weights therefore describe the allocation at inception and drift with
 * prices afterwards, exactly like a real buy-and-hold portfolio.
 *
 * High / low. Summing component highs (or lows) would assume every leg peaks at
 * the same instant – a price the basket almost never trades at, which lets
 * take-profits fill on prices that never existed. Instead the basket range is
 * its body (open ↔ close, which the basket certainly traded) widened by the
 * root-sum-square of the legs' wicks:
 *
 *     high = max(open, close) + √Σ (unitsᵢ × (highᵢ − max(openᵢ, closeᵢ)))²
 *     low  = min(open, close) − √Σ (unitsᵢ × (min(openᵢ, closeᵢ) − lowᵢ))²
 *
 * This always lies inside the summed envelope, never inside the body, equals the
 * real candle for a single leg, and keeps the low above Σ unitsᵢ × lowᵢ > 0.
 *
 * Volume is the weight-averaged RELATIVE volume of the components. Each leg's
 * volume is divided by its own trailing mean over the last `VOLUME_NORM_WINDOW`
 * bars (bar i only uses bars ≤ i), so no later volume – least of all the
 * holdout's – leaks into earlier candles.
 */
export function buildDataset(
  seriesBySymbol: Readonly<Record<string, CandleSeries>>,
  components: readonly BasketComponent[],
): Dataset {
  if (components.length === 0) throw new Error('Select at least one asset with a positive weight.');
  const legs = components.map((component) => {
    const series = seriesBySymbol[component.symbol];
    if (!series || series.t.length === 0) throw new Error(`No market data available for ${component.symbol}.`);
    return { component, series };
  });
  const sources = [...new Set(legs.map((leg) => leg.series.source))];
  const label = describeAllocation(components);

  if (legs.length === 1) {
    return {
      label,
      components: [{ symbol: legs[0].component.symbol, weight: 1 }],
      units: [],
      data: toTyped(legs[0].series),
      sources,
      isBasket: false,
    };
  }

  // Timestamps present in every component.
  const indexMaps = legs.map((leg) => {
    const map = new Map<number, number>();
    leg.series.t.forEach((time, index) => map.set(time, index));
    return map;
  });
  const shared = legs[0].series.t.filter((time) => indexMaps.every((map) => map.has(time))).sort((a, b) => a - b);
  if (shared.length < 2) {
    throw new Error(
      `The basket components share almost no common trading timestamps (${shared.length}). Try the 1d timeframe.`,
    );
  }

  const n = shared.length;
  const data: OHLCV = {
    t: Float64Array.from(shared),
    o: new Float64Array(n),
    h: new Float64Array(n),
    l: new Float64Array(n),
    c: new Float64Array(n),
    v: new Float64Array(n),
  };
  const units: number[] = [];
  // Squared dollar wicks of the legs, summed per bar (see the high / low note above).
  const upperWick = new Float64Array(n);
  const lowerWick = new Float64Array(n);

  legs.forEach((leg, legIndex) => {
    const { series, component } = leg;
    const map = indexMaps[legIndex];
    const rows = shared.map((time) => map.get(time) as number);
    const firstOpen = series.o[rows[0]];
    if (!(firstOpen > 0)) throw new Error(`Invalid opening price for ${component.symbol}.`);
    const unit = (component.weight * 100) / firstOpen;
    units.push(unit);

    // Trailing (causal) mean volume of this leg over the shared bars.
    let windowSum = 0;
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      const o = series.o[row];
      const c = series.c[row];
      data.o[i] += unit * o;
      data.c[i] += unit * c;
      const up = unit * (series.h[row] - Math.max(o, c));
      const down = unit * (Math.min(o, c) - series.l[row]);
      upperWick[i] += up > 0 ? up * up : 0;
      lowerWick[i] += down > 0 ? down * down : 0;

      const volume = series.v[row] > 0 ? series.v[row] : 0;
      windowSum += volume;
      if (i >= VOLUME_NORM_WINDOW) {
        const old = series.v[rows[i - VOLUME_NORM_WINDOW]];
        windowSum -= old > 0 ? old : 0;
      }
      const mean = windowSum / Math.min(i + 1, VOLUME_NORM_WINDOW);
      if (mean > 0) data.v[i] += component.weight * (volume / mean) * 1_000_000;
    }
  });

  for (let i = 0; i < n; i++) {
    const o = data.o[i];
    const c = data.c[i];
    data.h[i] = Math.max(o, c) + Math.sqrt(upperWick[i]);
    data.l[i] = Math.min(o, c) - Math.sqrt(lowerWick[i]);
  }

  return { label, components: [...components], units, data, sources, isBasket: true };
}
