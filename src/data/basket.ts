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
 *     basket X = Σ unitsᵢ × Xᵢ        for X ∈ {open, high, low, close}
 *
 * Weights therefore describe the allocation at inception and drift with
 * prices afterwards, exactly like a real buy-and-hold portfolio. The basket
 * high/low are the sums of component highs/lows (a conservative envelope),
 * and volume is the weight-averaged relative volume of the components.
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

  legs.forEach((leg, legIndex) => {
    const { series, component } = leg;
    const map = indexMaps[legIndex];
    const rows = shared.map((time) => map.get(time) as number);
    const firstOpen = series.o[rows[0]];
    if (!(firstOpen > 0)) throw new Error(`Invalid opening price for ${component.symbol}.`);
    const unit = (component.weight * 100) / firstOpen;
    units.push(unit);

    let volumeSum = 0;
    for (const row of rows) volumeSum += series.v[row];
    const meanVolume = volumeSum / n;

    for (let i = 0; i < n; i++) {
      const row = rows[i];
      data.o[i] += unit * series.o[row];
      data.h[i] += unit * series.h[row];
      data.l[i] += unit * series.l[row];
      data.c[i] += unit * series.c[row];
      if (meanVolume > 0) data.v[i] += component.weight * (series.v[row] / meanVolume) * 1_000_000;
    }
  });

  return { label, components: [...components], units, data, sources, isBasket: true };
}
