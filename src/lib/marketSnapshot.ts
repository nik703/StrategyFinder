/**
 * Latest candles for the Next Entry Forecast.
 *
 * A snapshot holds the traded instrument (a single asset or the basket index)
 * plus, for baskets and Robust-universe runs, every individual asset, so the
 * forecast can rank them. It is built from the run's own data when the run
 * starts and rebuilt from freshly fetched candles on "Refresh Latest Price".
 */

import { buildDataset } from '../data/basket';
import { loadMarketData } from '../data/marketData';
import type { CandleSeries, OHLCV } from '../types/trading';
import { venueLabel, type RunContext } from './config';

export interface ForecastAsset {
  /** Stable key, e.g. "BTC-USD" or "Basket". */
  key: string;
  label: string;
  data: OHLCV;
  /** The instrument the strategy was optimised on. */
  primary: boolean;
}

export interface MarketSnapshot {
  assets: ForecastAsset[];
  /** When the candles were fetched (ms). */
  fetchedAt: number;
  simulated: boolean;
  fallbackReason: string | null;
  venue: string;
}

type SnapshotContext = Pick<RunContext, 'label' | 'components' | 'isBasket' | 'universe' | 'interval' | 'range'>;

/** Every symbol the snapshot needs: basket components and universe assets. */
export function snapshotSymbols(ctx: SnapshotContext): string[] {
  const symbols = ctx.components.map((component) => component.symbol);
  for (const symbol of ctx.universe) if (symbol !== 'Basket' && !symbols.includes(symbol)) symbols.push(symbol);
  return symbols;
}

export function buildSnapshot(
  series: Readonly<Record<string, CandleSeries>>,
  ctx: SnapshotContext,
  meta: { simulated: boolean; fallbackReason: string | null; fetchedAt?: number },
): MarketSnapshot {
  const single = (symbol: string): OHLCV => buildDataset(series, [{ symbol, weight: 1 }]).data;
  const assets: ForecastAsset[] = [];
  if (ctx.isBasket) {
    assets.push({ key: 'Basket', label: ctx.label, data: buildDataset(series, ctx.components).data, primary: true });
    for (const { symbol } of ctx.components) assets.push({ key: symbol, label: symbol, data: single(symbol), primary: false });
  } else {
    const primary = ctx.components[0].symbol;
    assets.push({ key: primary, label: primary, data: single(primary), primary: true });
    for (const symbol of ctx.universe) {
      if (symbol !== primary && series[symbol]) assets.push({ key: symbol, label: symbol, data: single(symbol), primary: false });
    }
  }
  const traded = ctx.components.map((component) => series[component.symbol]).filter((s): s is CandleSeries => s !== undefined);
  return {
    assets,
    fetchedAt: meta.fetchedAt ?? Date.now(),
    simulated: meta.simulated,
    fallbackReason: meta.fallbackReason,
    venue: venueLabel(traded),
  };
}

/** Fetch the latest candles through the normal data pipeline (live, or the offline fallback). */
export async function fetchSnapshot(ctx: SnapshotContext): Promise<MarketSnapshot> {
  const loaded = await loadMarketData(snapshotSymbols(ctx), ctx.interval, ctx.range);
  return buildSnapshot(loaded.series, ctx, { simulated: loaded.usedFallback, fallbackReason: loaded.fallbackReason });
}
