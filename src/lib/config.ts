/** UI-level run configuration and the immutable context captured when a run starts. */

import { normalizeWeights, TICKER_PATTERN } from '../data/basket';
import type {
  BacktestSettings,
  BasketComponent,
  DataSource,
  HistoryRange,
  Interval,
  TradeMode,
} from '../types/trading';

export const PRESET_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'SPY', 'NVDA', 'AAPL'] as const;
export const MAX_BASKET_ASSETS = 8;

export interface BasketRow {
  id: number;
  symbol: string;
  weight: number;
}

export interface RunConfig {
  assetMode: 'single' | 'basket';
  symbol: string;
  basket: BasketRow[];
  interval: Interval;
  range: HistoryRange;
  capital: number;
  feePct: number;
  slippagePct: number;
  tradeMode: TradeMode;
}

export const DEFAULT_CONFIG: RunConfig = {
  assetMode: 'single',
  symbol: 'BTC-USD',
  basket: [
    { id: 1, symbol: 'BTC-USD', weight: 50 },
    { id: 2, symbol: 'ETH-USD', weight: 30 },
    { id: 3, symbol: 'NVDA', weight: 20 },
  ],
  interval: '1d',
  range: '5y',
  capital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  tradeMode: 'long-only',
};

export type ResolvedAssets = { ok: true; components: BasketComponent[] } | { ok: false; error: string };

/** Validate the asset selection and return normalised basket components. */
export function resolveAssets(config: RunConfig): ResolvedAssets {
  if (config.assetMode === 'single') {
    const symbol = config.symbol.trim().toUpperCase();
    if (!TICKER_PATTERN.test(symbol)) return { ok: false, error: 'Enter a valid ticker, for example BTC-USD or NVDA.' };
    return { ok: true, components: [{ symbol, weight: 1 }] };
  }
  const rows = config.basket.filter((row) => row.symbol.trim() !== '');
  const invalid = rows.find((row) => !TICKER_PATTERN.test(row.symbol.trim().toUpperCase()));
  if (invalid) return { ok: false, error: `"${invalid.symbol}" is not a valid ticker.` };
  const components = normalizeWeights(rows);
  if (components.length === 0) return { ok: false, error: 'Add at least one basket asset with a weight above zero.' };
  return { ok: true, components };
}

export function settingsFrom(config: RunConfig): BacktestSettings {
  return {
    initialCapital: config.capital,
    feePct: config.feePct,
    slippagePct: config.slippagePct,
    tradeMode: config.tradeMode,
  };
}

/** Everything about a run that must not change when the user edits the controls afterwards. */
export interface RunContext {
  label: string;
  interval: Interval;
  range: HistoryRange;
  settings: BacktestSettings;
  components: BasketComponent[];
  units: number[];
  isBasket: boolean;
  bars: number;
  windowStart: number;
  windowEnd: number;
  sources: DataSource[];
  /** True when the offline fallback generator supplied the candles. */
  simulated: boolean;
  fallbackReason: string | null;
  notes: string[];
}

export const SOURCE_LABEL: Record<DataSource, string> = {
  binance: 'Binance',
  yahoo: 'Yahoo Finance',
  synthetic: 'Simulated',
};
