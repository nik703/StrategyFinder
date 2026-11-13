/** UI-level run configuration and the immutable context captured when a run starts. */

import { normalizeWeights, TICKER_PATTERN } from '../data/basket';
import {
  DEFAULT_CRYPTO_UNIVERSE,
  DEFAULT_MAX_STAGNATION,
  DEFAULT_STOCK_UNIVERSE,
  INTERVAL_MS,
  type CandleSeries,
} from '../types/trading';
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
  /** Stop after this many consecutive iterations without improvement. */
  maxStagnation: number;
  /** fixed: risk % of starting capital · compound: risk % of current equity · full: 100% of equity. */
  sizingMode: 'fixed' | 'compound' | 'full';
  /** Share of capital lost when a trade is stopped out, in percent. */
  riskPct: number;
  /** Perpetual funding charged on open positions, in percent per 8 hours (0 = off). */
  fundingPct8h: number;
  /** Score every strategy on a whole universe of assets instead of one. */
  robustUniverse: boolean;
  /** Comma-separated tickers of the universe. */
  universe: string;
}

export const MAX_UNIVERSE_ASSETS = 8;
export const DEFAULT_FUNDING_PCT_8H = 0.01;

export const isCrypto = (symbol: string): boolean => /-(USD|USDT|USDC)$/.test(symbol.trim().toUpperCase());

export const defaultUniverse = (symbol: string): string =>
  (isCrypto(symbol) ? DEFAULT_CRYPTO_UNIVERSE : DEFAULT_STOCK_UNIVERSE).join(', ');

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
  maxStagnation: DEFAULT_MAX_STAGNATION,
  sizingMode: 'fixed',
  riskPct: 2,
  fundingPct8h: 0,
  // On by default for crypto (the default ticker is BTC-USD).
  robustUniverse: true,
  universe: DEFAULT_CRYPTO_UNIVERSE.join(', '),
};

/** Parse a comma-separated ticker list into unique, upper-case symbols. */
export function parseUniverse(text: string): string[] {
  const out: string[] = [];
  for (const part of text.split(/[,\s]+/)) {
    const symbol = part.trim().toUpperCase();
    if (symbol && !out.includes(symbol)) out.push(symbol);
  }
  return out;
}

export type ResolvedUniverse = { ok: true; symbols: string[] } | { ok: false; error: string };

/**
 * Symbols a Robust-universe run is scored on: the primary asset first, then the
 * rest of the list. Returns an empty list when the mode is off (or a basket is traded).
 */
export function resolveUniverse(config: RunConfig, primary: string): ResolvedUniverse {
  if (!config.robustUniverse || config.assetMode !== 'single') return { ok: true, symbols: [] };
  const listed = parseUniverse(config.universe);
  const invalid = listed.find((symbol) => !TICKER_PATTERN.test(symbol));
  if (invalid) return { ok: false, error: `"${invalid}" in the Robust universe is not a valid ticker.` };
  const symbols = [primary, ...listed.filter((symbol) => symbol !== primary)];
  if (symbols.length < 2) return { ok: false, error: 'A Robust universe needs at least one asset besides the one being traded.' };
  if (symbols.length > MAX_UNIVERSE_ASSETS) {
    return { ok: false, error: `A Robust universe can hold at most ${MAX_UNIVERSE_ASSETS} assets.` };
  }
  return { ok: true, symbols };
}

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
    sizing: {
      mode: config.sizingMode === 'full' ? 'full' : 'risk',
      riskPct: config.riskPct,
      compounding: config.sizingMode !== 'fixed',
    },
    // The setting is quoted per 8 hours; the engine charges it per candle.
    fundingPctPerBar: config.fundingPct8h * (INTERVAL_MS[config.interval] / (8 * 3_600_000)),
  };
}

/** Everything about a run that must not change when the user edits the controls afterwards. */
export interface RunContext {
  label: string;
  interval: Interval;
  range: HistoryRange;
  settings: BacktestSettings;
  /** Stagnation limit this run was started with. */
  maxStagnation: number;
  /** Assets the run was scored on (the primary first); a single entry when the Robust universe is off. */
  universe: string[];
  /** Instrument and venue actually traded, e.g. "XRPUSDT on Binance". */
  venue: string;
  /** Identifies this dataset in the holdout log. */
  fingerprint: string;
  /** How many times this dataset's holdout had been revealed before this run. */
  priorHoldoutViews: number;
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

/** "XRPUSDT on Binance" – the instrument the candles really came from. */
export function venueLabel(series: readonly CandleSeries[]): string {
  if (series.length === 0) return 'unknown';
  if (series.some((s) => s.source === 'synthetic')) return `${series.map((s) => s.symbol).join(' + ')} (simulated prices)`;
  return series.map((s) => `${s.venueSymbol ?? s.symbol} on ${SOURCE_LABEL[s.source]}`).join(' + ');
}
