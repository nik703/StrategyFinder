/** UI-level run configuration and the immutable context captured when a run starts. */

import { normalizeWeights, TICKER_PATTERN } from '../data/basket';
import type { HoldoutWindow } from './holdoutLog';
import type { TrialWindow } from './trialLedger';
import {
  DEFAULT_CRYPTO_UNIVERSE,
  DEFAULT_RUN_EVALUATIONS,
  DEFAULT_STOCK_UNIVERSE,
  INTERVAL_MS,
  type CandleSeries,
} from '../types/trading';
import type {
  Hypothesis,
  TrialStats,
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
  /** Fixed slippage floor per fill, in percent. */
  slippagePct: number;
  /** Volatility-scaled slippage: each fill slips at least this multiple of ATR(14) as a % of price. */
  slippageAtrMult: number;
  tradeMode: TradeMode;
  /** The market hypothesis the next run tests. */
  hypothesis: Hypothesis;
  /** Evaluations (trials) the next run may spend. */
  maxEvaluations: number;
  /** fixed: risk % of starting capital · compound: risk % of current equity · full: 100% of equity. */
  sizingMode: 'fixed' | 'compound' | 'full';
  /** Share of capital lost when a trade is stopped out, in percent. */
  riskPct: number;
  /**
   * Perpetual funding charged on every open position, in percent per 8 hours (0 = off).
   * Applies to crypto in Long/Short mode, where shorting means trading a perpetual.
   */
  fundingPct8h: number;
  /** Borrow fee charged on short stock / ETF positions, in percent per year (Long/Short mode only). */
  borrowPctYear: number;
  /** Score every strategy on a whole universe of assets instead of one. */
  robustUniverse: boolean;
  /** Comma-separated tickers of the universe. */
  universe: string;
}

export const MAX_UNIVERSE_ASSETS = 8;
/** Typical perpetual funding rate (≈ 11% a year). */
export const DEFAULT_FUNDING_PCT_8H = 0.01;
/** Retail borrow cost of shorting stocks and ETFs, in percent per year. */
export const DEFAULT_BORROW_PCT_YEAR = 2;
/** Default volatility-scaled slippage: 0.03 × ATR% (≈ 0.09% on daily BTC, ≈ 0.18% on daily DOGE). */
export const DEFAULT_SLIPPAGE_ATR_MULT = 0.03;

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
  // 0.1% matches crypto taker fees and is conservative for commission-free stock brokers.
  feePct: 0.1,
  slippagePct: 0.05,
  slippageAtrMult: DEFAULT_SLIPPAGE_ATR_MULT,
  tradeMode: 'long-only',
  hypothesis: 'TREND_FOLLOWING',
  maxEvaluations: DEFAULT_RUN_EVALUATIONS,
  sizingMode: 'fixed',
  riskPct: 2,
  // Financing is on by default; it only applies to Long/Short runs (Long-Only is treated as spot).
  fundingPct8h: DEFAULT_FUNDING_PCT_8H,
  borrowPctYear: DEFAULT_BORROW_PCT_YEAR,
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

/** Every symbol a run with this configuration scores strategies on (traded assets and universe). */
export function runSymbols(config: RunConfig): string[] {
  const assets = resolveAssets(config);
  if (!assets.ok) return [];
  const traded = assets.components.map((component) => component.symbol);
  const universe = resolveUniverse(config, traded[0]);
  return [...new Set([...traded, ...(universe.ok ? universe.symbols : [])])];
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

/** True when every traded instrument is a crypto pair (a crypto basket counts; a mixed basket does not). */
export function tradesCrypto(config: RunConfig): boolean {
  const symbols =
    config.assetMode === 'single' ? [config.symbol] : config.basket.filter((row) => row.symbol.trim() !== '' && row.weight > 0).map((row) => row.symbol);
  return symbols.length > 0 && symbols.every(isCrypto);
}

/** How Long/Short positions are financed for this configuration. */
export type Financing = 'none' | 'funding' | 'borrow';

/**
 * Long-Only runs are treated as spot positions: no financing. Long/Short crypto is traded as a
 * perpetual, so every position pays funding; Long/Short stocks pay a borrow fee on the shorts.
 */
export function financingFor(config: RunConfig): Financing {
  if (config.tradeMode !== 'long-short') return 'none';
  return tradesCrypto(config) ? 'funding' : 'borrow';
}

const MS_PER_YEAR = 365.25 * 86_400_000;

export function settingsFrom(config: RunConfig): BacktestSettings {
  const financing = financingFor(config);
  const barMs = INTERVAL_MS[config.interval];
  return {
    initialCapital: config.capital,
    feePct: config.feePct,
    slippagePct: config.slippagePct,
    slippageAtrMult: config.slippageAtrMult,
    tradeMode: config.tradeMode,
    sizing: {
      mode: config.sizingMode === 'full' ? 'full' : 'risk',
      riskPct: config.riskPct,
      compounding: config.sizingMode !== 'fixed',
    },
    // Quoted per 8 hours (funding) and per year (borrow); the engine charges both per candle.
    fundingPctPerBar: financing === 'funding' ? config.fundingPct8h * (barMs / (8 * 3_600_000)) : 0,
    borrowPctPerBar: financing === 'borrow' ? config.borrowPctYear * (barMs / MS_PER_YEAR) : 0,
  };
}

/** Everything about a run that must not change when the user edits the controls afterwards. */
export interface RunContext {
  label: string;
  interval: Interval;
  range: HistoryRange;
  settings: BacktestSettings;
  /** Evaluation limit this run was started with (capped by the window's remaining budget). */
  maxEvaluations: number;
  /** The market hypothesis this run tests. */
  hypothesis: Hypothesis;
  /** Identifies this run's record in the trial ledger. */
  runId: string;
  /** Symbols and dates of the search period, as recorded in the trial ledger. */
  trialWindow: TrialWindow;
  /** Trials spent on this window by earlier runs (from the trial ledger). */
  priorTrials: TrialStats & { runs: number };
  /** Assets the run was scored on (the primary first); a single entry when the Robust universe is off. */
  universe: string[];
  /** Instrument and venue actually traded, e.g. "XRPUSDT on Binance". */
  venue: string;
  /** Symbols and date range of this run's holdout, as recorded in the holdout log. */
  holdoutWindow: HoldoutWindow;
  /** Earlier reveals whose holdout overlaps this one (same symbols, overlapping dates). */
  priorHoldoutReveals: number;
  /** When the most recent of those reveals happened (ms), or null. */
  lastHoldoutReveal: number | null;
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
