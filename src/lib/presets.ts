/**
 * One-click quick-start templates. Each preset sets the assets, timeframe, history, realistic costs,
 * direction mode, market hypothesis and risk settings for a common way of trading; everything stays
 * editable afterwards (the Advanced Settings drawer holds the rest).
 */

import type { RunConfig } from './config';

export type PresetIcon = 'coins' | 'trend' | 'stock' | 'zap';

export interface Preset {
  id: string;
  title: string;
  /** One line: what the preset looks for. */
  tagline: string;
  icon: PresetIcon;
  /** Key facts shown on the card. */
  facts: string[];
  patch: Partial<RunConfig>;
}

export const PRESETS: readonly Preset[] = [
  {
    id: 'crypto-swing-4h',
    title: 'Crypto Multi-Asset Swing (4h)',
    tagline: 'Volatility breakouts and squeezes on four crypto majors, pooled so one rule set must work on all of them.',
    icon: 'coins',
    facts: ['BTC · ETH · SOL · BNB pool', '4h candles · 2 years', 'Long-Only spot', '0.1% fee + ATR slippage'],
    patch: {
      assetMode: 'pool',
      pool: 'BTC-USD, ETH-USD, SOL-USD, BNB-USD',
      interval: '4h',
      range: '2y',
      tradeMode: 'long-only',
      hypothesis: 'VOL_BREAKOUT',
      feePct: 0.1,
      slippagePct: 0.05,
      slippageAtrMult: 0.03,
      sizingMode: 'fixed',
      riskPct: 1,
      maPeriods: 'integer',
      maxEvaluations: 25_000,
      robustTarget: 3,
    },
  },
  {
    id: 'trend-breakout-1d',
    title: 'Trend Breakout (1d)',
    tagline: 'Daily channel breakouts and moving-average trends; shorts only in confirmed downtrends.',
    icon: 'trend',
    facts: ['BTC · ETH · SOL · XRP pool', 'Daily candles · 5 years', 'Asymmetric L/S + funding', '0.1% fee + ATR slippage'],
    patch: {
      assetMode: 'pool',
      pool: 'BTC-USD, ETH-USD, SOL-USD, XRP-USD',
      interval: '1d',
      range: '5y',
      tradeMode: 'asymmetric',
      hypothesis: 'TREND_FOLLOWING',
      feePct: 0.1,
      slippagePct: 0.05,
      slippageAtrMult: 0.03,
      fundingPct8h: 0.01,
      sizingMode: 'fixed',
      riskPct: 1,
      maPeriods: 'integer',
      maxEvaluations: 25_000,
      robustTarget: 3,
    },
  },
  {
    id: 'stock-momentum-1d',
    title: 'Stock Momentum (1d)',
    tagline: 'Buy pullbacks inside established uptrends on large-cap stocks and index ETFs.',
    icon: 'stock',
    facts: ['SPY · QQQ · NVDA · AAPL · MSFT · AMZN', 'Daily candles · 5 years', 'Long-Only', '0.02% fee + 0.03% slippage'],
    patch: {
      assetMode: 'pool',
      pool: 'SPY, QQQ, NVDA, AAPL, MSFT, AMZN',
      interval: '1d',
      range: '5y',
      tradeMode: 'long-only',
      hypothesis: 'DIP_IN_UPTREND',
      feePct: 0.02,
      slippagePct: 0.03,
      slippageAtrMult: 0.02,
      sizingMode: 'fixed',
      riskPct: 1,
      maPeriods: 'integer',
      maxEvaluations: 25_000,
      robustTarget: 3,
    },
  },
  {
    id: 'intraday-scalp-15m',
    title: 'Intraday Scalp (15m)',
    tagline: 'Fade short-term extremes inside quiet ranges on the two most liquid crypto pairs, both directions.',
    icon: 'zap',
    facts: ['BTC · ETH pool', '15m candles · 1 year', 'Mirrored L/S + funding', '0.05% fee + ATR slippage'],
    patch: {
      assetMode: 'pool',
      pool: 'BTC-USD, ETH-USD',
      interval: '15m',
      range: '1y',
      tradeMode: 'long-short',
      hypothesis: 'RANGE_REVERSION',
      feePct: 0.05,
      slippagePct: 0.02,
      slippageAtrMult: 0.05,
      fundingPct8h: 0.01,
      sizingMode: 'fixed',
      riskPct: 0.5,
      maPeriods: 'grid',
      maxEvaluations: 15_000,
      robustTarget: 3,
    },
  },
];

/** The preset the configuration currently matches exactly (every field the preset sets), or null. */
export function activePreset(config: RunConfig): Preset | null {
  return (
    PRESETS.find((preset) =>
      (Object.keys(preset.patch) as (keyof RunConfig)[]).every((key) => {
        const want = preset.patch[key];
        const have = config[key];
        return typeof want === 'string' && typeof have === 'string' ? want.replace(/\s+/g, '') === have.replace(/\s+/g, '') : want === have;
      }),
    ) ?? null
  );
}
