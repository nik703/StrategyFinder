/**
 * Shared type contracts for StrategyFinder.
 *
 * These types are imported by the React UI, the optimisation Web Worker, the
 * pure TypeScript backtesting engine and (type-only) by the Netlify Function,
 * so the wire formats on both sides of every boundary stay in sync.
 */

/* -------------------------------------------------------------------------- */
/*  Market data                                                               */
/* -------------------------------------------------------------------------- */

export type Interval = '1h' | '4h' | '1d';
export type HistoryRange = '1y' | '2y' | '5y';
export type DataSource = 'binance' | 'yahoo' | 'synthetic';

export const INTERVALS: readonly Interval[] = ['1h', '4h', '1d'];
export const HISTORY_RANGES: readonly HistoryRange[] = ['1y', '2y', '5y'];

export const INTERVAL_MS: Record<Interval, number> = {
  '1h': 3_600_000,
  '4h': 14_400_000,
  '1d': 86_400_000,
};

export const RANGE_DAYS: Record<HistoryRange, number> = {
  '1y': 365,
  '2y': 730,
  '5y': 1826,
};

/**
 * Columnar OHLCV candles for one symbol. Columnar JSON is ~35% smaller than an
 * array of candle objects and maps 1:1 onto the typed arrays used by the engine.
 * `t` is the bar OPEN time in milliseconds since the Unix epoch (UTC).
 */
export interface CandleSeries {
  symbol: string;
  source: DataSource;
  t: number[];
  o: number[];
  h: number[];
  l: number[];
  c: number[];
  v: number[];
  /** Optional human-readable note, e.g. when the upstream API truncated the range. */
  note?: string;
  /** Instrument actually quoted by the venue, e.g. "XRPUSDT" when XRP-USD is routed to Binance. */
  venueSymbol?: string;
}

/** JSON body returned by `/.netlify/functions/market-data`. */
export interface MarketDataResponse {
  interval: Interval;
  range: HistoryRange;
  generatedAt: number;
  series: Record<string, CandleSeries>;
  errors: Record<string, string>;
}

/** Typed-array OHLCV used inside the engine and transferred to the worker. */
export interface OHLCV {
  t: Float64Array;
  o: Float64Array;
  h: Float64Array;
  l: Float64Array;
  c: Float64Array;
  v: Float64Array;
}

/** One leg of a weighted basket. `weight` is normalised so all legs sum to 1. */
export interface BasketComponent {
  symbol: string;
  weight: number;
}

/* -------------------------------------------------------------------------- */
/*  Strategy genome                                                           */
/* -------------------------------------------------------------------------- */

export type TradeMode = 'long-only' | 'long-short';

/** Comparison operator applied as `left <op> right` on closed candles. */
export type CmpOp = 'gt' | 'lt' | 'cross_above' | 'cross_below';
export type MaType = 'EMA' | 'SMA';
export type GroupLogic = 'AND' | 'OR';

/**
 * A single indicator condition. Every variant resolves to `left <op> right`
 * where `right` is either another series or a constant level.
 */
export type Condition =
  /** RSI(period) <op> level */
  | { ind: 'RSI'; period: number; op: CmpOp; level: number }
  /**
   * signal: MACD line <op> signal line
   * zero:   MACD line <op> 0
   * hist:   histogram <op> previous-bar histogram (momentum rising / falling)
   */
  | { ind: 'MACD'; fast: number; slow: number; signal: number; variant: 'signal' | 'zero' | 'hist'; op: CmpOp }
  /** Close <op> MA(period) */
  | { ind: 'PRICE_MA'; maType: MaType; period: number; op: CmpOp }
  /** fastMA <op> slowMA */
  | { ind: 'MA_CROSS'; fastType: MaType; fastPeriod: number; slowType: MaType; slowPeriod: number; op: CmpOp }
  /**
   * pctb:      %B <op> level
   * bandwidth: Bandwidth% <op> level
   * upper:     Close <op> upper band
   * lower:     Close <op> lower band
   */
  | { ind: 'BB'; period: number; mult: number; variant: 'pctb' | 'bandwidth' | 'upper' | 'lower'; op: CmpOp; level: number }
  /** ATR(period) as a % of Close <op> level (volatility filter) */
  | { ind: 'ATR'; period: number; op: CmpOp; level: number }
  /**
   * level: %K <op> level
   * kd:    %K <op> %D
   */
  | { ind: 'STOCH'; kPeriod: number; smooth: number; dPeriod: number; variant: 'level' | 'kd'; op: CmpOp; level: number }
  /**
   * strength: ADX <op> level (trend-strength filter)
   * di:       +DI <op> -DI
   */
  | { ind: 'ADX'; period: number; variant: 'strength' | 'di'; op: CmpOp; level: number }
  /** Volume / SMA(Volume, period) <op> level */
  | { ind: 'VOLUME'; period: number; op: CmpOp; level: number }
  /** Close <op> Supertrend(period, mult) */
  | { ind: 'SUPERTREND'; period: number; mult: number; op: CmpOp }
  /** CCI(period) <op> level */
  | { ind: 'CCI'; period: number; op: CmpOp; level: number };

export type IndicatorFamily = Condition['ind'];

/** One trigger plus an optional filter, always combined with AND. */
export interface RuleGroup {
  logic: GroupLogic;
  conditions: Condition[];
}

/**
 * What a condition says about the market:
 *   TREND_UP / TREND_DOWN  – price or momentum is pointing up / down
 *   OVERSOLD / OVERBOUGHT  – an oscillator or band is stretched low / high
 *   FILTER                 – direction-agnostic (volatility, volume, trend strength)
 */
export type ConditionRole = 'TREND_UP' | 'TREND_DOWN' | 'OVERSOLD' | 'OVERBOUGHT' | 'FILTER';

/**
 * TREND:          buy strength – one TREND_UP trigger, optionally gated by a FILTER.
 * MEAN_REVERSION: buy weakness – one OVERSOLD or short-term TREND_DOWN trigger, optionally
 *                 gated by a slow up-trend regime filter or a FILTER.
 * Shorts are always the exact mirror image of the long rules.
 */
export type Archetype = 'TREND' | 'MEAN_REVERSION';

/** Fixed-percentage stop loss and take profit, measured from the entry price. */
export interface StaticExit {
  mode: 'static';
  /** % of entry price, from STATIC_PCT_GRID (≤ 15). */
  stopLossPct: number;
  /** % of entry price, from STATIC_PCT_GRID. */
  takeProfitPct: number;
}

/** ATR-driven exits: ATR stop, optional ATR target, optional trailing, optional indicator-reversal exit. */
export interface DynamicExit {
  mode: 'dynamic';
  /** ATR lookback shared by the stop, the target and the trail. */
  atrPeriod: number;
  /** 1.0 – 4.0 × ATR in steps of 0.5. */
  stopAtrMult: number;
  /** When true the ATR stop ratchets behind the best price since entry. */
  trailing: boolean;
  /** 1.5 – 8.0 × ATR in steps of 0.5, or null for "no fixed target". */
  takeProfitAtrMult: number | null;
  /** TREND only: close the trade when the entry trigger reverses. */
  reversalExit: boolean;
  /** Derived: mirror of the long entry trigger when `reversalExit` is on, else null. */
  longExit: RuleGroup | null;
  /** Derived: mirror of the short entry trigger when `reversalExit` is on, else null. */
  shortExit: RuleGroup | null;
}

export type ExitGene = StaticExit | DynamicExit;

export interface StrategyGenome {
  archetype: Archetype;
  /** conditions[0] is the trigger, conditions[1] the optional filter. */
  longEntry: RuleGroup;
  /** Derived: the mirror image of `longEntry` in Long/Short mode, null in Long-Only mode. */
  shortEntry: RuleGroup | null;
  exit: ExitGene;
  /** Time stop: close the trade at the open after this many bars (null = disabled). */
  maxBarsInTrade: number | null;
}

/* -------------------------------------------------------------------------- */
/*  Backtest configuration and results                                        */
/* -------------------------------------------------------------------------- */

/**
 * Position sizing is a run setting, not something the optimiser may tune.
 *   risk: being stopped out costs `riskPct` % of capital. Without compounding the risk is
 *         based on the starting capital; with compounding on current equity.
 *   full: every trade deploys 100% of current equity.
 * Exposure never exceeds 1× equity.
 */
export interface SizingSetting {
  mode: 'risk' | 'full';
  riskPct: number;
  compounding: boolean;
}

export const DEFAULT_SIZING: SizingSetting = { mode: 'risk', riskPct: 2, compounding: false };

export interface BacktestSettings {
  initialCapital: number;
  /** Trading fee charged on every fill, in percent (0.1 = 0.1% per side). */
  feePct: number;
  /** Adverse slippage applied to every fill, in percent (0.05 = 0.05%). */
  slippagePct: number;
  tradeMode: TradeMode;
  sizing: SizingSetting;
  /** Perpetual-funding cost charged on the notional of every open position per bar, in percent (0 = off). */
  fundingPctPerBar: number;
}

export interface PerformanceMetrics {
  /** Net profit in account currency after fees and slippage. */
  netPnl: number;
  /** Return on starting equity for the segment, in percent. */
  roi: number;
  /** Compound annual growth rate, in percent. */
  cagr: number;
  /** Largest peak-to-trough equity decline using intrabar worst-case marks, in percent. */
  maxDrawdown: number;
  /** Annualised Sharpe ratio of per-bar returns (risk-free rate = 0). */
  sharpe: number;
  /** Annualised Sortino ratio of per-bar returns. */
  sortino: number;
  /** Winning trades / total trades, in percent. */
  winRate: number;
  /** Gross profit / gross loss (Infinity when there are no losing trades). */
  profitFactor: number;
  /** Number of round-trip trades opened inside the segment. */
  trades: number;
  /** Share of bars spent in the market, in percent. */
  exposure: number;
  /** Number of bars in the segment. */
  bars: number;
  /** Per-bar (not annualised) Sharpe ratio – the input to the deflated Sharpe and its confidence interval. */
  sharpeRaw: number;
  /** Skewness of per-bar returns. */
  skew: number;
  /** Kurtosis of per-bar returns (3 = normal). */
  kurtosis: number;
  /** Longest run of consecutive losing trades. */
  maxConsecLosses: number;
  /** Average winning trade / average losing trade (Infinity without losers). */
  payoffRatio: number;
  /** Win rate needed to break even at this payoff ratio, in percent. */
  breakevenWinRate: number;
  /** Skewness of per-trade PnL. */
  tradeSkew: number;
  /** Trades closed on a candle that touched both the stop and the target. */
  ambiguousTrades: number;
}

/** Metrics of one strategy on one asset. */
export interface AssetResult {
  symbol: string;
  /** Train + Validation combined – everything the search is allowed to see. */
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
  /** Null until the run has ended. */
  holdout: PerformanceMetrics | null;
}

export type Verdict = 'Robust' | 'Fragile' | 'Overfit';

export interface RobustnessReport {
  sensitivity: {
    /** Number of ±20% single-parameter variants that were re-tested. */
    variants: number;
    baseRoi: number;
    medianRoi: number;
    worstRoi: number;
    baseFitness: number;
    medianFitness: number;
    worstFitness: number;
    /** True when the worst variant loses more than half of the original ROI. */
    fragile: boolean;
  };
  topTrades: {
    netPnl: number;
    /** Net PnL after removing the best trades. */
    netPnlWithout: number;
    removed: number;
    profitable: boolean;
  };
  /** Null when no other assets were available. */
  crossAsset: {
    rows: { symbol: string; roi: number; sharpe: number; trades: number; profitable: boolean }[];
    /** Share of assets with a positive ROI (0–1). */
    share: number;
  } | null;
}

export interface StrategyResult {
  /** Unique within one optimisation run, e.g. "S-4821". */
  id: string;
  genome: StrategyGenome;
  /** Composite fitness score used for ranking (higher is better). */
  fitness: number;
  /** Number of tunable numeric parameters. */
  params: number;
  /** Deflated Sharpe Ratio of the validation segment: probability that the Sharpe is real given every trial so far. */
  dsr: number;
  /** Primary-asset metrics. `search` = Train + Validation. */
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
  /** Null while the run is in progress – the holdout is only evaluated once the search has ended. */
  holdout: PerformanceMetrics | null;
  /** Search-period ROI minus Buy & Hold ROI, in percentage points. */
  alpha: number;
  /** Equity curve sampled on `ChartAxis.barIndex`; stops at the holdout boundary until the run ends. */
  equity: Float32Array;
  /** Iteration at which the strategy was discovered. */
  foundAt: number;
  /** One entry per asset (the primary asset first). */
  assets: AssetResult[];
  /** Filled in when the run ends, or on demand through an `analyze` request. */
  robustness: RobustnessReport | null;
  verdict: Verdict | null;
}

export interface BenchmarkResult {
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
  holdout: PerformanceMetrics | null;
  /** Buy & Hold equity curve sampled on `ChartAxis.barIndex`; stops at the holdout boundary until the run ends. */
  equity: Float32Array;
}

/** Shared, down-sampled X axis for every equity curve sent to the UI. */
export interface ChartAxis {
  /** Bar indices (into the full dataset) that were sampled. */
  barIndex: Int32Array;
  /** Bar open time (ms) for each sampled index. */
  time: Float64Array;
  /** Position inside the sampled arrays of the first validation bar. */
  validationPos: number;
  /** Position inside the sampled arrays of the first holdout bar. */
  holdoutPos: number;
  /** Total number of bars in the dataset. */
  bars: number;
  validationIndex: number;
  holdoutIndex: number;
}

/* -------------------------------------------------------------------------- */
/*  Optimiser worker protocol                                                 */
/* -------------------------------------------------------------------------- */

/** Default number of consecutive non-improving iterations after which the optimiser declares convergence. */
export const DEFAULT_MAX_STAGNATION = 5000;
/** Bounds for the user-configurable stagnation limit. */
export const MIN_MAX_STAGNATION = 100;
export const MAX_MAX_STAGNATION = 1_000_000;
/** Size of the live leaderboard. */
export const LEADERBOARD_SIZE = 25;
/** The worker posts a progress message every N iterations. */
export const PROGRESS_EVERY = 25;

/** Chronological split: the search sees Train + Validation, the Holdout is only scored when the run ends. */
export const TRAIN_FRACTION = 0.6;
export const VALIDATION_FRACTION = 0.2;
export const HOLDOUT_FRACTION = 0.2;

/** Default "Robust universe": the same rules must work on all of these, untuned. */
export const DEFAULT_CRYPTO_UNIVERSE: readonly string[] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD'];
export const DEFAULT_STOCK_UNIVERSE: readonly string[] = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'AMZN'];

export interface NamedDataset {
  symbol: string;
  data: OHLCV;
}

export interface OptimizerStartPayload {
  /** The primary asset first; more than one entry switches on Robust-universe scoring. */
  datasets: NamedDataset[];
  settings: BacktestSettings;
  /** Stop after this many consecutive iterations without improvement (default 5,000). */
  maxStagnation?: number;
  /** Optional seed for reproducible runs. */
  seed?: number;
}

export type WorkerRequest =
  | { type: 'start'; payload: OptimizerStartPayload }
  | { type: 'stop' }
  /** Robustness analysis of one strategy. `universe` supplies extra assets for the cross-asset test. */
  | { type: 'analyze'; strategyId: string; genome: StrategyGenome; universe?: NamedDataset[] };

export interface ProgressSnapshot {
  iteration: number;
  stagnation: number;
  bestFitness: number | null;
  bestRoi: number | null;
  itersPerSec: number;
  elapsedMs: number;
}

export type StopReason = 'stopped' | 'converged';

export type WorkerResponse =
  | { type: 'ready'; benchmark: BenchmarkResult; axis: ChartAxis; barsPerYear: number }
  | ({ type: 'progress' } & ProgressSnapshot)
  | { type: 'leaderboard'; iteration: number; strategies: StrategyResult[] }
  /** The run has ended: the holdout is revealed for the final leaderboard and the benchmark. */
  | ({ type: 'done'; reason: StopReason; strategies: StrategyResult[]; benchmark: BenchmarkResult } & ProgressSnapshot)
  | { type: 'analysis'; strategyId: string; report: RobustnessReport; verdict: Verdict | null }
  | { type: 'error'; message: string };

export type EngineStatus = 'idle' | 'loading' | 'optimizing' | 'stopped' | 'converged' | 'error';
