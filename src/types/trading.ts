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

/** 1–4 conditions combined with AND / OR. */
export interface RuleGroup {
  logic: GroupLogic;
  conditions: Condition[];
}

/** Fixed-percentage stop loss and take profit, measured from the entry price. */
export interface StaticExit {
  mode: 'static';
  /** 0.5 – 15 (% of entry price). */
  stopLossPct: number;
  /** 1 – 60 (% of entry price). */
  takeProfitPct: number;
}

/** ATR-driven exits: ATR stop, optional ATR target, optional trailing, optional indicator-reversal exit. */
export interface DynamicExit {
  mode: 'dynamic';
  /** ATR lookback shared by the stop, the target and the trail. */
  atrPeriod: number;
  /** 1.0 – 4.0 × ATR. */
  stopAtrMult: number;
  /** When true the ATR stop ratchets behind the best price since entry. */
  trailing: boolean;
  /** 1.5 – 8.0 × ATR, or null for "no fixed target, let winners run". */
  takeProfitAtrMult: number | null;
  /** Indicator-reversal rules that close a long at the next open (null = disabled). */
  longExit: RuleGroup | null;
  /** Indicator-reversal rules that cover a short at the next open (null = disabled / long-only). */
  shortExit: RuleGroup | null;
}

export type ExitGene = StaticExit | DynamicExit;

/**
 * full: every trade deploys 100% of equity (compounding).
 * risk: volatility sizing – the position is scaled so that being stopped out
 *       costs `riskPct` % of equity (capped at 100% of equity, no leverage).
 */
export type SizingGene = { mode: 'full' } | { mode: 'risk'; riskPct: number };

export interface StrategyGenome {
  longEntry: RuleGroup;
  /** null in Long-Only mode. */
  shortEntry: RuleGroup | null;
  exit: ExitGene;
  sizing: SizingGene;
}

/* -------------------------------------------------------------------------- */
/*  Backtest configuration and results                                        */
/* -------------------------------------------------------------------------- */

export interface BacktestSettings {
  initialCapital: number;
  /** Trading fee charged on every fill, in percent (0.1 = 0.1% per side). */
  feePct: number;
  /** Adverse slippage applied to every fill, in percent (0.05 = 0.05%). */
  slippagePct: number;
  tradeMode: TradeMode;
}

export interface PerformanceMetrics {
  /** Net profit in account currency after fees and slippage. */
  netPnl: number;
  /** Return on starting equity for the segment, in percent. */
  roi: number;
  /** Compound annual growth rate, in percent. */
  cagr: number;
  /** Largest peak-to-trough equity decline, in percent (positive number). */
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
}

export interface StrategyResult {
  /** Unique within one optimisation run, e.g. "S-4821". */
  id: string;
  genome: StrategyGenome;
  /** Composite fitness score used for ranking (higher is better). */
  fitness: number;
  full: PerformanceMetrics;
  inSample: PerformanceMetrics;
  outOfSample: PerformanceMetrics;
  /** Full-period ROI minus Buy & Hold ROI, in percentage points. */
  alpha: number;
  /** Out-of-sample ROI minus Buy & Hold out-of-sample ROI, in percentage points. */
  alphaOos: number;
  /** Equity curve sampled on `ChartAxis.barIndex`. */
  equity: Float32Array;
  /** Iteration at which the strategy was discovered. */
  foundAt: number;
}

export interface BenchmarkResult {
  full: PerformanceMetrics;
  inSample: PerformanceMetrics;
  outOfSample: PerformanceMetrics;
  /** Buy & Hold equity curve sampled on `ChartAxis.barIndex`. */
  equity: Float32Array;
}

/** Shared, down-sampled X axis for every equity curve sent to the UI. */
export interface ChartAxis {
  /** Bar indices (into the full dataset) that were sampled. */
  barIndex: Int32Array;
  /** Bar open time (ms) for each sampled index. */
  time: Float64Array;
  /** Position inside the sampled arrays of the first out-of-sample bar. */
  splitPos: number;
  /** Total number of bars in the dataset. */
  bars: number;
  /** Index of the first out-of-sample bar in the full dataset. */
  splitIndex: number;
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
/** Chronological share of the data used as the in-sample segment. */
export const IN_SAMPLE_FRACTION = 0.7;

export interface OptimizerStartPayload {
  data: OHLCV;
  settings: BacktestSettings;
  /** Stop after this many consecutive iterations without improvement (default 5,000). */
  maxStagnation?: number;
  /** Optional seed for reproducible runs. */
  seed?: number;
}

export type WorkerRequest = { type: 'start'; payload: OptimizerStartPayload } | { type: 'stop' };

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
  | ({ type: 'done'; reason: StopReason } & ProgressSnapshot)
  | { type: 'error'; message: string };

export type EngineStatus = 'idle' | 'loading' | 'optimizing' | 'stopped' | 'converged' | 'error';
