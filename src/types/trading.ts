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

export type Interval = '15m' | '30m' | '1h' | '4h' | '1d';
export type HistoryRange = '1y' | '2y' | '5y';
export type DataSource = 'binance' | 'yahoo' | 'synthetic';

export const INTERVALS: readonly Interval[] = ['15m', '30m', '1h', '4h', '1d'];
export const HISTORY_RANGES: readonly HistoryRange[] = ['1y', '2y', '5y'];

export const INTERVAL_MS: Record<Interval, number> = {
  '15m': 900_000,
  '30m': 1_800_000,
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
 * Longest history each timeframe may load. Short timeframes are capped at roughly
 * 35,000 crypto candles: beyond that the download outgrows a serverless response
 * and every backtest gets proportionally slower.
 */
export const MAX_RANGE: Record<Interval, HistoryRange> = {
  '15m': '1y',
  '30m': '2y',
  '1h': '5y',
  '4h': '5y',
  '1d': '5y',
};

/** True when `range` is allowed for `interval`. */
export function rangeAllowed(interval: Interval, range: HistoryRange): boolean {
  return RANGE_DAYS[range] <= RANGE_DAYS[MAX_RANGE[interval]];
}

/** The longest allowed range that does not exceed `range`. */
export function clampRange(interval: Interval, range: HistoryRange): HistoryRange {
  return rangeAllowed(interval, range) ? range : MAX_RANGE[interval];
}

/**
 * How far back Yahoo Finance serves each timeframe (stocks, ETFs and the crypto fallback).
 * Sub-hour data only covers the last 60 days; hourly data about two years.
 */
export const YAHOO_MAX_DAYS: Record<Interval, number> = {
  '15m': 59,
  '30m': 59,
  '1h': 729,
  '4h': 729,
  '1d': 100_000,
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

/**
 * Direction mode of a run:
 *   long-only   – never sells short (the default: equities and spot crypto drift upwards, and shorts
 *                 pay funding or borrow);
 *   asymmetric  – shorts use the mirrored entry trigger but ONLY inside their own, independently
 *                 tuned bear-regime filter (close below a slow EMA/SMA, or a falling slow EMA/SMA), so
 *                 the short side sits out bull markets instead of fading every pullback;
 *   long-short  – mirrored Long/Short: the short rules are the exact mirror image of the long rules.
 */
export type TradeMode = 'long-only' | 'asymmetric' | 'long-short';
export const TRADE_MODES: readonly TradeMode[] = ['long-only', 'asymmetric', 'long-short'];
export const TRADE_MODE_LABEL: Record<TradeMode, string> = {
  'long-only': 'Long-Only',
  asymmetric: 'Asymmetric L/S',
  'long-short': 'Mirrored L/S',
};
/** True when a run in this mode may open short positions. */
export const allowsShorts = (mode: TradeMode): boolean => mode !== 'long-only';

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
  | { ind: 'CCI'; period: number; op: CmpOp; level: number }
  /**
   * upper: Close <op> highest high of the previous `period` candles (a range breakout)
   * lower: Close <op> lowest low of the previous `period` candles
   */
  | { ind: 'DONCHIAN'; period: number; variant: 'upper' | 'lower'; op: CmpOp }
  /** MA(period) <op> the same MA `SLOPE_LAG` candles earlier – the sign of the trend's slope */
  | { ind: 'MA_SLOPE'; maType: MaType; period: number; op: CmpOp }
  /**
   * Keltner channel EMA(period) ± mult × ATR(period).
   * upper: Close <op> upper channel (a volatility breakout)
   * lower: Close <op> lower channel (a pullback / rejection at the lower channel)
   */
  | { ind: 'KELTNER'; period: number; mult: number; variant: 'upper' | 'lower'; op: CmpOp }
  /**
   * Volatility-squeeze release: Bollinger(period, 2σ) was inside Keltner(period, mult × ATR) on the
   * previous candle and is not on this one. The series is +1 when the release closes above the
   * Bollinger basis, −1 when it closes below, 0 otherwise; `level` is +0.5 (long) or −0.5 (short).
   */
  | { ind: 'SQUEEZE'; period: number; mult: number; op: CmpOp; level: number }
  /**
   * ATR range expansion: this candle's true range exceeds `mult` × ATR(period) of the previous
   * candle. +1 when it closed up, −1 when it closed down, 0 otherwise; `level` is ±0.5.
   */
  | { ind: 'RANGE_EXP'; period: number; mult: number; op: CmpOp; level: number }
  /**
   * Pullback to dynamic support: the candle touched MA(period) and closed back on the trend side –
   * +1 when Low ≤ MA < Close (support held), −1 when High ≥ MA > Close (resistance held), else 0;
   * `level` is ±0.5.
   */
  | { ind: 'PULLBACK'; maType: MaType; period: number; op: CmpOp; level: number };

/** Candles over which an MA_SLOPE condition measures the slope. */
export const SLOPE_LAG = 10;

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
 * Shorts mirror the long trigger; in Asymmetric mode they also need their own bear-regime filter.
 */
export type Archetype = 'TREND' | 'MEAN_REVERSION';

/**
 * The market hypothesis a run tests. Every run starts from exactly one, and every strategy it
 * generates is built from that hypothesis's single trigger primitive and its fixed-direction filter
 * (see `HYPOTHESES` in `engine/genome.ts`). The search tunes numbers; it does not shop for ideas.
 */
export type Hypothesis = 'TREND_FOLLOWING' | 'VOL_BREAKOUT' | 'DIP_IN_UPTREND' | 'RANGE_REVERSION';
export const HYPOTHESIS_IDS: readonly Hypothesis[] = ['TREND_FOLLOWING', 'VOL_BREAKOUT', 'DIP_IN_UPTREND', 'RANGE_REVERSION'];

/** Fixed-percentage stop loss and take profit, measured from the entry price. */
export interface StaticExit {
  mode: 'static';
  /** % of entry price, from STATIC_PCT_GRID (≤ 15). */
  stopLossPct: number;
  /** % of entry price, from STATIC_PCT_GRID. */
  takeProfitPct: number;
  /** Breakeven lock: once the trade has moved 1R in favour, the stop moves to the entry price. */
  breakeven?: boolean;
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
  /** Breakeven lock: once the trade has moved 1R (the initial stop distance) in favour, the stop moves to the entry price. */
  breakeven?: boolean;
}

export type ExitGene = StaticExit | DynamicExit;

export interface StrategyGenome {
  archetype: Archetype;
  /** The run's market hypothesis. Absent only on hand-built genomes; the search always sets it. */
  hypothesis?: Hypothesis;
  /** conditions[0] is the trigger, conditions[1] the optional filter. */
  longEntry: RuleGroup;
  /**
   * Derived: null in Long-Only mode; the mirror image of `longEntry` in Mirrored L/S mode; the
   * mirrored trigger AND `shortRegime` in Asymmetric L/S mode.
   */
  shortEntry: RuleGroup | null;
  /**
   * Asymmetric L/S only: the short side's own bear-regime filter (close below a slow EMA/SMA, or a
   * falling slow EMA/SMA), tuned independently of the long rules. Absent in the other modes.
   */
  shortRegime?: Condition | null;
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
  /**
   * Volatility-scaled slippage. Every fill slips by the larger of `slippagePct` and this multiple of
   * ATR(14) as a percentage of price, read on the last CLOSED candle so it is known before the fill.
   * Absent or 0 = fixed slippage only.
   */
  slippageAtrMult?: number;
  /** Borrow / financing cost charged on the notional of SHORT positions only, in percent per bar. Absent = 0. */
  borrowPctPerBar?: number;
  /**
   * How the search tunes EMA / SMA lengths (and the Bollinger basis): 'integer' tries every whole
   * number in range, 'grid' only the coarse geometric grid. Absent = 'grid'.
   */
  maPeriods?: MaPeriodSearch;
}

/** 'integer': fine-tune every EMA / SMA length; 'grid': the coarse geometric grid (5, 7, 10, 14, 20, 28 …). */
export type MaPeriodSearch = 'integer' | 'grid';

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
  /** The search period cut into equal, sequential folds (oldest first). */
  folds?: PerformanceMetrics[];
  /** Null until this strategy's holdout has been revealed. */
  holdout: PerformanceMetrics | null;
}

export type Verdict = 'Robust' | 'Fragile' | 'Overfit';

/**
 * Parameter-stability map: the search-period Sharpe of the strategy over a grid of its two most
 * sensitive parameters (the ones whose one-step neighbours move the result most), centred on the
 * chosen values. A broad warm region is a plateau; a single hot cell is a spike.
 */
export interface StabilityGrid {
  xLabel: string;
  yLabel: string | null;
  xValues: number[];
  /** A single entry when the strategy has only one tunable parameter. */
  yValues: number[];
  /** `cells[row][col]`: annualised search-period Sharpe, or null where the combination is not a valid strategy. */
  cells: (number | null)[][];
  /** Same shape: true where the combination fails the minimum-trade gates. */
  undersampled: boolean[][];
  baseX: number;
  baseY: number;
}

/**
 * Monte Carlo execution stress test of a strategy's search-period trades (see `engine/monteCarlo.ts`):
 * hundreds of re-played trade sequences with 10% of the trades skipped at random, the order
 * shuffled and random adverse slippage shocks on every fill.
 */
export interface MonteCarloSummary {
  simulations: number;
  /** Trades in the sample each path is drawn from. */
  trades: number;
  /** Share of paths that end with a profit (0–1). */
  probProfit: number;
  /** 95th-percentile worst-case max drawdown over the paths, in percent (intrabar marks included). */
  drawdownP95: number;
  /** Median and 5th-percentile total return of the paths, in percent. */
  medianReturn: number;
  p5Return: number;
  /** Max drawdown of the original trade sequence, measured the same way, in percent. */
  baseDrawdown: number;
  /** probProfit ≥ 80% and a 95th-percentile drawdown below 50%. */
  passed: boolean;
}

/** Percentile bands of the Monte Carlo equity paths (multiples of starting equity) by trade number. */
export interface MonteCarloBands {
  /** Trade numbers the bands are sampled at (0 = start). */
  x: number[];
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}

export interface MonteCarloReport extends MonteCarloSummary {
  bands: MonteCarloBands;
}

/** One unseen asset of a cross-market generalization check. */
export interface CrossMarketRow {
  symbol: string;
  /** False when the asset was part of the run (it is then not unseen and not counted). */
  unseen: boolean;
  /** Candles tested (the asset's history before the run's holdout boundary). */
  bars: number;
  roi: number;
  sharpe: number;
  maxDrawdown: number;
  trades: number;
  /** Buy & Hold return over the same candles, in percent. */
  holdRoi: number;
  profitable: boolean;
  /** Null when the asset could not be tested (too little history); the reason is in `note`. */
  note: string | null;
}

/**
 * Cross-market generalization check: the strategy's exact (frozen) rules on correlated assets the
 * search never saw, over the dates before the run's holdout boundary – so the holdout stays sealed.
 */
export interface CrossMarketReport {
  rows: CrossMarketRow[];
  /** Unseen assets with enough history. */
  tested: number;
  /** Share of those that made money (0–1). */
  share: number;
  medianSharpe: number;
  /** At least 2 assets tested, two thirds profitable and a positive median Sharpe. */
  passed: boolean;
  /** Last date tested (the run's holdout boundary), ms. */
  cutoff: number;
}

/** On-demand robustness analysis of one strategy, on the search period only (the holdout is not touched). */
export interface RobustnessReport {
  topTrades: {
    netPnl: number;
    /** Net PnL after removing the best trades. */
    netPnlWithout: number;
    removed: number;
    profitable: boolean;
  };
  /** Parameter-stability map (null when the strategy has no tunable parameter). */
  stability: StabilityGrid | null;
  /** Monte Carlo execution stress test with its equity-path bands (null without trades). */
  monteCarlo: MonteCarloReport | null;
}

/** The minimum-trade gates of a run's data (see `tradeGatesFor` in engine/backtester.ts). */
export interface TradeGates {
  /** Trades over train + validation. */
  search: number;
  /** Trades in the validation segment. */
  validation: number;
  /** Average trades per year of the search period. */
  perYear: number;
  /** Holdout trades below which a locked strategy cannot be rated Robust. */
  holdout: number;
}

/** The sample a strategy was gated on. */
export interface SampleInfo {
  /** Train + validation trades: the primary asset's, or summed over a pooled universe. */
  trades: number;
  validationTrades: number;
  /** Holdout trades counted the same way (null while the holdout is sealed). */
  holdoutTrades: number | null;
  gates: TradeGates;
  pooled: boolean;
}

export interface StrategyResult {
  /** Unique within one optimisation run, e.g. "S-4821". */
  id: string;
  genome: StrategyGenome;
  /** Composite fitness used for ranking (higher is better), after the parameter-neighbourhood blend. */
  fitness: number;
  /** Fitness of the exact parameter values, before the neighbourhood blend. */
  pointFitness?: number;
  /** Median fitness of the one-grid-step parameter neighbours (null without neighbours). */
  neighbourMedian?: number | null;
  /** Number of one-grid-step neighbours that were backtested. */
  neighbourCount?: number;
  /** Number of tunable numeric parameters. */
  params: number;
  /** Deflated Sharpe Ratio of the validation segment: probability that the Sharpe is real given every trial so far. */
  dsr: number;
  /** Primary-asset metrics. `search` = Train + Validation. */
  search: PerformanceMetrics;
  train: PerformanceMetrics;
  validation: PerformanceMetrics;
  /** Primary-asset folds of the search period (oldest first). */
  folds?: PerformanceMetrics[];
  /** Null until this strategy is locked and its holdout revealed (one strategy per run, after the search). */
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
  /** Only ever set for the one strategy whose holdout was locked and revealed. */
  verdict: Verdict | null;
  /** Why the locked strategy is not Robust (empty when it is). Absent until the holdout is revealed. */
  verdictNotes?: string[];
  /** The trade sample the strategy was gated on, and the gates. */
  sample?: SampleInfo;
  /** Robustness gates (search period only), re-checked whenever the luck hurdle moves. */
  robust?: RobustChecks;
  /** Monte Carlo execution stress test of the search-period trades (leaderboard entries only). */
  monteCarlo?: MonteCarloSummary | null;
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

/** Default number of strategy evaluations (trials) per run. */
export const DEFAULT_RUN_EVALUATIONS = 25_000;
/** Safety ceiling of one run (the window budget may be extended beyond it, a run never goes past it). */
export const MAX_RUN_EVALUATIONS = 50_000;
/** Smallest per-run evaluation limit the controls accept. */
export const MIN_RUN_EVALUATIONS = 100;
/**
 * Evaluations per data window, summed over EVERY run and session whose search period overlaps it.
 * At this point search stops by default: the user may extend the window by `BUDGET_EXTENSION`
 * trials at a time, knowingly – every extra trial raises the luck hurdle the deflated Sharpe uses.
 */
export const WINDOW_TRIAL_BUDGET = 50_000;
/** Trials one explicit "Extend budget" adds to a data window. */
export const BUDGET_EXTENSION = 10_000;
/**
 * Diversity restart: after this many evaluations in a row without a gain in the best composite
 * fitness, a run that has not yet found its target number of robust strategies keeps its elite and
 * re-seeds the rest of its parent population with fresh random genomes, instead of stopping.
 */
export const FITNESS_PATIENCE = 1_500;
/** The elite: the top strategies of the leaderboard, kept at every diversity restart. */
export const ELITE_SIZE = 5;
/** Default number of robust strategies a run scans for ("Stop after N robust strategies"); 0 = the full budget. */
export const DEFAULT_ROBUST_TARGET = 3;
export const MAX_ROBUST_TARGET = 10;
/** Safety stop: proposals in a row that yield no new (non-duplicate) evaluation, i.e. the search space is exhausted. */
export const EXHAUSTION_LIMIT = 2_000;
/** Size of the live leaderboard. */
export const LEADERBOARD_SIZE = 25;
/** The worker posts a progress message every N iterations. */
export const PROGRESS_EVERY = 25;

/** Chronological split: the search sees Train + Validation, the Holdout is only scored when the run ends. */
export const TRAIN_FRACTION = 0.6;
export const VALIDATION_FRACTION = 0.2;
export const HOLDOUT_FRACTION = 0.2;

export interface NamedDataset {
  symbol: string;
  data: OHLCV;
}

/** Summary statistics of a set of trials: count, mean and sum of squared deviations of their ANNUALISED Sharpe. */
export interface TrialStats {
  count: number;
  mean: number;
  m2: number;
}

export interface OptimizerStartPayload {
  /** The primary asset first; more than one entry is a pooled multi-asset run (the same rules on every asset). */
  datasets: NamedDataset[];
  settings: BacktestSettings;
  /** The market hypothesis every strategy of this run is built from. */
  hypothesis: Hypothesis;
  /** Evaluations this run may spend (default `DEFAULT_RUN_EVALUATIONS`, at most the window's remaining budget). */
  maxEvaluations?: number;
  /**
   * Trials already spent on this data window by earlier runs (from the persistent trial ledger).
   * They seed the selection-bias hurdle and count against `windowBudget`.
   */
  priorTrials?: TrialStats;
  /** Evaluations allowed on this data window across all runs (default `WINDOW_TRIAL_BUDGET`). */
  windowBudget?: number;
  /** Optional seed for reproducible runs. */
  seed?: number;
  /** Ports to evaluator workers (see `evaluator.worker.ts`); empty or absent = evaluate on the optimiser thread. */
  ports?: MessagePort[];
  /** Candidates per round. Defaults to 25 × the pool size; only tests need to set it. */
  batchSize?: number;
  /** Stop once this many strategies pass every robustness gate (1–10, default 3); 0 = run the full budget. */
  robustTarget?: number;
}

export type WorkerRequest =
  | { type: 'start'; payload: OptimizerStartPayload }
  | { type: 'stop' }
  /** An evaluator worker died: finish the run on the optimiser thread. */
  | { type: 'pool-failed' }
  /** Robustness analysis of one strategy (top-trade removal, stability map, Monte Carlo bands). */
  | { type: 'analyze'; strategyId: string; genome: StrategyGenome }
  /**
   * Lock ONE strategy of the final leaderboard as the run's answer and reveal its holdout.
   * Allowed once per run, only after the search has ended. `priorReveals` is how often this holdout
   * period had already been revealed (from the persistent holdout log); `simulated` marks fallback data.
   * Both stop the strategy from being rated Robust.
   */
  | { type: 'lock'; strategyId: string; priorReveals: number; simulated: boolean }
  /** Cross-market generalization check of one strategy on unseen assets (see `CrossMarketReport`). */
  | {
      type: 'cross-market';
      strategyId: string;
      genome: StrategyGenome;
      datasets: NamedDataset[];
      /** Every symbol the run itself used (basket legs included): those are never counted as unseen. */
      runSymbols?: string[];
    };

export interface ProgressSnapshot {
  /** Candidates proposed, including duplicates that were not evaluated. */
  iteration: number;
  /** Unique strategies evaluated by this run – the trials it added. */
  evaluations: number;
  /** Trials on this data window across all runs, this one included. */
  windowTrials: number;
  /** Evaluations since a candidate last improved the best fitness (a diversity restart at `FITNESS_PATIENCE`). */
  fitnessStall: number;
  /** Candidates dropped because their trade log repeated one already evaluated. */
  duplicates: number;
  /** This run's own trials, in annualised Sharpe units – persisted to the trial ledger. */
  runTrials: TrialStats;
  bestFitness: number | null;
  /** Highest deflated Sharpe ratio on the leaderboard (0–1). */
  bestDsr: number | null;
  /** Annualised validation Sharpe of the strategy with the highest deflated Sharpe. */
  bestDsrSharpe: number | null;
  /** Annualised validation Sharpe the luckiest of the strategies tried so far is expected to reach by chance. */
  hurdleSharpe: number;
  /** Candidates disqualified by the hard minimum-trade gates. */
  undersampled: number;
  /** Strategies currently passing every robustness gate. */
  robustFound: number;
  /** Robust strategies the run is scanning for (0 = it runs the full budget). */
  robustTarget: number;
  /** Population cycle: 1, then +1 at every diversity restart. */
  cycle: number;
  /** How many candidates cleared each robustness gate on its own. */
  funnel: GateFunnel;
  itersPerSec: number;
  elapsedMs: number;
}

/**
 * stopped       – the user pressed Stop;
 * target        – the run found its target number of robust strategies;
 * budget        – the run's evaluation limit was reached;
 * window-budget – the data window's cumulative trial budget is spent;
 * exhausted     – the hypothesis's search space yields no new strategies.
 */
export type StopReason = 'stopped' | 'target' | 'budget' | 'window-budget' | 'exhausted';

/**
 * The robustness gates a candidate must ALL pass to count towards the run's target. They use the
 * search period only – the holdout stays sealed – and are stricter than "on the leaderboard":
 *
 *   validation    validation Sharpe > 0.3 and a positive validation net profit (after all costs);
 *   folds         at least 3 of the 5 sequential folds profitable, none beyond the fold drawdown cap;
 *   neighbours    the one-grid-step parameter neighbours keep a positive median expectancy per trade;
 *   significance  the timeframe-scaled minimum-trade gates, and a deflated Sharpe of at least 50%:
 *                 the validation Sharpe beats the best a random walk is expected to reach after this
 *                 many trials.
 */
export interface RobustChecks {
  validation: boolean;
  folds: boolean;
  neighbours: boolean;
  significance: boolean;
  passed: boolean;
}

/** Candidates (fully scored trials) clearing each gate on its own, over the run so far. */
export interface GateFunnel {
  /** Fully scored trials. */
  scored: number;
  /** Cleared the minimum-trade gates. */
  trades: number;
  validation: number;
  folds: number;
  /** Deflated Sharpe ≥ 50%. */
  dsr: number;
  /** Neighbourhoods checked, and how many held up. */
  neighboursChecked: number;
  neighbours: number;
}

export type WorkerResponse =
  | { type: 'ready'; benchmark: BenchmarkResult; axis: ChartAxis; barsPerYear: number; gates: TradeGates; pooled: boolean }
  | ({ type: 'progress' } & ProgressSnapshot)
  | { type: 'leaderboard'; iteration: number; strategies: StrategyResult[] }
  /** The run has ended. The holdout stays sealed until the user locks one strategy (see `lock`). */
  | ({ type: 'done'; reason: StopReason; strategies: StrategyResult[]; benchmark: BenchmarkResult } & ProgressSnapshot)
  /** `verdict` is null for every strategy except the locked one. */
  | { type: 'analysis'; strategyId: string; report: RobustnessReport; verdict: Verdict | null }
  /** The locked strategy with its holdout, robustness report and verdict, plus the Buy & Hold holdout. */
  | { type: 'locked'; strategy: StrategyResult; benchmark: BenchmarkResult }
  /** `locked`: the locked strategy judged again with the new evidence, when the check was run on it. */
  | { type: 'cross-market'; strategyId: string; report: CrossMarketReport; locked?: StrategyResult }
  | { type: 'error'; message: string };

export type EngineStatus = 'idle' | 'loading' | 'optimizing' | 'stopped' | 'converged' | 'error';
