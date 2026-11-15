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
  | { ind: 'CCI'; period: number; op: CmpOp; level: number }
  /**
   * upper: Close <op> highest high of the previous `period` candles (a range breakout)
   * lower: Close <op> lowest low of the previous `period` candles
   */
  | { ind: 'DONCHIAN'; period: number; variant: 'upper' | 'lower'; op: CmpOp }
  /** MA(period) <op> the same MA `SLOPE_LAG` candles earlier – the sign of the trend's slope */
  | { ind: 'MA_SLOPE'; maType: MaType; period: number; op: CmpOp };

/** Candles over which an MA_SLOPE condition measures the slope. */
export const SLOPE_LAG = 10;

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
  /** The run's market hypothesis. Absent only on hand-built genomes; the search always sets it. */
  hypothesis?: Hypothesis;
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
  /**
   * Pooled multi-asset sample: every asset of the run is traded with the same rules and the
   * minimum-trade gates apply to the trades summed over all of them (default: off – each asset
   * must clear the gates on its own and the first asset is the one traded).
   */
  poolAssets?: boolean;
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

/** Market regime slices of the search period, in this order. A bar can be in a trend slice AND the high-volatility slice. */
export const REGIME_NAMES = ['Bull', 'Bear', 'High volatility'] as const;
export type RegimeName = (typeof REGIME_NAMES)[number];

/** How a strategy behaved on the search-period bars of one market regime. */
export interface RegimeMetrics {
  regime: RegimeName;
  /** Search-period bars labelled with this regime. */
  bars: number;
  /** Share of the search period in this regime (0–1). */
  coverage: number;
  /** Trades opened on a bar of this regime. */
  trades: number;
  /** Net profit of those trades. */
  netPnl: number;
  /** Annualised Sharpe ratio of the per-bar returns on this regime's bars. */
  sharpe: number;
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
  /** Behaviour in the bull, bear and high-volatility slices of the search period (null when unavailable). */
  regimes?: RegimeMetrics[] | null;
  /** Null until the run has ended. */
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
  /** Parameter-stability map (null when the strategy has no tunable parameter). */
  stability?: StabilityGrid | null;
  /** Null when no other assets were available. */
  crossAsset: {
    rows: { symbol: string; roi: number; sharpe: number; trades: number; profitable: boolean }[];
    /** Share of assets with a positive ROI (0–1). */
    share: number;
  } | null;
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
  /** Composite fitness score used for ranking (higher is better), after the parameter-neighbourhood cap. */
  fitness: number;
  /** Fitness of the exact parameter values, before the neighbourhood cap. */
  pointFitness?: number;
  /** Median fitness of the one-grid-step parameter neighbours (null without neighbours). */
  neighbourMedian?: number | null;
  /** Number of one-grid-step neighbours that were backtested. */
  neighbourCount?: number;
  /**
   * Haircut Sharpe of the validation segment: the annualised Sharpe that survives a Bonferroni
   * correction for every trial on this data window (Harvey & Liu). 0 when nothing survives.
   */
  haircutSharpe?: number;
  /** Share of the validation Sharpe removed by the haircut (0–1; 1 = entirely explained by multiple testing). */
  haircut?: number;
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
  /** Primary-asset regime slices of the search period. */
  regimes?: RegimeMetrics[] | null;
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
  /** Only ever set for the one strategy whose holdout was locked and revealed. */
  verdict: Verdict | null;
  /** Why the locked strategy is not Robust (empty when it is). Absent until the holdout is revealed. */
  verdictNotes?: string[];
  /** The trade sample the strategy was gated on, and the gates. */
  sample?: SampleInfo;
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
export const DEFAULT_RUN_EVALUATIONS = 5_000;
/** Smallest per-run evaluation limit the controls accept. */
export const MIN_RUN_EVALUATIONS = 100;
/**
 * Evaluations per data window, summed over EVERY run and session whose search period overlaps it.
 * At this point search stops by default: the user may extend the window by `BUDGET_EXTENSION`
 * trials at a time, knowingly – every extra trial raises the luck hurdle the deflated Sharpe uses.
 */
export const WINDOW_TRIAL_BUDGET = 25_000;
/** Trials one explicit "Extend budget" adds to a data window. */
export const BUDGET_EXTENSION = 10_000;
/**
 * Early stopping: a run ends after this many evaluations in a row fail to improve the best
 * composite fitness on the leaderboard. (Not the deflated Sharpe: it sits at 0% through the early
 * generations, which stopped runs before good combinations had a chance to evolve.)
 */
export const FITNESS_PATIENCE = 1_500;
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

/** Default "Robust universe": the same rules must work on all of these, untuned. */
export const DEFAULT_CRYPTO_UNIVERSE: readonly string[] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD'];
export const DEFAULT_STOCK_UNIVERSE: readonly string[] = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'AMZN'];

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
  /** The primary asset first; more than one entry switches on Robust-universe scoring. */
  datasets: NamedDataset[];
  settings: BacktestSettings;
  /** The market hypothesis every strategy of this run is built from. */
  hypothesis: Hypothesis;
  /** Evaluations this run may spend (default 5,000, at most the window's remaining budget). */
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
}

export type WorkerRequest =
  | { type: 'start'; payload: OptimizerStartPayload }
  | { type: 'stop' }
  /** An evaluator worker died: finish the run on the optimiser thread. */
  | { type: 'pool-failed' }
  /** Robustness analysis of one strategy. `universe` supplies extra assets for the cross-asset test. */
  | { type: 'analyze'; strategyId: string; genome: StrategyGenome; universe?: NamedDataset[] }
  /**
   * Lock ONE strategy of the final leaderboard as the run's answer and reveal its holdout.
   * Allowed once per run, only after the search has ended. `priorReveals` is how often this holdout
   * period had already been revealed (from the persistent holdout log); `simulated` marks fallback data.
   * Both stop the strategy from being rated Robust.
   */
  | { type: 'lock'; strategyId: string; priorReveals: number; simulated: boolean };

export interface ProgressSnapshot {
  /** Candidates proposed, including duplicates that were not evaluated. */
  iteration: number;
  /** Proposals since the top fitness last improved. */
  stagnation: number;
  /** Unique strategies evaluated by this run – the trials it added. */
  evaluations: number;
  /** Trials on this data window across all runs, this one included. */
  windowTrials: number;
  /** Evaluations since a candidate last improved the best fitness (early stop at `FITNESS_PATIENCE`). */
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
  itersPerSec: number;
  elapsedMs: number;
}

/**
 * stopped       – the user pressed Stop;
 * converged     – early stop: no candidate improved the best fitness for `FITNESS_PATIENCE` evaluations;
 * budget        – the run's evaluation limit was reached;
 * window-budget – the data window's cumulative trial budget is spent;
 * exhausted     – the hypothesis's search space yields no new strategies.
 */
export type StopReason = 'stopped' | 'converged' | 'budget' | 'window-budget' | 'exhausted';

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
  | { type: 'error'; message: string };

export type EngineStatus = 'idle' | 'loading' | 'optimizing' | 'stopped' | 'converged' | 'error';
