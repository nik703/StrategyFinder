/**
 * Strategy AST – the single source of truth for what a strategy's rules MEAN.
 *
 * A genome (`genome.ts`) is the search's representation: compact, mutable, snapped to grids. Before
 * anything executes it, it is compiled here into a small declarative tree:
 *
 *   SeriesNode     a numeric series: a price field, an indicator, a constant, or a series shifted back
 *                  N candles;
 *   ConditionNode  `left <op> right` on closed candles (op: >, <, crosses above, crosses below);
 *   StrategyAst    the long entry, the short entry (null in Long-Only; the mirror of the long rules in
 *                  Mirrored L/S; the mirrored trigger AND the bear-regime filter in Asymmetric L/S),
 *                  the opposite-signal exits and the exit spec (stop, target, trailing stop, breakeven
 *                  lock, time stop).
 *
 * Every consumer compiles from this tree and nothing else re-interprets a genome's conditions:
 *
 *   • the TypeScript engine evaluates SeriesNodes on an `IndicatorCache` (`evaluateSeries`) – the
 *     backtester's signals and therefore the Next Entry solver, which probes the backtester's own
 *     evaluator – and reads its stop / target / trailing / breakeven / time-stop parameters from
 *     `exitSpec`;
 *   • the Pine Script v5 exporter emits each SeriesNode as a Pine expression and its exit
 *     management from the same `exitSpec` (see `pine.ts`).
 *
 * Parity tests interpret the generated Pine and compare its signals and stop levels with the engine
 * bar by bar (`tests/pine-parity.test.ts`).
 */

import { SLOPE_LAG, type CmpOp, type Condition, type MaType, type RuleGroup, type StrategyGenome, type TradeMode } from '../types/trading';

/* -------------------------------------------------------------------------- */
/*  Nodes                                                                     */
/* -------------------------------------------------------------------------- */

export type SeriesNode =
  | { k: 'price'; field: 'close' | 'high' | 'low' | 'open' | 'volume' }
  | { k: 'const'; value: number }
  /** A series `bars` candles back (Pine: `x[bars]`). */
  | { k: 'lag'; of: SeriesNode; bars: number }
  | { k: 'ma'; type: MaType; period: number }
  | { k: 'rsi'; period: number }
  | { k: 'macd'; part: 'line' | 'signal' | 'hist'; fast: number; slow: number; signal: number }
  | { k: 'bb'; part: 'upper' | 'lower' | 'basis' | 'pctb' | 'width'; period: number; mult: number }
  | { k: 'atr'; period: number }
  /** ATR as a % of the close. */
  | { k: 'atrPct'; period: number }
  | { k: 'stoch'; part: 'k' | 'd'; kPeriod: number; smooth: number; dPeriod: number }
  | { k: 'dmi'; part: 'adx' | 'plus' | 'minus'; period: number }
  /** Volume / SMA(volume, period). */
  | { k: 'volRatio'; period: number }
  | { k: 'supertrend'; period: number; mult: number }
  | { k: 'cci'; period: number }
  /** Highest high / lowest low of the PREVIOUS `period` candles (the current candle excluded). */
  | { k: 'donchian'; part: 'upper' | 'lower'; period: number }
  /** EMA(period) ± mult × ATR(period). */
  | { k: 'keltner'; part: 'upper' | 'lower'; period: number; mult: number }
  /** Bollinger(period, 2) leaving the Keltner(period, mult) channel: +1 release up, −1 down, else 0. */
  | { k: 'squeeze'; period: number; mult: number }
  /** True range above mult × the previous ATR(period): +1 on an up candle, −1 on a down candle, else 0. */
  | { k: 'rangeExp'; period: number; mult: number }
  /** Touch-and-hold of a moving average: +1 support held, −1 resistance held, else 0. */
  | { k: 'pullback'; type: MaType; period: number };

export interface ConditionNode {
  left: SeriesNode;
  op: CmpOp;
  right: SeriesNode;
}

/** How far the protective stop sits from the fill: a fixed % of the fill, or a multiple of the signal candle's ATR. */
export type StopSpec = { kind: 'percent'; pct: number } | { kind: 'atr'; period: number; mult: number };

export interface ExitSpec {
  stop: StopSpec;
  /** Fixed target (same kinds as the stop), or null for none. */
  target: StopSpec | null;
  /** Trailing ATR stop behind the best price since entry, updated after every close (null = off). */
  trailing: { period: number; mult: number } | null;
  /** Move the stop to the entry once the trade has been this many R in profit (null = off). */
  breakevenR: number | null;
  /** Close at the open after this many candles in the trade (null = off). */
  timeStop: number | null;
}

export interface StrategyAst {
  mode: TradeMode;
  /** Long entry: every condition true on the same close (AND). */
  longEntry: ConditionNode[];
  /** Short entry (null when the strategy never sells short). */
  shortEntry: ConditionNode[] | null;
  /** Opposite-signal exits (null when off). */
  longExit: ConditionNode[] | null;
  shortExit: ConditionNode[] | null;
  exit: ExitSpec;
}

/* -------------------------------------------------------------------------- */
/*  Compilation                                                               */
/* -------------------------------------------------------------------------- */

const close: SeriesNode = { k: 'price', field: 'close' };
const constant = (value: number): SeriesNode => ({ k: 'const', value });

/** The canonical `left <op> right` form of one genome condition. */
export function compileCondition(c: Condition): ConditionNode {
  const node = (left: SeriesNode, right: SeriesNode): ConditionNode => ({ left, op: c.op, right });
  switch (c.ind) {
    case 'RSI':
      return node({ k: 'rsi', period: c.period }, constant(c.level));
    case 'MACD': {
      const macd = (part: 'line' | 'signal' | 'hist'): SeriesNode => ({ k: 'macd', part, fast: c.fast, slow: c.slow, signal: c.signal });
      if (c.variant === 'signal') return node(macd('line'), macd('signal'));
      if (c.variant === 'zero') return node(macd('line'), constant(0));
      return node(macd('hist'), { k: 'lag', of: macd('hist'), bars: 1 });
    }
    case 'PRICE_MA':
      return node(close, { k: 'ma', type: c.maType, period: c.period });
    case 'MA_CROSS':
      return node({ k: 'ma', type: c.fastType, period: c.fastPeriod }, { k: 'ma', type: c.slowType, period: c.slowPeriod });
    case 'BB': {
      const bb = (part: 'upper' | 'lower' | 'basis' | 'pctb' | 'width'): SeriesNode => ({ k: 'bb', part, period: c.period, mult: c.mult });
      if (c.variant === 'pctb') return node(bb('pctb'), constant(c.level));
      if (c.variant === 'bandwidth') return node(bb('width'), constant(c.level));
      return node(close, bb(c.variant));
    }
    case 'ATR':
      return node({ k: 'atrPct', period: c.period }, constant(c.level));
    case 'STOCH': {
      const stoch = (part: 'k' | 'd'): SeriesNode => ({ k: 'stoch', part, kPeriod: c.kPeriod, smooth: c.smooth, dPeriod: c.dPeriod });
      return c.variant === 'level' ? node(stoch('k'), constant(c.level)) : node(stoch('k'), stoch('d'));
    }
    case 'ADX':
      return c.variant === 'strength'
        ? node({ k: 'dmi', part: 'adx', period: c.period }, constant(c.level))
        : node({ k: 'dmi', part: 'plus', period: c.period }, { k: 'dmi', part: 'minus', period: c.period });
    case 'VOLUME':
      return node({ k: 'volRatio', period: c.period }, constant(c.level));
    case 'SUPERTREND':
      return node(close, { k: 'supertrend', period: c.period, mult: c.mult });
    case 'CCI':
      return node({ k: 'cci', period: c.period }, constant(c.level));
    case 'DONCHIAN':
      return node(close, { k: 'donchian', part: c.variant, period: c.period });
    case 'MA_SLOPE': {
      const ma: SeriesNode = { k: 'ma', type: c.maType, period: c.period };
      return node(ma, { k: 'lag', of: ma, bars: SLOPE_LAG });
    }
    case 'KELTNER':
      return node(close, { k: 'keltner', part: c.variant, period: c.period, mult: c.mult });
    case 'SQUEEZE':
      return node({ k: 'squeeze', period: c.period, mult: c.mult }, constant(c.level));
    case 'RANGE_EXP':
      return node({ k: 'rangeExp', period: c.period, mult: c.mult }, constant(c.level));
    case 'PULLBACK':
      return node({ k: 'pullback', type: c.maType, period: c.period }, constant(c.level));
  }
}

/** An AND rule group as condition nodes. */
export function compileGroup(group: RuleGroup): ConditionNode[] {
  if (group.logic !== 'AND' && group.conditions.length > 1) throw new Error('Only AND rule groups can be compiled.');
  return group.conditions.map(compileCondition);
}

/** The exit spec of a genome: what the backtester and the Pine exporter both execute. */
export function exitSpec(genome: StrategyGenome): ExitSpec {
  const exit = genome.exit;
  const breakevenR = exit.breakeven === true ? 1 : null;
  const timeStop = genome.maxBarsInTrade ?? null;
  if (exit.mode === 'static') {
    return {
      stop: { kind: 'percent', pct: exit.stopLossPct },
      target: { kind: 'percent', pct: exit.takeProfitPct },
      trailing: null,
      breakevenR,
      timeStop,
    };
  }
  return {
    stop: { kind: 'atr', period: exit.atrPeriod, mult: exit.stopAtrMult },
    target: exit.takeProfitAtrMult !== null ? { kind: 'atr', period: exit.atrPeriod, mult: exit.takeProfitAtrMult } : null,
    trailing: exit.trailing ? { period: exit.atrPeriod, mult: exit.stopAtrMult } : null,
    breakevenR,
    timeStop,
  };
}

/** Compile a (normalised) genome traded in `mode` into its AST. */
export function compileStrategy(genome: StrategyGenome, mode: TradeMode): StrategyAst {
  const shorts = mode !== 'long-only' && genome.shortEntry !== null;
  const exit = genome.exit;
  return {
    mode,
    longEntry: compileGroup(genome.longEntry),
    shortEntry: shorts && genome.shortEntry ? compileGroup(genome.shortEntry) : null,
    longExit: exit.mode === 'dynamic' && exit.longExit ? compileGroup(exit.longExit) : null,
    shortExit: shorts && exit.mode === 'dynamic' && exit.shortExit ? compileGroup(exit.shortExit) : null,
    exit: exitSpec(genome),
  };
}

/** Canonical text key of a series node (stable, used for memoisation and declaration names). */
export function seriesKey(node: SeriesNode): string {
  switch (node.k) {
    case 'price':
      return node.field;
    case 'const':
      return `c${node.value}`;
    case 'lag':
      return `${seriesKey(node.of)}[${node.bars}]`;
    default: {
      const { k, ...rest } = node as { k: string } & Record<string, unknown>;
      return `${k}(${Object.keys(rest)
        .sort()
        .map((key) => `${key}=${String(rest[key])}`)
        .join(',')})`;
    }
  }
}
