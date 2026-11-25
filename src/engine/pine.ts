/**
 * Pine Script v5 exporter.
 *
 * Generates a complete, ready-to-paste TradingView strategy that implements a
 * genome with the same rules, exits, sizing and execution timing as the
 * StrategyFinder backtester:
 *   • indicators use the matching `ta.*` built-ins,
 *   • orders are placed on bar close and fill at the next bar's open,
 *   • fee + slippage are charged through `commission_value`,
 *   • SL/TP brackets are attached to the entry so they are live from the fill, then managed after
 *     every close (trailing ATR stop, breakeven lock at +1R) exactly like the backtester,
 *   • all three direction modes: Long-Only, Asymmetric L/S and Mirrored L/S,
 *   • `use_bar_magnifier=false`, so the script runs on every TradingView plan.
 *
 * The script is plain ASCII so it survives any clipboard / editor round-trip.
 *
 * Rules are emitted from the strategy AST (`strategyAst.ts`) – the same compiled tree the engine
 * executes – never from the genome directly: `pineSeries` gives every series node its one Pine form
 * and the stop / target / trailing / breakeven / time-stop code comes from the AST's exit spec.
 */

import type {
  BacktestSettings,
  BasketComponent,
  CmpOp,
  Interval,
  PerformanceMetrics,
  RuleGroup,
  StrategyResult,
} from '../types/trading';
import { ARCHETYPE_LABEL, groupFormula } from './describe';
import { countParameters } from './genome';
import { compileStrategy, type ConditionNode, type SeriesNode } from './strategyAst';

export interface PineContext {
  /** Dataset label, e.g. "BTC-USD" or "50% BTC-USD · 30% ETH-USD · 20% NVDA". */
  label: string;
  /** Assets the run traded with these rules; more than one = a pooled multi-asset run. */
  assets: readonly string[];
  interval: Interval;
  settings: BacktestSettings;
  components: readonly BasketComponent[];
  /** Units per 100 of basket value (see `buildDataset`); empty for a single asset. */
  units: readonly number[];
  /** Open time (ms) of the first and last candle of the backtest. */
  windowStart: number;
  windowEnd: number;
  bars: number;
  benchmarkRoi: number;
  /** Instrument and venue the strategy was tested on, e.g. "XRPUSDT on Binance". */
  venue: string;
  /** True when the optimisation ran on simulated fallback data. */
  simulatedData: boolean;
}

/* -------------------------------------------------------------------------- */
/*  Formatting helpers                                                        */
/* -------------------------------------------------------------------------- */

const ascii = (text: string): string =>
  text
    .replace(/\s*·\s*/g, ', ')
    .replace(/[–—]/g, '-')
    .replace(/[^\x20-\x7e]/g, '');

/** Float literal that always carries a decimal point (2 -> "2.0"). */
function float(value: number): string {
  const text = String(Number(value.toFixed(6)));
  return /[.e]/.test(text) ? text : `${text}.0`;
}

/** Number as an identifier fragment (2.5 -> "2_5"). */
const ident = (value: number): string => String(Number(value.toFixed(6))).replace('.', '_').replace('-', 'n');

const signed = (value: number, decimals = 1): string => `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}`;

const isoDate = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

function metricsLine(m: PerformanceMetrics): string {
  return `ROI ${signed(m.roi)}% | Max DD ${m.maxDrawdown.toFixed(1)}% | Sharpe ${m.sharpe.toFixed(2)} | ${m.trades} trades`;
}

/* -------------------------------------------------------------------------- */
/*  Indicator declarations                                                    */
/* -------------------------------------------------------------------------- */

/** Ordered, de-duplicated set of indicator declarations. */
export class Declarations {
  private readonly lines = new Map<string, string>();

  add(key: string, line: string): void {
    if (!this.lines.has(key)) this.lines.set(key, line);
  }

  render(): string[] {
    return [...this.lines.values()];
  }
}

function maVar(decl: Declarations, type: 'EMA' | 'SMA', period: number): string {
  const name = `${type.toLowerCase()}_${period}`;
  decl.add(name, `${name} = ta.${type.toLowerCase()}(close, ${period})`);
  return name;
}

function atrVar(decl: Declarations, period: number): string {
  const name = `atr_${period}`;
  decl.add(name, `${name} = ta.atr(${period})`);
  return name;
}

/** Keltner channel EMA(period) ± mult × ATR(period) – written out so it matches StrategyFinder exactly. */
function keltnerVars(decl: Declarations, period: number, mult: number): [upper: string, lower: string] {
  const ema = maVar(decl, 'EMA', period);
  const atr = atrVar(decl, period);
  const id = `${period}_${ident(mult)}`;
  decl.add(`kcUpper_${id}`, `kcUpper_${id} = ${ema} + ${float(mult)} * ${atr}`);
  decl.add(`kcLower_${id}`, `kcLower_${id} = ${ema} - ${float(mult)} * ${atr}`);
  return [`kcUpper_${id}`, `kcLower_${id}`];
}

/** A number as a Pine literal: integers as written, other values with a decimal point. */
const literal = (value: number): string => (Number.isInteger(value) ? String(value) : float(value));

/**
 * The Pine expression of a strategy-AST series node, declaring whatever it needs. This is the only
 * place the exporter maps rules to Pine: every node kind has exactly one Pine form, mirroring the
 * engine's `evaluateSeries`.
 */
export function pineSeries(node: SeriesNode, decl: Declarations): string {
  switch (node.k) {
    case 'price':
      return node.field;
    case 'const':
      return literal(node.value);
    case 'lag':
      return `${pineSeries(node.of, decl)}[${node.bars}]`;
    case 'ma':
      return maVar(decl, node.type, node.period);
    case 'rsi': {
      const name = `rsi_${node.period}`;
      decl.add(name, `${name} = ta.rsi(close, ${node.period})`);
      return name;
    }
    case 'macd': {
      const id = `${node.fast}_${node.slow}_${node.signal}`;
      decl.add(`macd_${id}`, `[macdLine_${id}, macdSignal_${id}, macdHist_${id}] = ta.macd(close, ${node.fast}, ${node.slow}, ${node.signal})`);
      return node.part === 'line' ? `macdLine_${id}` : node.part === 'signal' ? `macdSignal_${id}` : `macdHist_${id}`;
    }
    case 'bb': {
      const id = `${node.period}_${ident(node.mult)}`;
      decl.add(`bb_${id}`, `[bbBasis_${id}, bbUpper_${id}, bbLower_${id}] = ta.bb(close, ${node.period}, ${float(node.mult)})`);
      switch (node.part) {
        case 'pctb':
          decl.add(`bbPctB_${id}`, `bbPctB_${id} = (close - bbLower_${id}) / (bbUpper_${id} - bbLower_${id})`);
          return `bbPctB_${id}`;
        case 'width':
          decl.add(`bbWidth_${id}`, `bbWidth_${id} = (bbUpper_${id} - bbLower_${id}) / bbBasis_${id} * 100`);
          return `bbWidth_${id}`;
        case 'upper':
          return `bbUpper_${id}`;
        case 'lower':
          return `bbLower_${id}`;
        case 'basis':
          return `bbBasis_${id}`;
      }
      break;
    }
    case 'atr':
      return atrVar(decl, node.period);
    case 'atrPct': {
      const atr = atrVar(decl, node.period);
      const name = `atrPct_${node.period}`;
      decl.add(name, `${name} = ${atr} / close * 100`);
      return name;
    }
    case 'stoch': {
      const k = `stochK_${node.kPeriod}_${node.smooth}`;
      decl.add(k, `${k} = ta.sma(ta.stoch(close, high, low, ${node.kPeriod}), ${node.smooth})`);
      if (node.part === 'k') return k;
      const d = `stochD_${node.kPeriod}_${node.smooth}_${node.dPeriod}`;
      decl.add(d, `${d} = ta.sma(${k}, ${node.dPeriod})`);
      return d;
    }
    case 'dmi': {
      const p = node.period;
      decl.add(`dmi_${p}`, `[diPlus_${p}, diMinus_${p}, adx_${p}] = ta.dmi(${p}, ${p})`);
      return node.part === 'adx' ? `adx_${p}` : node.part === 'plus' ? `diPlus_${p}` : `diMinus_${p}`;
    }
    case 'volRatio': {
      const name = `volRatio_${node.period}`;
      decl.add(name, `${name} = volume / ta.sma(volume, ${node.period})`);
      return name;
    }
    case 'supertrend': {
      const id = `${node.period}_${ident(node.mult)}`;
      decl.add(`st_${id}`, `[supertrend_${id}, supertrendDir_${id}] = ta.supertrend(${float(node.mult)}, ${node.period})`);
      return `supertrend_${id}`;
    }
    case 'cci': {
      const name = `cci_${node.period}`;
      decl.add(name, `${name} = ta.cci(hlc3, ${node.period})`);
      return name;
    }
    case 'donchian': {
      // Channel of the PREVIOUS candles: the current candle is excluded, as in StrategyFinder.
      const name = node.part === 'upper' ? `donchianHigh_${node.period}` : `donchianLow_${node.period}`;
      decl.add(name, node.part === 'upper' ? `${name} = ta.highest(high, ${node.period})[1]` : `${name} = ta.lowest(low, ${node.period})[1]`);
      return name;
    }
    case 'keltner': {
      const [upper, lower] = keltnerVars(decl, node.period, node.mult);
      return node.part === 'upper' ? upper : lower;
    }
    case 'squeeze': {
      const id = `${node.period}_${ident(node.mult)}`;
      const bb = `${node.period}_2`;
      decl.add(`bb_${bb}`, `[bbBasis_${bb}, bbUpper_${bb}, bbLower_${bb}] = ta.bb(close, ${node.period}, 2.0)`);
      const [kcUpper, kcLower] = keltnerVars(decl, node.period, node.mult);
      decl.add(`squeezeOn_${id}`, `squeezeOn_${id} = bbUpper_${bb} < ${kcUpper} and bbLower_${bb} > ${kcLower}`);
      decl.add(
        `squeeze_${id}`,
        `squeeze_${id} = squeezeOn_${id}[1] and not squeezeOn_${id} ? (close > bbBasis_${bb} ? 1 : close < bbBasis_${bb} ? -1 : 0) : 0 // +1 release up, -1 release down`,
      );
      return `squeeze_${id}`;
    }
    case 'rangeExp': {
      const atr = atrVar(decl, node.period);
      const name = `rangeExp_${node.period}_${ident(node.mult)}`;
      decl.add(
        name,
        `${name} = ta.tr(true) > ${float(node.mult)} * ${atr}[1] ? (close > close[1] ? 1 : close < close[1] ? -1 : 0) : 0 // wide-range candle: +1 up, -1 down`,
      );
      return name;
    }
    case 'pullback': {
      const ma = maVar(decl, node.type, node.period);
      const name = `pullback_${ma}`;
      decl.add(name, `${name} = low <= ${ma} and close > ${ma} ? 1 : high >= ${ma} and close < ${ma} ? -1 : 0 // +1 support held, -1 resistance held`);
      return name;
    }
  }
  throw new Error(`Unsupported series node: ${JSON.stringify(node)}`);
}

function compare(op: CmpOp, left: string, right: string): string {
  switch (op) {
    case 'gt':
      return `${left} > ${right}`;
    case 'lt':
      return `${left} < ${right}`;
    case 'cross_above':
      return `ta.crossover(${left}, ${right})`;
    case 'cross_below':
      return `ta.crossunder(${left}, ${right})`;
  }
}

/**
 * Emit `prefixCond1 = ...` lines from compiled condition nodes and the combined `name = a and b`
 * line. `group` only supplies the human-readable comment.
 */
function emitGroup(
  nodes: readonly ConditionNode[],
  group: RuleGroup,
  name: string,
  prefix: string,
  title: string,
  decl: Declarations,
  out: string[],
): void {
  out.push(`// ${title}: ${ascii(groupFormula(group))}`);
  const parts: string[] = [];
  nodes.forEach((node, index) => {
    const variable = `${prefix}Cond${index + 1}`;
    out.push(`${variable} = ${compare(node.op, pineSeries(node.left, decl), pineSeries(node.right, decl))}`);
    parts.push(variable);
  });
  out.push(`${name} = ${parts.join(' and ')}`);
}

/* -------------------------------------------------------------------------- */
/*  Script assembly                                                           */
/* -------------------------------------------------------------------------- */

const TV_INTERVAL: Record<Interval, string> = {
  '15m': '15 minute',
  '30m': '30 minute',
  '1h': '1 hour',
  '4h': '4 hour',
  '1d': '1 day',
};

/** Generate a complete Pine Script v5 strategy for `result`. */
export function generatePineScript(result: StrategyResult, ctx: PineContext): string {
  const { genome } = result;
  const { exit } = genome;
  const { sizing } = ctx.settings;
  const { settings } = ctx;
  // Everything below compiles from the strategy AST – the same tree the engine executes.
  const ast = compileStrategy(genome, settings.tradeMode);
  const spec = ast.exit;
  const timeStop = spec.timeStop;
  const hasShorts = ast.shortEntry !== null;
  const asymmetric = hasShorts && genome.shortRegime != null;
  const dynamic = spec.stop.kind === 'atr';
  const trailing = spec.trailing !== null;
  const breakeven = spec.breakevenR !== null;
  const hasTarget = spec.target !== null;
  const hasSignalExit = ast.longExit !== null;
  const hasShortSignalExit = ast.shortExit !== null;
  const costPct = Number((settings.feePct + settings.slippagePct).toFixed(6));
  const label = ascii(ctx.label);
  const title = ascii(`StrategyFinder ${result.id} | ${ctx.components.length > 1 ? 'Basket' : label} ${ctx.interval}`);

  const decl = new Declarations();
  const rules: string[] = [];

  emitGroup(ast.longEntry, genome.longEntry, 'longEntry', 'long', 'Long entry', decl, rules);
  if (ast.shortEntry && genome.shortEntry) {
    rules.push('');
    emitGroup(
      ast.shortEntry,
      genome.shortEntry,
      'shortEntry',
      'short',
      asymmetric ? 'Short entry (mirrored trigger, only inside its own bear-regime filter)' : 'Short entry (mirror of the long rules)',
      decl,
      rules,
    );
  }
  if (ast.longExit && exit.mode === 'dynamic' && exit.longExit) {
    rules.push('');
    emitGroup(ast.longExit, exit.longExit, 'longExitSignal', 'longExit', 'Long trigger-reversal exit', decl, rules);
  }
  if (ast.shortExit && exit.mode === 'dynamic' && exit.shortExit) {
    rules.push('');
    emitGroup(ast.shortExit, exit.shortExit, 'shortExitSignal', 'shortExit', 'Short trigger-reversal exit', decl, rules);
  }

  const lines: string[] = [];
  const push = (...text: string[]): void => {
    lines.push(...text);
  };

  /* ---- Header ------------------------------------------------------------ */
  push(
    '//@version=5',
    '// =============================================================================',
    `//  StrategyFinder export - Strategy ${result.id}`,
    `//  Market       : ${label}`,
    `//  Timeframe    : ${TV_INTERVAL[ctx.interval]} candles (apply this script to a ${ctx.interval} chart)`,
    `//  Mode         : ${
      !hasShorts
        ? 'Long-Only'
        : asymmetric
          ? 'Asymmetric Long/Short (shorts: mirrored trigger + their own bear-regime filter)'
          : 'Mirrored Long/Short (shorts mirror the long rules)'
    }`,
    `//  Archetype    : ${ARCHETYPE_LABEL[genome.archetype]}`,
    `//  Parameters   : ${countParameters(genome)} tunable values`,
    `//  Tested on    : ${ascii(ctx.venue)}`,
    `//  Period       : ${isoDate(ctx.windowStart)} to ${isoDate(ctx.windowEnd)} (${ctx.bars} candles: 60% train, 20% validation, 20% holdout)`,
    `//  Costs        : ${settings.feePct}% fee + ${settings.slippagePct}% slippage per side` +
      ((settings.slippageAtrMult ?? 0) > 0 ? ` (StrategyFinder: at least ${settings.slippageAtrMult} x ATR(14)% slippage)` : ''),
    '//',
    `//  StrategyFinder results${ctx.simulatedData ? ' (SIMULATED fallback data - re-validate on real data)' : ''}:`,
    `//    Train        : ${metricsLine(result.train)}`,
    `//    Validation   : ${metricsLine(result.validation)}`,
    `//    Holdout      : ${result.holdout ? metricsLine(result.holdout) : 'not revealed yet (sealed until this strategy is locked)'}`,
    `//    Deflated Sharpe (validation): ${(result.dsr * 100).toFixed(0)}%`,
    `//    Buy & Hold   : ROI ${signed(ctx.benchmarkRoi)}% over train + validation`,
    '//',
    '//  Notes',
    '//   - use_bar_magnifier=false so the script compiles on every TradingView plan. Premium users can',
    '//     set it to true in the strategy() call to resolve candles that touch both the stop and the',
    '//     target with lower-timeframe data (StrategyFinder assumes the stop fills first).',
    "//   - Orders are placed on bar close and filled at the next bar's open, exactly as in StrategyFinder.",
    '//   - Fee and slippage are both charged through commission_value (fee + slippage per side).',
    ...((settings.slippageAtrMult ?? 0) > 0 || (settings.borrowPctPerBar ?? 0) > 0 || settings.fundingPctPerBar > 0
      ? [
          '//   - StrategyFinder also charged volatility-scaled slippage and any funding / short borrow,',
          '//     which commission_value cannot express: expect TradingView results to look slightly better.',
        ]
      : []),
    '//   - Small differences are expected: TradingView uses its own data feed and intrabar fill model.',
    '//   - Educational tool, not financial advice. Past performance does not guarantee future results.',
  );
  if (ctx.assets.length > 1) {
    push(
      '//',
      '//  Asset pool',
      `//   This strategy was optimised on a pool of assets (${ascii(ctx.assets.join(', '))}), each traded`,
      '//   separately with these exact rules. Apply the script to each asset\'s chart; the trade-count',
      '//   gates were met by the pooled sample, so one chart alone shows fewer trades.',
    );
  }
  if (ctx.components.length > 1) {
    push(
      '//',
      '//  Basket',
      '//   This strategy was optimised on a synthetic buy-and-hold basket:',
      ...ctx.components.map(
        (component, index) =>
          `//     ${(component.weight * 100).toFixed(1)}% ${ascii(component.symbol)}${
            ctx.units[index] !== undefined ? `  (${Number(ctx.units[index].toPrecision(6))} units per 100 of basket value)` : ''
          }`,
      ),
      '//   To reproduce it, open a TradingView spread chart whose formula is the sum of',
      '//   units x symbol for each component (for example 0.5*BTCUSD+12*ETHUSD), then add this script.',
      '//   Volume-based conditions are not available on spread charts.',
    );
  }
  push(
    '// =============================================================================',
    `strategy("${title}", overlay=true, initial_capital=${Math.round(settings.initialCapital)}, currency=currency.USD, pyramiding=0, commission_type=strategy.commission.percent, commission_value=${costPct}, slippage=0, calc_on_every_tick=false, process_orders_on_close=false, use_bar_magnifier=false, margin_long=0, margin_short=0)`,
    '',
  );

  /* ---- Inputs ------------------------------------------------------------ */
  push(
    '// ---- Backtest window ---------------------------------------------------------',
    'useWindow   = input.bool(true, "Limit trading to the StrategyFinder test window", group="Backtest window")',
    `windowStart = input.time(${Math.round(ctx.windowStart)}, "Window start", group="Backtest window") // ${isoDate(ctx.windowStart)}`,
    `windowEnd   = input.time(${Math.round(ctx.windowEnd)}, "Window end", group="Backtest window") // ${isoDate(ctx.windowEnd)}`,
    'canEnter    = not useWindow or (time >= windowStart and time < windowEnd)',
    '',
    '// ---- Risk management ---------------------------------------------------------',
  );
  if (spec.stop.kind === 'percent') {
    push(
      `stopLossPct   = input.float(${float(spec.stop.pct)}, "Stop loss %", minval=0.1, step=0.1, group="Risk management")`,
      `takeProfitPct = input.float(${float(spec.target?.kind === 'percent' ? spec.target.pct : 0)}, "Take profit %", minval=0.1, step=0.5, group="Risk management")`,
    );
  } else {
    push(
      `atrLength     = input.int(${spec.stop.period}, "ATR length", minval=1, group="Risk management")`,
      `stopAtrMult   = input.float(${float(spec.stop.mult)}, "${trailing ? 'Trailing stop' : 'Stop loss'} (x ATR)", minval=0.1, step=0.1, group="Risk management")`,
    );
    if (spec.target?.kind === 'atr') {
      push(
        `targetAtrMult = input.float(${float(spec.target.mult)}, "Take profit (x ATR)", minval=0.1, step=0.1, group="Risk management")`,
      );
    }
    push('atrValue      = ta.atr(atrLength)');
  }
  if (sizing.mode === 'risk') {
    push(
      `riskPct       = input.float(${float(sizing.riskPct)}, "Risk per trade (%)", minval=0.05, step=0.05, group="Risk management")`,
      `compound      = input.bool(${sizing.compounding ? 'true' : 'false'}, "Compound (risk % of current equity instead of starting capital)", group="Risk management")`,
    );
  }
  if (timeStop !== null) {
    push(
      `maxBarsInTrade = input.int(${timeStop}, "Time stop (bars in trade)", minval=1, group="Risk management")`,
    );
  }
  push(
    `feePct        = ${float(settings.feePct)} // trading fee per side, in percent`,
    `costPct       = ${float(costPct)} // fee + slippage per side, in percent (charged as commission)`,
    '',
  );

  /* ---- Indicators and rules ---------------------------------------------- */
  push('// ---- Indicators --------------------------------------------------------------', ...decl.render(), '');
  push('// ---- Entry / exit rules ------------------------------------------------------', ...rules, '');

  const atrReady = dynamic ? ' and not na(atrValue)' : '';
  // A time stop blocks re-entering the SAME side on that open; an opposite signal still reverses.
  const longTimed = timeStop !== null ? ' and not (timedOut and strategy.position_size > 0)' : '';
  const shortTimed = timeStop !== null ? ' and not (timedOut and strategy.position_size < 0)' : '';
  if (timeStop !== null) {
    push(
      "// Time stop: a trade that has been open for maxBarsInTrade candles is closed at the next bar's open.",
      'timedOut = strategy.position_size != 0 and bar_index - strategy.opentrades.entry_bar_index(0) >= maxBarsInTrade - 1',
      '',
    );
  }
  push(
    '// An entry is actionable only when the opposite side and its own exit signal are silent.',
    '// An actionable entry against an open position reverses it (strategy.entry closes the other side first).',
  );
  push(
    `goLong  = longEntry${hasShorts ? ' and not shortEntry' : ''}${hasSignalExit ? ' and not longExitSignal' : ''} and canEnter${atrReady}${longTimed}`,
  );
  if (hasShorts) {
    push(
      `goShort = shortEntry and not longEntry${hasShortSignalExit ? ' and not shortExitSignal' : ''} and canEnter${atrReady}${shortTimed}`,
    );
  }
  push('');

  /* ---- Position size ----------------------------------------------------- */
  push('// ---- Position size -----------------------------------------------------------');
  push('// Never deploy more than the account can buy (no leverage).');
  push('maxQty = math.max(strategy.equity, 0) / (close * (1 + costPct / 100))');
  if (sizing.mode === 'full') {
    push('orderQty = maxQty // 100% of equity per trade', '');
  } else {
    push(
      '// Risk-based size: being stopped out costs riskPct of the risk capital (the entry fee is',
      '// budgeted inside the size, exactly as in StrategyFinder).',
      'riskCapital  = compound ? math.max(strategy.equity, 0) : strategy.initial_capital',
      exit.mode === 'static' ? 'stopDistance = close * stopLossPct / 100' : 'stopDistance = stopAtrMult * atrValue',
      'orderQty = na(stopDistance) or stopDistance <= 0 ? 0.0 : math.min(riskCapital * riskPct / 100 / (stopDistance * (1 + feePct / 100)), maxQty)',
      '',
    );
  }

  /* ---- Orders ------------------------------------------------------------ */
  push('// ---- Orders ------------------------------------------------------------------');
  if (exit.mode === 'static') {
    push(
      '// Bracket distances in ticks, attached to the entry so they are live from the fill.',
      'stopTicks   = math.round(close * stopLossPct / 100 / syminfo.mintick)',
      'profitTicks = math.round(close * takeProfitPct / 100 / syminfo.mintick)',
    );
  } else {
    push(
      '// Bracket distances in ticks from the ATR of the signal candle, live from the fill.',
      'stopTicks   = math.round(stopAtrMult * atrValue / syminfo.mintick)',
    );
    if (hasTarget) push('profitTicks = math.round(targetAtrMult * atrValue / syminfo.mintick)');
  }
  const bracket = hasTarget ? 'loss=stopTicks, profit=profitTicks' : 'loss=stopTicks';
  push(
    '',
    `if goLong and strategy.position_size ${hasShorts ? '<=' : '=='} 0`,
    '    strategy.entry("Long", strategy.long, qty=orderQty)',
    `    strategy.exit("Long exit", from_entry="Long", ${bracket})`,
  );
  if (hasShorts) {
    push(
      '',
      'if goShort and strategy.position_size >= 0',
      '    strategy.entry("Short", strategy.short, qty=orderQty)',
      `    strategy.exit("Short exit", from_entry="Short", ${bracket})`,
    );
  }
  push('');

  /* ---- Managed exits: exact levels, trailing stop, breakeven lock ------- */
  // 1R = the initial stop distance. On the fill candle's close the levels are pinned to the fill
  // price; afterwards the stop only ever moves in the trade's favour.
  const riskExpr = exit.mode === 'static' ? 'strategy.position_avg_price * stopLossPct / 100' : 'stopAtrMult * atrValue[1]';
  const targetExpr = (dir: 1 | -1): string =>
    exit.mode === 'static'
      ? `strategy.position_avg_price * (1 ${dir === 1 ? '+' : '-'} takeProfitPct / 100)`
      : exit.takeProfitAtrMult !== null
        ? `strategy.position_avg_price ${dir === 1 ? '+' : '-'} targetAtrMult * atrValue[1]`
        : 'na';
  push(
    '// ---- Exit management ---------------------------------------------------------',
    '// 1R = the initial stop distance. After every close the stop is re-sent: it is pinned to the fill',
    `// price${trailing ? ', ratchets behind the best price since entry (trailing ATR stop)' : ''}${
      breakeven ? ', moves to the entry once the trade has been 1R in profit (breakeven lock)' : ''
    } and never moves against the trade.`,
    'var float longStop   = na',
    'var float longTarget = na',
    'var float longRisk   = na',
    'var float longPeak   = na',
    'if strategy.position_size > 0',
    '    if strategy.position_size[1] <= 0',
    `        longRisk   := ${riskExpr}`,
    '        longStop   := strategy.position_avg_price - longRisk',
    `        longTarget := ${targetExpr(1)}`,
    '        longPeak   := math.max(strategy.position_avg_price, high)',
    '    else',
    '        longPeak   := math.max(longPeak, high)',
  );
  if (breakeven) {
    push('    if longPeak >= strategy.position_avg_price + longRisk // +1R reached: breakeven lock', '        longStop := math.max(longStop, strategy.position_avg_price)');
  }
  if (trailing) push('    longStop := math.max(longStop, longPeak - stopAtrMult * atrValue)');
  push(`    strategy.exit("Long exit", from_entry="Long", stop=longStop${hasTarget ? ', limit=longTarget' : ''})`);
  if (hasShorts) {
    push(
      '',
      'var float shortStop   = na',
      'var float shortTarget = na',
      'var float shortRisk   = na',
      'var float shortTrough = na',
      'if strategy.position_size < 0',
      '    if strategy.position_size[1] >= 0',
      `        shortRisk   := ${riskExpr}`,
      '        shortStop   := strategy.position_avg_price + shortRisk',
      `        shortTarget := ${targetExpr(-1)}`,
      '        shortTrough := math.min(strategy.position_avg_price, low)',
      '    else',
      '        shortTrough := math.min(shortTrough, low)',
    );
    if (breakeven) {
      push('    if shortTrough <= strategy.position_avg_price - shortRisk // +1R reached: breakeven lock', '        shortStop := math.min(shortStop, strategy.position_avg_price)');
    }
    if (trailing) push('    shortStop := math.min(shortStop, shortTrough + stopAtrMult * atrValue)');
    push(`    strategy.exit("Short exit", from_entry="Short", stop=shortStop${hasTarget ? ', limit=shortTarget' : ''})`);
  }
  push('');

  if (hasSignalExit) {
    push(
      "// Opposite-signal exits: close at the next bar's open (a reversal entry, when there is one, takes precedence).",
      `if longExitSignal and strategy.position_size > 0${hasShorts ? ' and not goShort' : ''}`,
      '    strategy.close("Long", comment="Signal exit")',
    );
    if (hasShortSignalExit) {
      push(
        'if shortExitSignal and strategy.position_size < 0 and not goLong',
        '    strategy.close("Short", comment="Signal exit")',
      );
    }
    push('');
  }

  if (timeStop !== null) {
    // A reversal entry on the same open already closes the position: closing it again would cancel the reversal.
    push(
      `if timedOut${hasShorts ? ' and not (strategy.position_size > 0 and goShort) and not (strategy.position_size < 0 and goLong)' : ''}`,
      '    strategy.close_all(comment="Time stop")',
      '',
    );
  }

  push(
    "// Flatten on the window's last candle (filled at the next open), as StrategyFinder liquidates at that close.",
    'if useWindow and time >= windowEnd and strategy.position_size != 0',
    '    strategy.close_all(comment="Window end")',
  );

  push(
    '',
    '// ---- Plots -------------------------------------------------------------------',
    'plot(strategy.position_size > 0 ? longStop : na, "Long stop", color=color.new(color.red, 0), style=plot.style_linebr)',
  );
  if (hasTarget) push('plot(strategy.position_size > 0 ? longTarget : na, "Long target", color=color.new(color.green, 0), style=plot.style_linebr)');
  if (hasShorts) {
    push('plot(strategy.position_size < 0 ? shortStop : na, "Short stop", color=color.new(color.orange, 0), style=plot.style_linebr)');
    if (hasTarget) push('plot(strategy.position_size < 0 ? shortTarget : na, "Short target", color=color.new(color.teal, 0), style=plot.style_linebr)');
  }

  return `${lines.join('\n')}\n`;
}
