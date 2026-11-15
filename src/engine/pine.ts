/**
 * Pine Script v5 exporter.
 *
 * Generates a complete, ready-to-paste TradingView strategy that implements a
 * genome with the same rules, exits, sizing and execution timing as the
 * StrategyFinder backtester:
 *   • indicators use the matching `ta.*` built-ins,
 *   • orders are placed on bar close and fill at the next bar's open,
 *   • fee + slippage are charged through `commission_value`,
 *   • SL/TP brackets are attached to the entry so they are live from the fill.
 *
 * The script is plain ASCII so it survives any clipboard / editor round-trip.
 */

import { SLOPE_LAG } from '../types/trading';
import type {
  BacktestSettings,
  BasketComponent,
  CmpOp,
  Condition,
  Interval,
  PerformanceMetrics,
  RuleGroup,
  StrategyResult,
} from '../types/trading';
import { ARCHETYPE_LABEL, groupFormula } from './describe';
import { countParameters } from './genome';

export interface PineContext {
  /** Dataset label, e.g. "BTC-USD" or "50% BTC-USD · 30% ETH-USD · 20% NVDA". */
  label: string;
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
class Declarations {
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

/** Declare what a condition needs and return its Pine operands. */
function operands(c: Condition, decl: Declarations): { left: string; right: string } {
  switch (c.ind) {
    case 'RSI': {
      const name = `rsi_${c.period}`;
      decl.add(name, `${name} = ta.rsi(close, ${c.period})`);
      return { left: name, right: String(c.level) };
    }
    case 'MACD': {
      const id = `${c.fast}_${c.slow}_${c.signal}`;
      decl.add(
        `macd_${id}`,
        `[macdLine_${id}, macdSignal_${id}, macdHist_${id}] = ta.macd(close, ${c.fast}, ${c.slow}, ${c.signal})`,
      );
      if (c.variant === 'signal') return { left: `macdLine_${id}`, right: `macdSignal_${id}` };
      if (c.variant === 'zero') return { left: `macdLine_${id}`, right: '0' };
      return { left: `macdHist_${id}`, right: `macdHist_${id}[1]` };
    }
    case 'PRICE_MA':
      return { left: 'close', right: maVar(decl, c.maType, c.period) };
    case 'MA_CROSS':
      return { left: maVar(decl, c.fastType, c.fastPeriod), right: maVar(decl, c.slowType, c.slowPeriod) };
    case 'BB': {
      const id = `${c.period}_${ident(c.mult)}`;
      decl.add(`bb_${id}`, `[bbBasis_${id}, bbUpper_${id}, bbLower_${id}] = ta.bb(close, ${c.period}, ${float(c.mult)})`);
      switch (c.variant) {
        case 'pctb':
          decl.add(`bbPctB_${id}`, `bbPctB_${id} = (close - bbLower_${id}) / (bbUpper_${id} - bbLower_${id})`);
          return { left: `bbPctB_${id}`, right: float(c.level) };
        case 'bandwidth':
          decl.add(`bbWidth_${id}`, `bbWidth_${id} = (bbUpper_${id} - bbLower_${id}) / bbBasis_${id} * 100`);
          return { left: `bbWidth_${id}`, right: float(c.level) };
        case 'upper':
          return { left: 'close', right: `bbUpper_${id}` };
        case 'lower':
          return { left: 'close', right: `bbLower_${id}` };
      }
      break;
    }
    case 'ATR': {
      const atr = atrVar(decl, c.period);
      const name = `atrPct_${c.period}`;
      decl.add(name, `${name} = ${atr} / close * 100`);
      return { left: name, right: float(c.level) };
    }
    case 'STOCH': {
      const k = `stochK_${c.kPeriod}_${c.smooth}`;
      decl.add(k, `${k} = ta.sma(ta.stoch(close, high, low, ${c.kPeriod}), ${c.smooth})`);
      if (c.variant === 'level') return { left: k, right: String(c.level) };
      const d = `stochD_${c.kPeriod}_${c.smooth}_${c.dPeriod}`;
      decl.add(d, `${d} = ta.sma(${k}, ${c.dPeriod})`);
      return { left: k, right: d };
    }
    case 'ADX': {
      const p = c.period;
      decl.add(`dmi_${p}`, `[diPlus_${p}, diMinus_${p}, adx_${p}] = ta.dmi(${p}, ${p})`);
      return c.variant === 'strength'
        ? { left: `adx_${p}`, right: String(c.level) }
        : { left: `diPlus_${p}`, right: `diMinus_${p}` };
    }
    case 'VOLUME': {
      const name = `volRatio_${c.period}`;
      decl.add(name, `${name} = volume / ta.sma(volume, ${c.period})`);
      return { left: name, right: float(c.level) };
    }
    case 'SUPERTREND': {
      const id = `${c.period}_${ident(c.mult)}`;
      decl.add(`st_${id}`, `[supertrend_${id}, supertrendDir_${id}] = ta.supertrend(${float(c.mult)}, ${c.period})`);
      return { left: 'close', right: `supertrend_${id}` };
    }
    case 'CCI': {
      const name = `cci_${c.period}`;
      decl.add(name, `${name} = ta.cci(hlc3, ${c.period})`);
      return { left: name, right: String(c.level) };
    }
    case 'DONCHIAN': {
      // Channel of the PREVIOUS candles: the current candle is excluded, as in StrategyFinder.
      const name = c.variant === 'upper' ? `donchianHigh_${c.period}` : `donchianLow_${c.period}`;
      decl.add(name, c.variant === 'upper' ? `${name} = ta.highest(high, ${c.period})[1]` : `${name} = ta.lowest(low, ${c.period})[1]`);
      return { left: 'close', right: name };
    }
    case 'MA_SLOPE': {
      const ma = maVar(decl, c.maType, c.period);
      return { left: ma, right: `${ma}[${SLOPE_LAG}]` };
    }
  }
  throw new Error(`Unsupported condition: ${JSON.stringify(c)}`);
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

/** Emit `prefixCond1 = ...` lines and the combined `name = a and b` line. */
function emitGroup(
  group: RuleGroup,
  name: string,
  prefix: string,
  title: string,
  decl: Declarations,
  out: string[],
): void {
  out.push(`// ${title}: ${ascii(groupFormula(group))}`);
  const parts: string[] = [];
  group.conditions.forEach((condition, index) => {
    const { left, right } = operands(condition, decl);
    const variable = `${prefix}Cond${index + 1}`;
    out.push(`${variable} = ${compare(condition.op, left, right)}`);
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
  const timeStop = genome.maxBarsInTrade;
  const { settings } = ctx;
  const hasShorts = genome.shortEntry !== null && settings.tradeMode === 'long-short';
  const dynamic = exit.mode === 'dynamic';
  const trailing = exit.mode === 'dynamic' && exit.trailing;
  const hasTarget = exit.mode === 'static' || exit.takeProfitAtrMult !== null;
  const hasSignalExit = exit.mode === 'dynamic' && exit.longExit !== null;
  const hasShortSignalExit = exit.mode === 'dynamic' && hasShorts && exit.shortExit !== null;
  const costPct = Number((settings.feePct + settings.slippagePct).toFixed(6));
  const label = ascii(ctx.label);
  const title = ascii(`StrategyFinder ${result.id} | ${ctx.components.length > 1 ? 'Basket' : label} ${ctx.interval}`);

  const decl = new Declarations();
  const rules: string[] = [];

  emitGroup(genome.longEntry, 'longEntry', 'long', 'Long entry', decl, rules);
  if (hasShorts && genome.shortEntry) {
    rules.push('');
    emitGroup(genome.shortEntry, 'shortEntry', 'short', 'Short entry', decl, rules);
  }
  if (exit.mode === 'dynamic' && exit.longExit) {
    rules.push('');
    emitGroup(exit.longExit, 'longExitSignal', 'longExit', 'Long trigger-reversal exit', decl, rules);
  }
  if (exit.mode === 'dynamic' && hasShorts && exit.shortExit) {
    rules.push('');
    emitGroup(exit.shortExit, 'shortExitSignal', 'shortExit', 'Short trigger-reversal exit', decl, rules);
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
    `//  Mode         : ${hasShorts ? 'Long/Short (shorts mirror the long rules)' : 'Long-Only'}`,
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
  if (settings.poolAssets) {
    push(
      '//',
      '//  Asset pool',
      `//   This strategy was optimised on a pool of assets (${label.replace(/^Pool:\s*/, '')}), each traded`,
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
  if (exit.mode === 'static') {
    push(
      `stopLossPct   = input.float(${float(exit.stopLossPct)}, "Stop loss %", minval=0.1, step=0.1, group="Risk management")`,
      `takeProfitPct = input.float(${float(exit.takeProfitPct)}, "Take profit %", minval=0.1, step=0.5, group="Risk management")`,
    );
  } else {
    push(
      `atrLength     = input.int(${exit.atrPeriod}, "ATR length", minval=1, group="Risk management")`,
      `stopAtrMult   = input.float(${float(exit.stopAtrMult)}, "${trailing ? 'Trailing stop' : 'Stop loss'} (x ATR)", minval=0.1, step=0.1, group="Risk management")`,
    );
    if (exit.takeProfitAtrMult !== null) {
      push(
        `targetAtrMult = input.float(${float(exit.takeProfitAtrMult)}, "Take profit (x ATR)", minval=0.1, step=0.1, group="Risk management")`,
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
  push(`costPct       = ${float(costPct)} // fee + slippage per side, in percent`, '');

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
      '// Risk-based size: being stopped out costs riskPct of the risk capital.',
      'riskCapital  = compound ? math.max(strategy.equity, 0) : strategy.initial_capital',
      exit.mode === 'static' ? 'stopDistance = close * stopLossPct / 100' : 'stopDistance = stopAtrMult * atrValue',
      'orderQty = na(stopDistance) or stopDistance <= 0 ? 0.0 : math.min(riskCapital * riskPct / 100 / stopDistance, maxQty)',
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

  if (exit.mode === 'static') {
    push(
      '// Once filled, pin the bracket to the exact fill price.',
      'if strategy.position_size > 0',
      '    strategy.exit("Long exit", from_entry="Long", stop=strategy.position_avg_price * (1 - stopLossPct / 100), limit=strategy.position_avg_price * (1 + takeProfitPct / 100))',
    );
    if (hasShorts) {
      push(
        'if strategy.position_size < 0',
        '    strategy.exit("Short exit", from_entry="Short", stop=strategy.position_avg_price * (1 + stopLossPct / 100), limit=strategy.position_avg_price * (1 - takeProfitPct / 100))',
      );
    }
    push('');
  } else if (trailing) {
    push(
      '// Trailing ATR stop: after every close the stop ratchets behind the best price since entry.',
      'var float longTrail  = na',
      'var float longPeak   = na',
    );
    if (hasTarget) push('var float longTarget = na');
    push(
      'if strategy.position_size > 0',
      '    if strategy.position_size[1] <= 0',
      '        longPeak  := math.max(strategy.position_avg_price, high)',
      '        longTrail := math.max(strategy.position_avg_price - stopAtrMult * atrValue[1], longPeak - stopAtrMult * atrValue)',
    );
    if (hasTarget) push('        longTarget := strategy.position_avg_price + targetAtrMult * atrValue[1]');
    push(
      '    else',
      '        longPeak  := math.max(longPeak, high)',
      '        longTrail := math.max(longTrail, longPeak - stopAtrMult * atrValue)',
      `    strategy.exit("Long exit", from_entry="Long", stop=longTrail${hasTarget ? ', limit=longTarget' : ''})`,
    );
    if (hasShorts) {
      push('', 'var float shortTrail  = na', 'var float shortTrough = na');
      if (hasTarget) push('var float shortTarget = na');
      push(
        'if strategy.position_size < 0',
        '    if strategy.position_size[1] >= 0',
        '        shortTrough := math.min(strategy.position_avg_price, low)',
        '        shortTrail  := math.min(strategy.position_avg_price + stopAtrMult * atrValue[1], shortTrough + stopAtrMult * atrValue)',
      );
      if (hasTarget) push('        shortTarget := strategy.position_avg_price - targetAtrMult * atrValue[1]');
      push(
        '    else',
        '        shortTrough := math.min(shortTrough, low)',
        '        shortTrail  := math.min(shortTrail, shortTrough + stopAtrMult * atrValue)',
        `    strategy.exit("Short exit", from_entry="Short", stop=shortTrail${hasTarget ? ', limit=shortTarget' : ''})`,
      );
    }
    push('');
  }

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
    push('if timedOut', '    strategy.close_all(comment="Time stop")', '');
  }

  push(
    '// Flatten once the test window has ended.',
    'if useWindow and time > windowEnd and strategy.position_size != 0',
    '    strategy.close_all(comment="Window end")',
  );

  if (trailing) {
    push(
      '',
      '// ---- Plots -------------------------------------------------------------------',
      'plot(strategy.position_size > 0 ? longTrail : na, "Long trailing stop", color=color.new(color.red, 0), style=plot.style_linebr)',
    );
    if (hasShorts) {
      push(
        'plot(strategy.position_size < 0 ? shortTrail : na, "Short trailing stop", color=color.new(color.orange, 0), style=plot.style_linebr)',
      );
    }
  }

  return `${lines.join('\n')}\n`;
}
