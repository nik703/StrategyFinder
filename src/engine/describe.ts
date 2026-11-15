/**
 * Human-readable rendering of strategy genomes: compact formulas for badges
 * and tables, and plain-English sentences for the Strategy Breakdown view.
 */

import { conditionRole } from './genome';
import { SLOPE_LAG } from '../types/trading';
import type {
  Archetype,
  BacktestSettings,
  CmpOp,
  Condition,
  RuleGroup,
  StrategyGenome,
} from '../types/trading';

const fmt = (value: number, decimals: number): string => value.toFixed(decimals);
const mult = (value: number): string => value.toFixed(1);

/* -------------------------------------------------------------------------- */
/*  Compact formulas                                                          */
/* -------------------------------------------------------------------------- */

const OP_SYMBOL: Record<CmpOp, string> = {
  gt: '>',
  lt: '<',
  cross_above: 'crosses above',
  cross_below: 'crosses below',
};

/** Left- and right-hand labels of a condition, e.g. { left: "RSI(14)", right: "34" }. */
export function conditionOperands(c: Condition): { left: string; right: string } {
  switch (c.ind) {
    case 'RSI':
      return { left: `RSI(${c.period})`, right: String(c.level) };
    case 'MACD':
      switch (c.variant) {
        case 'signal':
          return { left: `MACD(${c.fast},${c.slow})`, right: `Signal(${c.signal})` };
        case 'zero':
          return { left: `MACD(${c.fast},${c.slow})`, right: '0' };
        case 'hist':
          return { left: `MACD_Hist(${c.fast},${c.slow},${c.signal})`, right: 'MACD_Hist[1]' };
      }
      break;
    case 'PRICE_MA':
      return { left: 'Close', right: `${c.maType}(${c.period})` };
    case 'MA_CROSS':
      return { left: `${c.fastType}(${c.fastPeriod})`, right: `${c.slowType}(${c.slowPeriod})` };
    case 'BB': {
      const args = `${c.period},${mult(c.mult)}`;
      switch (c.variant) {
        case 'pctb':
          return { left: `BB_%B(${args})`, right: fmt(c.level, 2) };
        case 'bandwidth':
          return { left: `BB_Width(${args})`, right: `${fmt(c.level, 1)}%` };
        case 'upper':
          return { left: 'Close', right: `BB_Upper(${args})` };
        case 'lower':
          return { left: 'Close', right: `BB_Lower(${args})` };
      }
      break;
    }
    case 'ATR':
      return { left: `ATR%(${c.period})`, right: `${fmt(c.level, 2)}%` };
    case 'STOCH':
      return c.variant === 'level'
        ? { left: `Stoch_%K(${c.kPeriod},${c.smooth})`, right: String(c.level) }
        : { left: `Stoch_%K(${c.kPeriod},${c.smooth})`, right: `%D(${c.dPeriod})` };
    case 'ADX':
      return c.variant === 'strength'
        ? { left: `ADX(${c.period})`, right: String(c.level) }
        : { left: `+DI(${c.period})`, right: `-DI(${c.period})` };
    case 'VOLUME':
      return { left: `Volume_Ratio(${c.period})`, right: fmt(c.level, 2) };
    case 'SUPERTREND':
      return { left: 'Close', right: `Supertrend(${c.period},${mult(c.mult)})` };
    case 'CCI':
      return { left: `CCI(${c.period})`, right: String(c.level) };
    case 'DONCHIAN':
      return { left: 'Close', right: c.variant === 'upper' ? `Highest_High(${c.period})[1]` : `Lowest_Low(${c.period})[1]` };
    case 'MA_SLOPE':
      return { left: `${c.maType}(${c.period})`, right: `${c.maType}(${c.period})[${SLOPE_LAG}]` };
  }
  return { left: '?', right: '?' };
}

/** e.g. "RSI(14) < 34" */
export function conditionFormula(c: Condition): string {
  const { left, right } = conditionOperands(c);
  return `${left} ${OP_SYMBOL[c.op]} ${right}`;
}

/** e.g. "RSI(14) < 34 AND Close > EMA(200) AND Volume_Ratio(20) > 1.30" */
export function groupFormula(group: RuleGroup): string {
  return group.conditions.map(conditionFormula).join(' AND ');
}

/* -------------------------------------------------------------------------- */
/*  Plain-English sentences                                                   */
/* -------------------------------------------------------------------------- */

function relation(op: CmpOp, above: string, below: string, crossUp: string, crossDown: string): string {
  switch (op) {
    case 'gt':
      return above;
    case 'lt':
      return below;
    case 'cross_above':
      return crossUp;
    case 'cross_below':
      return crossDown;
  }
}

const generic = (op: CmpOp): string => relation(op, 'is above', 'is below', 'crosses above', 'crosses below');

/** One plain-English sentence describing when a condition is true. */
export function conditionSentence(c: Condition): string {
  switch (c.ind) {
    case 'RSI': {
      const note =
        c.op === 'lt'
          ? c.level <= 40
            ? ' (oversold / pullback zone)'
            : ''
          : c.op === 'gt'
            ? c.level >= 60
              ? ' (strong upside momentum)'
              : ''
            : '';
      return `${c.period}-period RSI ${generic(c.op)} ${c.level}${note}`;
    }
    case 'MACD':
      switch (c.variant) {
        case 'signal':
          return `MACD line (${c.fast}/${c.slow} EMA) ${generic(c.op)} its ${c.signal}-period signal line${relation(
            c.op,
            ' (histogram positive)',
            ' (histogram negative)',
            ' (bullish crossover)',
            ' (bearish crossover)',
          )}`;
        case 'zero':
          return `MACD line (${c.fast}/${c.slow} EMA) ${generic(c.op)} zero${relation(
            c.op,
            ' (fast EMA above slow EMA)',
            ' (fast EMA below slow EMA)',
            '',
            '',
          )}`;
        case 'hist':
          return `MACD histogram (${c.fast}, ${c.slow}, ${c.signal}) ${relation(
            c.op,
            'is higher than on the previous candle (momentum building)',
            'is lower than on the previous candle (momentum fading)',
            'turns up after falling',
            'turns down after rising',
          )}`;
      }
      break;
    case 'PRICE_MA':
      return `Close ${generic(c.op)} the ${c.period}-period ${c.maType}`;
    case 'MA_CROSS':
      return `${c.fastPeriod}-period ${c.fastType} ${generic(c.op)} the ${c.slowPeriod}-period ${c.slowType}${relation(
        c.op,
        ' (fast average leading)',
        ' (fast average lagging)',
        ' (bullish moving-average crossover)',
        ' (bearish moving-average crossover)',
      )}`;
    case 'BB': {
      const bands = `Bollinger Bands (${c.period}, ${mult(c.mult)}σ)`;
      switch (c.variant) {
        case 'pctb':
          return `%B of the ${bands} ${generic(c.op)} ${fmt(c.level, 2)} (0 = lower band, 1 = upper band)`;
        case 'bandwidth':
          return `Bandwidth of the ${bands} ${generic(c.op)} ${fmt(c.level, 1)}% of the middle band${relation(
            c.op,
            ' (volatility expanding)',
            ' (volatility squeeze)',
            '',
            '',
          )}`;
        case 'upper':
          return `Close ${generic(c.op)} the upper band of the ${bands}`;
        case 'lower':
          return `Close ${generic(c.op)} the lower band of the ${bands}`;
      }
      break;
    }
    case 'ATR':
      return `${c.period}-period ATR ${generic(c.op)} ${fmt(c.level, 2)}% of the close${relation(
        c.op,
        ' (elevated volatility)',
        ' (calm market)',
        '',
        '',
      )}`;
    case 'STOCH':
      return c.variant === 'level'
        ? `Stochastic %K (${c.kPeriod}-period, smoothed ${c.smooth}) ${generic(c.op)} ${c.level}`
        : `Stochastic %K (${c.kPeriod}-period, smoothed ${c.smooth}) ${generic(c.op)} its ${c.dPeriod}-period %D line`;
    case 'ADX':
      return c.variant === 'strength'
        ? `${c.period}-period ADX ${generic(c.op)} ${c.level}${relation(c.op, ' (a trend is in force)', ' (ranging market)', '', '')}`
        : `+DI ${generic(c.op)} −DI on the ${c.period}-period DMI${relation(
            c.op,
            ' (buyers in control)',
            ' (sellers in control)',
            '',
            '',
          )}`;
    case 'VOLUME':
      return c.op === 'lt'
        ? `Volume is below ${fmt(c.level, 2)}× its ${c.period}-bar average (quiet tape)`
        : `Volume is above ${fmt(c.level, 2)}× its ${c.period}-bar average${c.level >= 1.2 ? ' (volume spike)' : ''}`;
    case 'SUPERTREND':
      return `Close ${generic(c.op)} the Supertrend line (ATR ${c.period}, ×${mult(c.mult)})${relation(
        c.op,
        ' (uptrend)',
        ' (downtrend)',
        ' (trend flips up)',
        ' (trend flips down)',
      )}`;
    case 'CCI':
      return `${c.period}-period CCI ${generic(c.op)} ${c.level}`;
    case 'DONCHIAN':
      return c.variant === 'upper'
        ? `Close ${generic(c.op)} the highest high of the previous ${c.period} candles${relation(c.op, '', '', ' (range breakout)', '')}`
        : `Close ${generic(c.op)} the lowest low of the previous ${c.period} candles${relation(c.op, '', '', '', ' (range breakdown)')}`;
    case 'MA_SLOPE':
      return `The ${c.period}-period ${c.maType} ${relation(c.op, 'is rising', 'is falling', 'turns up', 'turns down')} (vs. ${SLOPE_LAG} candles earlier)`;
  }
  return conditionFormula(c);
}

export interface RuleDescription {
  /** Sentence that introduces the bullet list. */
  intro: string;
  bullets: { text: string; formula: string }[];
}

/** Label for the job a condition does inside its entry. */
function conditionJob(c: Condition, index: number): string {
  if (index === 0) return 'Entry trigger';
  return conditionRole(c) === 'FILTER' ? 'Confirmation filter' : 'Regime filter';
}

function describeGroup(group: RuleGroup, action: string, labelled = true): RuleDescription {
  const quantifier =
    group.conditions.length > 2
      ? 'when ALL of the following are true on the close of the same candle:'
      : group.conditions.length > 1
      ? 'when BOTH of the following are true on the close of the same candle:'
      : 'when, on the close of a candle:';
  return {
    intro: `${action} ${quantifier}`,
    bullets: group.conditions.map((c, index) => ({
      text: labelled ? `${conditionJob(c, index)}: ${conditionSentence(c)}` : conditionSentence(c),
      formula: conditionFormula(c),
    })),
  };
}

export function describeLongEntry(genome: StrategyGenome): RuleDescription {
  return describeGroup(genome.longEntry, "Buy (go long) at the next candle's open");
}

export function describeShortEntry(genome: StrategyGenome): RuleDescription | null {
  if (genome.shortEntry === null) return null;
  return describeGroup(genome.shortEntry, "Sell short at the next candle's open");
}

export const ARCHETYPE_LABEL: Record<Archetype, string> = {
  TREND: 'Trend-following',
  MEAN_REVERSION: 'Mean reversion',
};

/** One paragraph that states the idea behind the strategy in plain English. */
export function describeArchetype(genome: StrategyGenome): string {
  const second = genome.longEntry.conditions[1];
  const hasShorts = genome.shortEntry !== null;
  if (genome.archetype === 'TREND') {
    return (
      'Trend-following: it buys strength. The trigger fires when price or momentum turns up' +
      (second ? ', and the filter only lets trades through in the right market conditions' : '') +
      (hasShorts ? '. Shorts are the exact mirror image: it sells weakness under the same rules.' : '.')
    );
  }
  const regime = second && conditionRole(second) !== 'FILTER';
  return (
    'Mean reversion: it buys weakness and expects a bounce. The trigger fires on an oversold reading or a short-term downswing' +
    (regime
      ? ', but only while the slow trend is still up, so it fades dips inside an uptrend rather than catching falling knives'
      : second
        ? ', and the filter only lets trades through in the right market conditions'
        : '') +
    (hasShorts ? '. Shorts are the exact mirror image: it sells strength under the same rules.' : '.')
  );
}

/* -------------------------------------------------------------------------- */
/*  Exit and risk management                                                  */
/* -------------------------------------------------------------------------- */

export interface ExitDescription {
  mode: 'Static %' | 'Dynamic ATR';
  /** One-line summary of the exit construction. */
  headline: string;
  bullets: { title: string; text: string }[];
  longSignalExit: RuleDescription | null;
  shortSignalExit: RuleDescription | null;
}

export function describeExit(genome: StrategyGenome, settings: BacktestSettings): ExitDescription {
  const { exit } = genome;
  const { sizing } = settings;
  const hasShorts = genome.shortEntry !== null;
  const bullets: { title: string; text: string }[] = [];
  let headline: string;
  let longSignalExit: RuleDescription | null = null;
  let shortSignalExit: RuleDescription | null = null;

  if (exit.mode === 'static') {
    const sl = fmt(exit.stopLossPct, 1);
    const tp = fmt(exit.takeProfitPct, 1);
    headline = `Static exits: fixed ${sl}% stop loss and fixed ${tp}% take profit, both measured from the entry price.`;
    bullets.push({
      title: 'Stop loss',
      text: `Fixed ${sl}% from entry. Longs exit at entry × ${fmt(1 - exit.stopLossPct / 100, 4)}${
        hasShorts ? `; shorts cover at entry × ${fmt(1 + exit.stopLossPct / 100, 4)}` : ''
      }.`,
    });
    bullets.push({
      title: 'Take profit',
      text: `Fixed ${tp}% from entry. Longs exit at entry × ${fmt(1 + exit.takeProfitPct / 100, 4)}${
        hasShorts ? `; shorts cover at entry × ${fmt(1 - exit.takeProfitPct / 100, 4)}` : ''
      }. Reward-to-risk ${fmt(exit.takeProfitPct / exit.stopLossPct, 2)} : 1.`,
    });
  } else {
    const atr = `ATR(${exit.atrPeriod})`;
    const k = mult(exit.stopAtrMult);
    headline = `Dynamic exits driven by ${atr}: the distances adapt to current volatility instead of using fixed percentages.`;
    if (exit.trailing) {
      bullets.push({
        title: 'Trailing ATR stop',
        text:
          `Initial stop = entry − ${k} × ${atr} for longs${hasShorts ? ` (entry + ${k} × ${atr} for shorts)` : ''}, ` +
          `using the ATR of the signal candle. After every close the stop ratchets to ` +
          `highest high since entry − ${k} × ${atr}${hasShorts ? ` (lowest low since entry + ${k} × ${atr} for shorts)` : ''} ` +
          `and never moves against the position.`,
      });
    } else {
      bullets.push({
        title: 'ATR stop loss',
        text:
          `Stop = entry − ${k} × ${atr} for longs${hasShorts ? ` (entry + ${k} × ${atr} for shorts)` : ''}, ` +
          `fixed at entry using the ATR of the signal candle.`,
      });
    }
    if (exit.takeProfitAtrMult !== null) {
      const m = mult(exit.takeProfitAtrMult);
      bullets.push({
        title: 'ATR take profit',
        text:
          `Target = entry + ${m} × ${atr} for longs${hasShorts ? ` (entry − ${m} × ${atr} for shorts)` : ''}. ` +
          `Reward-to-risk ${fmt(exit.takeProfitAtrMult / exit.stopAtrMult, 2)} : 1 at entry.`,
      });
    } else {
      bullets.push({
        title: 'Take profit',
        text: `No fixed target – winners run until ${
          [
            exit.trailing ? 'the trailing stop' : null,
            exit.longExit ? 'the opposite signal' : null,
            genome.maxBarsInTrade !== null ? 'the time stop' : null,
          ]
            .filter((part): part is string => part !== null)
            .join(' or ')
        } closes the trade.`,
      });
    }
    if (exit.longExit) {
      longSignalExit = describeGroup(exit.longExit, "Close the long at the next candle's open", false);
      if (exit.shortExit) shortSignalExit = describeGroup(exit.shortExit, "Cover the short at the next candle's open", false);
      bullets.push({
        title: 'Opposite-signal exit',
        text: "A trade is also closed when the opposite side's entry trigger fires (rule listed below).",
      });
    }
  }

  if (genome.maxBarsInTrade !== null) {
    bullets.push({
      title: 'Time stop',
      text: `A trade still open after ${genome.maxBarsInTrade} candles is closed at the next candle's open.`,
    });
  }

  if (hasShorts) {
    bullets.push({
      title: 'Stop-and-reverse',
      text: "An opposite entry signal closes the open position and flips direction at the next candle's open.",
    });
  }

  if (sizing.mode === 'full') {
    bullets.push({
      title: 'Position size (run setting)',
      text: '100% of account equity on every trade (profits are compounded, no leverage).',
    });
  } else {
    const risk = fmt(sizing.riskPct, 2);
    const base = sizing.compounding ? 'current equity' : 'the starting capital (no compounding)';
    bullets.push({
      title: 'Position size (run setting)',
      text:
        `Each trade risks ${risk}% of ${base} at the stop: size = risk ÷ stop distance` +
        (exit.mode === 'dynamic'
          ? ` (${mult(exit.stopAtrMult)} × ATR(${exit.atrPeriod})), so positions shrink when volatility rises.`
          : ` (${fmt(exit.stopLossPct, 1)}% of price).`) +
        ' Exposure is capped at 1× equity – no leverage.',
    });
  }

  if (settings.fundingPctPerBar > 0) {
    bullets.push({
      title: 'Funding',
      text: `Every open position pays ${trimPct(settings.fundingPctPerBar)}% of its notional per candle as perpetual funding.`,
    });
  }
  const borrow = settings.borrowPctPerBar ?? 0;
  if (borrow > 0 && settings.tradeMode === 'long-short') {
    bullets.push({
      title: 'Short borrow',
      text: `Every open short pays a borrow fee of ${trimPct(borrow)}% of its notional per candle.`,
    });
  }

  const atrMult = settings.slippageAtrMult ?? 0;
  const slippage =
    atrMult > 0
      ? `slippage of ${trimPct(settings.slippagePct)}% or ${trimPct(atrMult)} × ATR(14) as a % of price, whichever is larger (volatility-scaled)`
      : `${trimPct(settings.slippagePct)}% slippage`;
  bullets.push({
    title: 'Execution',
    text:
      `Signals are read on the close of candle t and filled at the open of candle t+1 (no look-ahead). ` +
      `Every fill pays a ${trimPct(settings.feePct)}% fee plus ${slippage}. ` +
      `If a candle touches both the stop and the target, the stop is assumed to fill first.`,
  });

  return {
    mode: exit.mode === 'static' ? 'Static %' : 'Dynamic ATR',
    headline,
    bullets,
    longSignalExit,
    shortSignalExit,
  };
}

function trimPct(value: number): string {
  return String(Number(value.toFixed(4)));
}

/* -------------------------------------------------------------------------- */
/*  Badges and indicator inventory                                            */
/* -------------------------------------------------------------------------- */

export function exitModeLabel(genome: StrategyGenome): 'Static %' | 'Dynamic ATR' {
  return genome.exit.mode === 'static' ? 'Static %' : 'Dynamic ATR';
}

/** Short qualifiers shown next to the SL/TP mode, e.g. ["Trailing", "Signal exit"]. */
export function exitTags(genome: StrategyGenome): string[] {
  const tags: string[] = [];
  if (genome.exit.mode === 'dynamic') {
    if (genome.exit.trailing) tags.push('Trailing');
    if (genome.exit.longExit) tags.push('Opposite-signal exit');
    if (genome.exit.takeProfitAtrMult === null) tags.push('No target');
  }
  if (genome.maxBarsInTrade !== null) tags.push(`${genome.maxBarsInTrade}-bar time stop`);
  return tags;
}

function badgeFor(c: Condition): string[] {
  switch (c.ind) {
    case 'RSI':
      return ['RSI'];
    case 'MACD':
      return ['MACD'];
    case 'PRICE_MA':
      return [c.maType];
    case 'MA_CROSS':
      return c.fastType === c.slowType ? [c.fastType] : [c.fastType, c.slowType];
    case 'BB':
      return ['BB'];
    case 'ATR':
      return ['ATR'];
    case 'STOCH':
      return ['Stoch'];
    case 'ADX':
      return ['ADX'];
    case 'VOLUME':
      return ['Volume'];
    case 'SUPERTREND':
      return ['Supertrend'];
    case 'CCI':
      return ['CCI'];
    case 'DONCHIAN':
      return ['Donchian'];
    case 'MA_SLOPE':
      return [c.maType];
  }
}

function allGroups(genome: StrategyGenome): { group: RuleGroup; role: string }[] {
  const groups: { group: RuleGroup; role: string }[] = [{ group: genome.longEntry, role: 'Long entry' }];
  if (genome.shortEntry) groups.push({ group: genome.shortEntry, role: 'Short entry' });
  if (genome.exit.mode === 'dynamic') {
    if (genome.exit.longExit) groups.push({ group: genome.exit.longExit, role: 'Long exit' });
    if (genome.exit.shortExit) groups.push({ group: genome.exit.shortExit, role: 'Short exit' });
  }
  return groups;
}

const BADGE_ORDER = ['RSI', 'MACD', 'EMA', 'SMA', 'Donchian', 'BB', 'Stoch', 'ADX', 'Volume', 'Supertrend', 'CCI', 'ATR'];

/** Unique indicator names used anywhere in the strategy, in a stable order. */
export function indicatorBadges(genome: StrategyGenome): string[] {
  const used = new Set<string>();
  for (const { group } of allGroups(genome)) {
    for (const c of group.conditions) for (const badge of badgeFor(c)) used.add(badge);
  }
  if (genome.exit.mode === 'dynamic') used.add('ATR');
  return BADGE_ORDER.filter((badge) => used.has(badge));
}

export interface IndicatorUsage {
  name: string;
  /** Exact parameters, e.g. "period 14" or "fast 12 · slow 26 · signal 9". */
  params: string;
  /** Where the indicator is used, e.g. ["Long entry", "Stop loss"]. */
  roles: string[];
}

function usageFor(c: Condition): { name: string; params: string }[] {
  switch (c.ind) {
    case 'RSI':
      return [{ name: 'RSI', params: `period ${c.period}` }];
    case 'MACD':
      return [{ name: 'MACD', params: `fast EMA ${c.fast} · slow EMA ${c.slow} · signal ${c.signal}` }];
    case 'PRICE_MA':
      return [{ name: c.maType, params: `period ${c.period}` }];
    case 'MA_CROSS':
      return [
        { name: c.fastType, params: `period ${c.fastPeriod}` },
        { name: c.slowType, params: `period ${c.slowPeriod}` },
      ];
    case 'BB':
      return [{ name: 'Bollinger Bands', params: `period ${c.period} · ${mult(c.mult)} standard deviations` }];
    case 'ATR':
      return [{ name: 'ATR', params: `period ${c.period}` }];
    case 'STOCH':
      return [
        {
          name: 'Stochastic Oscillator',
          params: `%K period ${c.kPeriod} · %K smoothing ${c.smooth}${c.variant === 'kd' ? ` · %D period ${c.dPeriod}` : ''}`,
        },
      ];
    case 'ADX':
      return [{ name: c.variant === 'strength' ? 'ADX' : 'DMI (+DI / −DI)', params: `period ${c.period}` }];
    case 'VOLUME':
      return [{ name: 'Volume SMA & Spike Ratio', params: `Volume ÷ SMA(Volume, ${c.period})` }];
    case 'SUPERTREND':
      return [{ name: 'Supertrend', params: `ATR period ${c.period} · multiplier ${mult(c.mult)}` }];
    case 'CCI':
      return [{ name: 'CCI', params: `period ${c.period}` }];
    case 'DONCHIAN':
      return [{ name: 'Donchian channel', params: `previous ${c.period} candles` }];
    case 'MA_SLOPE':
      return [{ name: c.maType, params: `period ${c.period} · slope over ${SLOPE_LAG} candles` }];
  }
}

/** Every indicator instance with its exact parameters and where it is used. */
export function listIndicators(genome: StrategyGenome): IndicatorUsage[] {
  const byKey = new Map<string, IndicatorUsage>();
  const add = (name: string, params: string, role: string): void => {
    const key = `${name}|${params}`;
    const existing = byKey.get(key);
    if (existing) {
      if (!existing.roles.includes(role)) existing.roles.push(role);
    } else {
      byKey.set(key, { name, params, roles: [role] });
    }
  };
  for (const { group, role } of allGroups(genome)) {
    for (const c of group.conditions) for (const usage of usageFor(c)) add(usage.name, usage.params, role);
  }
  if (genome.exit.mode === 'dynamic') {
    const { exit } = genome;
    const stop = `${exit.trailing ? 'trailing stop' : 'stop'} ×${mult(exit.stopAtrMult)}`;
    const target = exit.takeProfitAtrMult !== null ? ` · target ×${mult(exit.takeProfitAtrMult)}` : '';
    add('ATR', `period ${exit.atrPeriod} · ${stop}${target}`, 'Stop / target');
  }
  return [...byKey.values()];
}
