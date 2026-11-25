/**
 * Plain-English diagnostics: every statistical check of a strategy as a status badge with a short
 * explanation, and a "Strategy verdict" that says in human terms why it looks strong or what to
 * watch out for. Pure functions of the strategy and its reports – no UI.
 */

import { judgedFolds, foldsNeeded, passesFoldGate } from '../engine/backtester';
import type { CrossMarketReport, MonteCarloSummary, RobustnessReport, StrategyResult } from '../types/trading';

export type BadgeStatus = 'pass' | 'warn' | 'fail' | 'pending';

export type DiagnosticKey = 'dsr' | 'folds' | 'monteCarlo' | 'neighbours' | 'crossMarket' | 'holdout';

export interface DiagnosticBadge {
  key: DiagnosticKey;
  /** Plain-English name, e.g. "Beats luck". */
  label: string;
  /** The jargon it stands for, e.g. "Deflated Sharpe". */
  metric: string;
  /** Short value, e.g. "97%" or "4 of 5 periods". */
  value: string;
  status: BadgeStatus;
  /** One or two sentences: what the check does and what this result means. */
  tooltip: string;
}

export type VerdictTone = 'strong' | 'promising' | 'mixed' | 'weak';

export interface StrategyVerdict {
  tone: VerdictTone;
  headline: string;
  summary: string;
  strengths: string[];
  risks: string[];
}

export interface DiagnosticsInput {
  strategy: StrategyResult;
  /** On-demand robustness report (null until it has arrived). */
  report: RobustnessReport | null;
  /** Cross-market generalization check, when the user has run one. */
  crossMarket: CrossMarketReport | null;
  /** Strategies tried on this data window (all runs). */
  trials: number;
  /** True when the run used simulated fallback candles. */
  simulated?: boolean;
}

const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

function dsrBadge(s: StrategyResult, trials: number): DiagnosticBadge {
  const status: BadgeStatus = s.dsr >= 0.95 ? 'pass' : s.dsr >= 0.5 ? 'warn' : 'fail';
  return {
    key: 'dsr',
    label: 'Beats luck',
    metric: 'Deflated Sharpe',
    value: pct(s.dsr),
    status,
    tooltip:
      `The chance that the validation result is a real edge rather than the luckiest of the ${trials.toLocaleString('en-US')} strategies tried on this data. ` +
      (status === 'pass'
        ? 'Above 95%: convincing.'
        : status === 'warn'
          ? 'Between 50% and 95%: better than a typical lucky pick, but not proof.'
          : 'Under 50%: most likely luck.'),
  };
}

function foldBadge(s: StrategyResult): DiagnosticBadge {
  const folds = s.folds ?? [];
  if (folds.length === 0) {
    return { key: 'folds', label: 'Consistent over time', metric: 'Fold consistency', value: '–', status: 'pending', tooltip: 'No fold breakdown is available.' };
  }
  const judged = judgedFolds(folds);
  const profitable = judged.filter((k) => folds[k].roi > 0).length;
  const needed = foldsNeeded(judged.length);
  const flat = folds.length - judged.length;
  // The run's own fold gate when it was checked (it also caps each slice's drawdown), else the count.
  const passes = passesFoldGate(folds);
  const status: BadgeStatus = passes ? 'pass' : profitable >= needed - 1 ? 'warn' : 'fail';
  return {
    key: 'folds',
    label: 'Consistent over time',
    metric: 'Fold consistency',
    value: `${profitable} of ${judged.length} periods`,
    status,
    tooltip:
      `The search period is cut into ${folds.length} equal time slices. It made money in ${profitable} of the ${judged.length} it traded` +
      (flat > 0 ? ` (${flat} flat ${flat === 1 ? 'slice' : 'slices'} with no position are not held against it)` : '') +
      `. The median slice must earn a positive risk-adjusted return (½ Sortino + ½ Calmar) – about ${needed} winning slices – with no slice losing more than 35% at its worst point.`,
  };
}

function monteCarloBadge(mc: MonteCarloSummary | null | undefined): DiagnosticBadge {
  if (!mc) {
    return {
      key: 'monteCarlo',
      label: 'Survives stress',
      metric: 'Monte Carlo',
      value: '–',
      status: 'pending',
      tooltip: 'Replays the strategy’s trades hundreds of times with random skips, shuffled order and slippage shocks.',
    };
  }
  const status: BadgeStatus = mc.passed ? 'pass' : mc.probProfit >= 0.6 && mc.drawdownP95 < 50 ? 'warn' : 'fail';
  return {
    key: 'monteCarlo',
    label: 'Survives stress',
    metric: 'Monte Carlo',
    value: `${pct(mc.probProfit)} profit · DD ${mc.drawdownP95.toFixed(0)}%`,
    status,
    tooltip:
      `${mc.simulations} replays of its ${mc.trades} trades with 10% skipped at random, the order shuffled and random slippage shocks that double the cost of the average fill: ` +
      `${pct(mc.probProfit)} still end in profit (80% needed), and 95% of them keep the worst drawdown under ${mc.drawdownP95.toFixed(1)}% ` +
      `(the backtest's own: ${mc.baseDrawdown.toFixed(1)}%).`,
  };
}

function neighbourBadge(s: StrategyResult): DiagnosticBadge {
  const checked = (s.neighbourCount ?? 0) > 0;
  if (!checked) {
    return {
      key: 'neighbours',
      label: 'Stable parameters',
      metric: 'Neighbourhood stability',
      value: '–',
      status: 'pending',
      tooltip: 'Every parameter nudged one step up and down: a real edge sits on a plateau where nearby settings also work.',
    };
  }
  const holds = (s.neighbourExpectancy ?? 0) > 0;
  const point = s.pointFitness ?? s.fitness;
  const median = s.neighbourMedian ?? null;
  const spike = median !== null && point - median > 1.5;
  const status: BadgeStatus = holds && !spike ? 'pass' : holds ? 'warn' : 'fail';
  return {
    key: 'neighbours',
    label: 'Stable parameters',
    metric: 'Neighbourhood stability',
    value: holds ? (spike ? 'narrow peak' : 'plateau') : 'spike',
    status,
    tooltip:
      `${s.neighbourCount} neighbouring settings (each parameter one step up and down; ×√2 for lengths) were backtested. ` +
      (holds
        ? spike
          ? 'They still make money on average, but score well below this exact setting: the optimum is narrow.'
          : 'They make money too: the result does not hinge on these exact numbers.'
        : 'They lose money on average: the result hinges on these exact numbers – a classic sign of curve-fitting.'),
  };
}

function crossMarketBadge(cross: CrossMarketReport | null): DiagnosticBadge {
  const base = { key: 'crossMarket' as const, label: 'Works on other markets', metric: 'Cross-market check' };
  if (cross && cross.tested > 0) {
    const profitable = Math.round(cross.share * cross.tested);
    return {
      ...base,
      value: `${profitable} of ${cross.tested} unseen`,
      status: cross.passed ? 'pass' : cross.share >= 0.5 ? 'warn' : 'fail',
      tooltip:
        `The exact rules, untouched, on ${cross.tested} related market${cross.tested === 1 ? '' : 's'} the search never saw (dates before the sealed holdout): ` +
        `profitable on ${profitable}, median Sharpe ${cross.medianSharpe.toFixed(2)}. Two thirds profitable and a positive median are needed.`,
    };
  }
  return {
    ...base,
    value: 'not run',
    status: 'pending',
    tooltip: 'Test the exact rules on related markets the search never saw (for example SOL and BNB for a BTC strategy) to see whether the edge is structural.',
  };
}

function holdoutBadge(s: StrategyResult): DiagnosticBadge {
  const base = { key: 'holdout' as const, label: 'Holdout', metric: 'Locked holdout' };
  if (!s.holdout) {
    return {
      ...base,
      value: 'sealed',
      status: 'pending',
      tooltip: 'The last 20% of history is sealed. Lock ONE strategy when the run ends to reveal it – the only test the search could not have fitted.',
    };
  }
  const verdict = s.verdict;
  const luckOnly = verdict === 'Overfit' && heldUp(s);
  return {
    ...base,
    value: `${verdict ?? 'revealed'} · Sharpe ${s.holdout.sharpe.toFixed(2)}`,
    status: verdict === 'Robust' ? 'pass' : verdict === 'Fragile' || luckOnly ? 'warn' : 'fail',
    tooltip:
      `On the untouched holdout: ROI ${s.holdout.roi.toFixed(1)}%, Sharpe ${s.holdout.sharpe.toFixed(2)}, ${s.holdout.trades} trades. ` +
      (verdict === 'Robust'
        ? 'Every Robust condition held.'
        : verdict === 'Fragile'
          ? 'It survived, but missed some Robust conditions (listed in the workbench).'
          : luckOnly
            ? 'It made money on the holdout, but its deflated Sharpe is under 50%: the search-period result is not distinguishable from luck, so one good holdout is not proof.'
            : 'It did not hold up out of sample.'),
  };
}

/** True when the revealed holdout made money with a positive Sharpe. */
function heldUp(s: StrategyResult): boolean {
  return s.holdout !== null && s.holdout.roi > 0 && s.holdout.sharpe > 0;
}

/** Every check as a badge, in the order the workbench shows them. */
export function diagnosticBadges(input: DiagnosticsInput): DiagnosticBadge[] {
  const { strategy: s, report, crossMarket, trials } = input;
  return [
    dsrBadge(s, trials),
    foldBadge(s),
    monteCarloBadge(report?.monteCarlo ?? s.monteCarlo),
    neighbourBadge(s),
    crossMarketBadge(crossMarket),
    holdoutBadge(s),
  ];
}

/** The plain-English verdict box: why the strategy looks strong, or what to watch out for. */
export function strategyVerdict(input: DiagnosticsInput): StrategyVerdict {
  const { strategy: s, report } = input;
  const badges = diagnosticBadges(input);
  const byKey = Object.fromEntries(badges.map((b) => [b.key, b])) as Record<DiagnosticKey, DiagnosticBadge>;
  const checks = badges.filter((b) => b.key !== 'holdout' && b.status !== 'pending');
  const passed = checks.filter((b) => b.status === 'pass').length;
  const mc = report?.monteCarlo ?? s.monteCarlo ?? null;

  const strengths: string[] = [];
  const risks: string[] = [];
  if (byKey.dsr.status === 'pass') strengths.push(`Clears the luck hurdle: a ${pct(s.dsr)} chance the edge is real after every strategy tried.`);
  else if (byKey.dsr.status === 'warn') risks.push(`Only a ${pct(s.dsr)} chance the validation result is more than luck – promising, not proven.`);
  else risks.push(`A ${pct(s.dsr)} chance the result is more than luck: most likely the luckiest of many tries.`);

  if (byKey.folds.status === 'pass') strengths.push(`Made money in ${byKey.folds.value} of the search history, not just one lucky stretch.`);
  else risks.push(`Uneven over time: profitable in ${byKey.folds.value} – the edge may belong to one market phase.`);

  if (mc) {
    if (mc.passed) strengths.push(`${pct(mc.probProfit)} of ${mc.simulations} stressed replays (skipped trades, shuffled order, slippage shocks) still end in profit.`);
    else if (mc.probProfit < 0.8) risks.push(`Only ${pct(mc.probProfit)} of stressed replays end in profit: small execution differences could erase the edge.`);
    if (mc.drawdownP95 >= 25) risks.push(`Plan for a ${mc.drawdownP95.toFixed(0)}% drawdown: 1 in 20 stressed replays is worse than that. Size positions so you can sit through it.`);
  }

  if (byKey.neighbours.status === 'pass') strengths.push('Nearby parameter settings work too – a plateau, not a fragile spike.');
  else if (byKey.neighbours.status === 'warn') risks.push('Nearby settings still make money, but clearly less: the optimum is narrow.');
  else if (byKey.neighbours.status === 'fail') risks.push('Nudging any parameter one step loses money: the exact numbers are probably fitted to noise.');

  if (byKey.crossMarket.status === 'pass') strengths.push(`The frozen rules also work on unseen markets (${byKey.crossMarket.value}): the edge looks structural.`);
  else if (byKey.crossMarket.status === 'pending') risks.push('Not yet tested on other markets – run the cross-market check before trusting a single-chart result.');
  else risks.push(`Weak on other markets (${byKey.crossMarket.value}): the edge may be specific to this chart.`);

  if (report && !report.topTrades.profitable) risks.push(`Unprofitable without its ${report.topTrades.removed} best trades: a few outliers carry the result.`);
  // The sample the gates counted: the primary asset's, or the total over a pooled run's assets.
  const validationTrades = s.sample?.validationTrades ?? s.validation.trades;
  if (validationTrades < 25) {
    risks.push(`Only ${validationTrades} trades in the validation period${s.sample?.pooled ? ' (pooled over the assets)' : ''}: a small sample.`);
  }
  const ambiguous = s.search.trades > 0 ? s.search.ambiguousTrades / s.search.trades : 0;
  if (ambiguous > 0.1) risks.push(`${pct(ambiguous)} of trades hit the stop and the target in the same candle; the stop was assumed to fill first.`);
  if (input.simulated) risks.push('Simulated fallback data: re-run on live market data before drawing conclusions.');

  let tone: VerdictTone;
  let headline: string;
  if (s.holdout && s.verdict) {
    // An Overfit verdict can come from the holdout (it lost money) or from luck alone (a deflated
    // Sharpe under 50% with a profitable holdout): say which.
    const luckOnly = s.verdict === 'Overfit' && heldUp(s);
    tone = s.verdict === 'Robust' ? 'strong' : s.verdict === 'Fragile' || luckOnly ? 'mixed' : 'weak';
    headline =
      s.verdict === 'Robust'
        ? 'Robust: held up on the untouched holdout and passed every check'
        : s.verdict === 'Fragile'
          ? 'Fragile: survived the holdout, but not every check'
          : luckOnly
            ? 'Unproven: profitable on the holdout, but not distinguishable from luck'
            : 'Overfit: did not hold up on the untouched holdout';
    if (s.verdict === 'Robust' || luckOnly) {
      strengths.unshift(`Made money on the sealed holdout: Sharpe ${s.holdout.sharpe.toFixed(2)} over ${s.holdout.trades} trades.`);
    }
    if (s.verdict !== 'Robust') risks.unshift(...(s.verdictNotes ?? []).slice(0, 3));
  } else if (byKey.dsr.status === 'fail' || byKey.monteCarlo.status === 'fail') {
    tone = 'weak';
    headline = `Unproven: passes ${passed} of ${checks.length} checks`;
  } else if (passed === checks.length && checks.length >= 4) {
    tone = 'promising';
    headline = `Promising: passes all ${checks.length} checks run so far`;
  } else if (passed >= checks.length - 1 && checks.length >= 3) {
    tone = 'promising';
    headline = `Promising: passes ${passed} of ${checks.length} checks`;
  } else {
    tone = 'mixed';
    headline = `Mixed evidence: passes ${passed} of ${checks.length} checks`;
  }

  const summary = s.holdout
    ? 'The holdout has been revealed for this strategy: the verdict below is final for this run.'
    : tone === 'weak'
      ? 'Treat it as unproven: the evidence that this is more than a lucky fit is thin.'
      : tone === 'promising'
        ? 'The in-sample evidence holds up. The real test is the sealed holdout: lock this strategy when the run ends to reveal it.'
        : 'Some checks pass and some do not. Read the risks before trusting it with money.';

  return { tone, headline, summary, strengths, risks };
}
