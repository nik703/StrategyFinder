/**
 * Overfitting risk score (0 = low, 100 = high) of a strategy, from evidence the search already
 * produced. Each component is a 0–1 risk, weighted to a 100-point total:
 *
 *   30  selection bias        1 − deflated Sharpe ratio (given every trial on this data window)
 *   15  multiple testing      share of the validation Sharpe removed by the haircut
 *   20  parameter spike       how far the strategy's own fitness sits above its one-step neighbours
 *   15  fold inconsistency    share of the five search-period folds that lost money
 *   10  regime collapse       share of the judged market regimes the strategy collapses in
 *   10  thin sample           trades below twice the minimum (100 → full risk, 200+ → none)
 *
 * Level: low under 35, moderate under 65, high from 65 – and always high when the deflated Sharpe is
 * under 50% (the same "most likely luck" line at which a locked strategy is rated Overfit).
 *
 * It summarises – it does not replace – the evidence: the drawer shows every component.
 */

import { collapsedRegimes, MIN_TRADES, testableRegimes } from '../engine/backtester';
import type { StrategyResult } from '../types/trading';

export type RiskLevel = 'low' | 'moderate' | 'high';

export interface RiskComponent {
  key: 'selection' | 'haircut' | 'spike' | 'folds' | 'regimes' | 'sample';
  label: string;
  weight: number;
  /** 0–1. */
  risk: number;
  detail: string;
}

export interface OverfittingRisk {
  /** 0–100. */
  score: number;
  level: RiskLevel;
  components: RiskComponent[];
}

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);
const pct = (x: number): string => `${Math.round(x * 100)}%`;

export function overfittingRisk(s: StrategyResult): OverfittingRisk {
  const components: RiskComponent[] = [];

  components.push({
    key: 'selection',
    label: 'Selection bias',
    weight: 30,
    risk: clamp01(1 - s.dsr),
    detail: `Deflated Sharpe ${pct(s.dsr)}`,
  });

  const haircut = s.haircut ?? 1;
  components.push({
    key: 'haircut',
    label: 'Multiple testing',
    weight: 15,
    risk: clamp01(haircut),
    detail: s.haircutSharpe === undefined ? 'Haircut not computed' : `Haircut removes ${pct(haircut)} of the validation Sharpe`,
  });

  const point = s.pointFitness;
  const neighbours = s.neighbourMedian;
  const spike =
    point === undefined || neighbours === undefined || neighbours === null ? 0.5 : clamp01((point - neighbours) / Math.max(0.5, Math.abs(point)));
  components.push({
    key: 'spike',
    label: 'Parameter spike',
    weight: 20,
    risk: spike,
    detail:
      point === undefined || neighbours === undefined || neighbours === null
        ? 'No neighbourhood data'
        : `Own score ${point.toFixed(2)} vs neighbours ${neighbours.toFixed(2)}`,
  });

  const folds = s.folds ?? [];
  const losing = folds.filter((fold) => fold.roi <= 0).length;
  components.push({
    key: 'folds',
    label: 'Fold inconsistency',
    weight: 15,
    risk: folds.length > 0 ? losing / folds.length : 0.5,
    detail: folds.length > 0 ? `${losing} of ${folds.length} folds lost money` : 'No fold data',
  });

  const testable = testableRegimes(s.regimes ?? null);
  const collapsed = collapsedRegimes(s.regimes ?? null);
  components.push({
    key: 'regimes',
    label: 'Regime collapse',
    weight: 10,
    risk: s.regimes ? (testable.length > 0 ? collapsed.length / testable.length : 0.5) : 0.5,
    detail: testable.length > 0 ? `Collapses in ${collapsed.length} of ${testable.length} judged regimes` : 'Too few trades per regime to judge',
  });

  const trades = s.search.trades;
  components.push({
    key: 'sample',
    label: 'Thin sample',
    weight: 10,
    risk: clamp01((2 * MIN_TRADES - trades) / MIN_TRADES),
    detail: `${trades} trades over train + validation`,
  });

  const score = Math.round(components.reduce((sum, c) => sum + c.weight * c.risk, 0));
  // A deflated Sharpe under 50% says "most likely luck" – the verdict would be Overfit – so the level
  // is high whatever the other components add up to.
  const level: RiskLevel = s.dsr < 0.5 || score >= 65 ? 'high' : score < 35 ? 'low' : 'moderate';
  return { score, level, components };
}
