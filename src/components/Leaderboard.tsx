/**
 * Live, sortable Top-25 leaderboard. Clicking a row previews the strategy on
 * the equity chart and opens the Strategy Breakdown drawer.
 *
 * Risk-adjusted, selection-bias-aware figures lead: the default order is the
 * deflated Sharpe ratio, and every return figure is labelled in-sample. There is
 * deliberately no holdout column – the holdout is revealed for one locked
 * strategy only, so it can never be sorted on.
 */

import { ChevronDown, ChevronsUpDown, ChevronUp, Lock, Trophy } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { exitModeLabel, exitTags, groupFormula, indicatorBadges } from '../engine/describe';
import { fmtInt, fmtNum, fmtPct, fmtUsd } from '../lib/format';
import { overfittingRisk } from '../lib/risk';
import { LEADERBOARD_SIZE, type EngineStatus, type StrategyResult, type TradeGates, type Verdict } from '../types/trading';
import { cx, Panel, Pill } from './ui';

interface LeaderboardProps {
  strategies: StrategyResult[];
  selectedId: string | null;
  /** The strategy whose holdout was revealed, if any. */
  lockedId: string | null;
  onSelect: (strategy: StrategyResult) => void;
  status: EngineStatus;
  /** The run's minimum-trade gates (null before the data has loaded). */
  gates: TradeGates | null;
  /** True when the gates apply to trades pooled over every asset. */
  pooled: boolean;
}

type SortKey =
  | 'rank'
  | 'params'
  | 'dsr'
  | 'haircut'
  | 'risk'
  | 'valSharpe'
  | 'sortino'
  | 'verdict'
  | 'netPnl'
  | 'roi'
  | 'alpha'
  | 'maxDrawdown'
  | 'winRate'
  | 'profitFactor'
  | 'sharpe'
  | 'trades'
  | 'mode'
  | 'fitness';

interface SortState {
  key: SortKey;
  dir: 'asc' | 'desc';
}

interface Row {
  rank: number;
  strategy: StrategyResult;
  locked: boolean;
}

/** Default order: highest deflated Sharpe first. */
const DEFAULT_SORT: SortState = { key: 'dsr', dir: 'desc' };

interface Column {
  key: SortKey | 'indicators';
  label: string;
  title?: string;
  align: 'left' | 'right';
  /** Value used for sorting; omitted for non-sortable columns. */
  value?: (row: Row) => number | string;
  /** Direction applied on the first click. */
  firstDir?: 'asc' | 'desc';
  render: (row: Row) => ReactNode;
}

export function VerdictPill({ verdict }: { verdict: Verdict | null }): JSX.Element {
  if (verdict === null) return <span className="text-[11px] text-ink-3">pending</span>;
  return <Pill tone={verdict === 'Robust' ? 'up' : verdict === 'Fragile' ? 'warn' : 'down'}>{verdict}</Pill>;
}

const COLUMNS: Column[] = [
  {
    key: 'rank',
    label: 'Rank',
    title: 'Rank by composite fitness score',
    align: 'left',
    value: (row) => row.rank,
    firstDir: 'asc',
    render: (row) => (
      <div className="flex items-center gap-1.5">
        <span className={cx('num font-semibold', row.rank === 1 ? 'text-accent-strong' : 'text-ink-2')}>#{row.rank}</span>
        {row.locked && (
          <span title="Locked: this strategy's holdout has been revealed" className="text-accent-strong">
            <Lock size={12} aria-label="Locked" />
          </span>
        )}
      </div>
    ),
  },
  {
    key: 'indicators',
    label: 'Indicators Used',
    align: 'left',
    render: ({ strategy }) => (
      <div className="min-w-[200px] max-w-[340px]">
        <div className="flex flex-wrap gap-1">
          {indicatorBadges(strategy.genome).map((badge) => (
            <Pill key={badge}>{badge}</Pill>
          ))}
          <Pill tone="bench">{strategy.genome.archetype === 'TREND' ? 'Trend' : 'Mean rev.'}</Pill>
          {strategy.genome.shortEntry !== null && <Pill tone="accent">L/S</Pill>}
        </div>
        <div className="num mt-1 truncate text-[11px] text-ink-3" title={groupFormula(strategy.genome.longEntry)}>
          {groupFormula(strategy.genome.longEntry)}
        </div>
      </div>
    ),
  },
  {
    key: 'dsr',
    label: 'Deflated Sharpe',
    title: 'Deflated Sharpe Ratio of the validation segment: probability the Sharpe is real given every strategy tried so far. 95% or more is needed for a Robust verdict.',
    align: 'right',
    value: ({ strategy }) => strategy.dsr,
    render: ({ strategy }) => (
      <span className={cx('font-semibold', strategy.dsr >= 0.95 ? 'text-up' : strategy.dsr >= 0.5 ? 'text-warn' : 'text-ink-2')}>
        {fmtPct(strategy.dsr * 100, false, 0)}
      </span>
    ),
  },
  {
    key: 'haircut',
    label: 'Haircut Sharpe',
    title:
      'Validation Sharpe after a Bonferroni haircut for every strategy tried on this data window, in any run or session (Harvey & Liu). 0 = fully explained by multiple testing.',
    align: 'right',
    value: ({ strategy }) => strategy.haircutSharpe ?? -Infinity,
    render: ({ strategy }) => (
      <span className={strategy.haircutSharpe !== undefined && strategy.haircutSharpe > 0 ? 'text-ink' : 'text-ink-3'}>
        {strategy.haircutSharpe === undefined ? '–' : fmtNum(strategy.haircutSharpe)}
      </span>
    ),
  },
  {
    key: 'risk',
    label: 'Overfit risk',
    title: 'Overfitting risk score 0–100: selection bias, multiple testing, parameter spike, fold inconsistency, regime collapse, thin sample. Open a row for the breakdown.',
    align: 'right',
    value: ({ strategy }) => overfittingRisk(strategy).score,
    firstDir: 'asc',
    render: ({ strategy }) => {
      const risk = overfittingRisk(strategy);
      return (
        <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <span className="num font-semibold text-ink">{risk.score}</span>
          <Pill tone={risk.level === 'low' ? 'up' : risk.level === 'moderate' ? 'warn' : 'down'}>{risk.level}</Pill>
        </span>
      );
    },
  },
  {
    key: 'valSharpe',
    label: 'Validation Sharpe',
    title: 'Annualised Sharpe ratio on the validation segment (the input to the deflated Sharpe)',
    align: 'right',
    value: ({ strategy }) => strategy.validation.sharpe,
    render: ({ strategy }) => <span className="text-ink">{fmtNum(strategy.validation.sharpe)}</span>,
  },
  {
    key: 'sharpe',
    label: 'Sharpe (in-sample)',
    title: 'Annualised Sharpe ratio over train + validation (risk-free rate 0)',
    align: 'right',
    value: ({ strategy }) => strategy.search.sharpe,
    render: ({ strategy }) => <span className="text-ink-2">{fmtNum(strategy.search.sharpe)}</span>,
  },
  {
    key: 'sortino',
    label: 'Sortino (in-sample)',
    title: 'Annualised Sortino ratio over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.sortino,
    render: ({ strategy }) => <span className="text-ink-2">{fmtNum(strategy.search.sortino)}</span>,
  },
  {
    key: 'maxDrawdown',
    label: 'Max Drawdown (%)',
    title: 'Largest peak-to-trough equity decline, measured to intrabar lows',
    align: 'right',
    value: ({ strategy }) => strategy.search.maxDrawdown,
    firstDir: 'asc',
    render: ({ strategy }) => <span className="text-ink-2">{fmtPct(strategy.search.maxDrawdown)}</span>,
  },
  {
    key: 'trades',
    label: 'Trades',
    title: 'Trades over train + validation (validation trades in brackets)',
    align: 'right',
    value: ({ strategy }) => strategy.search.trades,
    render: ({ strategy }) => (
      <span className="text-ink-2">
        {fmtInt(strategy.search.trades)} <span className="text-ink-3">({fmtInt(strategy.validation.trades)})</span>
      </span>
    ),
  },
  {
    key: 'winRate',
    label: 'Win Rate (%)',
    align: 'right',
    value: ({ strategy }) => strategy.search.winRate,
    render: ({ strategy }) => <span className="text-ink-3">{fmtPct(strategy.search.winRate)}</span>,
  },
  {
    key: 'profitFactor',
    label: 'Profit Factor',
    title: 'Gross profit ÷ gross loss',
    align: 'right',
    value: ({ strategy }) => strategy.search.profitFactor,
    render: ({ strategy }) => <span className="text-ink-3">{fmtNum(strategy.search.profitFactor)}</span>,
  },
  {
    key: 'roi',
    label: 'ROI (in-sample, %)',
    title: 'Return on starting capital over train + validation – the data the search optimised on, so it is biased upwards',
    align: 'right',
    value: ({ strategy }) => strategy.search.roi,
    render: ({ strategy }) => <span className="text-ink-3">{fmtPct(strategy.search.roi, true)}</span>,
  },
  {
    key: 'netPnl',
    label: 'Net PnL (in-sample, $)',
    title: 'Net profit after fees, slippage and financing over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.netPnl,
    render: ({ strategy }) => <span className="text-ink-3">{fmtUsd(strategy.search.netPnl, true)}</span>,
  },
  {
    key: 'alpha',
    label: 'Excess ROI vs. B&H (in-sample)',
    title: 'Strategy ROI minus Buy & Hold ROI over train + validation, in percentage points. Not risk-adjusted, so it is not alpha.',
    align: 'right',
    value: ({ strategy }) => strategy.alpha,
    render: ({ strategy }) => <span className="text-ink-3">{fmtPct(strategy.alpha, true)}</span>,
  },
  {
    key: 'mode',
    label: 'SL/TP Mode',
    title: 'How the stop loss and take profit are defined',
    align: 'left',
    value: ({ strategy }) => `${exitModeLabel(strategy.genome)} ${exitTags(strategy.genome).join(' ')}`,
    firstDir: 'asc',
    render: ({ strategy }) => (
      <div className="flex items-center gap-1.5 whitespace-nowrap">
        <Pill tone={strategy.genome.exit.mode === 'static' ? 'neutral' : 'accent'}>{exitModeLabel(strategy.genome)}</Pill>
        {exitTags(strategy.genome).length > 0 && (
          <span className="text-[11px] text-ink-3">{exitTags(strategy.genome).join(' · ')}</span>
        )}
      </div>
    ),
  },
  {
    key: 'params',
    label: 'Params',
    title: 'Number of tunable parameters (hard cap 8, fitness penalty above 4)',
    align: 'right',
    value: ({ strategy }) => strategy.params,
    firstDir: 'asc',
    render: ({ strategy }) => <span className={strategy.params > 4 ? 'text-warn' : 'text-ink-2'}>{strategy.params}</span>,
  },
  {
    key: 'verdict',
    label: 'Verdict',
    title: 'Robust / Fragile / Overfit – only the locked strategy gets one, once its holdout is revealed',
    align: 'left',
    value: ({ strategy }) => (strategy.verdict === 'Robust' ? 0 : strategy.verdict === 'Fragile' ? 1 : strategy.verdict === 'Overfit' ? 2 : 3),
    firstDir: 'asc',
    render: ({ strategy }) => <VerdictPill verdict={strategy.verdict} />,
  },
  {
    key: 'fitness',
    label: 'Score',
    title: 'Composite fitness: deflated validation Sharpe and the 25th-percentile fold score, minus drawdown, regime-collapse and complexity penalties, capped at the median of the one-step parameter neighbours',
    align: 'right',
    value: ({ strategy }) => strategy.fitness,
    render: ({ strategy }) => <span className="text-ink-3">{fmtNum(strategy.fitness)}</span>,
  },
];

export default function Leaderboard({ strategies, selectedId, lockedId, onSelect, status, gates, pooled }: LeaderboardProps): JSX.Element {
  const gateText =
    gates === null
      ? null
      : `at least ${fmtInt(gates.search)} trades (${fmtInt(gates.validation)} in validation) and ${fmtNum(gates.perYear, 0)} trades a year${pooled ? ', pooled over the assets' : ' on every asset'}`;
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);

  const rows = useMemo<Row[]>(() => {
    // `strategies` arrives ordered by fitness, so the index is the rank.
    const ranked = strategies.map((strategy, index) => ({ rank: index + 1, strategy, locked: strategy.id === lockedId }));
    const column = COLUMNS.find((c) => c.key === sort.key);
    if (!column?.value || sort.key === 'rank') return sort.dir === 'asc' ? ranked : ranked.reverse();
    const value = column.value;
    const direction = sort.dir === 'asc' ? 1 : -1;
    return ranked.sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      if (va === vb) return a.rank - b.rank;
      if (typeof va === 'string' || typeof vb === 'string') return String(va).localeCompare(String(vb)) * direction;
      if (!Number.isFinite(va as number) && !Number.isFinite(vb as number)) return a.rank - b.rank;
      return (va < vb ? -1 : 1) * direction;
    });
  }, [strategies, sort, lockedId]);

  const toggleSort = (column: Column): void => {
    if (!column.value) return;
    const key = column.key as SortKey;
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: column.firstDir ?? 'desc' },
    );
  };

  return (
    <Panel className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <Trophy size={16} className="text-ink-3" />
          <h2 className="text-sm font-semibold text-ink">Top Strategies Leaderboard</h2>
          <span className="num rounded-md border border-line bg-raised px-1.5 py-0.5 text-[11px] text-ink-2">
            {strategies.length} / {LEADERBOARD_SIZE}
          </span>
        </div>
        <p className="text-xs text-ink-3">
          Sorted by deflated Sharpe. Return figures are in-sample (train + validation). Click a row for the breakdown and to lock one
          strategy for the holdout.
        </p>
      </div>

      {strategies.length === 0 ? (
        <div className="px-4 py-14 text-center">
          <p className="text-sm text-ink-2">
            {status === 'optimizing' || status === 'loading'
              ? 'Searching for strategies that clear the minimum-trade gates…'
              : status === 'converged' || status === 'stopped'
                ? 'No strategy cleared the minimum-trade gates.'
                : 'No strategies yet.'}
          </p>
          <p className="mt-1 text-xs text-ink-3">
            {status === 'optimizing' || status === 'loading'
              ? gateText
                ? `Only strategies with ${gateText} can be listed.`
                : 'Loading market data…'
              : status === 'converged' || status === 'stopped'
                ? `No candidate had ${gateText ?? 'enough trades'}. Try a longer history, shorter candles, or pool several assets.`
                : 'Configure an asset above and press Start Optimization.'}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="num w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-line bg-raised/40">
                {COLUMNS.map((column) => {
                  const sortable = column.value !== undefined;
                  const active = sortable && sort.key === column.key;
                  return (
                    <th
                      key={column.key}
                      scope="col"
                      title={column.title}
                      aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                      className={cx(
                        'px-2.5 py-2 align-bottom text-[11px] font-medium uppercase leading-tight tracking-wide first:pl-4 last:pr-4',
                        column.align === 'right' ? 'max-w-[104px] text-right' : 'whitespace-nowrap text-left',
                        active ? 'text-ink' : 'text-ink-3',
                      )}
                    >
                      {sortable ? (
                        <button
                          type="button"
                          onClick={() => toggleSort(column)}
                          className={cx(
                            'inline-flex items-end gap-1 uppercase tracking-wide transition-colors hover:text-ink',
                            column.align === 'right' && 'flex-row-reverse text-right',
                          )}
                        >
                          {column.label}
                          {active ? (
                            sort.dir === 'asc' ? (
                              <ChevronUp size={12} className="shrink-0" />
                            ) : (
                              <ChevronDown size={12} className="shrink-0" />
                            )
                          ) : (
                            <ChevronsUpDown size={12} className="shrink-0 opacity-40" />
                          )}
                        </button>
                      ) : (
                        column.label
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const selected = row.strategy.id === selectedId;
                return (
                  <tr
                    key={row.strategy.id}
                    tabIndex={0}
                    aria-selected={selected}
                    aria-label={`Rank ${row.rank}, strategy ${row.strategy.id}. Open breakdown`}
                    onClick={() => onSelect(row.strategy)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelect(row.strategy);
                      }
                    }}
                    className={cx(
                      'cursor-pointer border-b border-line/70 transition-colors last:border-b-0 focus-visible:outline-none',
                      selected
                        ? 'bg-accent/10 shadow-[inset_3px_0_0_0_var(--color-accent)]'
                        : 'hover:bg-hover/70 focus-visible:bg-hover/70',
                    )}
                  >
                    {COLUMNS.map((column) => (
                      <td
                        key={column.key}
                        className={cx(
                          'px-2.5 py-2 align-middle first:pl-4 last:pr-4',
                          column.align === 'right' ? 'whitespace-nowrap text-right' : 'text-left',
                        )}
                      >
                        {column.render(row)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
