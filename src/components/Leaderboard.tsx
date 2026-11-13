/**
 * Live, sortable Top-25 leaderboard. Clicking a row previews the strategy on
 * the equity chart and opens the Strategy Breakdown drawer.
 */

import { ChevronDown, ChevronsUpDown, ChevronUp, Trophy } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { exitModeLabel, exitTags, groupFormula, indicatorBadges } from '../engine/describe';
import { fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from '../lib/format';
import { LEADERBOARD_SIZE, type EngineStatus, type StrategyResult, type Verdict } from '../types/trading';
import { cx, Panel, Pill } from './ui';

interface LeaderboardProps {
  strategies: StrategyResult[];
  selectedId: string | null;
  onSelect: (strategy: StrategyResult) => void;
  status: EngineStatus;
}

type SortKey =
  | 'rank'
  | 'params'
  | 'dsr'
  | 'holdout'
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
}

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
      <span className={cx('num font-semibold', row.rank === 1 ? 'text-accent-strong' : 'text-ink-2')}>#{row.rank}</span>
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
    key: 'netPnl',
    label: 'Net PnL ($)',
    title: 'Net profit after fees and slippage over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.netPnl,
    render: ({ strategy }) => (
      <span className={cx('font-medium', toneClass(strategy.search.netPnl))}>{fmtUsd(strategy.search.netPnl, true)}</span>
    ),
  },
  {
    key: 'roi',
    label: 'ROI (%)',
    title: 'Return on starting capital over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.roi,
    render: ({ strategy }) => (
      <span className={cx('font-medium', toneClass(strategy.search.roi))}>{fmtPct(strategy.search.roi, true)}</span>
    ),
  },
  {
    key: 'alpha',
    label: 'Alpha vs. B&H (%)',
    title: 'Strategy ROI minus Buy & Hold ROI, in percentage points',
    align: 'right',
    value: ({ strategy }) => strategy.alpha,
    render: ({ strategy }) => <span className={toneClass(strategy.alpha)}>{fmtPct(strategy.alpha, true)}</span>,
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
    key: 'winRate',
    label: 'Win Rate (%)',
    align: 'right',
    value: ({ strategy }) => strategy.search.winRate,
    render: ({ strategy }) => <span className="text-ink-2">{fmtPct(strategy.search.winRate)}</span>,
  },
  {
    key: 'profitFactor',
    label: 'Profit Factor',
    title: 'Gross profit ÷ gross loss',
    align: 'right',
    value: ({ strategy }) => strategy.search.profitFactor,
    render: ({ strategy }) => <span className="text-ink-2">{fmtNum(strategy.search.profitFactor)}</span>,
  },
  {
    key: 'sharpe',
    label: 'Sharpe Ratio',
    title: 'Annualised Sharpe ratio (risk-free rate 0)',
    align: 'right',
    value: ({ strategy }) => strategy.search.sharpe,
    render: ({ strategy }) => <span className="text-ink">{fmtNum(strategy.search.sharpe)}</span>,
  },
  {
    key: 'trades',
    label: 'Trades',
    align: 'right',
    value: ({ strategy }) => strategy.search.trades,
    render: ({ strategy }) => <span className="text-ink-2">{fmtInt(strategy.search.trades)}</span>,
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
    key: 'dsr',
    label: 'DSR',
    title: 'Deflated Sharpe Ratio of the validation segment: probability the Sharpe is real given every strategy tried so far',
    align: 'right',
    value: ({ strategy }) => strategy.dsr,
    render: ({ strategy }) => (
      <span className={strategy.dsr >= 0.95 ? 'text-up' : strategy.dsr >= 0.5 ? 'text-warn' : 'text-ink-2'}>
        {fmtPct(strategy.dsr * 100, false, 0)}
      </span>
    ),
  },
  {
    key: 'holdout',
    label: 'Holdout ROI',
    title: 'Return on the 20% holdout – hidden until the run ends, never used for ranking',
    align: 'right',
    value: ({ strategy }) => strategy.holdout?.roi ?? -Infinity,
    render: ({ strategy }) =>
      strategy.holdout ? (
        <span className={toneClass(strategy.holdout.roi)}>{fmtPct(strategy.holdout.roi, true)}</span>
      ) : (
        <span className="text-ink-3" title="Sealed until the run ends">
          sealed
        </span>
      ),
  },
  {
    key: 'verdict',
    label: 'Verdict',
    title: 'Robust / Fragile / Overfit – combines the holdout, the deflated Sharpe and the robustness tests',
    align: 'left',
    value: ({ strategy }) => (strategy.verdict === 'Robust' ? 0 : strategy.verdict === 'Fragile' ? 1 : strategy.verdict === 'Overfit' ? 2 : 3),
    firstDir: 'asc',
    render: ({ strategy }) => <VerdictPill verdict={strategy.verdict} />,
  },
  {
    key: 'fitness',
    label: 'Score',
    title: 'Composite fitness: deflated validation Sharpe and excess return, minus penalties for few trades, complexity and drawdown',
    align: 'right',
    value: ({ strategy }) => strategy.fitness,
    render: ({ strategy }) => <span className="text-ink-3">{fmtNum(strategy.fitness)}</span>,
  },
];

export default function Leaderboard({ strategies, selectedId, onSelect, status }: LeaderboardProps): JSX.Element {
  const [sort, setSort] = useState<SortState>({ key: 'rank', dir: 'asc' });

  const rows = useMemo<Row[]>(() => {
    // `strategies` arrives ordered by fitness, so the index is the rank.
    const ranked = strategies.map((strategy, index) => ({ rank: index + 1, strategy }));
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
  }, [strategies, sort]);

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
          Click a row to preview its equity curve and open the full strategy breakdown. Click a column header to sort.
        </p>
      </div>

      {strategies.length === 0 ? (
        <div className="px-4 py-14 text-center">
          <p className="text-sm text-ink-2">
            {status === 'optimizing' || status === 'loading'
              ? 'Searching for the first profitable strategies…'
              : 'No strategies yet.'}
          </p>
          <p className="mt-1 text-xs text-ink-3">
            {status === 'optimizing' || status === 'loading'
              ? 'The leaderboard fills in live as the optimiser discovers them.'
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
