/**
 * Step 3 – the live, sortable ranked leaderboard. Clicking a row opens the strategy in the
 * Strategy Workbench below; ticking up to three rows compares them side by side.
 *
 * The default order is the run's rank: strategies passing every robustness gate first, then the
 * rest by composite fitness. Risk-adjusted, selection-bias-aware figures lead (sparkline, deflated
 * Sharpe, stress test, Sortino, max drawdown, trades) and every return figure is labelled
 * in-sample. There is deliberately no holdout column – the holdout is revealed for one locked
 * strategy only, so it can never be sorted on.
 */

import { ChevronDown, ChevronsUpDown, ChevronUp, Columns3, Lock } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { exitModeLabel, exitTags, groupFormula, indicatorBadges } from '../engine/describe';
import { MAX_PARAMETERS } from '../engine/genome';
import { fmtInt, fmtNum, fmtPct, fmtUsd } from '../lib/format';
import { LEADERBOARD_SIZE, type EngineStatus, type StrategyResult, type TradeGates, type Verdict } from '../types/trading';
import { cx, Panel, Pill, Sparkline, StepHeader } from './ui';

/** Strategies that can be compared at once. */
export const MAX_COMPARE = 3;

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
  /** Strategies ticked for comparison (at most `MAX_COMPARE`). */
  compareIds: string[];
  onToggleCompare: (strategy: StrategyResult) => void;
}

type SortKey =
  | 'rank'
  | 'stress'
  | 'params'
  | 'gates'
  | 'dsr'
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

/** Default order: the run's rank (robust strategies first, then by fitness). */
const DEFAULT_SORT: SortState = { key: 'rank', dir: 'asc' };

interface Column {
  key: SortKey | 'indicators' | 'curve';
  /** Only shown with "All columns" switched on. */
  detail?: boolean;
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

/** Number of robustness gates a strategy passes (0–4). */
function gatesPassed(strategy: StrategyResult): number {
  const r = strategy.robust;
  return r ? [r.validation, r.folds, r.neighbours, r.significance].filter(Boolean).length : 0;
}

const COLUMNS: Column[] = [
  {
    key: 'rank',
    label: 'Rank',
    title: 'Strategies passing every robustness gate first, then by composite fitness',
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
    key: 'curve',
    label: 'Equity',
    title: 'Equity curve over train + validation (the holdout stays sealed)',
    align: 'left',
    render: ({ strategy }) => <Sparkline values={strategy.equity} />,
  },
  {
    key: 'gates',
    label: 'Checks',
    title: 'Robustness gates passed on the search period: validation, fold consistency, neighbour stability, significance (trades + DSR ≥ 50%)',
    align: 'left',
    value: ({ strategy }) => gatesPassed(strategy),
    render: ({ strategy }) => {
      const r = strategy.robust;
      if (!r) return <span className="text-ink-3">–</span>;
      const failed = [
        !r.validation && 'validation',
        !r.folds && 'folds',
        !r.neighbours && 'neighbours',
        !r.significance && 'significance',
      ].filter(Boolean);
      return (
        <span title={r.passed ? 'Passes every robustness gate on the search period' : `Fails: ${failed.join(', ')}`}>
          <Pill tone={r.passed ? 'up' : 'neutral'}>{r.passed ? 'All 4 ✓' : `${gatesPassed(strategy)}/4`}</Pill>
        </span>
      );
    },
  },
  {
    key: 'indicators',
    label: 'Strategy',
    align: 'left',
    render: ({ strategy }) => (
      <div className="min-w-[200px] max-w-[340px]">
        <div className="flex flex-wrap gap-1">
          {indicatorBadges(strategy.genome).map((badge) => (
            <Pill key={badge}>{badge}</Pill>
          ))}
          <Pill tone="bench">{strategy.genome.archetype === 'TREND' ? 'Trend' : 'Mean rev.'}</Pill>
          {strategy.genome.shortEntry !== null && <Pill tone="accent">{strategy.genome.shortRegime ? 'Asym. L/S' : 'L/S'}</Pill>}
        </div>
        <div className="num mt-1 truncate text-[11px] text-ink-3" title={groupFormula(strategy.genome.longEntry)}>
          {groupFormula(strategy.genome.longEntry)}
        </div>
      </div>
    ),
  },
  {
    key: 'dsr',
    label: 'Beats luck (DSR)',
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
    key: 'stress',
    label: 'Stress test',
    title:
      'Monte Carlo: 500 replays of the trades with 10% skipped, the order shuffled and random slippage shocks. Share of replays that end in profit, and the 95th-percentile worst drawdown.',
    align: 'right',
    value: ({ strategy }) => strategy.monteCarlo?.probProfit ?? -1,
    render: ({ strategy }) => {
      const mc = strategy.monteCarlo;
      if (!mc) return <span className="text-ink-3">–</span>;
      return (
        <span className="inline-flex flex-col items-end leading-tight">
          <span className={cx('font-semibold', mc.passed ? 'text-up' : mc.probProfit >= 0.6 ? 'text-warn' : 'text-down')}>
            {fmtPct(mc.probProfit * 100, false, 0)}
          </span>
          <span className="text-[10.5px] text-ink-3">DD95 {fmtPct(mc.drawdownP95, false, 0)}</span>
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
    detail: true,
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
    detail: true,
    label: 'Win Rate (%)',
    align: 'right',
    value: ({ strategy }) => strategy.search.winRate,
    render: ({ strategy }) => <span className="text-ink-3">{fmtPct(strategy.search.winRate)}</span>,
  },
  {
    key: 'profitFactor',
    detail: true,
    label: 'Profit Factor',
    title: 'Gross profit ÷ gross loss',
    align: 'right',
    value: ({ strategy }) => strategy.search.profitFactor,
    render: ({ strategy }) => <span className="text-ink-3">{fmtNum(strategy.search.profitFactor)}</span>,
  },
  {
    key: 'roi',
    detail: true,
    label: 'ROI (in-sample, %)',
    title: 'Return on starting capital over train + validation – the data the search optimised on, so it is biased upwards',
    align: 'right',
    value: ({ strategy }) => strategy.search.roi,
    render: ({ strategy }) => <span className="text-ink-3">{fmtPct(strategy.search.roi, true)}</span>,
  },
  {
    key: 'netPnl',
    detail: true,
    label: 'Net PnL (in-sample, $)',
    title: 'Net profit after fees, slippage and financing over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.netPnl,
    render: ({ strategy }) => <span className="text-ink-3">{fmtUsd(strategy.search.netPnl, true)}</span>,
  },
  {
    key: 'alpha',
    detail: true,
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
    detail: true,
    label: 'Params',
    title: `Number of tunable parameters (hard cap ${MAX_PARAMETERS}, +1 for an Asymmetric short regime filter)`,
    align: 'right',
    value: ({ strategy }) => strategy.params,
    firstDir: 'asc',
    render: ({ strategy }) => <span className="text-ink-2">{strategy.params}</span>,
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
    detail: true,
    label: 'Score',
    title:
      'Composite fitness: 1.5 × the median-fold risk-adjusted return (Sortino / Calmar) + expectancy × trade frequency − a drawdown guard beyond 25%, blended 30% towards the one-step parameter neighbours when they are worse. The deflated Sharpe is a gate, not part of the score.',
    align: 'right',
    value: ({ strategy }) => strategy.fitness,
    render: ({ strategy }) => <span className="text-ink-3">{fmtNum(strategy.fitness)}</span>,
  },
];

export default function Leaderboard({
  strategies,
  selectedId,
  lockedId,
  onSelect,
  status,
  gates,
  pooled,
  compareIds,
  onToggleCompare,
}: LeaderboardProps): JSX.Element {
  const [allColumns, setAllColumns] = useState(false);
  const columns = useMemo(() => COLUMNS.filter((column) => allColumns || !column.detail), [allColumns]);
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
      <div className="border-b border-line px-4 py-3">
        <StepHeader
          step={3}
          title={`Ranked leaderboard · ${strategies.length} / ${LEADERBOARD_SIZE}`}
          hint="Robust strategies first, then by composite fitness. Returns are in-sample. Click a row to open it in the workbench; tick up to three to compare."
        >
          <button
            type="button"
            onClick={() => setAllColumns((on) => !on)}
            aria-pressed={allColumns}
            className={cx(
              'inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs transition-colors',
              allColumns ? 'border-accent/50 bg-accent/10 text-accent-strong' : 'border-line text-ink-2 hover:bg-hover hover:text-ink',
            )}
          >
            <Columns3 size={13} /> {allColumns ? 'Key columns' : 'All columns'}
          </button>
        </StepHeader>
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
                : 'Pick a preset and press 1-Click Run in the top bar.'}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="num w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-line bg-raised/40">
                <th scope="col" className="w-8 pl-4 pr-1 py-2 text-left align-bottom text-[11px] font-medium uppercase tracking-wide text-ink-3" title={`Tick up to ${MAX_COMPARE} strategies to compare them side by side`}>
                  Cmp
                </th>
                {columns.map((column) => {
                  const sortable = column.value !== undefined;
                  const active = sortable && sort.key === column.key;
                  return (
                    <th
                      key={column.key}
                      scope="col"
                      title={column.title}
                      aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                      className={cx(
                        'px-2.5 py-2 align-bottom text-[11px] font-medium uppercase leading-tight tracking-wide last:pr-4',
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
                const compared = compareIds.includes(row.strategy.id);
                const compareFull = !compared && compareIds.length >= MAX_COMPARE;
                return (
                  <tr
                    key={row.strategy.id}
                    tabIndex={0}
                    aria-selected={selected}
                    aria-label={`Rank ${row.rank}, strategy ${row.strategy.id}. Open in the workbench`}
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
                    <td className="w-8 pl-4 pr-1 align-middle" onClick={(event) => event.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={compared}
                        disabled={compareFull}
                        onChange={() => onToggleCompare(row.strategy)}
                        onKeyDown={(event) => event.stopPropagation()}
                        aria-label={`Compare strategy ${row.strategy.id}`}
                        title={compareFull ? `At most ${MAX_COMPARE} strategies can be compared` : 'Compare side by side'}
                        className="h-3.5 w-3.5 cursor-pointer accent-[var(--color-accent)] disabled:cursor-not-allowed"
                      />
                    </td>
                    {columns.map((column) => (
                      <td
                        key={column.key}
                        className={cx(
                          'px-2.5 py-2 align-middle last:pr-4',
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
