/**
 * Step 3 – the live, sortable ranked leaderboard: a shadcn Table with status Badges (Robust /
 * Marginal / Overfit risk) and Tooltip explanations on every column header. Clicking a row opens
 * the strategy in the Strategy Workbench below; ticking up to three rows compares them side by side.
 *
 * The default order is the run's rank: strategies passing every robustness gate first, then the
 * rest by composite fitness. Risk-adjusted, selection-bias-aware figures lead (sparkline, deflated
 * Sharpe, stress test, Sortino, max drawdown, trades) and every return figure is labelled
 * in-sample. There is deliberately no holdout column – the holdout is revealed for one locked
 * strategy only, so it can never be sorted on.
 */

import { ChevronDown, ChevronsUpDown, ChevronUp, Columns3, Lock } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { CardHeader } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { judgedFolds, passesFoldGate } from '@/engine/backtester';
import { exitModeLabel, exitTags, groupFormula, indicatorBadges } from '@/engine/describe';
import { MAX_PARAMETERS } from '@/engine/genome';
import { fmtInt, fmtNum, fmtPct, fmtUsd } from '@/lib/format';
import { cn } from '@/lib/utils';
import { LEADERBOARD_SIZE, type EngineStatus, type StrategyResult, type TradeGates, type Verdict } from '@/types/trading';
import { InfoTip, Panel, Pill, Sparkline, StepHeader, type PillTone } from './common';

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
  | 'folds'
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
  if (verdict === null) return <span className="text-[11px] text-muted-foreground">pending</span>;
  return <Pill tone={verdict === 'Robust' ? 'up' : verdict === 'Fragile' ? 'warn' : 'down'}>{verdict}</Pill>;
}

/** Number of robustness gates a strategy passes (0–3). */
function gatesPassed(strategy: StrategyResult): number {
  const r = strategy.robust;
  return r ? [r.sample, r.dsr, r.stress === true].filter(Boolean).length : 0;
}

/** Search-period status: Robust (every gate), Marginal (one gate short) or Overfit risk (two or more short). */
function robustStatus(strategy: StrategyResult): { label: string; tone: PillTone } | null {
  if (!strategy.robust) return null;
  const passed = gatesPassed(strategy);
  if (strategy.robust.passed) return { label: 'Robust', tone: 'up' };
  return passed === 2 ? { label: 'Marginal', tone: 'warn' } : { label: 'Overfit risk', tone: 'down' };
}

/** Profitable folds among the folds the strategy is judged on (flat folds set aside, as in the fold gate). */
function foldTally(strategy: StrategyResult): { profitable: number; judged: number } | null {
  const folds = strategy.folds;
  if (!folds || folds.length === 0) return null;
  const judged = judgedFolds(folds);
  return { profitable: judged.filter((k) => folds[k].roi > 0).length, judged: judged.length };
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
        <span className={cn('num font-semibold', row.rank === 1 ? 'text-primary' : 'text-secondary-foreground')}>#{row.rank}</span>
        {row.locked && (
          <span title="Locked: this strategy's holdout has been revealed" className="text-primary">
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
    label: 'Status',
    title:
      'Three robustness gates on the search period – 1 effective sample + median-fold return + fold drawdown cap · 2 deflated Sharpe ≥ 50% · 3 Monte Carlo stress + neighbour stability (top 20 only). Robust: all 3 pass · Marginal: 2 of 3 · Overfit risk: 1 or none.',
    align: 'left',
    value: ({ strategy }) => gatesPassed(strategy),
    render: ({ strategy }) => {
      const r = strategy.robust;
      const status = robustStatus(strategy);
      if (!r || !status) return <span className="text-muted-foreground">–</span>;
      const failed = [!r.sample && 'sample', !r.dsr && 'deflated Sharpe', r.stress === false && 'stress', r.stress === null && 'stress (not checked yet)'].filter(Boolean);
      return (
        <Pill tone={status.tone} title={r.passed ? 'Passes every robustness gate on the search period' : `${gatesPassed(strategy)}/3 gates · open: ${failed.join(', ')}`}>
          {status.label}
        </Pill>
      );
    },
  },
  {
    key: 'folds',
    label: 'Fold consistency',
    title: 'Profitable time slices of the search period, out of the slices the strategy traded in (flat slices set aside). 3 of 5 are needed, none losing more than 35%.',
    align: 'right',
    value: ({ strategy }) => {
      const tally = foldTally(strategy);
      return tally ? tally.profitable / Math.max(1, tally.judged) : -1;
    },
    render: ({ strategy }) => {
      const tally = foldTally(strategy);
      if (!tally) return <span className="text-muted-foreground">–</span>;
      return (
        <span className={cn('font-semibold', strategy.folds && passesFoldGate(strategy.folds) ? 'text-up' : 'text-secondary-foreground')}>
          {tally.profitable}/{tally.judged}
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
        <div className="num mt-1 truncate text-[11px] text-muted-foreground" title={groupFormula(strategy.genome.longEntry)}>
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
      <span className={cn('font-semibold', strategy.dsr >= 0.95 ? 'text-up' : strategy.dsr >= 0.5 ? 'text-warn' : 'text-secondary-foreground')}>
        {fmtPct(strategy.dsr * 100, false, 0)}
      </span>
    ),
  },
  {
    key: 'stress',
    label: 'Monte Carlo · DD95',
    title:
      'Monte Carlo: 500 replays of the trades with 10% skipped, the order shuffled and random slippage shocks. Share of replays that end in profit, and the 95th-percentile worst drawdown.',
    align: 'right',
    value: ({ strategy }) => strategy.monteCarlo?.probProfit ?? -1,
    render: ({ strategy }) => {
      const mc = strategy.monteCarlo;
      if (!mc) return <span className="text-muted-foreground">–</span>;
      return (
        <span className="inline-flex flex-col items-end leading-tight">
          <span className={cn('font-semibold', mc.passed ? 'text-up' : mc.probProfit >= 0.6 ? 'text-warn' : 'text-down')}>
            {fmtPct(mc.probProfit * 100, false, 0)}
          </span>
          <span className="text-[10.5px] text-muted-foreground">DD95 {fmtPct(mc.drawdownP95, false, 0)}</span>
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
    render: ({ strategy }) => <span className="text-foreground">{fmtNum(strategy.validation.sharpe)}</span>,
  },
  {
    key: 'sharpe',
    detail: true,
    label: 'Sharpe (in-sample)',
    title: 'Annualised Sharpe ratio over train + validation (risk-free rate 0)',
    align: 'right',
    value: ({ strategy }) => strategy.search.sharpe,
    render: ({ strategy }) => <span className="text-secondary-foreground">{fmtNum(strategy.search.sharpe)}</span>,
  },
  {
    key: 'sortino',
    label: 'Sortino (in-sample)',
    title: 'Annualised Sortino ratio over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.sortino,
    render: ({ strategy }) => <span className="text-secondary-foreground">{fmtNum(strategy.search.sortino)}</span>,
  },
  {
    key: 'maxDrawdown',
    label: 'Max Drawdown (%)',
    title: 'Largest peak-to-trough equity decline, measured to intrabar lows',
    align: 'right',
    value: ({ strategy }) => strategy.search.maxDrawdown,
    firstDir: 'asc',
    render: ({ strategy }) => <span className="text-secondary-foreground">{fmtPct(strategy.search.maxDrawdown)}</span>,
  },
  {
    key: 'trades',
    label: 'Trades',
    title: 'Trades over train + validation (validation trades in brackets)',
    align: 'right',
    value: ({ strategy }) => strategy.search.trades,
    render: ({ strategy }) => (
      <span className="text-secondary-foreground">
        {fmtInt(strategy.search.trades)} <span className="text-muted-foreground">({fmtInt(strategy.validation.trades)})</span>
      </span>
    ),
  },
  {
    key: 'winRate',
    detail: true,
    label: 'Win Rate (%)',
    align: 'right',
    value: ({ strategy }) => strategy.search.winRate,
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtPct(strategy.search.winRate)}</span>,
  },
  {
    key: 'profitFactor',
    detail: true,
    label: 'Profit Factor',
    title: 'Gross profit ÷ gross loss',
    align: 'right',
    value: ({ strategy }) => strategy.search.profitFactor,
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtNum(strategy.search.profitFactor)}</span>,
  },
  {
    key: 'roi',
    detail: true,
    label: 'ROI (in-sample, %)',
    title: 'Return on starting capital over train + validation – the data the search optimised on, so it is biased upwards',
    align: 'right',
    value: ({ strategy }) => strategy.search.roi,
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtPct(strategy.search.roi, true)}</span>,
  },
  {
    key: 'netPnl',
    detail: true,
    label: 'Net PnL (in-sample, $)',
    title: 'Net profit after fees, slippage and financing over train + validation',
    align: 'right',
    value: ({ strategy }) => strategy.search.netPnl,
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtUsd(strategy.search.netPnl, true)}</span>,
  },
  {
    key: 'alpha',
    detail: true,
    label: 'Excess ROI vs. B&H (in-sample)',
    title: 'Strategy ROI minus Buy & Hold ROI over train + validation, in percentage points. Not risk-adjusted, so it is not alpha.',
    align: 'right',
    value: ({ strategy }) => strategy.alpha,
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtPct(strategy.alpha, true)}</span>,
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
          <span className="text-[11px] text-muted-foreground">{exitTags(strategy.genome).join(' · ')}</span>
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
    render: ({ strategy }) => <span className="text-secondary-foreground">{strategy.params}</span>,
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
    render: ({ strategy }) => <span className="text-muted-foreground">{fmtNum(strategy.fitness)}</span>,
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
      <CardHeader className="border-b px-4 py-3">
        <StepHeader
          step={3}
          title={`Ranked leaderboard · ${strategies.length} / ${LEADERBOARD_SIZE}`}
          hint="Robust strategies first, then by composite fitness. Returns are in-sample. Click a row to open it in the workbench; tick up to three to compare."
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAllColumns((on) => !on)}
            aria-pressed={allColumns}
            className={cn(allColumns && 'border-primary/50 bg-primary/10 text-primary')}
          >
            <Columns3 /> {allColumns ? 'Key columns' : 'All columns'}
          </Button>
        </StepHeader>
      </CardHeader>

      {strategies.length === 0 ? (
        <div className="px-4 py-14 text-center">
          <p className="text-sm text-secondary-foreground">
            {status === 'optimizing' || status === 'loading'
              ? 'Searching for strategies that clear the minimum-trade gates…'
              : status === 'converged' || status === 'stopped'
                ? 'No strategy cleared the minimum-trade gates.'
                : 'No strategies yet.'}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
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
        <Table className="num">
          <TableHeader>
            <TableRow className="bg-muted/40 hover:bg-muted/40">
              <TableHead className="w-8 pl-4 text-[11px] uppercase tracking-wide">
                <InfoTip tip={`Tick up to ${MAX_COMPARE} strategies to compare them side by side`}>Cmp</InfoTip>
              </TableHead>
              {columns.map((column) => {
                const sortable = column.value !== undefined;
                const active = sortable && sort.key === column.key;
                const label = column.title ? <InfoTip tip={column.title}>{column.label}</InfoTip> : column.label;
                return (
                  <TableHead
                    key={column.key}
                    aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
                    className={cn(
                      'h-auto py-2 align-bottom text-[11px] font-medium uppercase leading-tight tracking-wide last:pr-4',
                      column.align === 'right' ? 'max-w-[120px] text-right' : 'text-left',
                      active ? 'text-foreground' : 'text-muted-foreground',
                    )}
                  >
                    <span className={cn('inline-flex items-end gap-1', column.align === 'right' && 'flex-row-reverse')}>
                      {sortable ? (
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => toggleSort(column)}
                          aria-label={`Sort by ${column.label}`}
                          className={cn('shrink-0', !active && 'opacity-50')}
                        >
                          {active ? sort.dir === 'asc' ? <ChevronUp /> : <ChevronDown /> : <ChevronsUpDown />}
                        </Button>
                      ) : null}
                      <span className="whitespace-normal">{label}</span>
                    </span>
                  </TableHead>
                );
              })}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => {
              const selected = row.strategy.id === selectedId;
              const compared = compareIds.includes(row.strategy.id);
              const compareFull = !compared && compareIds.length >= MAX_COMPARE;
              return (
                <TableRow
                  key={row.strategy.id}
                  tabIndex={0}
                  data-state={selected ? 'selected' : undefined}
                  aria-selected={selected}
                  aria-label={`Rank ${row.rank}, strategy ${row.strategy.id}. Open in the workbench`}
                  onClick={() => onSelect(row.strategy)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      onSelect(row.strategy);
                    }
                  }}
                  className={cn('cursor-pointer focus-visible:bg-accent focus-visible:outline-none', selected && 'bg-primary/10 shadow-[inset_3px_0_0_0_var(--primary)]')}
                >
                  <TableCell className="w-8 pl-4" onClick={(event) => event.stopPropagation()}>
                    <Checkbox
                      checked={compared}
                      disabled={compareFull}
                      onCheckedChange={() => onToggleCompare(row.strategy)}
                      onKeyDown={(event) => event.stopPropagation()}
                      aria-label={`Compare strategy ${row.strategy.id}`}
                      title={compareFull ? `At most ${MAX_COMPARE} strategies can be compared` : 'Compare side by side'}
                    />
                  </TableCell>
                  {columns.map((column) => (
                    <TableCell key={column.key} className={cn('py-2 last:pr-4', column.align === 'right' ? 'text-right' : 'whitespace-normal text-left')}>
                      {column.render(row)}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}
