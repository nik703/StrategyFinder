/**
 * Live telemetry: engine status, evaluation counter, the trial-budget meter of
 * the data window (every run and session on overlapping data counts), the best
 * deflated Sharpe ratio against the luck hurdle, and the market-data source.
 *
 * The headline is deliberately NOT a return figure: in-sample returns of the
 * best of thousands of tried strategies are inflated by selection, while the
 * deflated Sharpe accounts for how many were tried.
 */

import { CircleCheck, CircleDot, Database, Loader2, OctagonX, TriangleAlert } from 'lucide-react';
import type { ReactNode } from 'react';
import type { RunContext } from '../lib/config';
import type { BudgetStatus } from '../lib/trialLedger';
import { fmtDate, fmtDuration, fmtInt, fmtNum, fmtPct } from '../lib/format';
import { FITNESS_PATIENCE, type EngineStatus, type ProgressSnapshot, type StopReason } from '../types/trading';
import { cx, Panel, StepHeader } from './ui';

interface TelemetryStripProps {
  status: EngineStatus;
  telemetry: ProgressSnapshot;
  /** Buy & Hold Sharpe over the validation segment, for comparison. */
  benchmarkSharpe: number | null;
  context: RunContext | null;
  /** Evaluation limit of the current (or next) run. */
  maxEvaluations: number;
  /** Why the last run ended (null while running or before the first run). */
  stopReason: StopReason | null;
  /** Trial budget of the active data window; `estimated` before the data has been downloaded. */
  budget: { status: BudgetStatus; estimated: boolean } | null;
}

const STOP_LABEL: Record<StopReason, string> = {
  stopped: 'Stopped by User',
  target: 'Robust target reached',
  budget: 'Run budget used',
  'window-budget': 'Window budget spent',
  exhausted: 'Search space exhausted',
};

const BUDGET_TEXT: Record<BudgetStatus['level'], { label: string; tone: string; bar: string }> = {
  fresh: { label: 'Fresh data', tone: 'text-up', bar: 'bg-accent' },
  degrading: { label: 'Significance degrading', tone: 'text-warn', bar: 'bg-warn' },
  critical: { label: 'Nearly spent', tone: 'text-down', bar: 'bg-down' },
  exhausted: { label: 'Spent – extend to continue', tone: 'text-down', bar: 'bg-down' },
};

const STATUS: Record<EngineStatus, { label: string; className: string; icon: ReactNode }> = {
  idle: {
    label: 'Idle',
    className: 'border-line-strong bg-raised text-ink-2',
    icon: <CircleDot size={13} />,
  },
  loading: {
    label: 'Loading Data',
    className: 'border-accent/40 bg-accent/10 text-accent-strong',
    icon: <Loader2 size={13} className="animate-spin" />,
  },
  optimizing: {
    label: 'Optimizing',
    className: 'border-accent/40 bg-accent/10 text-accent-strong',
    icon: (
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent-strong" />
      </span>
    ),
  },
  stopped: {
    label: 'Stopped by User',
    className: 'border-warn/40 bg-warn/10 text-warn',
    icon: <OctagonX size={13} />,
  },
  converged: {
    label: 'Converged',
    className: 'border-up/40 bg-up/10 text-up',
    icon: <CircleCheck size={13} />,
  },
  error: {
    label: 'Error',
    className: 'border-down/40 bg-down/10 text-down',
    icon: <TriangleAlert size={13} />,
  },
};

function Tile({ label, children, className }: { label: string; children: ReactNode; className?: string }): JSX.Element {
  return (
    <div className={cx('min-w-0 px-4 py-3', className)}>
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-3">{label}</div>
      {children}
    </div>
  );
}

/** One stage of the pass / fail funnel. */
interface FunnelStage {
  label: string;
  /** What the gate checks, in plain English. */
  title: string;
  value: number;
  /** What the share is measured against (the scored candidates, or the neighbourhoods checked). */
  of: number;
  tone: string;
}

/** Robust strategies found, evaluations, the restart cycle, and the pass / fail funnel of the robustness gates. */
function ScanProgress({ telemetry, maxEvaluations }: { telemetry: ProgressSnapshot; maxEvaluations: number }): JSX.Element {
  const { robustFound, robustTarget, cycle, funnel, evaluations } = telemetry;
  const share = maxEvaluations > 0 ? Math.min(1, evaluations / maxEvaluations) : 0;
  const reached = robustTarget > 0 && robustFound >= robustTarget;
  const stages: FunnelStage[] = [
    { label: 'Scored', title: 'Candidate strategies backtested on the search period (train + validation).', value: funnel.scored, of: funnel.scored, tone: 'bg-ink-3/60' },
    { label: 'Enough trades', title: 'Cleared the timeframe-scaled minimum-trade gates – fewer trades cannot be told apart from luck.', value: funnel.trades, of: funnel.scored, tone: 'bg-accent/70' },
    { label: 'Profitable out of sample', title: 'Validation Sharpe above 0.3 and a positive validation profit after every cost.', value: funnel.validation, of: funnel.scored, tone: 'bg-accent/70' },
    { label: 'Consistent over time', title: '3 of 5 time slices profitable (flat slices set aside) and none with a drawdown beyond 35%.', value: funnel.folds, of: funnel.scored, tone: 'bg-accent/70' },
    { label: 'Beats luck', title: 'Deflated Sharpe of 50% or more: beats the best a random walk is expected to show after this many trials.', value: funnel.dsr, of: funnel.scored, tone: 'bg-accent/70' },
    { label: 'Stable neighbours', title: 'Nearby parameter settings keep a positive median profit per trade (checked for promising candidates only).', value: funnel.neighbours, of: funnel.neighboursChecked, tone: 'bg-accent/70' },
  ];
  return (
    <div className="border-t border-line px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <div className="flex items-baseline gap-2" title="Strategies passing every robustness gate on the search period. The holdout stays sealed until you lock one.">
          <span className="text-[11px] font-medium uppercase tracking-wider text-ink-3">Robust strategies found</span>
          <span className={cx('num text-lg font-semibold', reached ? 'text-up' : robustFound > 0 ? 'text-warn' : 'text-ink')}>
            {fmtInt(robustFound)}
            <span className="text-sm font-normal text-ink-3"> / {robustTarget > 0 ? fmtInt(robustTarget) : 'full budget'}</span>
          </span>
        </div>
        <div className="num text-xs text-ink-3">
          <span className="font-semibold text-ink-2">{fmtInt(evaluations)}</span> / {fmtInt(maxEvaluations)} evaluations
        </div>
        <div
          className="num inline-flex items-center gap-1.5 rounded-md border border-line bg-raised px-2 py-0.5 text-xs text-ink-2"
          title={`When the best score has not improved for ${FITNESS_PATIENCE.toLocaleString('en-US')} evaluations, the search keeps its best strategies and re-seeds the rest with fresh random ones (a diversity restart).`}
        >
          Restart cycle <span className="font-semibold text-ink">{fmtInt(cycle)}</span>
          <span className="text-ink-3">· next restart {fmtInt(Math.max(0, FITNESS_PATIENCE - Math.min(telemetry.fitnessStall, FITNESS_PATIENCE)))} evals away</span>
        </div>
      </div>
      <div
        className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-accent/15"
        role="progressbar"
        aria-label="Evaluations used by this run"
        aria-valuemin={0}
        aria-valuemax={maxEvaluations}
        aria-valuenow={evaluations}
      >
        <div className="h-full rounded-full bg-accent transition-[width] duration-150 ease-linear" style={{ width: `${share * 100}%` }} />
      </div>
      <div className="mt-3 text-[11px] font-medium uppercase tracking-wider text-ink-3">Pass / fail funnel · each gate on its own</div>
      <ol className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3 xl:grid-cols-6">
        {stages.map((stage) => {
          const ratio = stage.of > 0 ? stage.value / stage.of : 0;
          return (
            <li key={stage.label} title={stage.title} className="min-w-0">
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className="truncate text-ink-2">{stage.label}</span>
                <span className="num shrink-0 font-semibold text-ink">{fmtInt(stage.value)}</span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line" aria-hidden="true">
                <div className={cx('h-full rounded-full', stage.tone)} style={{ width: `${Math.max(stage.value > 0 ? 2 : 0, ratio * 100)}%` }} />
              </div>
              <div className="num mt-0.5 text-[10.5px] text-ink-3">
                {stage.of > 0 ? `${fmtPct(ratio * 100, false, ratio < 0.1 && ratio > 0 ? 1 : 0)} of ${fmtInt(stage.of)}` : '–'}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function TelemetryStrip({
  status,
  telemetry,
  benchmarkSharpe,
  context,
  maxEvaluations,
  stopReason,
  budget,
}: TelemetryStripProps): JSX.Element {
  const badge = STATUS[status];
  const badgeLabel = (status === 'converged' || status === 'stopped') && stopReason !== null ? STOP_LABEL[stopReason] : badge.label;
  const hasRun = telemetry.iteration > 0;
  const level = budget ? BUDGET_TEXT[budget.status.level] : null;
  const dsr = telemetry.bestDsr;
  const dsrTone = dsr === null ? 'text-ink-3' : dsr >= 0.95 ? 'text-up' : dsr >= 0.5 ? 'text-warn' : 'text-ink';

  return (
    <Panel>
      <div className="border-b border-line px-4 py-3">
        <StepHeader
          step={2}
          title="Discovery scanner"
          hint="Searches the hypothesis until enough strategies pass every robustness gate – the holdout stays sealed throughout."
        />
      </div>
    <div className="grid grid-cols-1 divide-y divide-line sm:grid-cols-2 sm:divide-y-0 xl:grid-cols-[1.05fr_0.8fr_1.5fr_1.45fr_1.4fr] xl:divide-x">
      <Tile label="Engine status">
        <div
          role="status"
          aria-live="polite"
          className={cx('inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold', badge.className)}
        >
          {badge.icon}
          {badgeLabel}
        </div>
        <div className="num mt-1.5 text-xs text-ink-3">
          {hasRun ? (
            <>
              {fmtDuration(telemetry.elapsedMs)} elapsed
              {status === 'optimizing' && telemetry.itersPerSec > 0 && <> · {fmtInt(telemetry.itersPerSec)} it/s</>}
            </>
          ) : (
            'Waiting for a run'
          )}
        </div>
      </Tile>

      <Tile label="Evaluations this run">
        <div className="flex items-baseline gap-1.5 whitespace-nowrap">
          <span className="num text-2xl font-semibold leading-7 text-ink">{fmtInt(telemetry.evaluations)}</span>
          <span className="num text-sm text-ink-3">/ {fmtInt(maxEvaluations)}</span>
        </div>
        <div className="num mt-1 text-xs leading-snug text-ink-3" title="Repeated genomes and repeated trade logs are skipped: they are not new trials.">
          {fmtInt(telemetry.duplicates)} duplicates skipped · restart at stall {fmtInt(Math.min(telemetry.fitnessStall, FITNESS_PATIENCE))}/{fmtInt(FITNESS_PATIENCE)}
        </div>
      </Tile>

      <Tile label="Trial budget · this data window">
        {budget && level ? (
          <>
            <div className="flex items-baseline gap-1.5 whitespace-nowrap">
              <span className="num text-2xl font-semibold leading-7 text-ink">{fmtInt(budget.status.used)}</span>
              <span className="num text-sm text-ink-3">/ {fmtInt(budget.status.limit)}</span>
              <span className={cx('ml-1 truncate text-xs font-semibold', level.tone)}>{level.label}</span>
            </div>
            <div
              className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-accent/15"
              role="progressbar"
              aria-label="Trial budget used on this data window"
              aria-valuemin={0}
              aria-valuemax={budget.status.limit}
              aria-valuenow={budget.status.used}
            >
              <div className={cx('h-full rounded-full transition-[width] duration-150 ease-linear', level.bar)} style={{ width: `${budget.status.share * 100}%` }} />
            </div>
            {budget.status.extensions > 0 && (
              <div className="mt-1 text-xs font-semibold text-warn" role="note">
                Extended {budget.status.extensions}× beyond the default budget – multiple-testing significance is degraded
              </div>
            )}
            <div
              className="num mt-1 truncate text-xs text-ink-3"
              title="Annualised Sharpe the luckiest of that many skill-less strategies is expected to show over this search period by chance alone. Every trial on overlapping data – any run, any session – counts."
            >
              best-of-{fmtInt(budget.status.used)} by luck ≈ Sharpe {fmtNum(budget.status.luckSharpe)}
              {budget.estimated ? ' · estimated window' : context ? ` · ${context.priorTrials.runs} earlier run${context.priorTrials.runs === 1 ? '' : 's'}` : ''}
            </div>
          </>
        ) : (
          <div className="mt-1 text-xs text-ink-3">Enter an asset to see its trial budget</div>
        )}
      </Tile>

      <Tile label="Best deflated Sharpe">
        <div className="flex items-baseline gap-3">
          <span className={cx('num text-2xl font-semibold leading-7', dsrTone)}>{dsr === null ? '–' : fmtPct(dsr * 100, false, 0)}</span>
          <span className="text-xs text-ink-3">{dsr === null ? '' : dsr >= 0.95 ? 'distinguishable from luck' : 'not distinguishable from luck'}</span>
        </div>
        <div className="num mt-1 truncate text-xs text-ink-3">
          {telemetry.bestDsrSharpe === null ? (
            hasRun && telemetry.undersampled > 0 ? (
              `${fmtInt(telemetry.undersampled)} candidates rejected for too few trades`
            ) : (
              'Appears once a strategy clears the trade gates'
            )
          ) : (
            <>
              Validation Sharpe <span className="text-ink-2">{fmtNum(telemetry.bestDsrSharpe)}</span> vs luck hurdle{' '}
              <span className="text-ink-2">{fmtNum(telemetry.hurdleSharpe)}</span>
              {benchmarkSharpe !== null && (
                <>
                  {' '}
                  · B&amp;H <span className="text-ink-2">{fmtNum(benchmarkSharpe)}</span>
                </>
              )}
            </>
          )}
        </div>
      </Tile>

      <Tile label="Market data" className="sm:col-span-2 xl:col-span-1">
        {context ? (
          <>
            <div
              className={cx(
                'inline-flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-semibold',
                context.simulated ? 'border-warn/40 bg-warn/10 text-warn' : 'border-up/40 bg-up/10 text-up',
              )}
            >
              <Database size={13} className="shrink-0" />
              <span className="truncate">
                {context.simulated
                  ? 'Simulated · offline fallback'
                  : `Live · ${context.venue}`}
              </span>
            </div>
            <div
              className="num mt-1.5 truncate text-xs text-ink-3"
              title={`${fmtInt(context.bars)} × ${context.interval} candles · ${fmtDate(context.windowStart)} → ${fmtDate(context.windowEnd)}`}
            >
              {fmtInt(context.bars)} × {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)}
            </div>
            {context.universe.length > 1 && (
              <div className="num mt-0.5 truncate text-xs text-ink-3" title={context.universe.join(', ')}>
                {context.settings.poolAssets ? 'Asset pool' : 'Robust universe'}: {context.universe.length} assets
              </div>
            )}
          </>
        ) : (
          <>
            <div className="inline-flex items-center gap-1.5 rounded-md border border-line-strong bg-raised px-2 py-1 text-xs font-semibold text-ink-2">
              <Database size={13} />
              Not loaded
            </div>
            <div className="mt-1.5 text-xs text-ink-3">Binance + Yahoo Finance via Netlify Function</div>
          </>
        )}
      </Tile>
    </div>
      <ScanProgress telemetry={telemetry} maxEvaluations={maxEvaluations} />
    </Panel>
  );
}
