/**
 * Step 2 – the live discovery scanner (a shadcn Card): engine status, the best deflated Sharpe ratio
 * against the luck hurdle, the market-data source, and the scan itself – robust strategies found
 * against the target, the evaluation Progress bar, the diversity-restart cycle Badge and the pass /
 * fail funnel of the robustness gates. (The trial-budget meter of the data window sits in the top
 * action bar.)
 *
 * The headline is deliberately NOT a return figure: in-sample returns of the best of thousands of
 * tried strategies are inflated by selection, while the deflated Sharpe accounts for how many were
 * tried.
 */

import { CircleCheck, CircleDot, Database, Loader2, OctagonX, RefreshCcw, TriangleAlert } from 'lucide-react';
import type { ReactNode } from 'react';
import { Badge } from '@/components/ui/badge';
import { CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import type { RunContext } from '@/lib/config';
import { fmtDate, fmtDuration, fmtInt, fmtNum, fmtPct } from '@/lib/format';
import { cn } from '@/lib/utils';
import { FITNESS_PATIENCE, type EngineStatus, type ProgressSnapshot, type StopReason } from '@/types/trading';
import { InfoTip, Panel, StepHeader } from './common';

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
  /** Robust strategies the run scans for (0 = the full budget). */
  robustTarget: number;
}

const STOP_LABEL: Record<StopReason, string> = {
  stopped: 'Stopped by User',
  target: 'Robust target reached',
  budget: 'Run budget used',
  'window-budget': 'Window budget spent',
  exhausted: 'Search space exhausted',
};

const STATUS: Record<EngineStatus, { label: string; className: string; icon: ReactNode }> = {
  idle: { label: 'Idle', className: 'border-input bg-muted text-secondary-foreground', icon: <CircleDot /> },
  loading: { label: 'Loading Data', className: 'border-primary/40 bg-primary/10 text-primary', icon: <Loader2 className="animate-spin" /> },
  optimizing: {
    label: 'Optimizing',
    className: 'border-primary/40 bg-primary/10 text-primary',
    icon: (
      <span className="relative flex size-2">
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-60" />
        <span className="relative inline-flex size-2 rounded-full bg-primary" />
      </span>
    ),
  },
  stopped: { label: 'Stopped by User', className: 'border-warn/40 bg-warn/10 text-warn', icon: <OctagonX /> },
  converged: { label: 'Converged', className: 'border-up/40 bg-up/10 text-up', icon: <CircleCheck /> },
  error: { label: 'Error', className: 'border-down/40 bg-down/10 text-down', icon: <TriangleAlert /> },
};

function Tile({ label, children, className }: { label: string; children: ReactNode; className?: string }): JSX.Element {
  return (
    <div className={cn('min-w-0 px-4 py-3', className)}>
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{label}</div>
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
  muted?: boolean;
}

/** Robust strategies found, evaluations, the restart cycle, and the pass / fail funnel of the robustness gates. */
function ScanProgress({ telemetry, maxEvaluations, robustTarget }: { telemetry: ProgressSnapshot; maxEvaluations: number; robustTarget: number }): JSX.Element {
  const { robustFound, cycle, funnel, evaluations } = telemetry;
  const share = maxEvaluations > 0 ? Math.min(1, evaluations / maxEvaluations) : 0;
  const reached = robustTarget > 0 && robustFound >= robustTarget;
  const stages: FunnelStage[] = [
    { label: 'Scored', title: 'Candidate strategies backtested on the search period (train + validation).', value: funnel.scored, of: funnel.scored, muted: true },
    {
      label: '1 · Sample & folds',
      title:
        'Gate 1: enough effective independent trades (correlated, overlapping trades across a basket count once), a positive median-fold Sortino/Calmar and no time slice with a drawdown beyond 35%.',
      value: funnel.sample,
      of: funnel.scored,
    },
    { label: '2 · Beats luck', title: 'Gate 2: deflated Sharpe of 50% or more – beats the best a random walk is expected to show after this many trials.', value: funnel.dsr, of: funnel.scored },
    {
      label: '3 · Stress & stability',
      title:
        'Gate 3 (second tier, top 20 contenders only): the Monte Carlo stress test holds and nearby parameter settings keep a positive median profit per trade.',
      value: funnel.stress,
      of: funnel.stressChecked,
    },
  ];
  return (
    <CardContent className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <div className="flex items-baseline gap-2" title="Strategies passing every robustness gate on the search period. The holdout stays sealed until you lock one.">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Robust strategies found</span>
          <span className={cn('num text-lg font-semibold', reached ? 'text-up' : robustFound > 0 ? 'text-warn' : 'text-foreground')}>
            {fmtInt(robustFound)}
            <span className="text-sm font-normal text-muted-foreground"> / {robustTarget > 0 ? fmtInt(robustTarget) : 'full budget'}</span>
          </span>
        </div>
        <div className="num text-xs text-muted-foreground" title="Repeated genomes and repeated trade logs are skipped: they are not new trials.">
          <span className="font-semibold text-secondary-foreground">{fmtInt(evaluations)}</span> / {fmtInt(maxEvaluations)} evaluations
          {telemetry.duplicates > 0 ? ` · ${fmtInt(telemetry.duplicates)} duplicates skipped` : ''}
        </div>
        <Badge
          variant="outline"
          className="num rounded-md text-secondary-foreground"
          title={`When the best score has not improved for ${FITNESS_PATIENCE.toLocaleString('en-US')} evaluations, the search keeps its best strategies and re-seeds the rest with fresh random ones (a diversity restart).`}
        >
          <RefreshCcw /> Restart cycle <span className="font-semibold text-foreground">{fmtInt(cycle)}</span>
          <span className="text-muted-foreground">· next restart {fmtInt(Math.max(0, FITNESS_PATIENCE - Math.min(telemetry.fitnessStall, FITNESS_PATIENCE)))} evals away</span>
        </Badge>
      </div>
      <Progress value={share * 100} aria-label="Evaluations used by this run" className="mt-2 h-1.5" />
      <div className="mt-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Pass / fail funnel · each gate on its own</div>
      <ol className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3 xl:grid-cols-6">
        {stages.map((stage) => {
          const ratio = stage.of > 0 ? stage.value / stage.of : 0;
          return (
            <li key={stage.label} className="min-w-0">
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <InfoTip tip={stage.title} className="truncate text-secondary-foreground">
                  {stage.label}
                </InfoTip>
                <span className="num shrink-0 font-semibold text-foreground">{fmtInt(stage.value)}</span>
              </div>
              <Progress
                value={stage.value > 0 ? Math.max(2, ratio * 100) : 0}
                aria-label={`${stage.label}: ${fmtInt(stage.value)} of ${fmtInt(stage.of)}`}
                className={cn('mt-1 h-1.5', stage.muted && '*:data-[slot=progress-indicator]:bg-muted-foreground/60')}
              />
              <div className="num mt-0.5 text-[10.5px] text-muted-foreground">
                {stage.of > 0 ? `${fmtPct(ratio * 100, false, ratio < 0.1 && ratio > 0 ? 1 : 0)} of ${fmtInt(stage.of)}` : '–'}
              </div>
            </li>
          );
        })}
      </ol>
    </CardContent>
  );
}

export default function TelemetryStrip({ status, telemetry, benchmarkSharpe, context, maxEvaluations, stopReason, robustTarget }: TelemetryStripProps): JSX.Element {
  const badge = STATUS[status];
  const badgeLabel = (status === 'converged' || status === 'stopped') && stopReason !== null ? STOP_LABEL[stopReason] : badge.label;
  const hasRun = telemetry.iteration > 0;
  const dsr = telemetry.bestDsr;
  const dsrTone = dsr === null ? 'text-muted-foreground' : dsr >= 0.95 ? 'text-up' : dsr >= 0.5 ? 'text-warn' : 'text-foreground';

  return (
    <Panel>
      <CardHeader className="border-b px-4 py-3">
        <StepHeader
          step={2}
          title="Live discovery scanner"
          hint="Searches the hypothesis until enough strategies pass every robustness gate – the holdout stays sealed throughout."
        />
      </CardHeader>
      <div className="grid grid-cols-1 divide-y divide-border sm:grid-cols-[1fr_1.4fr] sm:divide-y-0 lg:grid-cols-[0.9fr_1.4fr_1.4fr] lg:divide-x">
        <Tile label="Engine status">
          <Badge role="status" aria-live="polite" variant="outline" className={cn('h-6 rounded-md px-2 text-xs font-semibold', badge.className)}>
            {badge.icon}
            {badgeLabel}
          </Badge>
          <div className="num mt-1.5 text-xs text-muted-foreground">
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

        <Tile label="Best deflated Sharpe">
          <div className="flex items-baseline gap-3">
            <span className={cn('num text-2xl font-semibold leading-7', dsrTone)}>{dsr === null ? '–' : fmtPct(dsr * 100, false, 0)}</span>
            <span className="text-xs text-muted-foreground">{dsr === null ? '' : dsr >= 0.95 ? 'distinguishable from luck' : 'not distinguishable from luck'}</span>
          </div>
          <div className="num mt-1 truncate text-xs text-muted-foreground">
            {telemetry.bestDsrSharpe === null ? (
              hasRun && telemetry.undersampled > 0 ? (
                `${fmtInt(telemetry.undersampled)} candidates rejected for too few trades`
              ) : (
                'Appears once a strategy clears the trade gates'
              )
            ) : (
              <>
                Validation Sharpe <span className="text-secondary-foreground">{fmtNum(telemetry.bestDsrSharpe)}</span> vs luck hurdle{' '}
                <span className="text-secondary-foreground">{fmtNum(telemetry.hurdleSharpe)}</span>
                {benchmarkSharpe !== null && (
                  <>
                    {' '}
                    · B&amp;H <span className="text-secondary-foreground">{fmtNum(benchmarkSharpe)}</span>
                  </>
                )}
              </>
            )}
          </div>
        </Tile>

        <Tile label="Market data" className="sm:col-span-2 lg:col-span-1">
          {context ? (
            <>
              <Badge
                variant="outline"
                className={cn('h-6 max-w-full rounded-md px-2 text-xs font-semibold', context.simulated ? 'border-warn/40 bg-warn/10 text-warn' : 'border-up/40 bg-up/10 text-up')}
              >
                <Database />
                <span className="truncate">{context.simulated ? 'Simulated · offline fallback' : `Live · ${context.venue}`}</span>
              </Badge>
              <div
                className="num mt-1.5 truncate text-xs text-muted-foreground"
                title={`${fmtInt(context.bars)} × ${context.interval} candles · ${fmtDate(context.windowStart)} → ${fmtDate(context.windowEnd)}`}
              >
                {fmtInt(context.bars)} × {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)}
              </div>
              {context.universe.length > 1 && (
                <div className="num mt-0.5 truncate text-xs text-muted-foreground" title={context.universe.join(', ')}>
                  Asset pool: {context.universe.length} assets, same rules on each
                </div>
              )}
            </>
          ) : (
            <>
              <Badge variant="outline" className="h-6 rounded-md px-2 text-xs font-semibold text-secondary-foreground">
                <Database />
                Not loaded
              </Badge>
              <div className="mt-1.5 text-xs text-muted-foreground">Binance + Yahoo Finance via Netlify Function</div>
            </>
          )}
        </Tile>
      </div>
      <Separator />
      <ScanProgress telemetry={telemetry} maxEvaluations={maxEvaluations} robustTarget={robustTarget} />
    </Panel>
  );
}
