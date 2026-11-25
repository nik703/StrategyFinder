/**
 * Top action bar (sticky): the brand, the primary "1-Click Run" button that launches a scan with the
 * active preset, Stop, the trial-budget meter of the selected data window (Progress + Badge, with an
 * Alert banner once the budget degrades significance or is spent), the "Reset Runs & Clear Cache"
 * button with its Dialog confirmation, the Export / Import Snapshot (.json) controls, and the four-step
 * workflow indicator.
 */

import { Activity, Loader2, Play, RotateCcw, Square, TriangleAlert } from 'lucide-react';
import { useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import SnapshotDialogs from '@/components/SnapshotDialogs';
import { fmtInt, fmtNum } from '@/lib/format';
import type { BudgetStatus } from '@/lib/trialLedger';
import { cn } from '@/lib/utils';
import { BUDGET_EXTENSION, type EngineStatus } from '@/types/trading';

interface TopActionBarProps {
  status: EngineStatus;
  /** Title of the preset the settings match, or null for a custom setup. */
  presetTitle: string | null;
  onRun: () => void;
  onStop: () => void;
  /** Why a run cannot start (the data window's trial budget is spent), or null. */
  runBlocked: string | null;
  /** Trial budget of the selected data window (`estimated` before its data has been downloaded). */
  budget: { status: BudgetStatus; estimated: boolean } | null;
  onExtendBudget: () => void;
  /** Wipe every run, cached strategy and trial / holdout record (asked for confirmation first). */
  onReset: () => void;
  /** A snapshot was imported: re-read the trial ledger and holdout log. */
  onSnapshotImported: () => void;
  /** Current step of the guided workflow (1–4). */
  stage: number;
}

const BUDGET_TONE: Record<BudgetStatus['level'], { label: string; badge: string; bar: string }> = {
  fresh: { label: 'Fresh data', badge: 'border-up/35 bg-up/10 text-up', bar: '' },
  degrading: { label: 'Significance degrading', badge: 'border-warn/35 bg-warn/10 text-warn', bar: '*:data-[slot=progress-indicator]:bg-warn' },
  critical: { label: 'Nearly spent', badge: 'border-down/35 bg-down/10 text-down', bar: '*:data-[slot=progress-indicator]:bg-down' },
  exhausted: { label: 'Budget spent', badge: 'border-down/35 bg-down/10 text-down', bar: '*:data-[slot=progress-indicator]:bg-down' },
};

const STEPS = [
  { n: 1, label: 'Presets', target: 'step-setup' },
  { n: 2, label: 'Discover', target: 'step-scan' },
  { n: 3, label: 'Leaderboard', target: 'step-rank' },
  { n: 4, label: 'Workbench', target: 'step-workbench' },
];

/** Compact trial-budget meter: trials used on the selected data window across every run and session. */
function TrialBudgetMeter({ budget }: { budget: TopActionBarProps['budget'] }): JSX.Element {
  if (!budget) return <div className="text-xs text-muted-foreground">Trial budget: pick valid assets</div>;
  const { status, estimated } = budget;
  const tone = BUDGET_TONE[status.level];
  const title =
    `Strategies already evaluated on this data window – any run, any session – out of its default budget. ` +
    `Every trial raises the luck hurdle: the best of ${fmtInt(status.used)} skill-less strategies is expected to reach a Sharpe of ` +
    `${fmtNum(status.luckSharpe)} by chance alone.${status.extensions > 0 ? ` Extended ${status.extensions}× – significance is degraded.` : ''}` +
    (estimated ? ' (Estimated until the data is downloaded.)' : '');
  return (
    <div className="min-w-[170px]" title={title}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Trial budget</span>
        <Badge variant="outline" className={cn('h-4 rounded-md px-1.5 text-[10.5px]', tone.badge)}>
          {tone.label}
        </Badge>
      </div>
      <Progress value={Math.min(100, status.share * 100)} aria-label="Trial budget used on this data window" className={cn('mt-1.5 h-1.5', tone.bar)} />
      <div className="num mt-0.5 text-[11px] text-muted-foreground">
        <span className="font-semibold text-secondary-foreground">{fmtInt(status.used)}</span> / {fmtInt(status.limit)} trials
        {status.extensions > 0 ? ` · extended ${status.extensions}×` : ''}
      </div>
    </div>
  );
}

/** Banner under the bar once the data window's budget is degrading significance, nearly spent or spent. */
function BudgetAlert({ budget, onExtend }: { budget: TopActionBarProps['budget']; onExtend: () => void }): JSX.Element | null {
  if (!budget || budget.status.level === 'fresh') return null;
  const { status } = budget;
  const spent = status.level === 'exhausted';
  const severe = spent || status.level === 'critical';
  return (
    <Alert className={cn('mt-2.5', severe ? 'border-down/40 bg-down/10 text-down' : 'border-warn/40 bg-warn/10 text-warn')}>
      <TriangleAlert />
      <AlertTitle>
        {spent
          ? 'The trial budget of this data window is spent.'
          : status.level === 'critical'
            ? 'The trial budget of this data window is nearly spent.'
            : 'Multiple-testing significance is degrading on this data window.'}
      </AlertTitle>
      <AlertDescription className="flex flex-wrap items-center gap-x-3 gap-y-2 text-current">
        <span className="min-w-0 flex-1">
          {fmtInt(status.used)} of {fmtInt(status.limit)} evaluations have been made on these symbols and dates. Every extra trial raises the luck
          hurdle (now Sharpe ≈ {fmtNum(status.luckSharpe)}).{' '}
          {spent ? 'Pick other assets or another period, or extend the budget knowingly.' : 'Consider fresh data for new research.'}
        </span>
        {spent && (
          <Button size="sm" variant="outline" onClick={onExtend} className="border-current bg-transparent text-current">
            Extend +{fmtInt(BUDGET_EXTENSION)} trials
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}

/** "Reset Runs & Clear Cache" with its Dialog: what it clears and what that does to multiple-testing tracking. */
function ResetButton({ running, onReset }: { running: boolean; onReset: () => void }): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="lg"
          title="Stop any scan and delete every cached run, strategy and trial / holdout record from this browser"
          className="hover:border-destructive/50 hover:bg-destructive/10 hover:text-destructive"
        >
          <RotateCcw /> Reset Runs &amp; Clear Cache
        </Button>
      </DialogTrigger>
      <DialogContent role="alertdialog" className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TriangleAlert className="size-5 text-warn" /> Reset runs and clear the cache?
          </DialogTitle>
          <DialogDescription>
            {running ? 'Every pooled worker is stopped at once, pending work is dropped and ' : ''}Every cached strategy, the leaderboard, the elite kept
            across restarts, the trial and restart counters, the holdout reveal log, the saved runs and the cached candles are deleted from this
            browser.
          </DialogDescription>
        </DialogHeader>
        <Alert className="border-warn/40 bg-warn/10 text-warn">
          <TriangleAlert />
          <AlertDescription className="text-current">
            Clearing the history resets multiple-testing tracking: earlier trials stop raising the luck hurdle of the deflated Sharpe, and holdouts
            you have already seen will look untouched again. Only reset when you start genuinely new research.
          </AlertDescription>
        </Alert>
        <p className="text-xs text-muted-foreground">Your settings and saved tickers are kept. Export a snapshot first if you may want the history back.</p>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <Button
            variant="destructive"
            onClick={() => {
              setOpen(false);
              onReset();
            }}
          >
            <RotateCcw /> Reset everything
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function TopActionBar({
  status,
  presetTitle,
  onRun,
  onStop,
  runBlocked,
  budget,
  onExtendBudget,
  onReset,
  onSnapshotImported,
  stage,
}: TopActionBarProps): JSX.Element {
  const busy = status === 'loading' || status === 'optimizing';
  return (
    <header className="sticky top-0 z-30 -mx-4 border-b border-border bg-background/90 px-4 py-2.5 backdrop-blur sm:-mx-6 sm:px-6">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
        <div className="flex items-center gap-2.5">
          <div className="grid size-9 place-items-center rounded-lg border border-primary/40 bg-primary/15 text-primary">
            <Activity className="size-[18px]" />
          </div>
          <div className="leading-tight">
            <h1 className="text-base font-semibold text-foreground">StrategyFinder</h1>
            <p className="hidden text-[11px] text-muted-foreground 2xl:block">Discover, stress-test and export robust strategies</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {busy ? (
            <Button variant="destructive" size="lg" onClick={onStop} className="h-10 px-4">
              {status === 'loading' ? <Loader2 className="animate-spin" /> : <Square className="fill-current" />}
              {status === 'loading' ? 'Loading data… Stop' : 'Stop scan'}
            </Button>
          ) : (
            <Button
              size="lg"
              onClick={onRun}
              disabled={runBlocked !== null}
              title={runBlocked ?? `Start a discovery scan with ${presetTitle ? `the "${presetTitle}" preset` : 'the current custom settings'}`}
              className="h-10 px-5 font-semibold"
            >
              <Play className="fill-current" /> 1-Click Run
            </Button>
          )}
          <span className="max-w-[220px] truncate text-xs text-muted-foreground" title={presetTitle ?? 'Custom settings'}>
            {presetTitle ? (
              <>
                Preset: <span className="text-secondary-foreground">{presetTitle}</span>
              </>
            ) : (
              <span className="text-secondary-foreground">Custom settings</span>
            )}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <TrialBudgetMeter budget={budget} />
          <ResetButton running={busy} onReset={onReset} />
          <SnapshotDialogs onImported={onSnapshotImported} />
        </div>

        <nav aria-label="Workflow" className="flex flex-wrap items-center gap-1 lg:ml-auto">
          {STEPS.map((step, index) => (
            <span key={step.n} className="inline-flex items-center gap-1">
              <Button
                asChild
                size="sm"
                variant="ghost"
                className={cn('rounded-full border', stage === step.n ? 'border-primary/60 bg-primary/15 font-semibold text-foreground' : 'border-border text-muted-foreground')}
              >
                <a href={`#${step.target}`} aria-current={stage === step.n ? 'step' : undefined}>
                  <span className="num">{step.n}</span> {step.label}
                </a>
              </Button>
              {index < STEPS.length - 1 && (
                <span aria-hidden="true" className="text-muted-foreground">
                  ›
                </span>
              )}
            </span>
          ))}
        </nav>
      </div>
      <BudgetAlert budget={budget} onExtend={onExtendBudget} />
    </header>
  );
}
