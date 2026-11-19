/**
 * Top action bar (sticky): the brand, the primary "1-Click Run" button that launches a scan with the
 * active preset, Stop, the compact trial-budget meter of the selected data window, the "Reset Runs &
 * Clear Cache" button (with a confirmation that spells out what a reset does to multiple-testing
 * tracking) and the four-step workflow indicator.
 */

import { Activity, Loader2, Play, RotateCcw, Square, TriangleAlert } from 'lucide-react';
import { useEffect, useId, useRef, useState } from 'react';
import { fmtInt, fmtNum } from '../lib/format';
import type { BudgetStatus } from '../lib/trialLedger';
import { BUDGET_EXTENSION, type EngineStatus } from '../types/trading';
import { cx } from './ui';

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
  /** Current step of the guided workflow (1–4). */
  stage: number;
}

const BUDGET_TONE: Record<BudgetStatus['level'], { label: string; text: string; bar: string }> = {
  fresh: { label: 'Fresh data', text: 'text-up', bar: 'bg-accent' },
  degrading: { label: 'Significance degrading', text: 'text-warn', bar: 'bg-warn' },
  critical: { label: 'Nearly spent', text: 'text-down', bar: 'bg-down' },
  exhausted: { label: 'Budget spent', text: 'text-down', bar: 'bg-down' },
};

const STEPS = [
  { n: 1, label: 'Presets', target: 'step-setup' },
  { n: 2, label: 'Discover', target: 'step-scan' },
  { n: 3, label: 'Leaderboard', target: 'step-rank' },
  { n: 4, label: 'Workbench', target: 'step-workbench' },
];

/** Compact trial-budget meter: trials used on the selected data window across every run and session. */
function TrialBudgetMeter({
  budget,
  onExtend,
}: {
  budget: TopActionBarProps['budget'];
  onExtend: () => void;
}): JSX.Element {
  if (!budget) {
    return <div className="text-xs text-ink-3">Trial budget: pick valid assets</div>;
  }
  const { status, estimated } = budget;
  const tone = BUDGET_TONE[status.level];
  const title =
    `Strategies already evaluated on this data window – any run, any session – out of its default budget. ` +
    `Every trial raises the luck hurdle: the best of ${fmtInt(status.used)} skill-less strategies is expected to reach a Sharpe of ` +
    `${fmtNum(status.luckSharpe)} by chance alone.${status.extensions > 0 ? ` Extended ${status.extensions}× – significance is degraded.` : ''}` +
    (estimated ? ' (Estimated until the data is downloaded.)' : '');
  return (
    <div className="flex min-w-0 items-center gap-2" title={title}>
      <div className="min-w-[150px]">
        <div className="flex items-baseline justify-between gap-2 text-[11px]">
          <span className="font-medium uppercase tracking-wider text-ink-3">Trial budget</span>
          <span className={cx('truncate font-semibold', tone.text)}>{tone.label}</span>
        </div>
        <div
          className="mt-1 h-1.5 overflow-hidden rounded-full bg-accent/15"
          role="progressbar"
          aria-label="Trial budget used on this data window"
          aria-valuemin={0}
          aria-valuemax={status.limit}
          aria-valuenow={status.used}
        >
          <div className={cx('h-full rounded-full transition-[width] duration-150 ease-linear', tone.bar)} style={{ width: `${status.share * 100}%` }} />
        </div>
        <div className="num mt-0.5 text-[11px] text-ink-3">
          <span className="font-semibold text-ink-2">{fmtInt(status.used)}</span> / {fmtInt(status.limit)} trials
          {status.extensions > 0 ? ` · extended ${status.extensions}×` : ''}
        </div>
      </div>
      {status.level === 'exhausted' && (
        <button
          type="button"
          onClick={onExtend}
          title={`Allow ${BUDGET_EXTENSION.toLocaleString('en-US')} more trials on this data window. Earlier trials keep counting, so the luck hurdle keeps rising.`}
          className="inline-flex h-8 shrink-0 items-center rounded-md border border-warn/60 px-2 text-xs font-semibold text-warn transition-colors hover:bg-warn/15"
        >
          Extend +{fmtInt(BUDGET_EXTENSION)}
        </button>
      )}
    </div>
  );
}

/** Confirmation dialog of the reset: what it clears and what that does to multiple-testing tracking. */
function ResetDialog({ onCancel, onConfirm, running }: { onCancel: () => void; onConfirm: () => void; running: boolean }): JSX.Element {
  const titleId = useId();
  const textId = useId();
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-base/70 p-4 backdrop-blur-sm" onMouseDown={onCancel}>
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={textId}
        onMouseDown={(event) => event.stopPropagation()}
        className="w-full max-w-md rounded-xl border border-line-strong bg-panel p-5 shadow-2xl"
      >
        <div className="flex items-start gap-3">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-warn/15 text-warn">
            <TriangleAlert size={18} />
          </span>
          <div className="min-w-0">
            <h2 id={titleId} className="text-base font-semibold text-ink">
              Reset runs and clear the cache?
            </h2>
            <div id={textId} className="mt-1.5 space-y-2 text-sm text-ink-2">
              <p>
                {running ? 'The running scan is stopped and ' : ''}Every cached strategy, the leaderboard, the elite kept across restarts, the
                trial and restart counters and the holdout reveal log are deleted from this browser.
              </p>
              <p className="rounded-lg border border-warn/35 bg-warn/10 px-3 py-2 text-xs text-warn">
                Clearing the history resets multiple-testing tracking: earlier trials stop raising the luck hurdle of the deflated Sharpe, and
                holdouts you have already seen will look untouched again. Only reset when you start genuinely new research.
              </p>
              <p className="text-xs text-ink-3">Your settings and saved tickers are kept.</p>
            </div>
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            className="inline-flex h-9 items-center rounded-lg border border-line px-4 text-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-down/90 px-4 text-sm font-semibold text-white transition-colors hover:bg-down"
          >
            <RotateCcw size={14} /> Reset everything
          </button>
        </div>
      </div>
    </div>
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
  stage,
}: TopActionBarProps): JSX.Element {
  const [confirming, setConfirming] = useState(false);
  const busy = status === 'loading' || status === 'optimizing';
  return (
    <header className="sticky top-0 z-30 -mx-4 border-b border-line bg-base/90 px-4 py-2.5 backdrop-blur sm:-mx-6 sm:px-6">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
        <div className="flex items-center gap-2.5">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-accent/40 bg-accent/15 text-accent-strong">
            <Activity size={18} />
          </div>
          <div className="leading-tight">
            <h1 className="text-base font-semibold text-ink">StrategyFinder</h1>
            <p className="hidden text-[11px] text-ink-3 2xl:block">Discover, stress-test and export robust strategies</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-down/50 bg-down/10 px-4 text-sm font-semibold text-down transition-colors hover:bg-down/20"
            >
              {status === 'loading' ? <Loader2 size={15} className="animate-spin" /> : <Square size={13} fill="currentColor" />}
              {status === 'loading' ? 'Loading data… Stop' : 'Stop scan'}
            </button>
          ) : (
            <button
              type="button"
              onClick={onRun}
              disabled={runBlocked !== null}
              title={runBlocked ?? `Start a discovery scan with ${presetTitle ? `the "${presetTitle}" preset` : 'the current custom settings'}`}
              className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-5 text-sm font-semibold text-white shadow-[0_0_0_1px_rgba(90,156,236,0.4)] transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Play size={15} fill="currentColor" /> 1-Click Run
            </button>
          )}
          <span className="max-w-[220px] truncate text-xs text-ink-3" title={presetTitle ?? 'Custom settings'}>
            {presetTitle ? (
              <>
                Preset: <span className="text-ink-2">{presetTitle}</span>
              </>
            ) : (
              <span className="text-ink-2">Custom settings</span>
            )}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <TrialBudgetMeter budget={budget} onExtend={onExtendBudget} />
          <button
            type="button"
            onClick={() => setConfirming(true)}
            title="Stop any scan and delete every cached run, strategy and trial / holdout record from this browser"
            className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line px-3 text-xs font-semibold text-ink-2 transition-colors hover:border-down/50 hover:bg-down/10 hover:text-down"
          >
            <RotateCcw size={13} /> Reset Runs &amp; Clear Cache
          </button>
        </div>

        <nav aria-label="Workflow" className="flex flex-wrap items-center gap-1 lg:ml-auto">
          {STEPS.map((step, index) => (
            <span key={step.n} className="inline-flex items-center gap-1">
              <a
                href={`#${step.target}`}
                aria-current={stage === step.n ? 'step' : undefined}
                className={cx(
                  'inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-xs transition-colors',
                  stage === step.n
                    ? 'border-accent/60 bg-accent/15 font-semibold text-ink'
                    : stage > step.n
                      ? 'border-line text-ink-2 hover:text-ink'
                      : 'border-line text-ink-3 hover:text-ink-2',
                )}
              >
                <span className="num">{step.n}</span> {step.label}
              </a>
              {index < STEPS.length - 1 && (
                <span aria-hidden="true" className="text-ink-3">
                  ›
                </span>
              )}
            </span>
          ))}
        </nav>
      </div>
      {confirming && (
        <ResetDialog
          running={busy}
          onCancel={() => setConfirming(false)}
          onConfirm={() => {
            setConfirming(false);
            onReset();
          }}
        />
      )}
    </header>
  );
}
