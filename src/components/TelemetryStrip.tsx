/**
 * Live telemetry: engine status, iteration counter, stagnation progress
 * (X / N iterations without improvement), best strategy vs. Buy & Hold
 * and the market-data source.
 */

import { CircleCheck, CircleDot, Database, Loader2, OctagonX, TriangleAlert } from 'lucide-react';
import type { ReactNode } from 'react';
import type { RunContext } from '../lib/config';
import { fmtDate, fmtDuration, fmtInt, fmtPct, toneClass } from '../lib/format';
import { type EngineStatus, type ProgressSnapshot } from '../types/trading';
import { cx, Panel } from './ui';

interface TelemetryStripProps {
  status: EngineStatus;
  telemetry: ProgressSnapshot;
  benchmarkRoi: number | null;
  context: RunContext | null;
  /** Stagnation limit of the current (or next) run. */
  maxStagnation: number;
}

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

export default function TelemetryStrip({ status, telemetry, benchmarkRoi, context, maxStagnation }: TelemetryStripProps): JSX.Element {
  const badge = STATUS[status];
  const badgeLabel = status === 'converged' ? `Converged at ${fmtInt(maxStagnation)}` : badge.label;
  const stagnation = Math.min(telemetry.stagnation, maxStagnation);
  const stagnationPct = (stagnation / maxStagnation) * 100;
  const hasRun = telemetry.iteration > 0;
  const alpha = telemetry.bestRoi !== null && benchmarkRoi !== null ? telemetry.bestRoi - benchmarkRoi : null;

  return (
    <Panel className="grid grid-cols-1 divide-y divide-line sm:grid-cols-2 sm:divide-y-0 xl:grid-cols-[1.05fr_0.8fr_1.5fr_1.45fr_1.4fr] xl:divide-x">
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

      <Tile label="Total iterations">
        <div className="num text-2xl font-semibold leading-7 text-ink">{fmtInt(telemetry.iteration)}</div>
        <div className="mt-1 text-xs text-ink-3">strategies evaluated</div>
      </Tile>

      <Tile label="Stagnation progress">
        <div className="flex items-baseline gap-1.5 whitespace-nowrap">
          <span className="num text-2xl font-semibold leading-7 text-ink">{fmtInt(stagnation)}</span>
          <span className="num text-sm text-ink-3">/ {fmtInt(maxStagnation)}</span>
        </div>
        <div className="mt-1 truncate text-xs text-ink-3">iterations without improvement</div>
        <div
          className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-accent/15"
          role="progressbar"
          aria-label="Stagnation progress"
          aria-valuemin={0}
          aria-valuemax={maxStagnation}
          aria-valuenow={stagnation}
        >
          <div
            className={cx(
              'h-full rounded-full transition-[width] duration-150 ease-linear',
              status === 'converged' ? 'bg-up' : stagnationPct >= 75 ? 'bg-warn' : 'bg-accent',
            )}
            style={{ width: `${stagnationPct}%` }}
          />
        </div>
      </Tile>

      <Tile label="Best strategy ROI vs. Buy & Hold ROI">
        <div className="flex items-baseline gap-3">
          <div>
            <span className={cx('num text-2xl font-semibold leading-7', toneClass(telemetry.bestRoi))}>
              {fmtPct(telemetry.bestRoi, true)}
            </span>
          </div>
          <span className="text-xs text-ink-3">vs</span>
          <div>
            <span className="num text-lg font-medium leading-7 text-ink-2">{fmtPct(benchmarkRoi, true)}</span>
          </div>
        </div>
        <div className="num mt-1 text-xs text-ink-3">
          {alpha === null ? (
            'Alpha appears once a strategy is found'
          ) : (
            <>
              Alpha <span className={cx('font-medium', toneClass(alpha))}>{fmtPct(alpha, true)}</span> · best fitness{' '}
              <span className="text-ink-2">{telemetry.bestFitness === null ? '–' : telemetry.bestFitness.toFixed(2)}</span>
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
                Robust universe: {context.universe.length} assets
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
    </Panel>
  );
}
