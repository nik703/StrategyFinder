/**
 * Strategy Breakdown drawer: a human-readable explanation of the selected
 * strategy – indicators and exact parameters, plain-English entry and exit
 * rules, in-sample vs. out-of-sample metrics, and a Pine Script v5 export.
 */

import { Check, CircleCheck, CircleX, Copy, Info, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  describeExit,
  describeLongEntry,
  describeShortEntry,
  indicatorBadges,
  listIndicators,
  type RuleDescription,
} from '../engine/describe';
import { generatePineScript } from '../engine/pine';
import type { RunContext } from '../lib/config';
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from '../lib/format';
import { expectedMaxSharpe } from '../lib/stats';
import { IN_SAMPLE_FRACTION, type BenchmarkResult, type PerformanceMetrics, type StrategyResult } from '../types/trading';
import { cx, Pill } from './ui';

interface StrategyModalProps {
  strategy: StrategyResult;
  /** Current leaderboard rank, or null when the strategy has since dropped out of the Top 25. */
  rank: number | null;
  benchmark: BenchmarkResult;
  context: RunContext;
  /** Total number of strategies the optimiser has evaluated so far. */
  testedCount: number;
  /** Length of the out-of-sample segment in years. */
  oosYears: number;
  onClose: () => void;
}

/* -------------------------------------------------------------------------- */

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }): JSX.Element {
  return (
    <section className="border-t border-line px-5 py-5">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-2">{title}</h3>
        {hint && <span className="text-[11px] text-ink-3">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

function RuleList({ rule, tone }: { rule: RuleDescription; tone: 'up' | 'down' | 'neutral' }): JSX.Element {
  const dot = tone === 'up' ? 'bg-up' : tone === 'down' ? 'bg-down' : 'bg-ink-3';
  return (
    <div>
      <p className="text-sm text-ink-2">{rule.intro}</p>
      <ul className="mt-2 space-y-2">
        {rule.bullets.map((bullet) => (
          <li key={bullet.formula} className="flex gap-2.5">
            <span aria-hidden="true" className={cx('mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full', dot)} />
            <div className="min-w-0">
              <div className="text-sm text-ink">{bullet.text}</div>
              <code className="num mt-0.5 block text-xs text-ink-3">{bullet.formula}</code>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function StatTile({ label, value, tone }: { label: string; value: string; tone?: string }): JSX.Element {
  return (
    <div className="rounded-lg border border-line bg-raised/50 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wider text-ink-3">{label}</div>
      <div className={cx('num mt-0.5 text-base font-semibold', tone ?? 'text-ink')}>{value}</div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

interface MetricRow {
  label: string;
  format: (m: PerformanceMetrics) => string;
  tone?: (m: PerformanceMetrics) => string;
  /** Buy & Hold value for the same segment, when it is meaningful. */
  bench?: (m: PerformanceMetrics) => string;
}

const METRIC_ROWS: MetricRow[] = [
  { label: 'Net PnL', format: (m) => fmtUsd(m.netPnl, true), tone: (m) => toneClass(m.netPnl), bench: (m) => fmtUsd(m.netPnl, true) },
  { label: 'ROI', format: (m) => fmtPct(m.roi, true), tone: (m) => toneClass(m.roi), bench: (m) => fmtPct(m.roi, true) },
  { label: 'CAGR', format: (m) => fmtPct(m.cagr, true), tone: (m) => toneClass(m.cagr), bench: (m) => fmtPct(m.cagr, true) },
  { label: 'Max drawdown', format: (m) => fmtPct(m.maxDrawdown), bench: (m) => fmtPct(m.maxDrawdown) },
  { label: 'Sharpe ratio', format: (m) => fmtNum(m.sharpe), bench: (m) => fmtNum(m.sharpe) },
  { label: 'Sortino ratio', format: (m) => fmtNum(m.sortino), bench: (m) => fmtNum(m.sortino) },
  { label: 'Win rate', format: (m) => fmtPct(m.winRate) },
  { label: 'Profit factor', format: (m) => fmtNum(m.profitFactor) },
  { label: 'Trades', format: (m) => fmtInt(m.trades) },
  { label: 'Time in market', format: (m) => fmtPct(m.exposure, false, 0) },
];

interface Check {
  label: string;
  detail: string;
  pass: boolean;
}

function robustnessChecks(strategy: StrategyResult, benchmark: BenchmarkResult): Check[] {
  const { inSample: is, outOfSample: oos } = strategy;
  return [
    {
      label: 'Profitable out-of-sample after costs',
      detail: `${fmtUsd(oos.netPnl, true)} (${fmtPct(oos.roi, true)})`,
      pass: oos.netPnl > 0,
    },
    {
      label: 'Beats Buy & Hold out-of-sample',
      detail: `${fmtPct(oos.roi, true)} vs ${fmtPct(benchmark.outOfSample.roi, true)}`,
      pass: oos.roi > benchmark.outOfSample.roi,
    },
    {
      label: 'Sharpe keeps at least half of its in-sample level',
      detail: `${fmtNum(oos.sharpe)} out-of-sample vs ${fmtNum(is.sharpe)} in-sample`,
      pass: is.sharpe > 0 ? oos.sharpe >= 0.5 * is.sharpe : oos.sharpe > 0,
    },
    {
      label: 'Drawdown no worse than 1.5× in-sample',
      detail: `${fmtPct(oos.maxDrawdown)} out-of-sample vs ${fmtPct(is.maxDrawdown)} in-sample`,
      pass: oos.maxDrawdown <= 1.5 * Math.max(is.maxDrawdown, 5),
    },
    {
      label: 'Enough out-of-sample trades to judge (≥ 6)',
      detail: `${fmtInt(oos.trades)} trades`,
      pass: oos.trades >= 6,
    },
  ];
}

/* -------------------------------------------------------------------------- */

async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

export default function StrategyModal({
  strategy,
  rank,
  benchmark,
  context,
  testedCount,
  oosYears,
  onClose,
}: StrategyModalProps): JSX.Element {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [showCode, setShowCode] = useState(false);
  const { genome } = strategy;

  // Close on Escape, lock background scroll, and move focus into the dialog.
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = overflow;
      previous?.focus?.();
    };
  }, [onClose]);

  useEffect(() => {
    setCopyState('idle');
  }, [strategy.id]);

  const indicators = useMemo(() => listIndicators(genome), [genome]);
  const longEntry = useMemo(() => describeLongEntry(genome), [genome]);
  const shortEntry = useMemo(() => describeShortEntry(genome), [genome]);
  const exit = useMemo(() => describeExit(genome, context.settings), [genome, context.settings]);
  const checks = useMemo(() => robustnessChecks(strategy, benchmark), [strategy, benchmark]);
  const passed = checks.filter((check) => check.pass).length;
  const luckSharpe = useMemo(() => expectedMaxSharpe(testedCount, oosYears), [testedCount, oosYears]);

  const pine = useMemo(
    () =>
      generatePineScript(strategy, {
        label: context.label,
        interval: context.interval,
        settings: context.settings,
        components: context.components,
        units: context.units,
        windowStart: context.windowStart,
        windowEnd: context.windowEnd,
        bars: context.bars,
        benchmarkRoi: benchmark.full.roi,
        simulatedData: context.simulated,
      }),
    [strategy, context, benchmark.full.roi],
  );

  const handleCopy = useCallback(async () => {
    const ok = await copyText(pine);
    setCopyState(ok ? 'copied' : 'failed');
    if (!ok) setShowCode(true);
    window.setTimeout(() => setCopyState('idle'), 2500);
  }, [pine]);

  const verdict =
    passed === checks.length
      ? { label: 'Holds up out-of-sample', tone: 'up' as const }
      : passed >= 3
        ? { label: 'Partially holds up out-of-sample', tone: 'warn' as const }
        : { label: 'Degrades out-of-sample', tone: 'down' as const };

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="strategy-modal-title">
      <div className="animate-fade absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick={onClose} aria-hidden="true" />

      <div className="animate-drawer absolute inset-y-0 right-0 flex w-full max-w-[760px] flex-col border-l border-line-strong bg-panel shadow-2xl">
        {/* Header */}
        <header className="flex items-start justify-between gap-4 px-5 py-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id="strategy-modal-title" className="num text-lg font-semibold text-ink">
                Strategy {strategy.id}
              </h2>
              {rank !== null ? <Pill tone="accent">Rank #{rank}</Pill> : <Pill>No longer in Top 25</Pill>}
              <Pill tone={genome.exit.mode === 'static' ? 'neutral' : 'accent'}>{exit.mode}</Pill>
              <Pill>{genome.shortEntry ? 'Long/Short' : 'Long-Only'}</Pill>
            </div>
            <p className="num mt-1 truncate text-xs text-ink-3">
              {context.label} · {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)} · found at
              iteration {fmtInt(strategy.foundAt)}
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close strategy breakdown"
            className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-line text-ink-2 transition-colors hover:bg-hover hover:text-ink"
          >
            <X size={16} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {/* Headline numbers */}
          <div className="grid grid-cols-2 gap-2 px-5 pb-5 sm:grid-cols-3">
            <StatTile label="Net PnL" value={fmtUsd(strategy.full.netPnl, true)} tone={toneClass(strategy.full.netPnl)} />
            <StatTile label="ROI" value={fmtPct(strategy.full.roi, true)} tone={toneClass(strategy.full.roi)} />
            <StatTile label="Alpha vs. B&H" value={fmtPct(strategy.alpha, true)} tone={toneClass(strategy.alpha)} />
            <StatTile label="Max drawdown" value={fmtPct(strategy.full.maxDrawdown)} />
            <StatTile label="Sharpe" value={fmtNum(strategy.full.sharpe)} />
            <StatTile label="Trades" value={fmtInt(strategy.full.trades)} />
          </div>

          {context.simulated && (
            <div className="mx-5 mb-5 flex gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2 text-xs text-warn">
              <Info size={14} className="mt-0.5 shrink-0" />
              <span>
                This run used simulated fallback candles, not real market data. Treat every number here as an illustration and re-run on
                live data before drawing conclusions.
              </span>
            </div>
          )}

          <Section title="Indicators & exact parameters" hint={`${indicators.length} indicator${indicators.length === 1 ? '' : 's'}`}>
            <div className="mb-3 flex flex-wrap gap-1">
              {indicatorBadges(genome).map((badge) => (
                <Pill key={badge}>{badge}</Pill>
              ))}
            </div>
            <div className="overflow-hidden rounded-lg border border-line">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line bg-raised/50 text-left text-[11px] uppercase tracking-wider text-ink-3">
                    <th scope="col" className="px-3 py-2 font-medium">Indicator</th>
                    <th scope="col" className="px-3 py-2 font-medium">Parameters</th>
                    <th scope="col" className="px-3 py-2 font-medium">Used for</th>
                  </tr>
                </thead>
                <tbody>
                  {indicators.map((usage) => (
                    <tr key={`${usage.name}|${usage.params}`} className="border-b border-line/70 last:border-b-0">
                      <td className="px-3 py-2 font-medium text-ink">{usage.name}</td>
                      <td className="num px-3 py-2 text-ink-2">{usage.params}</td>
                      <td className="px-3 py-2 text-ink-3">{usage.roles.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Entry rules">
            <div className="space-y-4">
              <div>
                <div className="mb-1.5 text-xs font-semibold text-up">Long entry</div>
                <RuleList rule={longEntry} tone="up" />
              </div>
              <div>
                <div className="mb-1.5 text-xs font-semibold text-down">Short entry</div>
                {shortEntry ? (
                  <RuleList rule={shortEntry} tone="down" />
                ) : (
                  <p className="text-sm text-ink-3">None – this strategy was optimised in Long-Only mode and never sells short.</p>
                )}
              </div>
            </div>
          </Section>

          <Section title="Exit & risk management rules" hint={`SL/TP mode: ${exit.mode}`}>
            <p className="text-sm text-ink-2">{exit.headline}</p>
            <dl className="mt-3 space-y-2.5">
              {exit.bullets.map((bullet) => (
                <div key={bullet.title} className="flex gap-2.5">
                  <span aria-hidden="true" className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent-strong" />
                  <div className="min-w-0 text-sm">
                    <dt className="inline font-semibold text-ink">{bullet.title}. </dt>
                    <dd className="inline text-ink-2">{bullet.text}</dd>
                  </div>
                </div>
              ))}
            </dl>
            {exit.longSignalExit && (
              <div className="mt-4 space-y-4 rounded-lg border border-line bg-raised/40 p-3">
                <RuleList rule={exit.longSignalExit} tone="neutral" />
                {exit.shortSignalExit && <RuleList rule={exit.shortSignalExit} tone="neutral" />}
              </div>
            )}
          </Section>

          <Section
            title="In-sample vs. out-of-sample"
            hint={`First ${Math.round(IN_SAMPLE_FRACTION * 100)}% of candles vs. last ${Math.round((1 - IN_SAMPLE_FRACTION) * 100)}%`}
          >
            <div className="overflow-x-auto rounded-lg border border-line">
              <table className="num w-full text-sm">
                <thead>
                  <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                    <th scope="col" className="px-3 py-2 text-left font-medium">Metric</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">In-sample</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Out-of-sample</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Full period</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">B&amp;H out-of-sample</th>
                  </tr>
                </thead>
                <tbody>
                  {METRIC_ROWS.map((row) => (
                    <tr key={row.label} className="border-b border-line/70 last:border-b-0">
                      <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink-2">{row.label}</th>
                      <td className={cx('px-3 py-1.5 text-right', row.tone?.(strategy.inSample) ?? 'text-ink')}>
                        {row.format(strategy.inSample)}
                      </td>
                      <td className={cx('px-3 py-1.5 text-right font-semibold', row.tone?.(strategy.outOfSample) ?? 'text-ink')}>
                        {row.format(strategy.outOfSample)}
                      </td>
                      <td className={cx('px-3 py-1.5 text-right', row.tone?.(strategy.full) ?? 'text-ink')}>
                        {row.format(strategy.full)}
                      </td>
                      <td className="px-3 py-1.5 text-right text-ink-3">{row.bench ? row.bench(benchmark.outOfSample) : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 rounded-lg border border-line bg-raised/40 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <Pill tone={verdict.tone}>{verdict.label}</Pill>
                <span className="num text-xs text-ink-3">
                  {passed} of {checks.length} robustness checks passed
                </span>
              </div>
              <ul className="space-y-1.5">
                {checks.map((check) => (
                  <li key={check.label} className="flex items-start gap-2 text-sm">
                    {check.pass ? (
                      <CircleCheck size={15} className="mt-0.5 shrink-0 text-up" aria-label="Passed" />
                    ) : (
                      <CircleX size={15} className="mt-0.5 shrink-0 text-down" aria-label="Failed" />
                    )}
                    <span className="text-ink-2">
                      {check.label} <span className="num text-ink-3">· {check.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-3 flex gap-2 rounded-lg border border-line px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              <Info size={14} className="mt-0.5 shrink-0 text-ink-3" />
              <p>
                <span className="font-semibold text-ink">Selection-bias check.</span> The ranking score uses the out-of-sample segment, and{' '}
                <span className="num">{fmtInt(testedCount)}</span> strategies were tested, so these figures are a validation result, not an
                untouched holdout. The best of that many skill-less strategies would be expected to show an out-of-sample Sharpe of roughly{' '}
                <span className="num font-semibold text-ink">{fmtNum(luckSharpe, 1)}</span> by luck alone (upper estimate); this strategy
                shows <span className="num font-semibold text-ink">{fmtNum(strategy.outOfSample.sharpe)}</span>
                {strategy.outOfSample.sharpe > luckSharpe ? ', which is above that bar' : ', which is within that range'}. Forward-test
                on new data before risking capital.
              </p>
            </div>
          </Section>

          <Section title="Export to TradingView" hint="Pine Script v5 strategy">
            <p className="text-sm text-ink-2">
              Copy the script, open the Pine Editor on a <span className="num font-medium text-ink">{context.interval}</span> chart of{' '}
              <span className="num font-medium text-ink">{context.isBasket ? 'the basket (see the script header)' : context.label}</span>,
              paste it and press <em>Add to chart</em>. The script reproduces these exact rules, exits, costs and next-bar-open
              execution.
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => void handleCopy()}
                className={cx(
                  'inline-flex h-9 items-center gap-2 rounded-lg px-4 text-sm font-semibold transition-colors',
                  copyState === 'copied' ? 'bg-up/15 text-up ring-1 ring-up/40' : 'bg-accent text-white hover:bg-accent-strong',
                )}
              >
                {copyState === 'copied' ? <Check size={15} /> : <Copy size={15} />}
                {copyState === 'copied' ? 'Copied to clipboard' : 'Copy Pine Script v5'}
              </button>
              <button
                type="button"
                onClick={() => setShowCode((value) => !value)}
                aria-expanded={showCode}
                className="inline-flex h-9 items-center rounded-lg border border-line px-3 text-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink"
              >
                {showCode ? 'Hide code' : 'View code'}
              </button>
              <span role="status" aria-live="polite" className="text-xs text-down">
                {copyState === 'failed' ? 'Clipboard blocked – select the code below and copy it manually.' : ''}
              </span>
            </div>
            {showCode && (
              <pre
                tabIndex={0}
                aria-label="Generated Pine Script"
                className="num mt-3 max-h-[420px] overflow-auto rounded-lg border border-line bg-base p-3 text-[11.5px] leading-relaxed text-ink-2"
              >
                <code>{pine}</code>
              </pre>
            )}
          </Section>

          <p className="border-t border-line px-5 py-4 text-[11px] leading-relaxed text-ink-3">
            Backtests are hypothetical and ignore liquidity, funding, borrow costs and taxes. Nothing here is financial advice.
          </p>
        </div>
      </div>
    </div>
  );
}
