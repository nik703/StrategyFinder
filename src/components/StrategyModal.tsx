/**
 * Strategy Breakdown drawer: a human-readable explanation of the selected
 * strategy – indicators and exact parameters, plain-English entry and exit
 * rules, in-sample vs. out-of-sample metrics, and a Pine Script v5 export.
 */

import { Check, CircleCheck, CircleX, Copy, Info, Loader2, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  ARCHETYPE_LABEL,
  describeArchetype,
  describeExit,
  describeLongEntry,
  describeShortEntry,
  indicatorBadges,
  listIndicators,
  type RuleDescription,
} from '../engine/describe';
import { FREE_PARAMETERS, MAX_PARAMETERS } from '../engine/genome';
import { generatePineScript } from '../engine/pine';
import type { RunContext } from '../lib/config';
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from '../lib/format';
import { sharpeConfidenceInterval } from '../lib/stats';
import type { BenchmarkResult, PerformanceMetrics, RobustnessReport, StrategyResult, Verdict } from '../types/trading';
import { VerdictPill } from './Leaderboard';
import { cx, Pill } from './ui';

interface StrategyModalProps {
  strategy: StrategyResult;
  /** Current leaderboard rank, or null when the strategy has since dropped out of the Top 25. */
  rank: number | null;
  benchmark: BenchmarkResult;
  context: RunContext;
  /** Total number of strategies the optimiser has evaluated so far. */
  testedCount: number;
  barsPerYear: number;
  /** Result of an on-demand `analyze` request; overrides the report carried by the strategy. */
  analysis: { report: RobustnessReport; verdict: Verdict | null } | null;
  /** True while an analysis request is in flight. */
  analysisPending: boolean;
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
  title?: string;
  format: (m: PerformanceMetrics) => string;
  tone?: (m: PerformanceMetrics) => string;
  /** False for trade statistics that do not apply to Buy & Hold. */
  bench?: boolean;
}

const METRIC_ROWS: MetricRow[] = [
  { label: 'Net PnL', format: (m) => fmtUsd(m.netPnl, true), tone: (m) => toneClass(m.netPnl), bench: true },
  { label: 'ROI', format: (m) => fmtPct(m.roi, true), tone: (m) => toneClass(m.roi), bench: true },
  { label: 'CAGR', format: (m) => fmtPct(m.cagr, true), tone: (m) => toneClass(m.cagr), bench: true },
  { label: 'Max drawdown', title: 'Measured from the highest close to the worst intrabar mark', format: (m) => fmtPct(m.maxDrawdown), bench: true },
  { label: 'Sharpe ratio', format: (m) => fmtNum(m.sharpe), bench: true },
  { label: 'Sortino ratio', format: (m) => fmtNum(m.sortino), bench: true },
  { label: 'Trades', format: (m) => fmtInt(m.trades) },
  { label: 'Win rate', format: (m) => fmtPct(m.winRate) },
  { label: 'Break-even win rate', title: 'Win rate needed to break even at this payoff ratio', format: (m) => (m.trades > 0 ? fmtPct(m.breakevenWinRate) : '–') },
  { label: 'Payoff ratio', title: 'Average win ÷ average loss', format: (m) => (m.trades > 0 ? fmtNum(m.payoffRatio) : '–') },
  { label: 'Profit factor', format: (m) => (m.trades > 0 ? fmtNum(m.profitFactor) : '–') },
  { label: 'Max consecutive losses', format: (m) => fmtInt(m.maxConsecLosses) },
  { label: 'Trade-return skew', title: 'Negative skew: many small wins and a few large losses', format: (m) => (m.trades > 2 ? fmtNum(m.tradeSkew, 2, true) : '–') },
  { label: 'Ambiguous candles', title: 'Trades closed on a candle that touched both the stop and the target (the stop is assumed to fill first)', format: (m) => fmtInt(m.ambiguousTrades) },
  { label: 'Time in market', format: (m) => fmtPct(m.exposure, false, 0) },
];

function TestRow({ pass, label, detail }: { pass: boolean; label: string; detail: string }): JSX.Element {
  return (
    <li className="flex items-start gap-2 text-sm">
      {pass ? (
        <CircleCheck size={15} className="mt-0.5 shrink-0 text-up" aria-label="Passed" />
      ) : (
        <CircleX size={15} className="mt-0.5 shrink-0 text-down" aria-label="Failed" />
      )}
      <span className="text-ink-2">
        <span className="font-medium text-ink">{label}</span> <span className="num text-ink-3">· {detail}</span>
      </span>
    </li>
  );
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
  barsPerYear,
  analysis,
  analysisPending,
  onClose,
}: StrategyModalProps): JSX.Element {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [showCode, setShowCode] = useState(false);
  const { genome } = strategy;
  const report = analysis?.report ?? strategy.robustness;
  const verdict = analysis?.verdict ?? strategy.verdict;
  const revealed = strategy.holdout !== null;

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
  const holdoutCi = useMemo(
    () => (strategy.holdout ? sharpeConfidenceInterval(strategy.holdout, barsPerYear) : null),
    [strategy.holdout, barsPerYear],
  );

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
        benchmarkRoi: benchmark.search.roi,
        venue: context.venue,
        simulatedData: context.simulated,
      }),
    [strategy, context, benchmark.search.roi],
  );

  const handleCopy = useCallback(async () => {
    const ok = await copyText(pine);
    setCopyState(ok ? 'copied' : 'failed');
    if (!ok) setShowCode(true);
    window.setTimeout(() => setCopyState('idle'), 2500);
  }, [pine]);

  const ambiguousShare = strategy.search.trades > 0 ? strategy.search.ambiguousTrades / strategy.search.trades : 0;
  const benchSegment = benchmark.holdout ?? benchmark.validation;

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="strategy-modal-title">
      <div className="animate-fade absolute inset-0 bg-black/60 backdrop-blur-[2px]" onClick={onClose} aria-hidden="true" />

      <div className="animate-drawer absolute inset-y-0 right-0 flex w-full max-w-[820px] flex-col border-l border-line-strong bg-panel shadow-2xl">
        {/* Header */}
        <header className="flex items-start justify-between gap-4 px-5 py-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id="strategy-modal-title" className="num text-lg font-semibold text-ink">
                Strategy {strategy.id}
              </h2>
              {rank !== null ? <Pill tone="accent">Rank #{rank}</Pill> : <Pill>No longer in Top 25</Pill>}
              <Pill tone="bench">{ARCHETYPE_LABEL[genome.archetype]}</Pill>
              <Pill tone={genome.exit.mode === 'static' ? 'neutral' : 'accent'}>{exit.mode}</Pill>
              <Pill>{genome.shortEntry ? 'Long/Short' : 'Long-Only'}</Pill>
              <Pill tone={strategy.params > FREE_PARAMETERS ? 'warn' : 'neutral'}>{strategy.params} params</Pill>
              {verdict !== null && <VerdictPill verdict={verdict} />}
            </div>
            <p className="num mt-1 truncate text-xs text-ink-3">
              {context.venue} · {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)} · found at
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
          {/* Headline numbers: everything the search was allowed to see (train + validation). */}
          <div className="grid grid-cols-2 gap-2 px-5 pb-2 sm:grid-cols-3">
            <StatTile label="Net PnL" value={fmtUsd(strategy.search.netPnl, true)} tone={toneClass(strategy.search.netPnl)} />
            <StatTile label="ROI" value={fmtPct(strategy.search.roi, true)} tone={toneClass(strategy.search.roi)} />
            <StatTile label="Alpha vs. B&H" value={fmtPct(strategy.alpha, true)} tone={toneClass(strategy.alpha)} />
            <StatTile label="Max drawdown" value={fmtPct(strategy.search.maxDrawdown)} />
            <StatTile label="Sharpe" value={fmtNum(strategy.search.sharpe)} />
            <StatTile label="Trades" value={fmtInt(strategy.search.trades)} />
          </div>
          <p className="px-5 pb-5 text-[11px] text-ink-3">Headline figures cover train + validation, the 80% of history the search could see.</p>

          {context.simulated && (
            <div className="mx-5 mb-5 flex gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2 text-xs text-warn">
              <Info size={14} className="mt-0.5 shrink-0" />
              <span>
                This run used simulated fallback candles, not real market data. Treat every number here as an illustration and re-run on
                live data before drawing conclusions.
              </span>
            </div>
          )}

          <Section title="The idea" hint={ARCHETYPE_LABEL[genome.archetype]}>
            <p className="text-sm leading-relaxed text-ink-2">{describeArchetype(genome)}</p>
          </Section>

          <Section
            title="Indicators & exact parameters"
            hint={`${strategy.params} tunable values (cap ${MAX_PARAMETERS}, penalty above ${FREE_PARAMETERS})`}
          >
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
                <div className="mb-1.5 text-xs font-semibold text-down">Short entry (mirror of the long rules)</div>
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

          <Section title="Train vs. validation vs. holdout" hint="First 60% · next 20% · last 20% of candles">
            <div className="overflow-x-auto rounded-lg border border-line">
              <table className="num w-full text-sm">
                <thead>
                  <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                    <th scope="col" className="px-3 py-2 text-left font-medium">Metric</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Train</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Validation</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Holdout</th>
                    <th scope="col" className="whitespace-nowrap px-3 py-2 text-right font-medium">
                      B&amp;H {benchmark.holdout ? 'holdout' : 'validation'}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {METRIC_ROWS.map((row) => (
                    <tr key={row.label} className="border-b border-line/70 last:border-b-0">
                      <th scope="row" title={row.title} className="whitespace-nowrap px-3 py-1.5 text-left font-normal text-ink-2">
                        {row.label}
                      </th>
                      <td className={cx('px-3 py-1.5 text-right', row.tone?.(strategy.train) ?? 'text-ink')}>{row.format(strategy.train)}</td>
                      <td className={cx('px-3 py-1.5 text-right', row.tone?.(strategy.validation) ?? 'text-ink')}>
                        {row.format(strategy.validation)}
                      </td>
                      <td className={cx('px-3 py-1.5 text-right font-semibold', strategy.holdout ? (row.tone?.(strategy.holdout) ?? 'text-ink') : 'font-normal text-ink-3')}>
                        {strategy.holdout ? row.format(strategy.holdout) : 'sealed'}
                      </td>
                      <td className="px-3 py-1.5 text-right text-ink-3">{row.bench ? row.format(benchSegment) : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {!revealed && (
              <p className="mt-2 flex items-start gap-2 text-xs text-ink-3">
                <Info size={14} className="mt-0.5 shrink-0" />
                The holdout is not simulated while the search is running. It is evaluated once, when the run stops or converges.
              </p>
            )}

            {ambiguousShare > 0.1 && (
              <div className="mt-3 flex gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2 text-xs text-warn">
                <TriangleAlert size={14} className="mt-0.5 shrink-0" />
                <span>
                  <span className="num">{fmtPct(ambiguousShare * 100, false, 0)}</span> of trades closed on a candle that touched both the
                  stop and the target. The stop is assumed to fill first, but the real order of events inside those candles is unknown – a
                  shorter timeframe would settle it.
                </span>
              </div>
            )}

            <div className="mt-3 flex gap-2 rounded-lg border border-line px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              <Info size={14} className="mt-0.5 shrink-0 text-ink-3" />
              <div>
                <p>
                  <span className="font-semibold text-ink">Selection bias.</span>{' '}
                  <span className="num">{fmtInt(testedCount)}</span> strategies have been tried on this data. Deflated Sharpe Ratio of the
                  validation segment: <span className="num font-semibold text-ink">{fmtPct(strategy.dsr * 100, false, 1)}</span> – the
                  probability that the validation Sharpe of <span className="num">{fmtNum(strategy.validation.sharpe)}</span> is real
                  rather than the luckiest of that many attempts. Below 95% it is not statistically distinguishable from luck.
                </p>
                {strategy.holdout && holdoutCi ? (
                  <p className="mt-1.5">
                    <span className="font-semibold text-ink">Holdout.</span> Sharpe{' '}
                    <span className="num font-semibold text-ink">{fmtNum(strategy.holdout.sharpe)}</span>, 95% confidence interval{' '}
                    <span className="num font-semibold text-ink">
                      {fmtNum(holdoutCi[0], 1)} to {fmtNum(holdoutCi[1], 1)}
                    </span>{' '}
                    from <span className="num">{fmtInt(strategy.holdout.bars)}</span> candles and{' '}
                    <span className="num">{fmtInt(strategy.holdout.trades)}</span> trades.
                    {holdoutCi[0] <= 0 ? ' The interval includes zero, so the holdout alone does not prove an edge.' : ''}
                    {context.priorHoldoutViews > 0
                      ? ` This holdout had already been viewed ${context.priorHoldoutViews} time${context.priorHoldoutViews === 1 ? '' : 's'} before this run, so it is no longer unseen data.`
                      : ''}
                  </p>
                ) : null}
              </div>
            </div>
          </Section>

          <Section title="Robustness" hint="Tested on train + validation; the holdout is not used">
            {report ? (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  {verdict !== null ? <VerdictPill verdict={verdict} /> : <Pill>Verdict after the run ends</Pill>}
                  <span className="text-xs text-ink-3">
                    {verdict === 'Robust'
                      ? 'Profitable on the holdout, deflated Sharpe of 95% or more, and every test below passed.'
                      : verdict === 'Fragile'
                        ? 'Survived the holdout, but a test below failed or the deflated Sharpe is under 95%.'
                        : verdict === 'Overfit'
                          ? 'Lost money on the holdout, or the deflated Sharpe says the result is probably luck (under 50%).'
                          : 'The verdict needs the holdout, which is revealed when the run ends.'}
                  </span>
                </div>
                <ul className="space-y-1.5">
                  <TestRow
                    pass={!report.sensitivity.fragile}
                    label="Parameter sensitivity (±20%)"
                    detail={`${report.sensitivity.variants} variants · ROI ${fmtPct(report.sensitivity.baseRoi, true)} → median ${fmtPct(report.sensitivity.medianRoi, true)}, worst ${fmtPct(report.sensitivity.worstRoi, true)} · score ${fmtNum(report.sensitivity.baseFitness)} → median ${fmtNum(report.sensitivity.medianFitness)}, worst ${fmtNum(report.sensitivity.worstFitness)}`}
                  />
                  <TestRow
                    pass={report.topTrades.profitable}
                    label={`Without the ${report.topTrades.removed} best trades`}
                    detail={`net PnL ${fmtUsd(report.topTrades.netPnl, true)} → ${fmtUsd(report.topTrades.netPnlWithout, true)}`}
                  />
                  {report.crossAsset ? (
                    <TestRow
                      pass={report.crossAsset.share >= 2 / 3 - 1e-9}
                      label="Same rules on other assets"
                      detail={`profitable on ${report.crossAsset.rows.filter((row) => row.profitable).length} of ${report.crossAsset.rows.length}`}
                    />
                  ) : (
                    <li className="flex items-start gap-2 text-sm text-ink-3">
                      {analysisPending ? <Loader2 size={15} className="mt-0.5 shrink-0 animate-spin" /> : <Info size={15} className="mt-0.5 shrink-0" />}
                      {analysisPending ? 'Loading other assets for the cross-asset test…' : 'Cross-asset test not available for this run.'}
                    </li>
                  )}
                  <TestRow
                    pass={strategy.dsr >= 0.95}
                    label="Deflated Sharpe ≥ 95%"
                    detail={fmtPct(strategy.dsr * 100, false, 1)}
                  />
                  {strategy.holdout && (
                    <TestRow
                      pass={strategy.holdout.roi > 0}
                      label="Profitable on the holdout"
                      detail={`${fmtPct(strategy.holdout.roi, true)} vs Buy & Hold ${fmtPct(benchmark.holdout?.roi ?? null, true)}`}
                    />
                  )}
                </ul>
                {report.crossAsset && (
                  <div className="overflow-x-auto rounded-lg border border-line">
                    <table className="num w-full text-sm">
                      <thead>
                        <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                          <th scope="col" className="px-3 py-2 text-left font-medium">Asset (untuned)</th>
                          <th scope="col" className="px-3 py-2 text-right font-medium">ROI</th>
                          <th scope="col" className="px-3 py-2 text-right font-medium">Sharpe</th>
                          <th scope="col" className="px-3 py-2 text-right font-medium">Trades</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.crossAsset.rows.map((row) => (
                          <tr key={row.symbol} className="border-b border-line/70 last:border-b-0">
                            <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">{row.symbol}</th>
                            <td className={cx('px-3 py-1.5 text-right', toneClass(row.roi))}>{fmtPct(row.roi, true)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-2">{fmtNum(row.sharpe)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-2">{fmtInt(row.trades)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : (
              <p className="flex items-center gap-2 text-sm text-ink-3">
                <Loader2 size={15} className="animate-spin" /> Running the robustness tests…
              </p>
            )}
          </Section>

          {strategy.assets.length > 1 && (
            <Section title="Robust universe" hint="The same rules, scored separately on every asset">
              <div className="overflow-x-auto rounded-lg border border-line">
                <table className="num w-full text-sm">
                  <thead>
                    <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                      <th scope="col" className="px-3 py-2 text-left font-medium">Asset</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Train ROI</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Validation ROI</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Validation Sharpe</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Holdout ROI</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategy.assets.map((asset) => (
                      <tr key={asset.symbol} className="border-b border-line/70 last:border-b-0">
                        <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">{asset.symbol}</th>
                        <td className={cx('px-3 py-1.5 text-right', toneClass(asset.train.roi))}>{fmtPct(asset.train.roi, true)}</td>
                        <td className={cx('px-3 py-1.5 text-right', toneClass(asset.validation.roi))}>{fmtPct(asset.validation.roi, true)}</td>
                        <td className="px-3 py-1.5 text-right text-ink-2">{fmtNum(asset.validation.sharpe)}</td>
                        <td className={cx('px-3 py-1.5 text-right', asset.holdout ? toneClass(asset.holdout.roi) : 'text-ink-3')}>
                          {asset.holdout ? fmtPct(asset.holdout.roi, true) : 'sealed'}
                        </td>
                        <td className="px-3 py-1.5 text-right text-ink-2">{fmtInt(asset.search.trades)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          <Section title="Export to TradingView" hint="Pine Script v5 strategy">
            <p className="text-sm text-ink-2">
              Copy the script, open the Pine Editor on a <span className="num font-medium text-ink">{context.interval}</span> chart of{' '}
              <span className="num font-medium text-ink">{context.isBasket ? 'the basket (see the script header)' : context.label}</span>,
              paste it and press <em>Add to chart</em>. The script reproduces these exact rules, exits, sizing, costs and next-bar-open
              execution. It switches on Bar Magnifier, which needs a TradingView Premium plan.
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
            Backtests are hypothetical and ignore liquidity, borrow costs and taxes. Nothing here is financial advice.
          </p>
        </div>
      </div>
    </div>
  );
}
