/**
 * Step 4 – the Strategy Workbench: everything about the selected strategy on one page.
 *
 *   • a plain-English verdict and a status badge for every robustness check,
 *   • the equity curve against Buy & Hold and the Next Entry Forecast (trigger prices, readiness),
 *   • the Monte Carlo stress curve and the parameter-stability heatmap,
 *   • the cross-market generalization check and the one-shot locked holdout reveal,
 *   • the exact rules, the train / validation / holdout figures and the 1-click Pine Script v5 export.
 */

import {
  Check,
  CircleCheck,
  CircleDashed,
  CircleX,
  Copy,
  Download,
  Globe,
  Info,
  Loader2,
  Lock,
  Pin,
  TriangleAlert,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ARCHETYPE_LABEL,
  describeArchetype,
  describeExit,
  describeLongEntry,
  describeShortEntry,
  indicatorBadges,
  listIndicators,
  shortRuleLabel,
  type RuleDescription,
} from '../engine/describe';
import { HYPOTHESES, MAX_PARAMETERS } from '../engine/genome';
import StabilityHeatmap from './StabilityHeatmap';
import { MIN_HOLDOUT_TRADES_FLOOR } from '../engine/backtester';
import { generatePineScript } from '../engine/pine';
import type { RunContext } from '../lib/config';
import { diagnosticBadges, strategyVerdict, type BadgeStatus, type DiagnosticBadge } from '../lib/diagnostics';
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from '../lib/format';
import { sharpeConfidenceInterval } from '../lib/stats';
import type { BenchmarkResult, ChartAxis, CrossMarketReport, PerformanceMetrics, RobustnessReport, StrategyResult, Verdict } from '../types/trading';
import type { MarketSnapshot } from '../lib/marketSnapshot';
import EquityChart, { BENCHMARK_COLOR, STRATEGY_COLOR } from './EquityChart';
import { VerdictPill } from './Leaderboard';
import MonteCarloChart from './MonteCarloChart';
import NextEntryForecast from './NextEntryForecast';
import { cx, LineKey, Panel, Pill, StepHeader } from './ui';

/** State of a cross-market check for one strategy. */
export interface CrossMarketState {
  status: 'loading' | 'done' | 'error';
  symbols: string[];
  report?: CrossMarketReport;
  error?: string;
}

interface StrategyWorkbenchProps {
  strategy: StrategyResult;
  /** Current leaderboard rank, or null when the strategy has since dropped out of the Top 25. */
  rank: number | null;
  benchmark: BenchmarkResult;
  context: RunContext;
  axis: ChartAxis | null;
  /** Strategies this run has evaluated (earlier runs on the window are in `context.priorTrials`). */
  testedCount: number;
  barsPerYear: number;
  /** Result of an on-demand `analyze` request; overrides the report carried by the strategy. */
  analysis: { report: RobustnessReport; verdict: Verdict | null } | null;
  /** True while an analysis request is in flight. */
  analysisPending: boolean;
  /** True once the search has ended (locking is only possible then). */
  finished: boolean;
  /** The run's locked strategy, if one has been chosen. */
  lockedId: string | null;
  /** True while the lock request is in flight. */
  lockPending: boolean;
  /** Lock this strategy and reveal its holdout. */
  onLock: () => void;
  /** Latest candles for the Next Entry Forecast (null until available). */
  snapshot: MarketSnapshot | null;
  /** Fetch fresh candles for the forecast. */
  onRefreshSnapshot: () => Promise<void>;
  /** True when the user picked this strategy (false: the workbench follows the current #1). */
  pinned: boolean;
  onFollowBest: () => void;
  /** Cross-market check of this strategy, if one was run. */
  crossMarket: CrossMarketState | null;
  /** Default unseen tickers for the cross-market check. */
  crossMarketSuggestion: string;
  onRunCrossMarket: (symbols: string[]) => void;
}

/* -------------------------------------------------------------------------- */

function Section({ title, hint, children, className }: { title: string; hint?: string; children: ReactNode; className?: string }): JSX.Element {
  return (
    <section className={cx('border-t border-line px-5 py-5', className)}>
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

/* -------------------------------------------------------------------------- */

const BADGE_STYLE: Record<BadgeStatus, { box: string; icon: ReactNode; word: string }> = {
  pass: { box: 'border-up/35 bg-up/10', icon: <CircleCheck size={15} className="text-up" />, word: 'Pass' },
  warn: { box: 'border-warn/35 bg-warn/10', icon: <TriangleAlert size={15} className="text-warn" />, word: 'Caution' },
  fail: { box: 'border-down/35 bg-down/10', icon: <CircleX size={15} className="text-down" />, word: 'Fail' },
  pending: { box: 'border-line bg-raised/50', icon: <CircleDashed size={15} className="text-ink-3" />, word: 'Not yet' },
};

function Badge({ badge }: { badge: DiagnosticBadge }): JSX.Element {
  const style = BADGE_STYLE[badge.status];
  return (
    <div className={cx('min-w-0 rounded-lg border px-3 py-2', style.box)} title={badge.tooltip}>
      <div className="flex items-center gap-1.5">
        {style.icon}
        <span className="truncate text-xs font-semibold text-ink">{badge.label}</span>
        <span className="sr-only">{style.word}</span>
      </div>
      <div className="num mt-0.5 truncate text-sm font-semibold text-ink">{badge.value}</div>
      <div className="mt-0.5 truncate text-[10.5px] uppercase tracking-wider text-ink-3">{badge.metric}</div>
    </div>
  );
}

const VERDICT_STYLE: Record<ReturnType<typeof strategyVerdict>['tone'], string> = {
  strong: 'border-up/40 bg-up/10',
  promising: 'border-accent/40 bg-accent/10',
  mixed: 'border-warn/40 bg-warn/10',
  weak: 'border-down/40 bg-down/10',
};

/** Compact Strategy vs Buy & Hold figures next to the equity curve. */
const VS_ROWS: { label: string; format: (m: PerformanceMetrics) => string; benchmark?: boolean }[] = [
  { label: 'Sharpe', format: (m) => fmtNum(m.sharpe) },
  { label: 'Sortino', format: (m) => fmtNum(m.sortino) },
  { label: 'Max drawdown', format: (m) => fmtPct(m.maxDrawdown) },
  { label: 'CAGR', format: (m) => fmtPct(m.cagr, true) },
  { label: 'ROI', format: (m) => fmtPct(m.roi, true) },
  { label: 'Trades', format: (m) => fmtInt(m.trades), benchmark: false },
  { label: 'Win rate', format: (m) => fmtPct(m.winRate), benchmark: false },
  { label: 'Time in market', format: (m) => fmtPct(m.exposure, false, 0) },
];

function download(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function StrategyWorkbench({
  strategy,
  rank,
  benchmark,
  context,
  axis,
  testedCount,
  barsPerYear,
  analysis,
  analysisPending,
  finished,
  lockedId,
  lockPending,
  onLock,
  snapshot,
  onRefreshSnapshot,
  pinned,
  onFollowBest,
  crossMarket,
  crossMarketSuggestion,
  onRunCrossMarket,
}: StrategyWorkbenchProps): JSX.Element {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  /** Locking is irreversible, so it takes two clicks. */
  const [confirmLock, setConfirmLock] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [crossInput, setCrossInput] = useState(crossMarketSuggestion);
  const { genome } = strategy;
  const report = analysis?.report ?? strategy.robustness;
  const verdict = analysis?.verdict ?? strategy.verdict;
  const revealed = strategy.holdout !== null;
  const pooled = context.universe.length > 1;
  const hypothesis = genome.hypothesis ? HYPOTHESES[genome.hypothesis] : null;
  const trials = context.priorTrials.count + testedCount;

  useEffect(() => {
    setCopyState('idle');
    setConfirmLock(false);
  }, [strategy.id]);
  useEffect(() => setCrossInput(crossMarket?.symbols.join(', ') ?? crossMarketSuggestion), [strategy.id, crossMarket, crossMarketSuggestion]);

  const indicators = useMemo(() => listIndicators(genome), [genome]);
  const longEntry = useMemo(() => describeLongEntry(genome), [genome]);
  const shortEntry = useMemo(() => describeShortEntry(genome), [genome]);
  const exit = useMemo(() => describeExit(genome, context.settings), [genome, context.settings]);
  const holdoutCi = useMemo(
    () => (strategy.holdout ? sharpeConfidenceInterval(strategy.holdout, barsPerYear) : null),
    [strategy.holdout, barsPerYear],
  );
  const cross = crossMarket?.report ?? null;
  const diagnostics = useMemo(() => {
    const input = { strategy: { ...strategy, verdict }, report, crossMarket: cross, trials, simulated: context.simulated };
    return { badges: diagnosticBadges(input), verdict: strategyVerdict(input) };
  }, [strategy, verdict, report, cross, trials, context.simulated]);

  const pine = useMemo(
    () =>
      generatePineScript(strategy, {
        label: context.label,
        assets: context.universe,
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
  const mc = report?.monteCarlo ?? null;
  const mcSummary = mc ?? strategy.monteCarlo ?? null;
  const crossSymbols = crossInput
    .split(/[,\s]+/)
    .map((symbol) => symbol.trim().toUpperCase())
    .filter(Boolean);

  return (
    <Panel className="overflow-hidden">
      {/* Header */}
      <header className="border-b border-line px-5 py-4">
        <StepHeader step={4} title={`Strategy workbench · ${strategy.id}`} hint="Everything about the selected strategy – from the verdict to the TradingView script.">
          <div className="flex flex-wrap items-center gap-2">
            {rank !== null ? <Pill tone="accent">Rank #{rank}</Pill> : <Pill>No longer in Top 25</Pill>}
            {pinned ? (
              <button
                type="button"
                onClick={onFollowBest}
                className="inline-flex h-7 items-center gap-1.5 rounded-md border border-line px-2 text-xs text-ink-2 transition-colors hover:bg-hover hover:text-ink"
                title="Stop inspecting the clicked strategy and follow the current #1"
              >
                <Pin size={12} /> Follow #1
              </button>
            ) : (
              <Pill tone="accent">Following #1</Pill>
            )}
          </div>
        </StepHeader>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Pill tone="bench">{hypothesis ? hypothesis.label : ARCHETYPE_LABEL[genome.archetype]}</Pill>
          <Pill>{shortRuleLabel(genome)}</Pill>
          <Pill tone={genome.exit.mode === 'static' ? 'neutral' : 'accent'}>{exit.mode}</Pill>
          {genome.exit.breakeven && <Pill tone="accent">Breakeven lock</Pill>}
          {genome.exit.mode === 'dynamic' && genome.exit.trailing && <Pill tone="accent">Trailing ATR stop</Pill>}
          <Pill>{strategy.params} params</Pill>
          {verdict !== null && <VerdictPill verdict={verdict} />}
          <span className="num ml-1 truncate text-xs text-ink-3">
            {context.venue} · {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)} · found at
            iteration {fmtInt(strategy.foundAt)}
          </span>
        </div>
      </header>

      {/* Verdict and checks */}
      <div className="grid grid-cols-1 gap-4 px-5 py-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className={cx('rounded-xl border p-4', VERDICT_STYLE[diagnostics.verdict.tone])} aria-live="polite">
          <div className="text-[11px] font-medium uppercase tracking-wider text-ink-3">Strategy verdict</div>
          <h3 className="mt-0.5 text-base font-semibold text-ink">{diagnostics.verdict.headline}</h3>
          <p className="mt-1 text-sm text-ink-2">{diagnostics.verdict.summary}</p>
          {diagnostics.verdict.strengths.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-semibold text-up">Why it looks strong</div>
              <ul className="mt-1 space-y-1">
                {diagnostics.verdict.strengths.map((line) => (
                  <li key={line} className="flex gap-2 text-sm text-ink-2">
                    <CircleCheck size={14} className="mt-0.5 shrink-0 text-up" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {diagnostics.verdict.risks.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-semibold text-warn">What to watch out for</div>
              <ul className="mt-1 space-y-1">
                {diagnostics.verdict.risks.slice(0, 6).map((line) => (
                  <li key={line} className="flex gap-2 text-sm text-ink-2">
                    <TriangleAlert size={14} className="mt-0.5 shrink-0 text-warn" />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div>
          <div className="mb-2 flex items-baseline justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-2">Robustness checks</h3>
            <span className="text-[11px] text-ink-3">Hover a badge for what it means</span>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {diagnostics.badges.map((badge) => (
              <Badge key={badge.key} badge={badge} />
            ))}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
            {fmtInt(trials)} strategies have been tried on this data window (all runs). Figures cover train + validation – the 80% of
            history the search optimised on – unless marked holdout. In-sample ROI {fmtPct(strategy.search.roi, true)},{' '}
            {fmtPct(strategy.alpha, true)} vs. Buy &amp; Hold.
          </p>
        </div>
      </div>

      {context.simulated && (
        <div className="mx-5 mb-4 flex gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2 text-xs text-warn">
          <Info size={14} className="mt-0.5 shrink-0" />
          <span>
            This run used simulated fallback candles, not real market data. Treat every number here as an illustration and re-run on live
            data before drawing conclusions.
          </span>
        </div>
      )}

      {/* Equity curve and the forecast */}
      <div className="grid grid-cols-1 gap-4 border-t border-line px-5 py-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-2">Equity curve vs. Buy &amp; Hold</h3>
            <span className="num text-[11px] text-ink-3">
              {context.label} · {context.interval} · starting capital {fmtUsd(context.settings.initialCapital)}
            </span>
          </div>
          <div className="h-[300px]">
            <EquityChart
              axis={axis}
              benchmark={benchmark.equity}
              strategy={strategy.equity}
              strategyName={`Strategy ${strategy.id}${rank !== null ? ` · #${rank}` : ''}`}
              revealed={benchmark.holdout != null}
              initialCapital={context.settings.initialCapital}
              interval={context.interval}
              placeholder="Waiting for the first strategy…"
            />
          </div>
          <div className="mt-3 overflow-x-auto rounded-lg border border-line">
            <table className="num w-full text-xs">
              <thead>
                <tr className="border-b border-line bg-raised/50 text-[10.5px] uppercase tracking-wider text-ink-3">
                  <th scope="col" className="px-3 py-1.5 text-left font-medium">Train + validation</th>
                  {VS_ROWS.map((row) => (
                    <th key={row.label} scope="col" className="whitespace-nowrap px-2 py-1.5 text-right font-medium">
                      {row.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line/60">
                  <th scope="row" className="whitespace-nowrap px-3 py-1.5 text-left font-normal text-ink-2">
                    <span className="inline-flex items-center gap-1.5">
                      <LineKey color={STRATEGY_COLOR} /> Strategy
                    </span>
                  </th>
                  {VS_ROWS.map((row) => (
                    <td key={row.label} className="px-2 py-1.5 text-right font-medium text-ink">
                      {row.format(strategy.search)}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row" className="whitespace-nowrap px-3 py-1.5 text-left font-normal text-ink-2">
                    <span className="inline-flex items-center gap-1.5">
                      <LineKey color={BENCHMARK_COLOR} /> Buy &amp; Hold
                    </span>
                  </th>
                  {VS_ROWS.map((row) => (
                    <td key={row.label} className="px-2 py-1.5 text-right text-ink-3">
                      {row.benchmark === false ? '–' : row.format(benchmark.search)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <NextEntryForecast className="m-0" genome={genome} settings={context.settings} snapshot={snapshot} onRefresh={onRefreshSnapshot} />
      </div>

      {/* Stress curve and stability map */}
      <div className="grid grid-cols-1 border-t border-line xl:grid-cols-2 xl:divide-x xl:divide-line">
        <Section className="border-t-0" title="Monte Carlo stress test" hint="Replays · 10% of trades skipped · shuffled order · 2× cost shocks">
          {mcSummary ? (
            <>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatTile
                  label="P(profit)"
                  value={fmtPct(mcSummary.probProfit * 100, false, 0)}
                  tone={mcSummary.passed ? 'text-up' : mcSummary.probProfit >= 0.6 ? 'text-warn' : 'text-down'}
                />
                <StatTile label="95% worst DD" value={fmtPct(mcSummary.drawdownP95, false, 1)} tone={mcSummary.drawdownP95 < 50 ? 'text-ink' : 'text-down'} />
                <StatTile label="Median return" value={fmtPct(mcSummary.medianReturn, true, 0)} tone={toneClass(mcSummary.medianReturn)} />
                <StatTile label="5th pct return" value={fmtPct(mcSummary.p5Return, true, 0)} tone={toneClass(mcSummary.p5Return)} />
              </div>
              <div className="mt-3">
                {mc ? (
                  <MonteCarloChart report={mc} />
                ) : (
                  <p className="flex items-center gap-2 text-sm text-ink-3">
                    <Loader2 size={15} className="animate-spin" /> Drawing the stress curve…
                  </p>
                )}
              </div>
              <p className="mt-2 text-xs text-ink-3">
                {mcSummary.simulations} replays of its {mcSummary.trades} search-period trades
                {pooled ? ` (pooled over the ${context.universe.length} assets)` : ''}, each fill shocked by on average
                twice its modelled cost. The backtest&apos;s own drawdown on the same
                trades was {fmtPct(mcSummary.baseDrawdown, false, 1)}
                {mcSummary.drawdownP95 > 1.5 * mcSummary.baseDrawdown
                  ? ' – the stressed worst case is much deeper, so the backtest saw a lucky order of trades.'
                  : '.'}{' '}
                Pass: at least 80% of replays profitable and the 95th-percentile drawdown under 50%.
              </p>
            </>
          ) : (
            <p className="text-sm text-ink-3">No trades to replay.</p>
          )}
        </Section>
        <Section className="border-t xl:border-t-0" title="Parameter stability" hint="Train + validation · primary asset">
          {report?.stability ? (
            <StabilityHeatmap grid={report.stability} />
          ) : report ? (
            <p className="text-sm text-ink-3">This strategy has no tunable parameter to map.</p>
          ) : (
            <p className="flex items-center gap-2 text-sm text-ink-3">
              <Loader2 size={15} className="animate-spin" /> {analysisPending ? 'Mapping the parameter neighbourhood…' : 'Waiting for the analysis…'}
            </p>
          )}
        </Section>
      </div>

      {/* Cross-market check and holdout */}
      <div className="grid grid-cols-1 border-t border-line xl:grid-cols-2 xl:divide-x xl:divide-line">
        <Section className="border-t-0" title="Cross-market check" hint="Frozen rules · unseen markets · holdout dates excluded">
          <p className="text-sm text-ink-2">
            Test these exact rules, untouched, on related markets the search never saw – for example SOL and BNB for a strategy found on
            BTC and ETH. A structural edge carries over; a fit to one chart&apos;s noise does not. Only candles before{' '}
            {axis ? fmtDate(axis.time[axis.holdoutPos]) : 'the holdout'} are used, so the holdout stays sealed.
          </p>
          <form
            className="mt-3 flex flex-wrap items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (crossSymbols.length > 0) onRunCrossMarket(crossSymbols);
            }}
          >
            <Globe size={15} className="text-ink-3" />
            <input
              type="text"
              value={crossInput}
              onChange={(event) => setCrossInput(event.target.value.toUpperCase())}
              aria-label="Unseen tickers for the cross-market check"
              spellCheck={false}
              placeholder="SOL-USD, BNB-USD, …"
              className="num h-9 min-w-[200px] flex-1 rounded-lg border border-line bg-base px-2.5 text-sm text-ink outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={crossMarket?.status === 'loading' || crossSymbols.length === 0}
              className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-accent/40 bg-accent/10 px-3 text-sm font-semibold text-accent-strong transition-colors hover:bg-accent/20 disabled:opacity-60"
            >
              {crossMarket?.status === 'loading' ? <Loader2 size={14} className="animate-spin" /> : <Globe size={14} />}
              Run check
            </button>
          </form>
          {crossMarket?.status === 'error' && <p className="mt-2 text-xs text-down">{crossMarket.error}</p>}
          {cross && (
            <div className="mt-3">
              <div className="mb-2 flex items-center gap-2 text-sm">
                {cross.passed ? <CircleCheck size={15} className="text-up" /> : <CircleX size={15} className="text-down" />}
                <span className="text-ink-2">
                  {cross.tested === 0
                    ? 'No unseen asset had enough history to test.'
                    : `Profitable on ${Math.round(cross.share * cross.tested)} of ${cross.tested} unseen markets, median Sharpe ${fmtNum(cross.medianSharpe)}${
                        cross.passed ? ' – the edge generalises.' : ' – two thirds profitable and a positive median are needed.'
                      }`}
                </span>
              </div>
              <div className="overflow-x-auto rounded-lg border border-line">
                <table className="num w-full text-sm">
                  <thead>
                    <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                      <th scope="col" className="px-3 py-2 text-left font-medium">Market</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">ROI</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">B&amp;H ROI</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Sharpe</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Max DD</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cross.rows.map((row) => (
                      <tr key={row.symbol} className="border-b border-line/70 last:border-b-0">
                        <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">
                          {row.symbol}
                          {!row.unseen && <span className="ml-1.5 text-[11px] text-ink-3">(in the run – not counted)</span>}
                        </th>
                        {row.note ? (
                          <td colSpan={5} className="px-3 py-1.5 text-right text-xs text-ink-3">
                            {row.note}
                          </td>
                        ) : (
                          <>
                            <td className={cx('px-3 py-1.5 text-right', toneClass(row.roi))}>{fmtPct(row.roi, true)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-3">{fmtPct(row.holdRoi, true)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-2">{fmtNum(row.sharpe)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-2">{fmtPct(row.maxDrawdown)}</td>
                            <td className="px-3 py-1.5 text-right text-ink-2">{fmtInt(row.trades)}</td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Section>
        <Section className="border-t xl:border-t-0" title="Locked holdout reveal" hint="The last 20% of history · one strategy per run">
          {revealed && strategy.holdout ? (
            <div className="space-y-3 text-sm text-ink-2">
              <div className="flex flex-wrap items-center gap-2">
                <Lock size={14} className="text-accent-strong" />
                <span className="font-semibold text-ink">Revealed for {strategy.id}.</span>
                {verdict !== null && <VerdictPill verdict={verdict} />}
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatTile label="Holdout ROI" value={fmtPct(strategy.holdout.roi, true)} tone={toneClass(strategy.holdout.roi)} />
                <StatTile label="Holdout Sharpe" value={fmtNum(strategy.holdout.sharpe)} />
                <StatTile label="B&H Sharpe" value={fmtNum(benchmark.holdout?.sharpe ?? null)} />
                <StatTile label="Holdout trades" value={fmtInt(strategy.sample?.holdoutTrades ?? strategy.holdout.trades)} />
              </div>
              <ul className="space-y-1.5">
                <TestRow pass={strategy.dsr >= 0.95} label="Deflated Sharpe ≥ 95%" detail={fmtPct(strategy.dsr * 100, false, 1)} />
                {holdoutCi && (
                  <TestRow
                    pass={holdoutCi[0] > 0}
                    label="Holdout Sharpe 95% interval above zero"
                    detail={`${fmtNum(holdoutCi[0], 2)} to ${fmtNum(holdoutCi[1], 2)}`}
                  />
                )}
                <TestRow
                  pass={strategy.holdout.sharpe > (benchmark.holdout?.sharpe ?? Infinity)}
                  label="Holdout Sharpe beats Buy & Hold"
                  detail={`${fmtNum(strategy.holdout.sharpe)} vs ${fmtNum(benchmark.holdout?.sharpe ?? null)} · ROI ${fmtPct(strategy.holdout.roi, true)} vs ${fmtPct(benchmark.holdout?.roi ?? null, true)}`}
                />
                <TestRow
                  pass={(strategy.sample?.holdoutTrades ?? strategy.holdout.trades) >= (strategy.sample?.gates.holdout ?? MIN_HOLDOUT_TRADES_FLOOR)}
                  label={`At least ${strategy.sample?.gates.holdout ?? MIN_HOLDOUT_TRADES_FLOOR} holdout trades${pooled ? ' (pooled over the assets)' : ''}`}
                  detail={fmtInt(strategy.sample?.holdoutTrades ?? strategy.holdout.trades)}
                />
              </ul>
              {strategy.verdictNotes && strategy.verdictNotes.length > 0 ? (
                <div>
                  <div className="text-xs font-semibold text-ink">{verdict === 'Overfit' ? 'Rejected:' : 'Why it is not Robust:'}</div>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-ink-2">
                    {strategy.verdictNotes.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </div>
              ) : verdict === 'Robust' ? (
                <p className="text-xs text-up">
                  Every Robust condition held: the search-period gates, the untouched holdout, the stress test and generalisation beyond one chart.
                </p>
              ) : null}
              {context.priorHoldoutReveals > 0 && (
                <p className="text-xs text-warn">
                  This holdout period had already been revealed {context.priorHoldoutReveals} time{context.priorHoldoutReveals === 1 ? '' : 's'} before this run,
                  so it is no longer untouched data.
                </p>
              )}
            </div>
          ) : (
            <>
              <div className="rounded-lg border border-line px-3 py-2.5 text-xs leading-relaxed text-ink-2">
                {!finished ? (
                  <p className="flex items-start gap-2">
                    <Info size={14} className="mt-0.5 shrink-0 text-ink-3" />
                    The holdout is not simulated while the search is running. When the run ends you can lock ONE strategy to reveal its
                    holdout.
                  </p>
                ) : lockedId !== null ? (
                  <p className="flex items-start gap-2">
                    <Lock size={14} className="mt-0.5 shrink-0 text-ink-3" />
                    This run&apos;s holdout was spent on strategy <span className="num font-semibold text-ink">{lockedId}</span>. Every other
                    strategy&apos;s holdout stays sealed, so the holdout cannot be used to pick a winner.
                  </p>
                ) : (
                  <div className="flex flex-wrap items-center gap-3">
                    <p className="min-w-0 flex-1">
                      <span className="font-semibold text-ink">Lock this strategy as the run&apos;s final answer?</span> Its holdout is revealed
                      once. No other strategy of this run can be revealed afterwards, and the reveal is recorded for later runs on the same
                      symbols and dates.
                    </p>
                    {confirmLock ? (
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setConfirmLock(false)}
                          className="inline-flex h-8 items-center rounded-lg border border-line px-3 text-xs text-ink-2 transition-colors hover:bg-hover hover:text-ink"
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          onClick={onLock}
                          disabled={lockPending}
                          className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-xs font-semibold text-white transition-colors hover:bg-accent-strong disabled:opacity-60"
                        >
                          {lockPending ? <Loader2 size={13} className="animate-spin" /> : <Lock size={13} />}
                          Yes, lock {strategy.id} and reveal
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setConfirmLock(true)}
                        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-accent/40 bg-accent/10 px-3 text-xs font-semibold text-accent-strong transition-colors hover:bg-accent/20"
                      >
                        <Lock size={13} /> Lock &amp; reveal holdout
                      </button>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </Section>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 xl:divide-x xl:divide-line">
          <Section title="The idea" hint={hypothesis ? `Hypothesis: ${hypothesis.label}` : ARCHETYPE_LABEL[genome.archetype]}>
            {hypothesis && (
              <p className="mb-2 text-sm leading-relaxed text-ink">
                <span className="font-semibold">{hypothesis.label}.</span> {hypothesis.rationale}
              </p>
            )}
            <p className="text-sm leading-relaxed text-ink-2">{describeArchetype(genome)}</p>
            {hypothesis && <p className="mt-2 text-xs text-ink-3">One primitive per idea: it stands in for {hypothesis.replaces}</p>}
          </Section>

          <Section
            title="Indicators & exact parameters"
            hint={`${strategy.params} tunable values (cap ${MAX_PARAMETERS}${genome.shortRegime ? ' + 1 short regime filter' : ''})`}
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
                <div className="mb-1.5 text-xs font-semibold text-down">
                  Short entry
                  {genome.shortEntry !== null &&
                    ` (${genome.shortRegime ? 'mirrored trigger, only inside its own bear-regime filter' : 'mirror of the long rules'})`}
                </div>
                {shortEntry ? (
                  <RuleList rule={shortEntry} tone="down" />
                ) : (
                  <p className="text-sm text-ink-3">None – Long-Only: this strategy never sells short.</p>
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

      </div>

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

          </Section>

          {strategy.assets.length > 1 && (
            <Section title="Per-asset results" hint="The same rules on every asset of the pool">
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

          <Section title="Export to TradingView" hint="Pine Script v5 strategy · runs on every TradingView plan">
            <p className="text-sm text-ink-2">
              Copy or download the script, open the Pine Editor on a <span className="num font-medium text-ink">{context.interval}</span> chart of{' '}
              <span className="num font-medium text-ink">{context.isBasket ? 'the basket (see the script header)' : context.label}</span>,
              paste it and press <em>Add to chart</em>. It reproduces these exact rules – entries for every direction mode, the trailing stop
              and breakeven lock, sizing, costs and next-bar-open execution – with <code className="num text-xs">use_bar_magnifier=false</code>,
              so it runs on every TradingView plan.
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
                onClick={() => download(`StrategyFinder-${strategy.id}-${context.interval}.pine`, pine)}
                className="inline-flex h-9 items-center gap-2 rounded-lg border border-line px-3 text-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink"
              >
                <Download size={15} /> Download .pine
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
            Backtests are hypothetical and ignore liquidity, borrow availability and taxes. Nothing here is financial advice.
          </p>
    </Panel>
  );
}
