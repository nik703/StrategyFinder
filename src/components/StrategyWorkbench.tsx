/**
 * Step 4 – the Strategy Workbench: everything about the selected strategy in one shadcn Card,
 * organised in Tabs:
 *
 *   • Verdict – the plain-English Strategy Verdict (an Alert), a status Card for every robustness
 *     check and the equity curve against Buy & Hold;
 *   • Next entry – the Next Entry Forecast (readiness gauges, trigger / fill / SL / TP, checklist);
 *   • Validation & stress – Monte Carlo stress test, parameter-stability heatmap, cross-market
 *     generalisation check and the one-shot locked holdout reveal (with a Dialog confirmation);
 *   • Rules – the idea, the indicators and the exact entry and exit rules;
 *   • Performance – train / validation / holdout figures and the per-asset results;
 *   • Pine Script – the 1-click TradingView Pine Script v5 export with copy feedback.
 */

import { Check, CircleCheck, CircleDashed, CircleX, Code, Copy, Download, Globe, Info, Loader2, Lock, Pin, TriangleAlert } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
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
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MIN_HOLDOUT_TRADES_FLOOR } from '@/engine/backtester';
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
} from '@/engine/describe';
import { HYPOTHESES, MAX_PARAMETERS } from '@/engine/genome';
import { generatePineScript } from '@/engine/pine';
import type { RunContext } from '@/lib/config';
import { diagnosticBadges, strategyVerdict, type BadgeStatus, type DiagnosticBadge } from '@/lib/diagnostics';
import { fmtDate, fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from '@/lib/format';
import type { MarketSnapshot } from '@/lib/marketSnapshot';
import { sharpeConfidenceInterval } from '@/lib/stats';
import { cn } from '@/lib/utils';
import type { BenchmarkResult, ChartAxis, CrossMarketReport, PerformanceMetrics, RobustnessReport, StrategyResult, Verdict } from '@/types/trading';
import { LineKey, Panel, Pill, StepHeader } from './common';
import EquityChart, { BENCHMARK_COLOR, STRATEGY_COLOR } from './EquityChart';
import { VerdictPill } from './Leaderboard';
import MonteCarloChart from './MonteCarloChart';
import NextEntryForecast from './NextEntryForecast';
import StabilityHeatmap from './StabilityHeatmap';

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

/** A titled Card inside a workbench tab. */
function SectionCard({ title, hint, children, className }: { title: string; hint?: string; children: ReactNode; className?: string }): JSX.Element {
  return (
    <Card className={cn('gap-3', className)}>
      <CardHeader>
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-secondary-foreground">{title}</CardTitle>
        {hint && <CardDescription className="text-[11px]">{hint}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function RuleList({ rule, tone }: { rule: RuleDescription; tone: 'up' | 'down' | 'neutral' }): JSX.Element {
  const dot = tone === 'up' ? 'bg-up' : tone === 'down' ? 'bg-down' : 'bg-muted-foreground';
  return (
    <div>
      <p className="text-sm text-secondary-foreground">{rule.intro}</p>
      <ul className="mt-2 space-y-2">
        {rule.bullets.map((bullet) => (
          <li key={bullet.formula} className="flex gap-2.5">
            <span aria-hidden="true" className={cn('mt-[7px] size-1.5 shrink-0 rounded-full', dot)} />
            <div className="min-w-0">
              <div className="text-sm text-foreground">{bullet.text}</div>
              <code className="num mt-0.5 block text-xs text-muted-foreground">{bullet.formula}</code>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A small figure Card (label + value). */
function StatTile({ label, value, tone }: { label: string; value: string; tone?: string }): JSX.Element {
  return (
    <Card size="sm" className="gap-0 bg-muted/50 py-2">
      <CardContent className="px-3">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className={cn('num mt-0.5 text-base font-semibold', tone ?? 'text-foreground')}>{value}</div>
      </CardContent>
    </Card>
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
      {pass ? <CircleCheck className="mt-0.5 size-4 shrink-0 text-up" aria-label="Passed" /> : <CircleX className="mt-0.5 size-4 shrink-0 text-down" aria-label="Failed" />}
      <span className="text-secondary-foreground">
        <span className="font-medium text-foreground">{label}</span> <span className="num text-muted-foreground">· {detail}</span>
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

/* -------------------------------------------------------------------------- */

const BADGE_STYLE: Record<BadgeStatus, { box: string; icon: ReactNode; word: string }> = {
  pass: { box: 'bg-up/10 ring-up/35', icon: <CircleCheck className="size-4 text-up" />, word: 'Pass' },
  warn: { box: 'bg-warn/10 ring-warn/35', icon: <TriangleAlert className="size-4 text-warn" />, word: 'Caution' },
  fail: { box: 'bg-down/10 ring-down/35', icon: <CircleX className="size-4 text-down" />, word: 'Fail' },
  pending: { box: 'bg-muted/50', icon: <CircleDashed className="size-4 text-muted-foreground" />, word: 'Not yet' },
};

/** One robustness check as a small status Card. */
function CheckCard({ badge }: { badge: DiagnosticBadge }): JSX.Element {
  const style = BADGE_STYLE[badge.status];
  return (
    <Card size="sm" className={cn('min-w-0 gap-0 py-2', style.box)} title={badge.tooltip}>
      <CardContent className="px-3">
        <div className="flex items-center gap-1.5">
          {style.icon}
          <span className="truncate text-xs font-semibold text-foreground">{badge.label}</span>
          <span className="sr-only">{style.word}</span>
        </div>
        <div className="num mt-0.5 truncate text-sm font-semibold text-foreground">{badge.value}</div>
        <div className="mt-0.5 truncate text-[10.5px] uppercase tracking-wider text-muted-foreground">{badge.metric}</div>
      </CardContent>
    </Card>
  );
}

const VERDICT_STYLE: Record<ReturnType<typeof strategyVerdict>['tone'], string> = {
  strong: 'border-up/40 bg-up/10',
  promising: 'border-primary/40 bg-primary/10',
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

type WorkbenchTab = 'verdict' | 'forecast' | 'validation' | 'rules' | 'performance' | 'export';

const HEAD_ROW = 'bg-muted/50 text-[11px] uppercase tracking-wider hover:bg-muted/50';

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
  const [tab, setTab] = useState<WorkbenchTab>('verdict');
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [lockDialog, setLockDialog] = useState(false);
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
    setLockDialog(false);
  }, [strategy.id]);
  useEffect(() => setCrossInput(crossMarket?.symbols.join(', ') ?? crossMarketSuggestion), [strategy.id, crossMarket, crossMarketSuggestion]);

  const indicators = useMemo(() => listIndicators(genome), [genome]);
  const longEntry = useMemo(() => describeLongEntry(genome), [genome]);
  const shortEntry = useMemo(() => describeShortEntry(genome), [genome]);
  const exit = useMemo(() => describeExit(genome, context.settings), [genome, context.settings]);
  const holdoutCi = useMemo(() => (strategy.holdout ? sharpeConfidenceInterval(strategy.holdout, barsPerYear) : null), [strategy.holdout, barsPerYear]);
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
  const holdoutTrades = strategy.holdout ? (strategy.sample?.holdoutTrades ?? strategy.holdout.trades) : 0;
  const holdoutGate = strategy.sample?.gates.holdout ?? MIN_HOLDOUT_TRADES_FLOOR;

  return (
    <Panel className="overflow-hidden">
      <CardHeader className="border-b px-5 py-4">
        <StepHeader step={4} title={`Strategy workbench · ${strategy.id}`} hint="Everything about the selected strategy – from the verdict to the TradingView script.">
          <div className="flex flex-wrap items-center gap-2">
            {rank !== null ? <Pill tone="accent">Rank #{rank}</Pill> : <Pill>No longer in Top 25</Pill>}
            {pinned ? (
              <Button variant="outline" size="sm" onClick={onFollowBest} title="Stop inspecting the clicked strategy and follow the current #1">
                <Pin /> Follow #1
              </Button>
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
          <span className="num ml-1 truncate text-xs text-muted-foreground">
            {context.venue} · {context.interval} candles · {fmtDate(context.windowStart)} → {fmtDate(context.windowEnd)} · found at iteration {fmtInt(strategy.foundAt)}
          </span>
        </div>
      </CardHeader>

      <CardContent className="px-5 py-4">
        <Tabs value={tab} onValueChange={(value) => setTab(value as WorkbenchTab)} className="gap-4">
          <TabsList className="h-auto w-full flex-wrap justify-start">
            <TabsTrigger value="verdict" className="flex-none">
              Verdict
            </TabsTrigger>
            <TabsTrigger value="forecast" className="flex-none">
              Next entry
            </TabsTrigger>
            <TabsTrigger value="validation" className="flex-none">
              Validation &amp; stress
            </TabsTrigger>
            <TabsTrigger value="rules" className="flex-none">
              Rules
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex-none">
              Performance
            </TabsTrigger>
            <TabsTrigger value="export" className="flex-none">
              Pine Script v5
            </TabsTrigger>
          </TabsList>

          {/* Verdict */}
          <TabsContent value="verdict" className="space-y-4">
            {context.simulated && (
              <Alert className="border-warn/40 bg-warn/10 text-warn">
                <Info />
                <AlertDescription className="text-current">
                  This run used simulated fallback candles, not real market data. Treat every number here as an illustration and re-run on live data
                  before drawing conclusions.
                </AlertDescription>
              </Alert>
            )}
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
              <Alert aria-live="polite" className={cn('block p-4', VERDICT_STYLE[diagnostics.verdict.tone])}>
                <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Strategy verdict</div>
                <AlertTitle className="mt-0.5 text-base font-semibold text-foreground">{diagnostics.verdict.headline}</AlertTitle>
                <AlertDescription className="mt-1 block text-sm text-secondary-foreground">
                  <p>{diagnostics.verdict.summary}</p>
                  {diagnostics.verdict.strengths.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs font-semibold text-up">Why it looks strong</div>
                      <ul className="mt-1 space-y-1">
                        {diagnostics.verdict.strengths.map((line) => (
                          <li key={line} className="flex gap-2">
                            <CircleCheck className="mt-0.5 size-3.5 shrink-0 text-up" />
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
                          <li key={line} className="flex gap-2">
                            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warn" />
                            <span>{line}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
              <div>
                <div className="mb-2 flex items-baseline justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-secondary-foreground">Robustness checks</h3>
                  <span className="text-[11px] text-muted-foreground">Hover a check for what it means</span>
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {diagnostics.badges.map((badge) => (
                    <CheckCard key={badge.key} badge={badge} />
                  ))}
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
                  {fmtInt(trials)} strategies have been tried on this data window (all runs). Figures cover train + validation – the 80% of history the
                  search optimised on – unless marked holdout. In-sample ROI {fmtPct(strategy.search.roi, true)}, {fmtPct(strategy.alpha, true)} vs. Buy
                  &amp; Hold.
                </p>
              </div>
            </div>

            <SectionCard title="Equity curve vs. Buy & Hold" hint={`${context.label} · ${context.interval} · starting capital ${fmtUsd(context.settings.initialCapital)}`}>
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
              <div className="mt-3 rounded-lg border border-border">
                <Table className="num text-xs">
                  <TableHeader>
                    <TableRow className={HEAD_ROW}>
                      <TableHead>Train + validation</TableHead>
                      {VS_ROWS.map((row) => (
                        <TableHead key={row.label} className="text-right">
                          {row.label}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableHead scope="row" className="font-normal text-secondary-foreground">
                        <span className="inline-flex items-center gap-1.5">
                          <LineKey color={STRATEGY_COLOR} /> Strategy
                        </span>
                      </TableHead>
                      {VS_ROWS.map((row) => (
                        <TableCell key={row.label} className="text-right font-medium text-foreground">
                          {row.format(strategy.search)}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableHead scope="row" className="font-normal text-secondary-foreground">
                        <span className="inline-flex items-center gap-1.5">
                          <LineKey color={BENCHMARK_COLOR} /> Buy &amp; Hold
                        </span>
                      </TableHead>
                      {VS_ROWS.map((row) => (
                        <TableCell key={row.label} className="text-right text-muted-foreground">
                          {row.benchmark === false ? '–' : row.format(benchmark.search)}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableBody>
                </Table>
              </div>
            </SectionCard>
          </TabsContent>

          {/* Next entry */}
          <TabsContent value="forecast">
            <NextEntryForecast genome={genome} settings={context.settings} snapshot={snapshot} onRefresh={onRefreshSnapshot} />
          </TabsContent>

          {/* Validation & stress */}
          <TabsContent value="validation" className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <SectionCard title="Monte Carlo stress test" hint="Replays · 10% of trades skipped · shuffled order · 2× cost shocks">
              {mcSummary ? (
                <>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile label="P(profit)" value={fmtPct(mcSummary.probProfit * 100, false, 0)} tone={mcSummary.passed ? 'text-up' : mcSummary.probProfit >= 0.6 ? 'text-warn' : 'text-down'} />
                    <StatTile label="95% worst DD" value={fmtPct(mcSummary.drawdownP95, false, 1)} tone={mcSummary.drawdownP95 < 50 ? 'text-foreground' : 'text-down'} />
                    <StatTile label="Median return" value={fmtPct(mcSummary.medianReturn, true, 0)} tone={toneClass(mcSummary.medianReturn)} />
                    <StatTile label="5th pct return" value={fmtPct(mcSummary.p5Return, true, 0)} tone={toneClass(mcSummary.p5Return)} />
                  </div>
                  <div className="mt-3">
                    {mc ? (
                      <MonteCarloChart report={mc} />
                    ) : (
                      <p className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="size-4 animate-spin" /> Drawing the stress curve…
                      </p>
                    )}
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {mcSummary.simulations} replays of its {mcSummary.trades} search-period trades
                    {pooled ? ` (pooled over the ${context.universe.length} assets)` : ''}, each fill shocked by on average twice its modelled cost. The
                    backtest&apos;s own drawdown on the same trades was {fmtPct(mcSummary.baseDrawdown, false, 1)}
                    {mcSummary.drawdownP95 > 1.5 * mcSummary.baseDrawdown ? ' – the stressed worst case is much deeper, so the backtest saw a lucky order of trades.' : '.'}{' '}
                    Pass: at least 80% of replays profitable and the 95th-percentile drawdown under 50%.
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">No trades to replay.</p>
              )}
            </SectionCard>

            <SectionCard title="Parameter stability" hint="Train + validation · primary asset">
              {report?.stability ? (
                <StabilityHeatmap grid={report.stability} />
              ) : report ? (
                <p className="text-sm text-muted-foreground">This strategy has no tunable parameter to map.</p>
              ) : (
                <p className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" /> {analysisPending ? 'Mapping the parameter neighbourhood…' : 'Waiting for the analysis…'}
                </p>
              )}
            </SectionCard>

            <SectionCard title="Cross-market check" hint="Frozen rules · unseen markets · holdout dates excluded">
              <p className="text-sm text-secondary-foreground">
                Test these exact rules, untouched, on related markets the search never saw – for example SOL and BNB for a strategy found on BTC and
                ETH. A structural edge carries over; a fit to one chart&apos;s noise does not. Only candles before{' '}
                {axis ? fmtDate(axis.time[axis.holdoutPos]) : 'the holdout'} are used, so the holdout stays sealed.
              </p>
              <form
                className="mt-3 flex flex-wrap items-center gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (crossSymbols.length > 0) onRunCrossMarket(crossSymbols);
                }}
              >
                <Input
                  type="text"
                  value={crossInput}
                  onChange={(event) => setCrossInput(event.target.value.toUpperCase())}
                  aria-label="Unseen tickers for the cross-market check"
                  spellCheck={false}
                  placeholder="SOL-USD, BNB-USD, …"
                  className="num h-9 min-w-[200px] flex-1"
                />
                <Button type="submit" variant="outline" disabled={crossMarket?.status === 'loading' || crossSymbols.length === 0} className="h-9 border-primary/40 text-primary">
                  {crossMarket?.status === 'loading' ? <Loader2 className="animate-spin" /> : <Globe />}
                  Run check
                </Button>
              </form>
              {crossMarket?.status === 'error' && (
                <Alert variant="destructive" className="mt-2">
                  <TriangleAlert />
                  <AlertDescription>{crossMarket.error}</AlertDescription>
                </Alert>
              )}
              {cross && (
                <div className="mt-3">
                  <div className="mb-2 flex items-center gap-2 text-sm">
                    {cross.passed ? <CircleCheck className="size-4 text-up" /> : <CircleX className="size-4 text-down" />}
                    <span className="text-secondary-foreground">
                      {cross.tested === 0
                        ? 'No unseen asset had enough history to test.'
                        : `Profitable on ${Math.round(cross.share * cross.tested)} of ${cross.tested} unseen markets, median Sharpe ${fmtNum(cross.medianSharpe)}${
                            cross.passed ? ' – the edge generalises.' : ' – two thirds profitable and a positive median are needed.'
                          }`}
                    </span>
                  </div>
                  <div className="rounded-lg border border-border">
                    <Table className="num">
                      <TableHeader>
                        <TableRow className={HEAD_ROW}>
                          <TableHead>Market</TableHead>
                          <TableHead className="text-right">ROI</TableHead>
                          <TableHead className="text-right">B&amp;H ROI</TableHead>
                          <TableHead className="text-right">Sharpe</TableHead>
                          <TableHead className="text-right">Max DD</TableHead>
                          <TableHead className="text-right">Trades</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {cross.rows.map((row) => (
                          <TableRow key={row.symbol}>
                            <TableHead scope="row" className="font-normal text-foreground">
                              {row.symbol}
                              {!row.unseen && <span className="ml-1.5 text-[11px] text-muted-foreground">(in the run – not counted)</span>}
                            </TableHead>
                            {row.note ? (
                              <TableCell colSpan={5} className="whitespace-normal text-right text-xs text-muted-foreground">
                                {row.note}
                              </TableCell>
                            ) : (
                              <>
                                <TableCell className={cn('text-right', toneClass(row.roi))}>{fmtPct(row.roi, true)}</TableCell>
                                <TableCell className="text-right text-muted-foreground">{fmtPct(row.holdRoi, true)}</TableCell>
                                <TableCell className="text-right text-secondary-foreground">{fmtNum(row.sharpe)}</TableCell>
                                <TableCell className="text-right text-secondary-foreground">{fmtPct(row.maxDrawdown)}</TableCell>
                                <TableCell className="text-right text-secondary-foreground">{fmtInt(row.trades)}</TableCell>
                              </>
                            )}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </SectionCard>

            <SectionCard title="Locked holdout reveal" hint="The last 20% of history · one strategy per run">
              {revealed && strategy.holdout ? (
                <div className="space-y-3 text-sm text-secondary-foreground">
                  <div className="flex flex-wrap items-center gap-2">
                    <Lock className="size-3.5 text-primary" />
                    <span className="font-semibold text-foreground">Revealed for {strategy.id}.</span>
                    {verdict !== null && <VerdictPill verdict={verdict} />}
                  </div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <StatTile label="Holdout ROI" value={fmtPct(strategy.holdout.roi, true)} tone={toneClass(strategy.holdout.roi)} />
                    <StatTile label="Holdout Sharpe" value={fmtNum(strategy.holdout.sharpe)} />
                    <StatTile label="B&H Sharpe" value={fmtNum(benchmark.holdout?.sharpe ?? null)} />
                    <StatTile label="Holdout trades" value={fmtInt(holdoutTrades)} />
                  </div>
                  <ul className="space-y-1.5">
                    <TestRow pass={strategy.dsr >= 0.95} label="Deflated Sharpe ≥ 95%" detail={fmtPct(strategy.dsr * 100, false, 1)} />
                    {holdoutCi && <TestRow pass={holdoutCi[0] > 0} label="Holdout Sharpe 95% interval above zero" detail={`${fmtNum(holdoutCi[0], 2)} to ${fmtNum(holdoutCi[1], 2)}`} />}
                    <TestRow
                      pass={strategy.holdout.sharpe > (benchmark.holdout?.sharpe ?? Infinity)}
                      label="Holdout Sharpe beats Buy & Hold"
                      detail={`${fmtNum(strategy.holdout.sharpe)} vs ${fmtNum(benchmark.holdout?.sharpe ?? null)} · ROI ${fmtPct(strategy.holdout.roi, true)} vs ${fmtPct(benchmark.holdout?.roi ?? null, true)}`}
                    />
                    <TestRow
                      pass={holdoutTrades >= holdoutGate}
                      label={`At least ${holdoutGate} holdout trades${pooled ? ' (pooled over the assets)' : ''}`}
                      detail={fmtInt(holdoutTrades)}
                    />
                  </ul>
                  {strategy.verdictNotes && strategy.verdictNotes.length > 0 ? (
                    <div>
                      <div className="text-xs font-semibold text-foreground">{verdict === 'Overfit' ? 'Rejected:' : 'Why it is not Robust:'}</div>
                      <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs">
                        {strategy.verdictNotes.map((note) => (
                          <li key={note}>{note}</li>
                        ))}
                      </ul>
                    </div>
                  ) : verdict === 'Robust' ? (
                    <p className="text-xs text-up">Every Robust condition held: the search-period gates, the untouched holdout, the stress test and generalisation beyond one chart.</p>
                  ) : null}
                  {context.priorHoldoutReveals > 0 && (
                    <p className="text-xs text-warn">
                      This holdout period had already been revealed {context.priorHoldoutReveals} time{context.priorHoldoutReveals === 1 ? '' : 's'} before
                      this run, so it is no longer untouched data.
                    </p>
                  )}
                </div>
              ) : !finished ? (
                <Alert>
                  <Info />
                  <AlertDescription>The holdout is not simulated while the search is running. When the run ends you can lock ONE strategy to reveal its holdout.</AlertDescription>
                </Alert>
              ) : lockedId !== null ? (
                <Alert>
                  <Lock />
                  <AlertDescription>
                    <span>
                      This run&apos;s holdout was spent on strategy <span className="num font-semibold text-foreground">{lockedId}</span>. Every other
                      strategy&apos;s holdout stays sealed, so the holdout cannot be used to pick a winner.
                    </span>
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="flex flex-wrap items-center gap-3 text-xs leading-relaxed text-secondary-foreground">
                  <p className="min-w-0 flex-1">
                    <span className="font-semibold text-foreground">Lock this strategy as the run&apos;s final answer?</span> Its holdout is revealed once.
                    No other strategy of this run can be revealed afterwards, and the reveal is recorded for later runs on the same symbols and dates.
                  </p>
                  <Dialog open={lockDialog} onOpenChange={setLockDialog}>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm" className="border-primary/40 bg-primary/10 text-primary">
                        <Lock /> Lock &amp; reveal holdout
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                          <Lock className="size-4 text-primary" /> Lock {strategy.id} and reveal its holdout?
                        </DialogTitle>
                        <DialogDescription>
                          The holdout – the last 20% of history – is revealed once, for this strategy only. No other strategy of this run can be revealed
                          afterwards, and the reveal is recorded for every later run on the same symbols and dates.
                        </DialogDescription>
                      </DialogHeader>
                      <DialogFooter>
                        <DialogClose asChild>
                          <Button variant="outline">Cancel</Button>
                        </DialogClose>
                        <Button
                          onClick={() => {
                            setLockDialog(false);
                            onLock();
                          }}
                          disabled={lockPending}
                        >
                          {lockPending ? <Loader2 className="animate-spin" /> : <Lock />}
                          Yes, lock {strategy.id} and reveal
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              )}
            </SectionCard>
          </TabsContent>

          {/* Rules */}
          <TabsContent value="rules" className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <SectionCard title="The idea" hint={hypothesis ? `Hypothesis: ${hypothesis.label}` : ARCHETYPE_LABEL[genome.archetype]}>
              {hypothesis && (
                <p className="mb-2 text-sm leading-relaxed text-foreground">
                  <span className="font-semibold">{hypothesis.label}.</span> {hypothesis.rationale}
                </p>
              )}
              <p className="text-sm leading-relaxed text-secondary-foreground">{describeArchetype(genome)}</p>
              {hypothesis && <p className="mt-2 text-xs text-muted-foreground">One primitive per idea: it stands in for {hypothesis.replaces}</p>}
            </SectionCard>

            <SectionCard title="Indicators & exact parameters" hint={`${strategy.params} tunable values (cap ${MAX_PARAMETERS}${genome.shortRegime ? ' + 1 short regime filter' : ''})`}>
              <div className="mb-3 flex flex-wrap gap-1">
                {indicatorBadges(genome).map((badge) => (
                  <Pill key={badge}>{badge}</Pill>
                ))}
              </div>
              <div className="rounded-lg border border-border">
                <Table>
                  <TableHeader>
                    <TableRow className={HEAD_ROW}>
                      <TableHead>Indicator</TableHead>
                      <TableHead>Parameters</TableHead>
                      <TableHead>Used for</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {indicators.map((usage) => (
                      <TableRow key={`${usage.name}|${usage.params}`}>
                        <TableCell className="font-medium text-foreground">{usage.name}</TableCell>
                        <TableCell className="num whitespace-normal text-secondary-foreground">{usage.params}</TableCell>
                        <TableCell className="whitespace-normal text-muted-foreground">{usage.roles.join(', ')}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </SectionCard>

            <SectionCard title="Entry rules">
              <div className="space-y-4">
                <div>
                  <div className="mb-1.5 text-xs font-semibold text-up">Long entry</div>
                  <RuleList rule={longEntry} tone="up" />
                </div>
                <div>
                  <div className="mb-1.5 text-xs font-semibold text-down">
                    Short entry
                    {genome.shortEntry !== null && ` (${genome.shortRegime ? 'mirrored trigger, only inside its own bear-regime filter' : 'mirror of the long rules'})`}
                  </div>
                  {shortEntry ? <RuleList rule={shortEntry} tone="down" /> : <p className="text-sm text-muted-foreground">None – Long-Only: this strategy never sells short.</p>}
                </div>
              </div>
            </SectionCard>

            <SectionCard title="Exit & risk management rules" hint={`SL/TP mode: ${exit.mode}`}>
              <p className="text-sm text-secondary-foreground">{exit.headline}</p>
              <dl className="mt-3 space-y-2.5">
                {exit.bullets.map((bullet) => (
                  <div key={bullet.title} className="flex gap-2.5">
                    <span aria-hidden="true" className="mt-[7px] size-1.5 shrink-0 rounded-full bg-primary" />
                    <div className="min-w-0 text-sm">
                      <dt className="inline font-semibold text-foreground">{bullet.title}. </dt>
                      <dd className="inline text-secondary-foreground">{bullet.text}</dd>
                    </div>
                  </div>
                ))}
              </dl>
              {exit.longSignalExit && (
                <div className="mt-4 space-y-4 rounded-lg border border-border bg-muted/40 p-3">
                  <RuleList rule={exit.longSignalExit} tone="neutral" />
                  {exit.shortSignalExit && <RuleList rule={exit.shortSignalExit} tone="neutral" />}
                </div>
              )}
            </SectionCard>
          </TabsContent>

          {/* Performance */}
          <TabsContent value="performance" className="space-y-4">
            <SectionCard title="Train vs. validation vs. holdout" hint="First 60% · next 20% · last 20% of candles">
              <div className="rounded-lg border border-border">
                <Table className="num">
                  <TableHeader>
                    <TableRow className={HEAD_ROW}>
                      <TableHead>Metric</TableHead>
                      <TableHead className="text-right">Train</TableHead>
                      <TableHead className="text-right">Validation</TableHead>
                      <TableHead className="text-right">Holdout</TableHead>
                      <TableHead className="text-right">B&amp;H {benchmark.holdout ? 'holdout' : 'validation'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {METRIC_ROWS.map((row) => (
                      <TableRow key={row.label}>
                        <TableHead scope="row" title={row.title} className="font-normal text-secondary-foreground">
                          {row.label}
                        </TableHead>
                        <TableCell className={cn('text-right', row.tone?.(strategy.train) ?? 'text-foreground')}>{row.format(strategy.train)}</TableCell>
                        <TableCell className={cn('text-right', row.tone?.(strategy.validation) ?? 'text-foreground')}>{row.format(strategy.validation)}</TableCell>
                        <TableCell className={cn('text-right font-semibold', strategy.holdout ? (row.tone?.(strategy.holdout) ?? 'text-foreground') : 'font-normal text-muted-foreground')}>
                          {strategy.holdout ? row.format(strategy.holdout) : 'sealed'}
                        </TableCell>
                        <TableCell className="text-right text-muted-foreground">{row.bench ? row.format(benchSegment) : '–'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              {ambiguousShare > 0.1 && (
                <Alert className="mt-3 border-warn/40 bg-warn/10 text-warn">
                  <TriangleAlert />
                  <AlertDescription className="text-current">
                    <span>
                      <span className="num">{fmtPct(ambiguousShare * 100, false, 0)}</span> of trades closed on a candle that touched both the stop and the
                      target. The stop is assumed to fill first, but the real order of events inside those candles is unknown – a shorter timeframe would
                      settle it.
                    </span>
                  </AlertDescription>
                </Alert>
              )}
            </SectionCard>

            {strategy.assets.length > 1 && (
              <SectionCard title="Per-asset results" hint="The same rules on every asset of the pool">
                <div className="rounded-lg border border-border">
                  <Table className="num">
                    <TableHeader>
                      <TableRow className={HEAD_ROW}>
                        <TableHead>Asset</TableHead>
                        <TableHead className="text-right">Train ROI</TableHead>
                        <TableHead className="text-right">Validation ROI</TableHead>
                        <TableHead className="text-right">Validation Sharpe</TableHead>
                        <TableHead className="text-right">Holdout ROI</TableHead>
                        <TableHead className="text-right">Trades</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {strategy.assets.map((asset) => (
                        <TableRow key={asset.symbol}>
                          <TableHead scope="row" className="font-normal text-foreground">
                            {asset.symbol}
                          </TableHead>
                          <TableCell className={cn('text-right', toneClass(asset.train.roi))}>{fmtPct(asset.train.roi, true)}</TableCell>
                          <TableCell className={cn('text-right', toneClass(asset.validation.roi))}>{fmtPct(asset.validation.roi, true)}</TableCell>
                          <TableCell className="text-right text-secondary-foreground">{fmtNum(asset.validation.sharpe)}</TableCell>
                          <TableCell className={cn('text-right', asset.holdout ? toneClass(asset.holdout.roi) : 'text-muted-foreground')}>
                            {asset.holdout ? fmtPct(asset.holdout.roi, true) : 'sealed'}
                          </TableCell>
                          <TableCell className="text-right text-secondary-foreground">{fmtInt(asset.search.trades)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </SectionCard>
            )}
          </TabsContent>

          {/* Pine Script */}
          <TabsContent value="export">
            <SectionCard title="Export to TradingView" hint="Pine Script v5 strategy · runs on every TradingView plan">
              <p className="text-sm text-secondary-foreground">
                Copy or download the script, open the Pine Editor on a <span className="num font-medium text-foreground">{context.interval}</span> chart of{' '}
                <span className="num font-medium text-foreground">{context.isBasket ? 'the basket (see the script header)' : context.label}</span>, paste it and
                press <em>Add to chart</em>. It reproduces these exact rules – entries for every direction mode, the trailing stop and breakeven lock,
                sizing, costs and next-bar-open execution – with <code className="num text-xs">use_bar_magnifier=false</code>, so it runs on every
                TradingView plan.
              </p>
              <Collapsible open={showCode} onOpenChange={setShowCode}>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button onClick={() => void handleCopy()} className={cn('h-9 px-4', copyState === 'copied' && 'bg-up/15 text-up ring-1 ring-up/40 hover:bg-up/20')}>
                    {copyState === 'copied' ? <Check /> : <Copy />}
                    {copyState === 'copied' ? 'Copied to clipboard' : 'Copy Pine Script v5'}
                  </Button>
                  <Button variant="outline" onClick={() => download(`StrategyFinder-${strategy.id}-${context.interval}.pine`, pine)} className="h-9">
                    <Download /> Download .pine
                  </Button>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="h-9">
                      <Code /> {showCode ? 'Hide code' : 'View code'}
                    </Button>
                  </CollapsibleTrigger>
                  <span role="status" aria-live="polite" className="text-xs text-down">
                    {copyState === 'failed' ? 'Clipboard blocked – select the code below and copy it manually.' : ''}
                  </span>
                </div>
                <CollapsibleContent>
                  <pre
                    tabIndex={0}
                    aria-label="Generated Pine Script"
                    className="num mt-3 max-h-[420px] overflow-auto rounded-lg border border-border bg-background p-3 text-[11.5px] leading-relaxed text-secondary-foreground"
                  >
                    <code>{pine}</code>
                  </pre>
                </CollapsibleContent>
              </Collapsible>
            </SectionCard>
          </TabsContent>
        </Tabs>
      </CardContent>

      <p className="border-t px-5 py-4 text-[11px] leading-relaxed text-muted-foreground">
        Backtests are hypothetical and ignore liquidity, borrow availability and taxes. Nothing here is financial advice.
      </p>
    </Panel>
  );
}
