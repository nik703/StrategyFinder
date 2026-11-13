/**
 * StrategyFinder dashboard.
 *
 * The main thread only loads market data, renders the UI and relays messages:
 * every indicator, backtest and optimisation step runs in `optimizer.worker.ts`.
 */

import { Activity, Pin, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ControlBar from './components/ControlBar';
import EquityChart, { BENCHMARK_COLOR, STRATEGY_COLOR } from './components/EquityChart';
import Leaderboard from './components/Leaderboard';
import StrategyModal from './components/StrategyModal';
import TelemetryStrip from './components/TelemetryStrip';
import { cx, LineKey, Panel, Pill } from './components/ui';
import { buildDataset } from './data/basket';
import { loadMarketData } from './data/marketData';
import { exitModeLabel, indicatorBadges } from './engine/describe';
import { DEFAULT_CONFIG, resolveAssets, settingsFrom, type RunConfig, type RunContext } from './lib/config';
import { fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from './lib/format';
import type {
  BenchmarkResult,
  ChartAxis,
  EngineStatus,
  PerformanceMetrics,
  ProgressSnapshot,
  StrategyResult,
  WorkerRequest,
  WorkerResponse,
} from './types/trading';

const EMPTY_TELEMETRY: ProgressSnapshot = {
  iteration: 0,
  stagnation: 0,
  bestFitness: null,
  bestRoi: null,
  itersPerSec: 0,
  elapsedMs: 0,
};

interface ComparisonRow {
  label: string;
  format: (m: PerformanceMetrics) => string;
  tone?: (m: PerformanceMetrics) => string;
  /** False for trade statistics that do not apply to Buy & Hold. */
  benchmark?: boolean;
}

const COMPARISON_ROWS: ComparisonRow[] = [
  { label: 'Net PnL', format: (m) => fmtUsd(m.netPnl, true), tone: (m) => toneClass(m.netPnl) },
  { label: 'ROI', format: (m) => fmtPct(m.roi, true), tone: (m) => toneClass(m.roi) },
  { label: 'CAGR', format: (m) => fmtPct(m.cagr, true), tone: (m) => toneClass(m.cagr) },
  { label: 'Max drawdown', format: (m) => fmtPct(m.maxDrawdown) },
  { label: 'Sharpe ratio', format: (m) => fmtNum(m.sharpe) },
  { label: 'Sortino ratio', format: (m) => fmtNum(m.sortino) },
  { label: 'Win rate', format: (m) => fmtPct(m.winRate), benchmark: false },
  { label: 'Profit factor', format: (m) => fmtNum(m.profitFactor), benchmark: false },
  { label: 'Trades', format: (m) => fmtInt(m.trades), benchmark: false },
  { label: 'Time in market', format: (m) => fmtPct(m.exposure, false, 0) },
];

export default function App(): JSX.Element {
  const [config, setConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<EngineStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<ProgressSnapshot>(EMPTY_TELEMETRY);
  const [leaderboard, setLeaderboard] = useState<StrategyResult[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [axis, setAxis] = useState<ChartAxis | null>(null);
  const [barsPerYear, setBarsPerYear] = useState(365);
  const [context, setContext] = useState<RunContext | null>(null);
  /** Strategy pinned by clicking a leaderboard row; null = follow the current #1. */
  const [pinned, setPinned] = useState<StrategyResult | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const runIdRef = useRef(0);
  const pendingProgress = useRef<ProgressSnapshot | null>(null);
  const frameRef = useRef<number | null>(null);

  const disposeWorker = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    pendingProgress.current = null;
  }, []);

  useEffect(() => disposeWorker, [disposeWorker]);

  /** Progress arrives every 25 iterations (hundreds of times per second on fast data) – paint at most once per frame. */
  const queueProgress = useCallback((snapshot: ProgressSnapshot) => {
    pendingProgress.current = snapshot;
    if (frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = null;
      if (pendingProgress.current) setTelemetry(pendingProgress.current);
      pendingProgress.current = null;
    });
  }, []);

  const handleMessage = useCallback(
    (message: WorkerResponse, runId: number) => {
      if (runId !== runIdRef.current) return; // stale message from a superseded run
      switch (message.type) {
        case 'ready':
          setBenchmark(message.benchmark);
          setAxis(message.axis);
          setBarsPerYear(message.barsPerYear);
          break;
        case 'progress': {
          const { type: _type, ...snapshot } = message;
          queueProgress(snapshot);
          break;
        }
        case 'leaderboard':
          setLeaderboard(message.strategies);
          break;
        case 'done': {
          const { type: _type, reason, ...snapshot } = message;
          if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
          frameRef.current = null;
          pendingProgress.current = null;
          setTelemetry(snapshot);
          setStatus(reason === 'converged' ? 'converged' : 'stopped');
          workerRef.current?.terminate();
          workerRef.current = null;
          break;
        }
        case 'error':
          setError(message.message);
          setStatus('error');
          disposeWorker();
          break;
      }
    },
    [disposeWorker, queueProgress],
  );

  const start = useCallback(async () => {
    const assets = resolveAssets(config);
    if (!assets.ok) {
      setError(assets.error);
      return;
    }
    disposeWorker();
    const runId = ++runIdRef.current;
    setError(null);
    setStatus('loading');
    setTelemetry(EMPTY_TELEMETRY);
    setLeaderboard([]);
    setBenchmark(null);
    setAxis(null);
    setContext(null);
    setPinned(null);
    setModalOpen(false);

    try {
      const loaded = await loadMarketData(
        assets.components.map((component) => component.symbol),
        config.interval,
        config.range,
      );
      if (runId !== runIdRef.current) return; // cancelled while downloading
      const dataset = buildDataset(loaded.series, assets.components);
      const settings = settingsFrom(config);
      const bars = dataset.data.t.length;
      setContext({
        label: dataset.label,
        interval: config.interval,
        range: config.range,
        settings,
        maxStagnation: config.maxStagnation,
        components: dataset.components,
        units: dataset.units,
        isBasket: dataset.isBasket,
        bars,
        windowStart: dataset.data.t[0],
        windowEnd: dataset.data.t[bars - 1],
        sources: dataset.sources,
        simulated: loaded.usedFallback,
        fallbackReason: loaded.fallbackReason,
        notes: loaded.notes,
      });

      const worker = new Worker(new URL('./workers/optimizer.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => handleMessage(event.data, runId);
      worker.onerror = (event) => {
        if (runId !== runIdRef.current) return;
        setError(event.message || 'The optimisation worker crashed.');
        setStatus('error');
        disposeWorker();
      };
      const request: WorkerRequest = { type: 'start', payload: { data: dataset.data, settings, maxStagnation: config.maxStagnation } };
      worker.postMessage(request);
      setStatus('optimizing');
    } catch (cause) {
      if (runId !== runIdRef.current) return;
      setError(cause instanceof Error ? cause.message : 'Could not start the optimisation.');
      setStatus('error');
    }
  }, [config, disposeWorker, handleMessage]);

  const stop = useCallback(() => {
    if (status === 'optimizing' && workerRef.current) {
      // Cooperative stop: the worker finishes its current batch, posts the final leaderboard and exits.
      const request: WorkerRequest = { type: 'stop' };
      workerRef.current.postMessage(request);
    } else if (status === 'loading') {
      runIdRef.current++;
      setStatus('stopped');
    }
  }, [status]);

  const selectStrategy = useCallback((strategy: StrategyResult) => {
    setPinned(strategy);
    setModalOpen(true);
  }, []);

  const closeModal = useCallback(() => setModalOpen(false), []);

  const displayed = pinned ?? leaderboard[0] ?? null;
  const rankOf = useCallback(
    (strategy: StrategyResult | null): number | null => {
      if (!strategy) return null;
      const index = leaderboard.findIndex((entry) => entry.id === strategy.id);
      return index >= 0 ? index + 1 : null;
    },
    [leaderboard],
  );
  const displayedRank = rankOf(displayed);
  const oosYears = useMemo(
    () => (axis ? (axis.bars - axis.splitIndex) / Math.max(1, barsPerYear) : 0),
    [axis, barsPerYear],
  );

  const strategyName = displayed
    ? `Strategy ${displayed.id}${displayedRank !== null ? ` · #${displayedRank}` : ''}`
    : null;
  const capital = context?.settings.initialCapital ?? config.capital;

  return (
    <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col gap-4 px-4 py-5 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-accent/40 bg-accent/15 text-accent-strong">
            <Activity size={18} />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-5 text-ink">StrategyFinder</h1>
            <p className="text-xs text-ink-3">Discover, backtest and optimise trading strategies for stocks, crypto and weighted baskets</p>
          </div>
        </div>
        <p className="text-xs text-ink-3">
          Signals on candle close · fills at next open · fees &amp; slippage included · 70 / 30 in-sample / out-of-sample
        </p>
      </header>

      <ControlBar
        config={config}
        onChange={(patch) => setConfig((current) => ({ ...current, ...patch }))}
        status={status}
        onStart={() => void start()}
        onStop={stop}
      />

      <TelemetryStrip
        maxStagnation={context?.maxStagnation ?? config.maxStagnation}
        status={status} telemetry={telemetry} benchmarkRoi={benchmark?.full.roi ?? null} context={context} />

      {error && (
        <div role="alert" className="flex items-start gap-2 rounded-lg border border-down/40 bg-down/10 px-3 py-2.5 text-sm text-down">
          <TriangleAlert size={16} className="mt-0.5 shrink-0" />
          <span className="min-w-0 flex-1">{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error" className="shrink-0 rounded p-0.5 hover:bg-down/20">
            <X size={14} />
          </button>
        </div>
      )}

      {context?.simulated && (
        <div className="flex items-start gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2.5 text-sm text-warn">
          <TriangleAlert size={16} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <span className="font-semibold">Simulated market data.</span> The market-data function could not supply live candles, so the
            built-in generator produced realistic placeholder prices. Results are illustrative only.
            {context.fallbackReason && <span className="num mt-0.5 block truncate text-xs opacity-80">Reason: {context.fallbackReason}</span>}
          </div>
        </div>
      )}

      {context && !context.simulated && context.notes.length > 0 && (
        <div className="rounded-lg border border-line bg-panel px-3 py-2 text-xs text-ink-2">{context.notes.join(' · ')}</div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        {/* Equity curve */}
        <Panel className="flex h-[440px] flex-col pt-3 xl:h-auto xl:min-h-[480px]">
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-3">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-ink">Equity curve</h2>
              <p className="num truncate text-xs text-ink-3">
                {context ? `${context.label} · ${context.interval} · starting capital ${fmtUsd(capital)}` : 'Selected strategy vs. Buy & Hold'}
              </p>
            </div>
            {displayed && (
              <div className="flex items-center gap-2">
                {pinned ? (
                  <button
                    type="button"
                    onClick={() => setPinned(null)}
                    className="inline-flex h-7 items-center gap-1.5 rounded-md border border-line px-2 text-xs text-ink-2 transition-colors hover:bg-hover hover:text-ink"
                    title="Stop previewing the clicked strategy and follow the current #1"
                  >
                    <Pin size={12} /> Follow best
                  </button>
                ) : (
                  <Pill tone="accent">Following #1</Pill>
                )}
                <button
                  type="button"
                  onClick={() => setModalOpen(true)}
                  className="inline-flex h-7 items-center rounded-md border border-accent/40 bg-accent/10 px-2 text-xs font-medium text-accent-strong transition-colors hover:bg-accent/20"
                >
                  View breakdown
                </button>
              </div>
            )}
          </div>
          <div className="min-h-0 flex-1">
            <EquityChart
              axis={axis}
              benchmark={benchmark?.equity ?? null}
              strategy={displayed?.equity ?? null}
              strategyName={strategyName}
              initialCapital={capital}
              interval={context?.interval ?? config.interval}
              placeholder={
                status === 'loading'
                  ? 'Loading market data…'
                  : 'Pick an asset or build a basket, then press Start Optimization to see the equity curve.'
              }
            />
          </div>
        </Panel>

        {/* Strategy vs. Buy & Hold */}
        <Panel className="flex flex-col">
          <div className="border-b border-line px-4 py-3">
            <h2 className="text-sm font-semibold text-ink">Strategy vs. Buy &amp; Hold</h2>
            <p className="text-xs text-ink-3">Same candles, same starting capital, same costs</p>
          </div>
          {benchmark ? (
            <div className="flex min-h-0 flex-1 flex-col">
              <table className="num w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-[11px] uppercase tracking-wider text-ink-3">
                    <th scope="col" className="px-4 py-2 text-left font-medium">Metric</th>
                    <th scope="col" className="whitespace-nowrap px-2 py-2 text-right font-medium">
                      <span className="inline-flex items-center gap-1.5">
                        <LineKey color={STRATEGY_COLOR} /> Strategy
                      </span>
                    </th>
                    <th scope="col" className="whitespace-nowrap px-4 py-2 text-right font-medium">
                      <span className="inline-flex items-center gap-1.5">
                        <LineKey color={BENCHMARK_COLOR} /> Buy &amp; Hold
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON_ROWS.map((row) => (
                    <tr key={row.label} className="border-b border-line/60 last:border-b-0">
                      <th scope="row" className="whitespace-nowrap px-4 py-[7px] text-left font-normal text-ink-2">{row.label}</th>
                      <td className={cx('px-2 py-[7px] text-right font-medium', displayed ? (row.tone?.(displayed.full) ?? 'text-ink') : 'text-ink-3')}>
                        {displayed ? row.format(displayed.full) : '–'}
                      </td>
                      <td className={cx('px-4 py-[7px] text-right', row.benchmark === false ? 'text-ink-3' : (row.tone?.(benchmark.full) ?? 'text-ink-2'))}>
                        {row.benchmark === false ? '–' : row.format(benchmark.full)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-auto border-t border-line px-4 py-3">
                {displayed ? (
                  <>
                    <div className="flex flex-wrap items-center gap-1">
                      {indicatorBadges(displayed.genome).map((badge) => (
                        <Pill key={badge}>{badge}</Pill>
                      ))}
                      <Pill tone={displayed.genome.exit.mode === 'static' ? 'neutral' : 'accent'}>{exitModeLabel(displayed.genome)}</Pill>
                    </div>
                    <p className="num mt-2 text-xs text-ink-3">
                      Alpha vs. Buy &amp; Hold{' '}
                      <span className={cx('font-semibold', toneClass(displayed.alpha))}>{fmtPct(displayed.alpha, true)}</span> · out-of-sample
                      ROI <span className={cx('font-semibold', toneClass(displayed.outOfSample.roi))}>{fmtPct(displayed.outOfSample.roi, true)}</span>{' '}
                      vs {fmtPct(benchmark.outOfSample.roi, true)}
                    </p>
                  </>
                ) : (
                  <p className="text-xs text-ink-3">The Buy &amp; Hold baseline is ready. Strategy figures appear as soon as one is found.</p>
                )}
              </div>
            </div>
          ) : (
            <div className="grid flex-1 place-items-center px-6 py-10 text-center">
              <p className="text-sm text-ink-3">
                Net PnL, ROI, CAGR, max drawdown and Sharpe for Buy &amp; Hold appear here once market data has loaded.
              </p>
            </div>
          )}
        </Panel>
      </div>

      <Leaderboard strategies={leaderboard} selectedId={displayed?.id ?? null} onSelect={selectStrategy} status={status} />

      <footer className="pb-2 text-[11px] leading-relaxed text-ink-3">
        StrategyFinder is a research tool. Backtests are hypothetical, are fitted to past data and do not guarantee future results. Nothing
        here is financial advice.
      </footer>

      {modalOpen && displayed && benchmark && context && (
        <StrategyModal
          strategy={displayed}
          rank={displayedRank}
          benchmark={benchmark}
          context={context}
          testedCount={telemetry.iteration}
          oosYears={oosYears}
          onClose={closeModal}
        />
      )}
    </div>
  );
}
