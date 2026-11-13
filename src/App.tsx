/**
 * StrategyFinder dashboard.
 *
 * The main thread only loads market data, renders the UI and relays messages:
 * every indicator, backtest and optimisation step runs in `optimizer.worker.ts`.
 */

import { Activity, Info, Pin, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import ControlBar from './components/ControlBar';
import EquityChart, { BENCHMARK_COLOR, STRATEGY_COLOR } from './components/EquityChart';
import Leaderboard from './components/Leaderboard';
import StrategyModal from './components/StrategyModal';
import TelemetryStrip from './components/TelemetryStrip';
import { VerdictPill } from './components/Leaderboard';
import { cx, LineKey, Panel, Pill } from './components/ui';
import { buildDataset } from './data/basket';
import { loadMarketData } from './data/marketData';
import { exitModeLabel, indicatorBadges } from './engine/describe';
import {
  DEFAULT_CONFIG,
  defaultUniverse,
  isCrypto,
  parseUniverse,
  resolveAssets,
  resolveUniverse,
  settingsFrom,
  venueLabel,
  type RunConfig,
  type RunContext,
} from './lib/config';
import { fmtInt, fmtNum, fmtPct, fmtUsd, toneClass } from './lib/format';
import { datasetFingerprint, holdoutViews, recordHoldoutView } from './lib/holdoutLog';
import {
  DEFAULT_CRYPTO_UNIVERSE,
  DEFAULT_STOCK_UNIVERSE,
  type BenchmarkResult,
  type ChartAxis,
  type EngineStatus,
  type NamedDataset,
  type PerformanceMetrics,
  type ProgressSnapshot,
  type RobustnessReport,
  type StrategyResult,
  type Verdict,
  type WorkerRequest,
  type WorkerResponse,
} from './types/trading';

type Analysis = { report: RobustnessReport; verdict: Verdict | null };

const DEFAULT_UNIVERSES = [DEFAULT_CRYPTO_UNIVERSE.join(', '), DEFAULT_STOCK_UNIVERSE.join(', ')];

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
  /** On-demand robustness reports, by strategy id. */
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({});
  const [pendingAnalyses, setPendingAnalyses] = useState<Record<string, boolean>>({});

  const workerRef = useRef<Worker | null>(null);
  const runIdRef = useRef(0);
  const pendingProgress = useRef<ProgressSnapshot | null>(null);
  const frameRef = useRef<number | null>(null);
  const contextRef = useRef<RunContext | null>(null);
  const finishedRef = useRef(false);
  /** Analysis requests already sent, keyed by strategy id and run phase. */
  const requestedRef = useRef(new Set<string>());
  /** Extra assets for the cross-asset test of single-asset runs: loaded once per run. */
  const extraUniverseRef = useRef<Promise<NamedDataset[]> | null>(null);
  const extraUniverseSentRef = useRef(false);

  const disposeWorker = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    pendingProgress.current = null;
  }, []);

  useEffect(() => disposeWorker, [disposeWorker]);

  /** Changing the ticker's asset class switches the Robust universe to that class's default. */
  const updateConfig = useCallback((patch: Partial<RunConfig>) => {
    setConfig((current) => {
      const next = { ...current, ...patch };
      if (patch.symbol !== undefined && isCrypto(patch.symbol) !== isCrypto(current.symbol)) {
        next.robustUniverse = isCrypto(patch.symbol); // on by default for crypto, off for stocks
        if (DEFAULT_UNIVERSES.includes(current.universe.trim())) next.universe = defaultUniverse(patch.symbol);
      }
      return next;
    });
  }, []);

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
          const { type: _type, reason, strategies, benchmark: finalBenchmark, ...snapshot } = message;
          if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
          frameRef.current = null;
          pendingProgress.current = null;
          finishedRef.current = true;
          setTelemetry(snapshot);
          // The search is over: the final leaderboard and benchmark now include the holdout.
          setLeaderboard(strategies);
          setBenchmark(finalBenchmark);
          setPinned((current) => (current ? (strategies.find((s) => s.id === current.id) ?? null) : null));
          setAnalyses({});
          setPendingAnalyses({});
          if (contextRef.current) recordHoldoutView(contextRef.current.fingerprint);
          setStatus(reason === 'converged' ? 'converged' : 'stopped');
          // The worker stays alive to answer robustness-analysis requests.
          break;
        }
        case 'analysis':
          setAnalyses((current) => ({ ...current, [message.strategyId]: { report: message.report, verdict: message.verdict } }));
          setPendingAnalyses((current) => ({ ...current, [message.strategyId]: false }));
          break;
        case 'error':
          setError(message.message);
          setPendingAnalyses({});
          if (!finishedRef.current) {
            setStatus('error');
            disposeWorker();
          }
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
    const universe = resolveUniverse(config, assets.components[0].symbol);
    if (!universe.ok) {
      setError(universe.error);
      return;
    }
    disposeWorker();
    const runId = ++runIdRef.current;
    finishedRef.current = false;
    requestedRef.current = new Set();
    extraUniverseRef.current = null;
    extraUniverseSentRef.current = false;
    contextRef.current = null;
    setError(null);
    setStatus('loading');
    setTelemetry(EMPTY_TELEMETRY);
    setLeaderboard([]);
    setBenchmark(null);
    setAxis(null);
    setContext(null);
    setPinned(null);
    setModalOpen(false);
    setAnalyses({});
    setPendingAnalyses({});

    try {
      const tradedSymbols = assets.components.map((component) => component.symbol);
      const symbols = [...new Set([...tradedSymbols, ...universe.symbols])];
      const loaded = await loadMarketData(symbols, config.interval, config.range);
      if (runId !== runIdRef.current) return; // cancelled while downloading
      const dataset = buildDataset(loaded.series, assets.components);
      const settings = settingsFrom(config);
      const bars = dataset.data.t.length;
      // The primary asset first; every other universe asset is backtested separately with the same rules.
      const datasets: NamedDataset[] = [{ symbol: dataset.isBasket ? 'Basket' : tradedSymbols[0], data: dataset.data }];
      for (const symbol of universe.symbols.slice(1)) {
        datasets.push({ symbol, data: buildDataset(loaded.series, [{ symbol, weight: 1 }]).data });
      }
      const fingerprint = datasetFingerprint(symbols, config.interval, config.range, dataset.data.t[bars - 1]);
      const runContext: RunContext = {
        label: dataset.label,
        interval: config.interval,
        range: config.range,
        settings,
        maxStagnation: config.maxStagnation,
        universe: datasets.map((entry) => entry.symbol),
        venue: venueLabel(tradedSymbols.map((symbol) => loaded.series[symbol])),
        fingerprint,
        priorHoldoutViews: holdoutViews(fingerprint),
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
      };
      contextRef.current = runContext;
      setContext(runContext);

      const worker = new Worker(new URL('./workers/optimizer.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => handleMessage(event.data, runId);
      worker.onerror = (event) => {
        if (runId !== runIdRef.current) return;
        setError(event.message || 'The optimisation worker crashed.');
        setStatus('error');
        disposeWorker();
      };
      const request: WorkerRequest = { type: 'start', payload: { datasets, settings, maxStagnation: config.maxStagnation } };
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
      // Cooperative stop: the worker finishes its current batch, reveals the holdout for the final leaderboard and idles.
      const request: WorkerRequest = { type: 'stop' };
      workerRef.current.postMessage(request);
    } else if (status === 'loading') {
      runIdRef.current++;
      setStatus('stopped');
    }
  }, [status]);

  /**
   * Ask the worker for a robustness report. Single-asset runs have no other assets
   * loaded, so the default universe for the asset class is fetched once and sent along
   * for the cross-asset test.
   */
  const requestAnalysis = useCallback(async (strategy: StrategyResult) => {
    const worker = workerRef.current;
    const runContext = contextRef.current;
    const runId = runIdRef.current;
    if (!worker || !runContext) return;
    setPendingAnalyses((current) => ({ ...current, [strategy.id]: true }));

    let universe: NamedDataset[] | undefined;
    const needsUniverse = !runContext.isBasket && runContext.universe.length <= 1 && !extraUniverseSentRef.current;
    if (needsUniverse) {
      const primary = runContext.components[0].symbol;
      extraUniverseRef.current ??= (async () => {
        const symbols = parseUniverse(defaultUniverse(primary)).filter((symbol) => symbol !== primary);
        const loaded = await loadMarketData(symbols, runContext.interval, runContext.range);
        // Never mix live and simulated candles in one comparison.
        if (loaded.usedFallback !== runContext.simulated) return [];
        return symbols.map((symbol) => ({ symbol, data: buildDataset(loaded.series, [{ symbol, weight: 1 }]).data }));
      })().catch(() => []);
      universe = await extraUniverseRef.current;
      if (runId !== runIdRef.current || workerRef.current !== worker) return;
      extraUniverseSentRef.current = true;
    }
    const request: WorkerRequest = {
      type: 'analyze',
      strategyId: strategy.id,
      genome: strategy.genome,
      universe: universe && universe.length > 0 ? universe : undefined,
    };
    worker.postMessage(request);
  }, []);

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
  const revealed = benchmark?.holdout != null;
  const finished = status === 'converged' || status === 'stopped';

  // Opening the breakdown requests a robustness report: once while the run is live
  // (no verdict yet) and once after it has ended (with the holdout and other assets).
  useEffect(() => {
    if (!modalOpen || !displayed || !workerRef.current) return;
    if (status !== 'optimizing' && !finished) return;
    const crossAssetMissing = !context?.isBasket && displayed.robustness?.crossAsset == null;
    if (displayed.robustness !== null && !crossAssetMissing) return;
    const key = `${displayed.id}|${finished ? 'final' : 'live'}`;
    if (requestedRef.current.has(key)) return;
    requestedRef.current.add(key);
    void requestAnalysis(displayed);
  }, [modalOpen, displayed, status, finished, context, requestAnalysis]);

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
          Signals on candle close · fills at next open · fees &amp; slippage included · 60 / 20 / 20 train / validation / holdout
        </p>
      </header>

      <ControlBar
        config={config}
        onChange={updateConfig}
        status={status}
        onStart={() => void start()}
        onStop={stop}
      />

      <TelemetryStrip
        maxStagnation={context?.maxStagnation ?? config.maxStagnation}
        status={status}
        telemetry={telemetry}
        benchmarkRoi={benchmark?.search.roi ?? null}
        context={context}
      />

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

      {context && context.priorHoldoutViews > 0 && (
        <div role="status" className="flex items-start gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2.5 text-sm text-warn">
          <Info size={16} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <span className="font-semibold">
              Holdout already viewed {context.priorHoldoutViews} time{context.priorHoldoutViews === 1 ? '' : 's'}. It is no longer unseen
              data.
            </span>{' '}
            Every run on the same symbols, timeframe, range and end date reveals the same holdout, so each look makes it a weaker test.
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
              revealed={revealed}
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
            <p className="text-xs text-ink-3">Train + validation · same candles, capital and costs</p>
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
                      <td className={cx('px-2 py-[7px] text-right font-medium', displayed ? (row.tone?.(displayed.search) ?? 'text-ink') : 'text-ink-3')}>
                        {displayed ? row.format(displayed.search) : '–'}
                      </td>
                      <td className={cx('px-4 py-[7px] text-right', row.benchmark === false ? 'text-ink-3' : (row.tone?.(benchmark.search) ?? 'text-ink-2'))}>
                        {row.benchmark === false ? '–' : row.format(benchmark.search)}
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
                      Deflated Sharpe{' '}
                      <span className={cx('font-semibold', displayed.dsr >= 0.95 ? 'text-up' : 'text-ink-2')}>
                        {fmtPct(displayed.dsr * 100, false, 0)}
                      </span>{' '}
                      · {displayed.params} params ·{' '}
                      {displayed.holdout ? (
                        <>
                          holdout ROI{' '}
                          <span className={cx('font-semibold', toneClass(displayed.holdout.roi))}>{fmtPct(displayed.holdout.roi, true)}</span> vs{' '}
                          {fmtPct(benchmark.holdout?.roi ?? null, true)}
                        </>
                      ) : (
                        'holdout sealed until the run ends'
                      )}
                    </p>
                    {(analyses[displayed.id]?.verdict ?? displayed.verdict) !== null && (
                      <div className="mt-2">
                        <VerdictPill verdict={analyses[displayed.id]?.verdict ?? displayed.verdict} />
                      </div>
                    )}
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
          barsPerYear={barsPerYear}
          analysis={analyses[displayed.id] ?? null}
          analysisPending={pendingAnalyses[displayed.id] === true}
          onClose={closeModal}
        />
      )}
    </div>
  );
}
