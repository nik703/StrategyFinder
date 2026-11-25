/**
 * StrategyFinder dashboard.
 *
 * The main thread only loads market data, renders the UI and relays messages:
 * every indicator, backtest and optimisation step runs in `optimizer.worker.ts`.
 *
 * Layout – one guided flow: (1) the top action bar and the presets, (2) the live discovery
 * scanner, (3) the ranked leaderboard, (4) the strategy workbench.
 */

import { Info, Lock, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { TooltipProvider } from '@/components/ui/tooltip';
import ComparePanel from './components/ComparePanel';
import ControlBar from './components/ControlBar';
import Leaderboard, { MAX_COMPARE } from './components/Leaderboard';
import StrategyWorkbench, { type CrossMarketState } from './components/StrategyWorkbench';
import TelemetryStrip from './components/TelemetryStrip';
import TopActionBar from './components/TopActionBar';
import { Panel, StepHeader } from './components/common';
import { buildDataset } from './data/basket';
import { loadMarketData, type LoadedMarketData } from './data/marketData';
import {
  crossMarketSuggestion,
  DEFAULT_CONFIG,
  resolveAssets,
  resolvePool,
  runSymbols,
  settingsFrom,
  venueLabel,
  type RunConfig,
  type RunContext,
} from './lib/config';
import { resetForecastWorker } from './lib/forecastClient';
import { fmtDate, fmtInt } from './lib/format';
import { holdoutWindowFor, priorReveals, recordHoldoutReveal } from './lib/holdoutLog';
import { savedRuns, saveRun } from './lib/storage';
import { activePreset } from './lib/presets';
import { clearRunStorage } from './lib/resetRuns';
import { budgetStatus, estimateSearchWindow, extendWindowBudget, priorTrials, recordRunTrials, searchWindowFor, windowBudget, type TrialWindow } from './lib/trialLedger';
import { buildSnapshot, fetchSnapshot, type MarketSnapshot } from './lib/marketSnapshot';
import { addSavedTickers, loadSavedTickers, removeSavedTicker } from './lib/savedTickers';
import { evaluatorBudgetBytes, WorkerPool } from './workers/workerPool';
import {
  clampRange,
  type BenchmarkResult,
  type ChartAxis,
  type EngineStatus,
  type NamedDataset,
  type ProgressSnapshot,
  type RobustnessReport,
  type StopReason,
  type StrategyResult,
  type Verdict,
  type WorkerRequest,
  MAX_RUN_EVALUATIONS,
  type TradeGates,
  type WorkerResponse,
} from './types/trading';

type Analysis = { report: RobustnessReport; verdict: Verdict | null };

/** One line per symbol on how its candles were loaded (cache hits, new candles, full downloads). */
function cacheNotes(cache: LoadedMarketData['cache']): string[] {
  return cache.map(({ symbol, cachedBars, fetchedBars, mode }) =>
    mode === 'full'
      ? `${symbol}: downloaded ${fetchedBars.toLocaleString('en-US')} candles (cached for later runs).`
      : `${symbol}: ${cachedBars.toLocaleString('en-US')} candles from the cache + ${fetchedBars.toLocaleString('en-US')} new.`,
  );
}

/** Save a finished run and its leaderboard snapshot in IndexedDB (exported with a snapshot). */
function persistRun(context: RunContext, strategies: StrategyResult[], lockedId: string | null): void {
  saveRun({
    id: context.runId,
    savedAt: Date.now(),
    label: context.label,
    interval: context.interval,
    range: context.range,
    hypothesis: context.hypothesis,
    tradeMode: context.settings.tradeMode,
    lockedId,
    strategies,
  });
}

const EMPTY_TELEMETRY: ProgressSnapshot = {
  iteration: 0,
  evaluations: 0,
  windowTrials: 0,
  fitnessStall: 0,
  duplicates: 0,
  runTrials: { count: 0, mean: 0, m2: 0 },
  bestFitness: null,
  bestDsr: null,
  bestDsrSharpe: null,
  hurdleSharpe: 0,
  undersampled: 0,
  robustFound: 0,
  robustTarget: 0,
  cycle: 1,
  funnel: { scored: 0, sample: 0, dsr: 0, stressChecked: 0, stress: 0 },
  itersPerSec: 0,
  elapsedMs: 0,
};

export default function App(): JSX.Element {
  const [config, setConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [savedTickers, setSavedTickers] = useState<string[]>(loadSavedTickers);
  const saveTickers = useCallback((symbols: string[]) => setSavedTickers((current) => addSavedTickers(current, symbols)), []);
  const forgetTicker = useCallback((symbol: string) => setSavedTickers((current) => removeSavedTicker(current, symbol)), []);
  const [status, setStatus] = useState<EngineStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<ProgressSnapshot>(EMPTY_TELEMETRY);
  const [leaderboard, setLeaderboard] = useState<StrategyResult[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [axis, setAxis] = useState<ChartAxis | null>(null);
  const [barsPerYear, setBarsPerYear] = useState(365);
  const [gates, setGates] = useState<TradeGates | null>(null);
  const [pooledRun, setPooledRun] = useState(false);
  const [context, setContext] = useState<RunContext | null>(null);
  /** Strategy pinned by clicking a leaderboard row; null = follow the current #1. */
  const [pinned, setPinned] = useState<StrategyResult | null>(null);
  /** Strategies ticked for side-by-side comparison, by id. */
  const [compareIds, setCompareIds] = useState<string[]>([]);
  /** Cross-market generalization checks, by strategy id. */
  const [crossMarket, setCrossMarket] = useState<Record<string, CrossMarketState>>({});
  const workbenchRef = useRef<HTMLDivElement | null>(null);
  /** On-demand robustness reports, by strategy id. */
  const [analyses, setAnalyses] = useState<Record<string, Analysis>>({});
  const [pendingAnalyses, setPendingAnalyses] = useState<Record<string, boolean>>({});
  /** Why the last run ended. */
  const [stopReason, setStopReason] = useState<StopReason | null>(null);
  /** Bumped whenever this tab writes the trial ledger, so the budget meter re-reads it. */
  const [ledgerVersion, setLedgerVersion] = useState(0);
  /** When this run's trials were last written to the ledger. */
  const lastLedgerWrite = useRef(0);
  /** The one strategy of this run whose holdout has been revealed. */
  const [lockedId, setLockedId] = useState<string | null>(null);
  const [lockPending, setLockPending] = useState(false);
  /** Latest candles for the Next Entry Forecast (the run's own data until refreshed). */
  const [snapshot, setSnapshot] = useState<MarketSnapshot | null>(null);

  const workerRef = useRef<Worker | null>(null);
  /** Evaluator workers that score candidate strategies in parallel for the optimiser. */
  /** The run's evaluator worker pool (first evaluation tier on every spare core). */
  const poolRef = useRef<WorkerPool | null>(null);
  const runIdRef = useRef(0);
  const pendingProgress = useRef<ProgressSnapshot | null>(null);
  const frameRef = useRef<number | null>(null);
  const contextRef = useRef<RunContext | null>(null);
  const finishedRef = useRef(false);
  /** Analysis requests already sent, keyed by strategy id and run phase. */
  const requestedRef = useRef(new Set<string>());

  const disposePool = useCallback(() => {
    poolRef.current?.terminate();
    poolRef.current = null;
  }, []);

  const disposeWorker = useCallback(() => {
    disposePool();
    workerRef.current?.terminate();
    workerRef.current = null;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    pendingProgress.current = null;
  }, [disposePool]);

  useEffect(() => disposeWorker, [disposeWorker]);

  const updateConfig = useCallback((patch: Partial<RunConfig>) => {
    setConfig((current) => {
      const next = { ...current, ...patch };
      // Short timeframes can only load a limited history: clamp the range when the timeframe changes.
      next.range = clampRange(next.interval, next.range);
      return next;
    });
  }, []);

  /** Forget everything about the current run (results, telemetry, selections) – before a new run or on a reset. */
  const clearRunState = useCallback(() => {
    finishedRef.current = false;
    requestedRef.current = new Set();
    contextRef.current = null;
    lastLedgerWrite.current = 0;
    setError(null);
    setTelemetry(EMPTY_TELEMETRY);
    setLeaderboard([]);
    setBenchmark(null);
    setAxis(null);
    setGates(null);
    setPooledRun(false);
    setContext(null);
    setSnapshot(null);
    setPinned(null);
    setCompareIds([]);
    setCrossMarket({});
    setAnalyses({});
    setPendingAnalyses({});
    setLockedId(null);
    setLockPending(false);
    setStopReason(null);
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
          setGates(message.gates);
          setPooledRun(message.pooled);
          break;
        case 'progress': {
          const { type: _type, ...snapshot } = message;
          queueProgress(snapshot);
          // Persist this run's trials as they accrue (at most once a second), so a reload keeps them.
          const now = Date.now();
          if (contextRef.current && now - lastLedgerWrite.current >= 1000) {
            lastLedgerWrite.current = now;
            recordRunTrials(contextRef.current.trialWindow, contextRef.current.runId, snapshot.runTrials, contextRef.current.hypothesis);
          }
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
          // The search is over. The holdout stays sealed until the user locks one strategy.
          setLeaderboard(strategies);
          setBenchmark(finalBenchmark);
          setPinned((current) => (current ? (strategies.find((s) => s.id === current.id) ?? null) : null));
          setAnalyses({});
          setPendingAnalyses({});
          if (contextRef.current) {
            recordRunTrials(contextRef.current.trialWindow, contextRef.current.runId, snapshot.runTrials, contextRef.current.hypothesis);
            persistRun(contextRef.current, strategies, null);
            setLedgerVersion((v) => v + 1);
          }
          setStopReason(reason);
          setStatus(reason === 'stopped' ? 'stopped' : 'converged');
          // The search is over, so the evaluator pool is no longer needed. The optimiser
          // worker stays alive to answer robustness-analysis requests.
          disposePool();
          break;
        }
        case 'locked': {
          const locked = message.strategy;
          setLockPending(false);
          setLockedId(locked.id);
          setLeaderboard((current) => current.map((entry) => (entry.id === locked.id ? locked : entry)));
          setPinned((current) => (current && current.id === locked.id ? locked : current));
          setBenchmark(message.benchmark);
          setAnalyses((current) => {
            const next = { ...current };
            delete next[locked.id];
            return next;
          });
          // Persist the reveal: every later run whose holdout overlaps this one will be warned.
          if (contextRef.current) {
            recordHoldoutReveal(contextRef.current.holdoutWindow, locked.id);
            // The saved run keeps the locked strategy (with its revealed holdout) in place of its sealed version.
            const saved = savedRuns().find((run) => run.id === contextRef.current?.runId);
            if (saved) saveRun({ ...saved, lockedId: locked.id, strategies: saved.strategies.map((s) => ((s as StrategyResult).id === locked.id ? locked : s)) });
            setLedgerVersion((v) => v + 1);
          }
          break;
        }
        case 'analysis':
          setAnalyses((current) => ({ ...current, [message.strategyId]: { report: message.report, verdict: message.verdict } }));
          setPendingAnalyses((current) => ({ ...current, [message.strategyId]: false }));
          break;
        case 'cross-market': {
          setCrossMarket((current) => {
            const previous = current[message.strategyId];
            return { ...current, [message.strategyId]: { status: 'done', symbols: previous?.symbols ?? [], report: message.report } };
          });
          // The locked strategy's verdict was judged again with the cross-market evidence.
          const locked = message.locked;
          if (locked) {
            setLeaderboard((current) => current.map((entry) => (entry.id === locked.id ? locked : entry)));
            setPinned((current) => (current && current.id === locked.id ? locked : current));
            setAnalyses((current) => {
              const next = { ...current };
              delete next[locked.id];
              return next;
            });
          }
          break;
        }
        case 'error':
          setError(message.message);
          setPendingAnalyses({});
          setLockPending(false);
          setCrossMarket((current) => {
            const next: Record<string, CrossMarketState> = {};
            for (const [id, state] of Object.entries(current)) next[id] = state.status === 'loading' ? { ...state, status: 'error', error: message.message } : state;
            return next;
          });
          if (!finishedRef.current) {
            setStatus('error');
            disposeWorker();
          }
          break;
      }
    },
    [disposePool, disposeWorker, queueProgress],
  );

  const start = useCallback(async () => {
    const assets = resolveAssets(config);
    if (!assets.ok) {
      setError(assets.error);
      return;
    }
    const pool = resolvePool(config);
    if (!pool.ok) {
      setError(pool.error);
      return;
    }
    // Remember the custom tickers of this run so they never have to be typed again.
    saveTickers(assets.components.map((component) => component.symbol));
    disposeWorker();
    const runId = ++runIdRef.current;
    clearRunState();
    setStatus('loading');
    const ledgerRunId = `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;

    try {
      const tradedSymbols = assets.components.map((component) => component.symbol);
      const symbols = [...new Set([...tradedSymbols, ...pool.symbols])];
      // Closed candles come from the IndexedDB cache; only the ones that closed since are downloaded
      // (the whole range with "Force Refresh Data", which applies to this one run).
      const loaded = await loadMarketData(symbols, config.interval, config.range, { forceRefresh: config.forceRefreshData });
      if (config.forceRefreshData) updateConfig({ forceRefreshData: false });
      if (runId !== runIdRef.current) return; // cancelled while downloading
      const dataset = buildDataset(loaded.series, assets.components);
      const settings = settingsFrom(config);
      const bars = dataset.data.t.length;
      // The primary asset first; every other pool asset is traded separately with the same rules.
      const datasets: NamedDataset[] = [{ symbol: dataset.isBasket ? 'Basket' : tradedSymbols[0], data: dataset.data }];
      for (const symbol of pool.symbols.slice(1)) {
        datasets.push({ symbol, data: buildDataset(loaded.series, [{ symbol, weight: 1 }]).data });
      }
      // Every symbol whose holdout this run can show, over the dates of the primary dataset's holdout.
      const holdoutWindow = holdoutWindowFor(symbols, config.interval, dataset.data.t);
      const earlier = priorReveals(holdoutWindow);
      // Every trial already spent on this search period – in any run, any session – counts against it.
      const trialWindow = searchWindowFor(symbols, config.interval, dataset.data.t);
      const prior = priorTrials(trialWindow);
      const limit = windowBudget(trialWindow);
      if (prior.count >= limit) {
        setError(
          `The trial budget of this data window is spent (${fmtInt(prior.count)} of ${fmtInt(limit)} evaluations over ` +
            `${prior.runs} runs). Extend the budget to keep searching – knowing every extra trial weakens significance – or test other data.`,
        );
        setLedgerVersion((v) => v + 1);
        setStatus('idle');
        return;
      }
      const runContext: RunContext = {
        label: config.assetMode === 'pool' ? `Pool: ${pool.symbols.join(' · ')}` : dataset.label,
        interval: config.interval,
        range: config.range,
        settings,
        maxEvaluations: Math.min(config.maxEvaluations, MAX_RUN_EVALUATIONS, limit - prior.count),
        hypothesis: config.hypothesis,
        runId: ledgerRunId,
        trialWindow,
        priorTrials: prior,
        universe: datasets.map((entry) => entry.symbol),
        venue: venueLabel((config.assetMode === 'pool' ? pool.symbols : tradedSymbols).map((symbol) => loaded.series[symbol])),
        holdoutWindow,
        priorHoldoutReveals: earlier.reduce((sum, entry) => sum + entry.views, 0),
        lastHoldoutReveal: earlier.length > 0 ? earlier[earlier.length - 1].revealedAt : null,
        components: dataset.components,
        units: dataset.units,
        isBasket: dataset.isBasket,
        bars,
        windowStart: dataset.data.t[0],
        windowEnd: dataset.data.t[bars - 1],
        sources: dataset.sources,
        simulated: loaded.usedFallback,
        fallbackReason: loaded.fallbackReason,
        notes: [...loaded.notes, ...cacheNotes(loaded.cache)],
      };
      contextRef.current = runContext;
      setContext(runContext);
      // The run's own candles seed the Next Entry Forecast until the user refreshes them.
      setSnapshot(
        buildSnapshot(loaded.series, runContext, { simulated: loaded.usedFallback, fallbackReason: loaded.fallbackReason }),
      );

      const worker = new Worker(new URL('./workers/optimizer.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => handleMessage(event.data, runId);
      worker.onerror = (event) => {
        if (runId !== runIdRef.current) return;
        setError(event.message || 'The optimisation worker crashed.');
        setStatus('error');
        disposeWorker();
      };
      // Worker pool: candidate strategies are scored on every spare core. Each evaluator gets its own
      // copy of the data and talks to the optimiser directly over a MessageChannel.
      const evaluators = new WorkerPool({
        createWorker: () => new Worker(new URL('./workers/evaluator.worker.ts', import.meta.url), { type: 'module' }),
        onFailure: () => {
          // A dead evaluator would never answer: the optimiser finishes the run single-threaded.
          if (runId !== runIdRef.current) return;
          const fallback: WorkerRequest = { type: 'pool-failed' };
          workerRef.current?.postMessage(fallback);
        },
      });
      poolRef.current = evaluators;
      const ports = evaluators.start({ datasets, settings, budgetBytes: evaluatorBudgetBytes(evaluators.size) });

      const request: WorkerRequest = {
        type: 'start',
        payload: {
          datasets,
          settings,
          hypothesis: config.hypothesis,
          maxEvaluations: config.maxEvaluations,
          robustTarget: config.robustTarget,
          priorTrials: { count: prior.count, mean: prior.mean, m2: prior.m2 },
          windowBudget: limit,
          ports,
        },
      };
      worker.postMessage(request, ports);
      setStatus('optimizing');
    } catch (cause) {
      if (runId !== runIdRef.current) return;
      setError(cause instanceof Error ? cause.message : 'Could not start the optimisation.');
      setStatus('error');
    }
  }, [config, clearRunState, disposePool, disposeWorker, handleMessage, saveTickers, updateConfig]);

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

  /** Ask the worker for a robustness report (top-trade removal, stability map, Monte Carlo bands). */
  const requestAnalysis = useCallback((strategy: StrategyResult) => {
    const worker = workerRef.current;
    if (!worker || !contextRef.current) return;
    setPendingAnalyses((current) => ({ ...current, [strategy.id]: true }));
    const request: WorkerRequest = { type: 'analyze', strategyId: strategy.id, genome: strategy.genome };
    worker.postMessage(request);
  }, []);

  /**
   * "Reset Runs & Clear Cache" (confirmed in the top bar): terminate every worker, drop every
   * result held in memory and delete the persistent trial ledger and holdout log, so the next run
   * starts from a clean 0-trial slate. Settings and saved tickers are kept.
   */
  const resetRuns = useCallback(() => {
    runIdRef.current++; // anything still in flight belongs to a run that no longer exists
    disposeWorker();
    resetForecastWorker();
    clearRunStorage();
    clearRunState();
    setStatus('idle');
    setLedgerVersion((v) => v + 1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [clearRunState, disposeWorker]);

  /** Lock ONE strategy of the finished run and reveal its holdout. The worker refuses a second lock. */
  const lockStrategy = useCallback(
    (strategy: StrategyResult) => {
      const worker = workerRef.current;
      const runContext = contextRef.current;
      if (!worker || !runContext || lockedId !== null || lockPending) return;
      if (status !== 'converged' && status !== 'stopped') return;
      setLockPending(true);
      const request: WorkerRequest = {
        type: 'lock',
        strategyId: strategy.id,
        priorReveals: runContext.priorHoldoutReveals,
        simulated: runContext.simulated,
      };
      worker.postMessage(request);
    },
    [lockedId, lockPending, status],
  );

  /** Open a strategy in the workbench (step 4) and bring the workbench into view. */
  const selectStrategy = useCallback((strategy: StrategyResult) => {
    setPinned(strategy);
    window.requestAnimationFrame(() => workbenchRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }, []);

  const toggleCompare = useCallback((strategy: StrategyResult) => {
    setCompareIds((current) =>
      current.includes(strategy.id) ? current.filter((id) => id !== strategy.id) : current.length >= MAX_COMPARE ? current : [...current, strategy.id],
    );
  }, []);

  /**
   * Cross-market generalization check: load the unseen markets with the run's timeframe and history,
   * then let the worker backtest the frozen rules on them (search-period dates only).
   */
  const runCrossMarket = useCallback(async (strategy: StrategyResult, symbols: string[]) => {
    const worker = workerRef.current;
    const runContext = contextRef.current;
    const runId = runIdRef.current;
    if (!worker || !runContext) return;
    const unique = [...new Set(symbols.map((symbol) => symbol.trim().toUpperCase()).filter(Boolean))].slice(0, 8);
    setCrossMarket((current) => ({ ...current, [strategy.id]: { status: 'loading', symbols: unique } }));
    try {
      const loaded = await loadMarketData(unique, runContext.interval, runContext.range);
      if (runId !== runIdRef.current || workerRef.current !== worker) return;
      if (loaded.usedFallback !== runContext.simulated) {
        throw new Error(
          runContext.simulated
            ? 'This run used simulated data, but the check could load live candles – live and simulated prices cannot be compared.'
            : `Could not load live candles for every market (${loaded.fallbackReason ?? 'unknown error'}).`,
        );
      }
      const datasets: NamedDataset[] = unique.map((symbol) => ({ symbol, data: buildDataset(loaded.series, [{ symbol, weight: 1 }]).data }));
      const request: WorkerRequest = {
        type: 'cross-market',
        strategyId: strategy.id,
        genome: strategy.genome,
        datasets,
        runSymbols: [...runContext.universe, ...runContext.components.map((component) => component.symbol)],
      };
      worker.postMessage(request);
    } catch (cause) {
      if (runId !== runIdRef.current) return;
      const message = cause instanceof Error ? cause.message : 'The cross-market check failed.';
      setCrossMarket((current) => ({ ...current, [strategy.id]: { status: 'error', symbols: unique, error: message } }));
    }
  }, []);

  /** "Refresh Latest Price": fetch the newest candles for the run's assets without re-optimising. */
  const refreshSnapshot = useCallback(async () => {
    const runContext = contextRef.current;
    if (!runContext) return;
    const runId = runIdRef.current;
    const fresh = await fetchSnapshot(runContext);
    if (runId === runIdRef.current) setSnapshot(fresh);
  }, []);

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
  const finished = status === 'converged' || status === 'stopped';
  const busy = status === 'loading' || status === 'optimizing';

  /**
   * Trial budget of the active data window. While a run is live (or its results are shown and the
   * selection is unchanged) it is the run's exact window: earlier trials plus this run's. Otherwise
   * it is estimated from the selected assets, timeframe and range, read from the persistent ledger.
   */
  const budget = useMemo((): { status: ReturnType<typeof budgetStatus>; estimated: boolean; window: TrialWindow } | null => {
    void ledgerVersion; // re-read the ledger after this tab wrote it
    const symbols = runSymbols(config);
    const sameSelection =
      context !== null &&
      context.interval === config.interval &&
      context.range === config.range &&
      [...symbols].sort().join('+') === context.trialWindow.symbols.join('+');
    if (context && (busy || sameSelection)) {
      const used = busy || finished ? context.priorTrials.count + telemetry.evaluations : priorTrials(context.trialWindow).count;
      return { status: budgetStatus(context.trialWindow, used), estimated: false, window: context.trialWindow };
    }
    if (symbols.length === 0) return null;
    const window = estimateSearchWindow(symbols, config.interval, config.range, Date.now());
    return { status: budgetStatus(window, priorTrials(window).count), estimated: true, window };
  }, [config, context, busy, finished, telemetry.evaluations, ledgerVersion]);
  const budgetBlocked =
    !busy && budget !== null && budget.status.level === 'exhausted'
      ? `The trial budget of this data window is spent (${fmtInt(budget.status.used)} of ${fmtInt(budget.status.limit)} evaluations). Search on the same data is paused: pick other assets or another period, or extend the budget knowingly.`
      : null;
  /** Explicitly allow `BUDGET_EXTENSION` more trials on the selected window (recorded in the ledger). */
  const extendBudget = useCallback(() => {
    if (!budget) return;
    extendWindowBudget(budget.window);
    setLedgerVersion((v) => v + 1);
    setError(null);
  }, [budget]);

  // The workbench requests a robustness report (stress curve, stability map, top-trade test): for a
  // strategy the user picked while the run is live, and for whatever it shows once the run has ended.
  // Following the moving #1 during a live run does not, so the search is not slowed down.
  useEffect(() => {
    if (!displayed || !workerRef.current) return;
    if (!(finished || (status === 'optimizing' && pinned !== null))) return;
    if (displayed.robustness !== null) return;
    const key = `${displayed.id}|${finished ? 'final' : 'live'}`;
    if (requestedRef.current.has(key)) return;
    requestedRef.current.add(key);
    requestAnalysis(displayed);
  }, [displayed, pinned, status, finished, requestAnalysis]);

  const compared = useMemo(
    () => compareIds.map((id) => leaderboard.find((entry) => entry.id === id)).filter((entry): entry is StrategyResult => entry !== undefined),
    [compareIds, leaderboard],
  );
  const crossSuggestion = useMemo(() => (context ? crossMarketSuggestion([...context.universe, ...context.components.map((c) => c.symbol)]).join(', ') : ''), [context]);
  const preset = useMemo(() => activePreset(config), [config]);
  /** Robust strategies the scan looks for: the run's, or before the first run the next run's. */
  const scanTarget = busy || telemetry.iteration > 0 ? telemetry.robustTarget : config.robustTarget;
  /** Where the user is in the guided workflow. */
  const stage = displayed ? 4 : leaderboard.length > 0 ? 3 : busy || telemetry.iteration > 0 ? 2 : 1;

  return (
    <TooltipProvider delayDuration={200}>
      <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col gap-4 px-4 pb-5 sm:px-6">
        <TopActionBar
          status={status}
          presetTitle={preset?.title ?? null}
          onRun={() => void start()}
          onStop={stop}
          runBlocked={budgetBlocked}
          budget={budget}
          onExtendBudget={extendBudget}
          onReset={resetRuns}
          onSnapshotImported={() => setLedgerVersion((v) => v + 1)}
          stage={stage}
        />

        <div id="step-setup" className="scroll-mt-20">
          <ControlBar
            config={config}
            onChange={updateConfig}
            savedTickers={savedTickers}
            onSaveTickers={saveTickers}
            onRemoveTicker={forgetTicker}
            status={status}
          />
        </div>

        <div id="step-scan" className="scroll-mt-20">
          <TelemetryStrip
            maxEvaluations={context?.maxEvaluations ?? config.maxEvaluations}
            robustTarget={scanTarget}
            stopReason={stopReason}
            status={status}
            telemetry={telemetry}
            benchmarkSharpe={benchmark?.validation.sharpe ?? null}
            context={context}
          />
        </div>

        {error && (
          <Alert variant="destructive" className="border-down/40 bg-down/10 pr-10">
            <TriangleAlert />
            <AlertDescription className="text-current">{error}</AlertDescription>
            <Button variant="ghost" size="icon-xs" onClick={() => setError(null)} aria-label="Dismiss error" className="absolute top-2 right-2 hover:bg-down/20">
              <X />
            </Button>
          </Alert>
        )}

        {context?.simulated && (
          <Alert className="border-warn/35 bg-warn/10 text-warn">
            <TriangleAlert />
            <AlertDescription className="block text-current">
              <span className="font-semibold">Simulated market data.</span> The market-data function could not supply live candles, so the
              built-in generator produced realistic placeholder prices. Results are illustrative only.
              {context.fallbackReason && <span className="num mt-0.5 block truncate text-xs opacity-80">Reason: {context.fallbackReason}</span>}
            </AlertDescription>
          </Alert>
        )}

        {context && context.priorHoldoutReveals > 0 && (
          <Alert role="status" className="border-warn/35 bg-warn/10 text-warn">
            <Info />
            <AlertDescription className="block text-current">
              <span className="font-semibold">
                Holdout already spent: this period ({fmtDate(context.holdoutWindow.start)} → {fmtDate(context.holdoutWindow.end)}) overlaps{' '}
                {context.priorHoldoutReveals} earlier reveal{context.priorHoldoutReveals === 1 ? '' : 's'}
                {context.lastHoldoutReveal !== null ? `, the last on ${fmtDate(context.lastHoldoutReveal)}` : ''}.
              </span>{' '}
              It is no longer untouched data, so no strategy from this run can be rated Robust. Any timeframe, range or end date over the
              same symbols and dates counts.
            </AlertDescription>
          </Alert>
        )}

        {finished && leaderboard.length > 0 && lockedId === null && (
          <Alert role="status" className="border-primary/40 bg-primary/10">
            <Lock className="text-primary" />
            <AlertDescription className="block text-secondary-foreground">
              <span className="font-semibold text-foreground">The holdout is still sealed.</span> Compare the candidates, choose ONE and press{' '}
              <span className="font-semibold text-foreground">Lock &amp; reveal holdout</span> in its workbench. Only that strategy&apos;s holdout is
              revealed, once, and the reveal is recorded for every later run on the same symbols and dates.
            </AlertDescription>
          </Alert>
        )}

        {context && !context.simulated && context.notes.length > 0 && (
          <Alert>
            <Info />
            <AlertDescription className="text-xs">{context.notes.join(' · ')}</AlertDescription>
          </Alert>
        )}

        <div id="step-rank" className="scroll-mt-20">
          <Leaderboard
            strategies={leaderboard}
            selectedId={displayed?.id ?? null}
            lockedId={lockedId}
            onSelect={selectStrategy}
            status={status}
            gates={gates}
            pooled={pooledRun}
            compareIds={compareIds}
            onToggleCompare={toggleCompare}
          />
        </div>

        {compared.length > 0 && (
          <ComparePanel
            strategies={compared}
            onRemove={(id) => setCompareIds((current) => current.filter((entry) => entry !== id))}
            onOpen={selectStrategy}
            onClear={() => setCompareIds([])}
          />
        )}

        <div id="step-workbench" ref={workbenchRef} className="scroll-mt-20">
          {displayed && benchmark && context ? (
            <StrategyWorkbench
              strategy={displayed}
              rank={displayedRank}
              benchmark={benchmark}
              context={context}
              axis={axis}
              testedCount={telemetry.evaluations}
              barsPerYear={barsPerYear}
              analysis={analyses[displayed.id] ?? null}
              analysisPending={pendingAnalyses[displayed.id] === true}
              finished={finished}
              lockedId={lockedId}
              lockPending={lockPending}
              onLock={() => lockStrategy(displayed)}
              snapshot={snapshot}
              onRefreshSnapshot={refreshSnapshot}
              pinned={pinned !== null}
              onFollowBest={() => setPinned(null)}
              crossMarket={crossMarket[displayed.id] ?? null}
              crossMarketSuggestion={crossSuggestion}
              onRunCrossMarket={(symbols) => void runCrossMarket(displayed, symbols)}
            />
          ) : (
            <Panel className="px-4 py-4">
              <StepHeader step={4} title="Strategy workbench" hint="Verdict, next-entry trigger prices, stress test, stability map, holdout reveal and Pine Script export." />
              <p className="mt-3 text-sm text-muted-foreground">
                {status === 'loading'
                  ? 'Loading market data…'
                  : busy
                    ? 'Appears as soon as the scanner finds its first strategy.'
                    : 'Pick a preset above (or keep the default) and press 1-Click Run. The best strategy opens here automatically; click any leaderboard row to inspect another one.'}
              </p>
            </Panel>
          )}
        </div>

        <footer className="pb-2 text-[11px] leading-relaxed text-muted-foreground">
          StrategyFinder is a research tool. Backtests are hypothetical, are fitted to past data and do not guarantee future results. Nothing
          here is financial advice. Signals on candle close · fills at next open · fees, slippage and financing included · 60 / 20 / 20
          train / validation / holdout.
        </footer>
      </div>
    </TooltipProvider>
  );
}
