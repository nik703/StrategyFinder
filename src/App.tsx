/**
 * StrategyFinder dashboard.
 *
 * The main thread only loads market data, renders the UI and relays messages:
 * every indicator, backtest and optimisation step runs in `optimizer.worker.ts`.
 */

import { Activity, Info, Lock, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ComparePanel from './components/ComparePanel';
import ControlBar from './components/ControlBar';
import Leaderboard, { MAX_COMPARE } from './components/Leaderboard';
import StrategyWorkbench, { type CrossMarketState } from './components/StrategyWorkbench';
import TelemetryStrip from './components/TelemetryStrip';
import { cx, Panel, StepHeader } from './components/ui';
import { buildDataset } from './data/basket';
import { loadMarketData } from './data/marketData';
import {
  crossMarketSuggestion,
  DEFAULT_CONFIG,
  defaultUniverse,
  isCrypto,
  parseUniverse,
  resolveAssets,
  resolveUniverse,
  runSymbols,
  settingsFrom,
  venueLabel,
  type RunConfig,
  type RunContext,
} from './lib/config';
import { fmtDate, fmtInt } from './lib/format';
import { holdoutWindowFor, priorReveals, recordHoldoutReveal } from './lib/holdoutLog';
import { budgetStatus, estimateSearchWindow, extendWindowBudget, priorTrials, recordRunTrials, searchWindowFor, windowBudget, type TrialWindow } from './lib/trialLedger';
import { buildSnapshot, fetchSnapshot, type MarketSnapshot } from './lib/marketSnapshot';
import { addSavedTickers, loadSavedTickers, removeSavedTicker } from './lib/savedTickers';
import type { EvaluatorConnect } from './engine/evaluation';
import {
  clampRange,
  DEFAULT_CRYPTO_UNIVERSE,
  DEFAULT_STOCK_UNIVERSE,
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

/** Most evaluator workers a run will start (each holds its own copy of the data and an indicator cache). */
const MAX_EVALUATORS = 6;

/**
 * Number of evaluator workers for this device: every core but one (left for the page and the
 * optimiser thread, which only proposes and ranks). With one or two cores the optimiser
 * scores candidates itself – a pool of one would only add messaging overhead.
 */
function evaluatorPoolSize(): number {
  const cores = typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 2;
  const size = Math.min(MAX_EVALUATORS, cores - 1);
  return size >= 2 ? size : 0;
}

const DEFAULT_UNIVERSES = [DEFAULT_CRYPTO_UNIVERSE.join(', '), DEFAULT_STOCK_UNIVERSE.join(', ')];

const EMPTY_TELEMETRY: ProgressSnapshot = {
  iteration: 0,
  stagnation: 0,
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
  funnel: { scored: 0, trades: 0, validation: 0, folds: 0, dsr: 0, neighboursChecked: 0, neighbours: 0 },
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
  const evaluatorsRef = useRef<Worker[]>([]);
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

  const disposePool = useCallback(() => {
    for (const evaluator of evaluatorsRef.current) evaluator.terminate();
    evaluatorsRef.current = [];
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

  /** Changing the ticker's asset class switches the Robust universe to that class's default. */
  const updateConfig = useCallback((patch: Partial<RunConfig>) => {
    setConfig((current) => {
      const next = { ...current, ...patch };
      // Short timeframes can only load a limited history: clamp the range when the timeframe changes.
      next.range = clampRange(next.interval, next.range);
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
          if (contextRef.current) recordHoldoutReveal(contextRef.current.holdoutWindow, locked.id);
          break;
        }
        case 'analysis':
          setAnalyses((current) => ({ ...current, [message.strategyId]: { report: message.report, verdict: message.verdict } }));
          setPendingAnalyses((current) => ({ ...current, [message.strategyId]: false }));
          break;
        case 'cross-market':
          setCrossMarket((current) => {
            const previous = current[message.strategyId];
            return { ...current, [message.strategyId]: { status: 'done', symbols: previous?.symbols ?? [], report: message.report } };
          });
          break;
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
    const universe = resolveUniverse(config, assets.components[0].symbol);
    if (!universe.ok) {
      setError(universe.error);
      return;
    }
    // Remember the custom tickers of this run so they never have to be typed again.
    saveTickers(assets.components.map((component) => component.symbol));
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
    setSnapshot(null);
    setPinned(null);
    setCompareIds([]);
    setCrossMarket({});
    setAnalyses({});
    setPendingAnalyses({});
    setLockedId(null);
    setLockPending(false);
    setStopReason(null);
    lastLedgerWrite.current = 0;
    const ledgerRunId = `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;

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
        label: config.assetMode === 'pool' ? `Pool: ${universe.symbols.join(' · ')}` : dataset.label,
        interval: config.interval,
        range: config.range,
        settings,
        maxEvaluations: Math.min(config.maxEvaluations, MAX_RUN_EVALUATIONS, limit - prior.count),
        hypothesis: config.hypothesis,
        runId: ledgerRunId,
        trialWindow,
        priorTrials: prior,
        universe: datasets.map((entry) => entry.symbol),
        venue: venueLabel((config.assetMode === 'pool' ? universe.symbols : tradedSymbols).map((symbol) => loaded.series[symbol])),
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
        notes: loaded.notes,
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
      // Evaluator pool: candidate strategies are scored on several cores. Each evaluator gets its
      // own copy of the data and talks to the optimiser directly over a MessageChannel.
      const poolSize = evaluatorPoolSize();
      const ports: MessagePort[] = [];
      const budgetBytes = Math.max(64, Math.floor(512 / Math.max(1, poolSize))) * 1024 * 1024;
      for (let w = 0; w < poolSize; w++) {
        const evaluator = new Worker(new URL('./workers/evaluator.worker.ts', import.meta.url), { type: 'module' });
        const channel = new MessageChannel();
        evaluator.onerror = () => {
          // A dead evaluator would never answer: tell the optimiser to finish the run single-threaded.
          if (runId !== runIdRef.current) return;
          disposePool();
          const fallback: WorkerRequest = { type: 'pool-failed' };
          workerRef.current?.postMessage(fallback);
        };
        const connect: EvaluatorConnect = { type: 'connect', port: channel.port2, datasets, settings, budgetBytes };
        evaluator.postMessage(connect, [channel.port2]);
        evaluatorsRef.current.push(evaluator);
        ports.push(channel.port1);
      }

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
  }, [config, disposePool, disposeWorker, handleMessage, saveTickers]);

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

  // The workbench requests a robustness report (stress curve, stability map, sensitivity): for a
  // strategy the user picked while the run is live, and for whatever it shows once the run has ended.
  // Following the moving #1 during a live run does not, so the search is not slowed down.
  useEffect(() => {
    if (!displayed || !workerRef.current) return;
    if (!(finished || (status === 'optimizing' && pinned !== null))) return;
    const crossAssetMissing = !context?.isBasket && displayed.robustness?.crossAsset == null;
    if (displayed.robustness !== null && !crossAssetMissing) return;
    const key = `${displayed.id}|${finished ? 'final' : 'live'}`;
    if (requestedRef.current.has(key)) return;
    requestedRef.current.add(key);
    void requestAnalysis(displayed);
  }, [displayed, pinned, status, finished, context, requestAnalysis]);

  const compared = useMemo(
    () => compareIds.map((id) => leaderboard.find((entry) => entry.id === id)).filter((entry): entry is StrategyResult => entry !== undefined),
    [compareIds, leaderboard],
  );
  const crossSuggestion = useMemo(() => (context ? crossMarketSuggestion([...context.universe, ...context.components.map((c) => c.symbol)]).join(', ') : ''), [context]);
  /** Where the user is in the guided workflow. */
  const stage = displayed ? 4 : leaderboard.length > 0 ? 3 : busy || telemetry.iteration > 0 ? 2 : 1;

  const STEPS = [
    { n: 1, label: 'Setup', target: 'step-setup' },
    { n: 2, label: 'Discover', target: 'step-scan' },
    { n: 3, label: 'Rank & compare', target: 'step-rank' },
    { n: 4, label: 'Workbench', target: 'step-workbench' },
  ];

  return (
    <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col gap-4 px-4 py-5 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-accent/40 bg-accent/15 text-accent-strong">
            <Activity size={18} />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-5 text-ink">StrategyFinder</h1>
            <p className="text-xs text-ink-3">From market to a verified, exportable strategy – with every overfitting check in plain English</p>
          </div>
        </div>
        <nav aria-label="Workflow" className="flex flex-wrap items-center gap-1">
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
              {index < STEPS.length - 1 && <span aria-hidden="true" className="text-ink-3">›</span>}
            </span>
          ))}
        </nav>
      </header>

      <div id="step-setup" className="scroll-mt-4">
        <ControlBar
          config={config}
          onChange={updateConfig}
          savedTickers={savedTickers}
          onSaveTickers={saveTickers}
          onRemoveTicker={forgetTicker}
          status={status}
          onStart={() => void start()}
          onStop={stop}
          budgetBlocked={budgetBlocked}
          onExtendBudget={extendBudget}
        />
      </div>

      <div id="step-scan" className="scroll-mt-4">
        <TelemetryStrip
          maxEvaluations={context?.maxEvaluations ?? config.maxEvaluations}
          stopReason={stopReason}
          budget={budget}
          status={status}
          telemetry={telemetry}
          benchmarkSharpe={benchmark?.validation.sharpe ?? null}
          context={context}
        />
      </div>

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

      {context && context.priorHoldoutReveals > 0 && (
        <div role="status" className="flex items-start gap-2 rounded-lg border border-warn/35 bg-warn/10 px-3 py-2.5 text-sm text-warn">
          <Info size={16} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <span className="font-semibold">
              Holdout already spent: this period ({fmtDate(context.holdoutWindow.start)} → {fmtDate(context.holdoutWindow.end)}) overlaps{' '}
              {context.priorHoldoutReveals} earlier reveal{context.priorHoldoutReveals === 1 ? '' : 's'}
              {context.lastHoldoutReveal !== null ? `, the last on ${fmtDate(context.lastHoldoutReveal)}` : ''}.
            </span>{' '}
            It is no longer untouched data, so no strategy from this run can be rated Robust. Any timeframe, range or end date over the
            same symbols and dates counts.
          </div>
        </div>
      )}

      {finished && leaderboard.length > 0 && lockedId === null && (
        <div role="status" className="flex items-start gap-2 rounded-lg border border-accent/40 bg-accent/10 px-3 py-2.5 text-sm text-ink-2">
          <Lock size={16} className="mt-0.5 shrink-0 text-accent-strong" />
          <div className="min-w-0">
            <span className="font-semibold text-ink">The holdout is still sealed.</span> Compare the candidates, choose ONE and press{' '}
            <span className="font-semibold text-ink">Lock &amp; reveal holdout</span> in its workbench. Only that strategy&apos;s holdout is
            revealed, once, and the reveal is recorded for every later run on the same symbols and dates.
          </div>
        </div>
      )}

      {context && !context.simulated && context.notes.length > 0 && (
        <div className="rounded-lg border border-line bg-panel px-3 py-2 text-xs text-ink-2">{context.notes.join(' · ')}</div>
      )}

      <div id="step-rank" className="scroll-mt-4">
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

      <div id="step-workbench" ref={workbenchRef} className="scroll-mt-4">
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
            <p className="mt-3 text-sm text-ink-3">
              {status === 'loading'
                ? 'Loading market data…'
                : busy
                  ? 'Appears as soon as the scanner finds its first strategy.'
                  : 'Pick a preset or a market above and press Start Discovery. The best strategy opens here automatically; click any leaderboard row to inspect another one.'}
            </p>
          </Panel>
        )}
      </div>

      <footer className="pb-2 text-[11px] leading-relaxed text-ink-3">
        StrategyFinder is a research tool. Backtests are hypothetical, are fitted to past data and do not guarantee future results. Nothing
        here is financial advice. Signals on candle close · fills at next open · fees, slippage and financing included · 60 / 20 / 20
        train / validation / holdout.
      </footer>
    </div>
  );
}
