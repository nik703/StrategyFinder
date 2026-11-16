import { appendFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  BacktestWorkspace,
  barsPerYear,
  benchmarkEquity,
  tradeLogHash,
  evaluateSearch,
  FOLDS,
  holdoutMetrics,
  holdoutShortfalls,
  plateauFitness,
  runBacktest,
  segmentMetrics,
  splitFor,
} from '../src/engine/backtester';
import { attachEvaluator } from '../src/engine/evaluation';
import { IndicatorCache } from '../src/engine/indicators';
import { countParameters, gaussian, isCoherent, MAX_PARAMETERS, mirrorGroup, mulberry32 } from '../src/engine/genome';
import { deflatedSharpeAgainst } from '../src/lib/stats';
import {
  DEFAULT_ROBUST_TARGET,
  DEFAULT_RUN_EVALUATIONS,
  DEFAULT_SIZING,
  FITNESS_PATIENCE,
  HYPOTHESIS_IDS,
  WINDOW_TRIAL_BUDGET,
  type Hypothesis,
  type TrialStats,
  LEADERBOARD_SIZE,
  PROGRESS_EVERY,
  type BacktestSettings,
  type NamedDataset,
  type OHLCV,
  type StrategyResult,
  type WorkerRequest,
  type WorkerResponse,
} from '../src/types/trading';

/** Minimal stand-in for the dedicated worker global scope. */
interface FakeScope {
  onmessage: ((event: { data: WorkerRequest }) => void) | null;
  postMessage: (message: WorkerResponse) => void;
}

const scope: FakeScope = { onmessage: null, postMessage: () => undefined };

type Of<T extends WorkerResponse['type']> = Extract<WorkerResponse, { type: T }>;

const settingsFor = (tradeMode: BacktestSettings['tradeMode']): BacktestSettings => ({
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  tradeMode,
  sizing: DEFAULT_SIZING,
  fundingPctPerBar: 0,
});

function dataset(symbol: string, interval: '15m' | '30m' | '1h' | '4h' | '1d', range: '1y' | '2y' | '5y'): NamedDataset {
  const series = generateSyntheticCandles(symbol, interval, range, Date.UTC(2026, 0, 15));
  return { symbol, data: buildDataset({ [symbol]: series }, [{ symbol, weight: 1 }]).data };
}

/** I.i.d. random walk with an optional constant drift per bar: there is no timing edge to find. */
function randomWalk(n: number, seed: number, sigma = 0.02, drift = 0): NamedDataset {
  const rng = mulberry32(seed);
  const data: OHLCV = {
    t: new Float64Array(n),
    o: new Float64Array(n),
    h: new Float64Array(n),
    l: new Float64Array(n),
    c: new Float64Array(n),
    v: new Float64Array(n),
  };
  let close = 100;
  for (let i = 0; i < n; i++) {
    const open = close;
    close = open * Math.exp(drift + sigma * gaussian(rng) - 0.5 * sigma * sigma);
    data.t[i] = i * 86_400_000;
    data.o[i] = open;
    data.c[i] = close;
    data.h[i] = Math.max(open, close) * (1 + Math.abs(gaussian(rng)) * sigma * 0.5);
    data.l[i] = Math.min(open, close) * (1 - Math.abs(gaussian(rng)) * sigma * 0.5);
    data.v[i] = 1000 * Math.exp(gaussian(rng) * 0.3);
  }
  return { symbol: `RW-${seed}`, data };
}

function runOptimizer(
  datasets: NamedDataset[],
  settings: BacktestSettings,
  options: {
    seed: number;
    hypothesis?: Hypothesis;
    stopAfterIterations?: number;
    maxEvaluations?: number;
    priorTrials?: TrialStats;
    windowBudget?: number;
    ports?: MessagePort[];
    batchSize?: number;
    robustTarget?: number;
  },
): Promise<WorkerResponse[]> {
  return new Promise((resolve) => {
    const messages: WorkerResponse[] = [];
    let stopSent = false;
    scope.postMessage = (message) => {
      messages.push(message);
      if (
        !stopSent &&
        options.stopAfterIterations !== undefined &&
        message.type === 'progress' &&
        message.iteration >= options.stopAfterIterations
      ) {
        stopSent = true;
        // Node drains MessagePort queues ahead of timers, so deliver the stop request directly;
        // the loop only notices it after its next yield, exactly as in a real worker.
        scope.onmessage?.({ data: { type: 'stop' } });
      }
      if (message.type === 'done' || message.type === 'error') resolve(messages);
    };
    scope.onmessage?.({
      data: {
        type: 'start',
        payload: {
          datasets,
          settings,
          seed: options.seed,
          hypothesis: options.hypothesis ?? 'TREND_FOLLOWING',
          maxEvaluations: options.maxEvaluations,
          priorTrials: options.priorTrials,
          windowBudget: options.windowBudget,
          ports: options.ports,
          batchSize: options.batchSize,
          robustTarget: options.robustTarget,
        },
      },
    });
  });
}

/** Send one request to the idle worker and return its single reply. */
function request(message: WorkerRequest): WorkerResponse {
  let reply: WorkerResponse | null = null;
  scope.postMessage = (response) => {
    reply = response;
  };
  scope.onmessage?.({ data: message });
  if (reply === null) throw new Error('The worker did not answer.');
  return reply;
}

/** Lock one strategy of the finished run and reveal its holdout. */
const lock = (strategyId: string, priorReveals = 0, simulated = false): WorkerResponse =>
  request({ type: 'lock', strategyId, priorReveals, simulated });

const ofType = <T extends WorkerResponse['type']>(messages: WorkerResponse[], type: T): Of<T>[] =>
  messages.filter((m): m is Of<T> => m.type === type);

function assertConstraints(strategies: StrategyResult[], mode: BacktestSettings['tradeMode']): void {
  for (const s of strategies) {
    expect(s.genome.longEntry.conditions.length).toBeLessThanOrEqual(3);
    expect(s.genome.longEntry.logic).toBe('AND');
    expect(countParameters(s.genome)).toBeLessThanOrEqual(MAX_PARAMETERS);
    expect(s.params).toBeLessThanOrEqual(MAX_PARAMETERS);
    expect(isCoherent(s.genome)).toBe(true);
    if (mode === 'long-only') expect(s.genome.shortEntry).toBeNull();
    else expect(s.genome.shortEntry).toEqual(mirrorGroup(s.genome.longEntry));
    expect('sizing' in s.genome).toBe(false);
  }
}

beforeAll(async () => {
  (globalThis as unknown as { self: FakeScope }).self = scope;
  await import('../src/workers/optimizer.worker');
});

describe('optimizer worker', () => {
  it('keeps the holdout sealed through the search and after it, and reveals it for one locked strategy only', async () => {
    const settings = settingsFor('long-short');
    const messages = await runOptimizer([dataset('BTC-USD', '1d', '5y')], settings, { seed: 12345 });

    const ready = messages[0] as Of<'ready'>;
    expect(ready.type).toBe('ready');
    const { axis } = ready;
    expect(axis.bars).toBeGreaterThan(1800);
    expect(axis.validationIndex).toBe(Math.floor(axis.bars * 0.6));
    expect(axis.holdoutIndex).toBe(Math.floor(axis.bars * 0.8));
    expect(axis.barIndex[axis.validationPos]).toBe(axis.validationIndex);
    expect(axis.barIndex[axis.holdoutPos]).toBe(axis.holdoutIndex);
    // Live benchmark: no holdout metrics and a curve that stops at the holdout boundary.
    expect(ready.benchmark.holdout).toBeNull();
    expect(ready.benchmark.equity.length).toBe(axis.holdoutPos);
    expect(ready.benchmark.train.netPnl + ready.benchmark.validation.netPnl).toBeCloseTo(ready.benchmark.search.netPnl, 4);

    // Progress every 25 iterations.
    const progress = ofType(messages, 'progress');
    expect(progress.length).toBeGreaterThan(10);
    progress.slice(0, -1).forEach((p, index) => expect(p.iteration).toBe((index + 1) * PROGRESS_EVERY));
    for (const p of progress) {
      expect(p.evaluations).toBeLessThanOrEqual(DEFAULT_RUN_EVALUATIONS);
      expect(p.windowTrials).toBe(p.evaluations); // no earlier runs on this window
      // A stall triggers a diversity restart at the next round boundary.
      expect(p.fitnessStall).toBeLessThanOrEqual(FITNESS_PATIENCE + PROGRESS_EVERY);
      expect(p.runTrials.count).toBe(p.evaluations);
    }

    // While the run is live nothing about the holdout leaves the worker.
    const boards = ofType(messages, 'leaderboard');
    expect(boards.length).toBeGreaterThan(2);
    for (const board of boards) {
      for (const s of board.strategies) {
        expect(s.holdout).toBeNull();
        expect(s.verdict).toBeNull();
        expect(s.robustness).toBeNull();
        expect(s.equity.length).toBe(axis.holdoutPos);
        for (const asset of s.assets) expect(asset.holdout).toBeNull();
      }
      assertConstraints(board.strategies, 'long-short');
    }

    // The run ends on its own: early stop (no fitness gain for 1,500 evaluations) or its 5,000-evaluation budget.
    // The holdout is STILL sealed: nothing is revealed for the board.
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(DEFAULT_RUN_EVALUATIONS).toBe(25_000);
    expect(FITNESS_PATIENCE).toBe(1500);
    // The run scans until it has DEFAULT_ROBUST_TARGET robust strategies, or its budget is spent.
    expect(['target', 'budget']).toContain(done.reason);
    if (done.reason === 'target') expect(done.robustFound).toBeGreaterThanOrEqual(DEFAULT_ROBUST_TARGET);
    else expect(done.evaluations).toBe(DEFAULT_RUN_EVALUATIONS);
    expect(done.strategies.filter((s) => s.robust?.passed).length).toBe(done.robustFound);
    expect(done.evaluations).toBeLessThanOrEqual(DEFAULT_RUN_EVALUATIONS);
    // Repeated genomes and repeated trade logs were skipped, not counted as trials.
    expect(done.iteration).toBeGreaterThan(done.evaluations);
    expect(done.duplicates).toBeGreaterThan(0);
    expect(done.benchmark.holdout).toBeNull();
    expect(done.benchmark.equity.length).toBe(axis.holdoutPos);

    const final = done.strategies;
    expect(final.length).toBeGreaterThan(5);
    // The leaderboard, plus robust strategies its fitness ranking pushed off.
    expect(final.length).toBeLessThanOrEqual(LEADERBOARD_SIZE + done.robustFound);
    for (let i = 1; i < final.length; i++) expect(final[i - 1].fitness).toBeGreaterThanOrEqual(final[i].fitness);
    expect(new Set(final.map((s) => s.id)).size).toBe(final.length);
    assertConstraints(final, 'long-short');
    for (const s of final) {
      expect(s.holdout).toBeNull();
      expect(s.verdict).toBeNull();
      expect(s.equity.length).toBe(axis.holdoutPos);
      for (const asset of s.assets) expect(asset.holdout).toBeNull();
      // Hard sample-size gates: nothing below them can be listed.
      expect(s.search.trades).toBeGreaterThanOrEqual(s.sample!.gates.search);
      expect(s.validation.trades).toBeGreaterThanOrEqual(s.sample!.gates.validation);
      expect(s.alpha).toBeCloseTo(s.search.roi - done.benchmark.search.roi, 6);
      expect(s.train.netPnl + s.validation.netPnl).toBeCloseTo(s.search.netPnl, 4);
      expect(s.dsr).toBeGreaterThanOrEqual(0);
      expect(s.dsr).toBeLessThanOrEqual(1);
      expect(s.assets).toHaveLength(1);
    }
    expect(done.bestFitness).toBeCloseTo(final[0].fitness, 9);
    expect(done.bestDsr).toBeCloseTo(Math.max(...final.map((s) => s.dsr)), 9);
    expect(done.hurdleSharpe).toBeGreaterThan(0);

    // Stage 2 ranking: every listed strategy carries its fold and regime breakdown, and is ranked by
    // its own fitness capped at the median of its one-grid-step neighbours (spikes sink, plateaus hold).
    for (const s of final) {
      expect(s.folds).toHaveLength(FOLDS);
      expect(s.regimes?.map((r) => r.regime)).toEqual(['Bull', 'Bear', 'High volatility']);
      expect(s.neighbourCount ?? 0).toBeGreaterThan(2);
      expect(s.pointFitness).toBeDefined();
      expect(s.fitness).toBeLessThanOrEqual((s.pointFitness as number) + 1e-12);
      expect(s.fitness).toBeCloseTo(plateauFitness(s.pointFitness as number, [s.neighbourMedian as number]), 9);
    }
    // Every listed strategy tests the run's hypothesis, and no two of them share a trade log.
    const btc = dataset('BTC-USD', '1d', '5y');
    const btcSplit = splitFor(btc.data.c.length);
    const btcInd = new IndicatorCache(btc.data);
    const btcWs = new BacktestWorkspace(btc.data.c.length);
    const logs = new Set<string>();
    for (const s of final) {
      expect(s.genome.hypothesis).toBe('TREND_FOLLOWING');
      expect(s.haircutSharpe).toBeDefined();
      expect(s.haircut ?? -1).toBeGreaterThanOrEqual(0);
      runBacktest(btcInd, s.genome, settings, btcWs, btcSplit.holdoutStart);
      logs.add(tradeLogHash(btcWs, btcSplit.holdoutStart));
    }
    expect(logs.size).toBe(final.length);
    // The cap actually bites on part of the board: those strategies were spikes at their exact parameters.
    expect(final.filter((s) => (s.pointFitness as number) > s.fitness + 1e-9).length).toBeGreaterThan(0);

    // Before the lock no strategy gets a verdict, even on request.
    const early = request({ type: 'analyze', strategyId: final[2].id, genome: final[2].genome }) as Of<'analysis'>;
    expect(early.verdict).toBeNull();

    // Lock ONE strategy: only its holdout is revealed.
    const chosen = final[1];
    const reply = lock(chosen.id) as Of<'locked'>;
    expect(reply.type).toBe('locked');
    const top = reply.strategy;
    expect(top.id).toBe(chosen.id);
    expect(top.holdout).not.toBeNull();
    expect(top.equity.length).toBe(axis.barIndex.length);
    expect(reply.benchmark.holdout).not.toBeNull();
    expect(reply.benchmark.equity.length).toBe(axis.barIndex.length);
    expect(['Robust', 'Fragile', 'Overfit']).toContain(top.verdict);
    expect(Array.isArray(top.verdictNotes)).toBe(true);
    expect((top.verdictNotes ?? []).length === 0).toBe(top.verdict === 'Robust');
    expect(top.robustness?.sensitivity.variants).toBeGreaterThan(2);
    expect(top.robustness?.topTrades.removed).toBeLessThanOrEqual(5);
    expect(top.robustness?.crossAsset).toBeNull(); // single-asset run
    // A strategy that lost money on the holdout, or whose DSR says luck, is always Overfit.
    if ((top.holdout?.roi ?? 0) <= 0 || top.dsr < 0.5) expect(top.verdict).toBe('Overfit');
    // Robust needs a holdout Sharpe CI above zero that beats Buy & Hold.
    if (top.verdict === 'Robust') expect(top.holdout?.sharpe ?? 0).toBeGreaterThan(reply.benchmark.holdout?.sharpe ?? Infinity);

    // The holdout is spent: no second strategy can be revealed. Re-locking the same one is idempotent.
    const second = lock(final[0].id) as Of<'error'>;
    expect(second.type).toBe('error');
    expect(second.message).toContain(chosen.id);
    const again = lock(chosen.id) as Of<'locked'>;
    expect(again.type).toBe('locked');
    expect(again.strategy).toBe(top);
    // Verdicts: the locked strategy gets one, every other strategy stays without.
    const lockedAnalysis = request({ type: 'analyze', strategyId: chosen.id, genome: chosen.genome }) as Of<'analysis'>;
    expect(lockedAnalysis.verdict).toBe(top.verdict);
    const otherAnalysis = request({ type: 'analyze', strategyId: final[0].id, genome: final[0].genome }) as Of<'analysis'>;
    expect(otherAnalysis.verdict).toBeNull();

    console.log(
      `converged after ${done.iteration} iterations in ${(done.elapsedMs / 1000).toFixed(1)}s, ${final.length} on the board, ` +
        `${done.undersampled} rejected for too few trades | locked ${top.id}: fitness ${top.fitness.toFixed(2)} ` +
        `params ${top.params} trades ${top.search.trades} DSR ${(top.dsr * 100).toFixed(0)}% ` +
        `validation Sharpe ${top.validation.sharpe.toFixed(2)} holdout Sharpe ${top.holdout?.sharpe.toFixed(2)} verdict ${top.verdict}`,
    );
  }, 300_000);

  it('stops promptly when the user presses Stop, and never shorts in Long-Only mode', async () => {
    const settings = settingsFor('long-only');
    const messages = await runOptimizer([dataset('NVDA', '1h', '1y')], settings, { seed: 7, stopAfterIterations: 300 });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(done.reason).toBe('stopped');
    expect(done.iteration).toBeGreaterThanOrEqual(300);
    expect(done.iteration).toBeLessThan(300 + 4 * PROGRESS_EVERY);
    expect(done.strategies.length).toBeGreaterThan(0);
    assertConstraints(done.strategies, 'long-only');
    // Stopping ends the search but keeps the holdout sealed until one strategy is locked.
    for (const s of done.strategies) {
      expect(s.holdout).toBeNull();
      if (s.genome.exit.mode === 'dynamic') expect(s.genome.exit.shortExit).toBeNull();
    }
    const reply = lock(done.strategies[0].id) as Of<'locked'>;
    expect(reply.type).toBe('locked');
    expect(reply.strategy.holdout).not.toBeNull();
  }, 120_000);

  it('refuses to reveal the holdout while the search is running, or for a strategy not on the final board', async () => {
    const replies: WorkerResponse[] = [];
    const done = await new Promise<Of<'done'>>((resolve) => {
      let asked = false;
      scope.postMessage = (message) => {
        if (message.type === 'progress' && !asked) {
          asked = true;
          // The worker answers synchronously: capture its reply to the mid-run lock request.
          const runPost = scope.postMessage;
          scope.postMessage = (reply) => replies.push(reply);
          scope.onmessage?.({ data: { type: 'lock', strategyId: 'S-1', priorReveals: 0, simulated: false } });
          scope.postMessage = runPost;
        }
        if (message.type === 'done') resolve(message);
      };
      scope.onmessage?.({
        data: {
          type: 'start',
          payload: { datasets: [dataset('NVDA', '1h', '1y')], settings: settingsFor('long-only'), hypothesis: 'TREND_FOLLOWING', seed: 8, maxEvaluations: 300 },
        },
      });
    });
    expect(replies).toHaveLength(1);
    expect(replies[0].type).toBe('error');
    expect((replies[0] as Of<'error'>).message).toContain('after the run has ended');
    const unknown = lock('S-not-on-the-board') as Of<'error'>;
    expect(unknown.type).toBe('error');
    expect(unknown.message).toContain('final leaderboard');
    // Neither refusal spent the holdout.
    if (done.strategies.length > 0) expect(lock(done.strategies[0].id).type).toBe('locked');
  }, 120_000);

  it('honours a custom evaluation limit', async () => {
    const messages = await runOptimizer([dataset('SPY', '1d', '2y')], settingsFor('long-only'), { seed: 3, maxEvaluations: 300 });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.reason).toBe('budget');
    expect(done.evaluations).toBe(300);
  }, 120_000);

  it('counts the trials of earlier sessions: they raise the hurdle, lower every DSR and use up the window budget', async () => {
    const eth = dataset('ETH-USD', '4h', '2y');
    const settings = settingsFor('long-only');
    const fresh = (await runOptimizer([eth], settings, { seed: 41, maxEvaluations: 600 })).pop() as Of<'done'>;
    expect(fresh.type).toBe('done');
    expect(fresh.windowTrials).toBe(fresh.evaluations);
    // Pretend all but 300 trials of the window's budget were spent on it in earlier sessions.
    const variance = fresh.runTrials.m2 / (fresh.runTrials.count - 1);
    const earlier = WINDOW_TRIAL_BUDGET - 300;
    const prior: TrialStats = { count: earlier, mean: fresh.runTrials.mean, m2: variance * (earlier - 1) };
    const messages = await runOptimizer([eth], settings, { seed: 41, maxEvaluations: 600, priorTrials: prior });
    const ready = messages[0] as Of<'ready'>;
    const later = messages[messages.length - 1] as Of<'done'>;
    expect(later.type).toBe('done');
    // Only 300 evaluations were left on the window: the run stops there, and the window is spent.
    expect(later.evaluations).toBe(300);
    expect(later.reason).toBe('window-budget');
    expect(later.windowTrials).toBe(WINDOW_TRIAL_BUDGET);
    // The hurdle reflects every trial on the window, so each strategy's DSR is lower than it would be against this run's alone.
    expect(later.hurdleSharpe).toBeGreaterThan(fresh.hurdleSharpe);
    const freshHurdle = fresh.hurdleSharpe / Math.sqrt(ready.barsPerYear);
    for (const s of later.strategies) {
      expect(s.dsr).toBeLessThanOrEqual(deflatedSharpeAgainst(s.validation, freshHurdle).probability + 1e-12);
    }
    // A spent window refuses to start at all.
    const refused = await runOptimizer([eth], settings, { seed: 41, priorTrials: { ...prior, count: WINDOW_TRIAL_BUDGET } });
    const error = refused[refused.length - 1] as Of<'error'>;
    expect(error.type).toBe('error');
    expect(error.message).toMatch(/trial budget/);
  }, 300_000);

  it('restarts the population instead of stopping when the search stalls', async () => {
    const messages = await runOptimizer([randomWalk(1826, 77, 0.035)], settingsFor('long-only'), {
      seed: 5,
      hypothesis: 'RANGE_REVERSION',
      maxEvaluations: 8_000,
      robustTarget: 0, // run the full budget
    });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    // No early exit: the whole budget is scanned ...
    expect(done.reason).toBe('budget');
    expect(done.evaluations).toBe(8_000);
    // ... in several population cycles, each started by a stall of FITNESS_PATIENCE evaluations.
    expect(done.cycle).toBeGreaterThanOrEqual(3);
    const cycles = ofType(messages, 'progress').map((p) => p.cycle);
    for (let k = 1; k < cycles.length; k++) expect(cycles[k]).toBeGreaterThanOrEqual(cycles[k - 1]);
    expect(done.evaluations / done.cycle).toBeGreaterThanOrEqual(FITNESS_PATIENCE * 0.9);
    // The funnel counts every scored candidate per gate.
    expect(done.funnel.scored).toBeGreaterThan(1000);
    for (const key of ['trades', 'validation', 'folds', 'dsr'] as const) expect(done.funnel[key]).toBeLessThanOrEqual(done.funnel.scored);
    expect(done.funnel.neighbours).toBeLessThanOrEqual(done.funnel.neighboursChecked);
    expect(done.robustTarget).toBe(0);
  }, 300_000);

  it('scores a Robust universe on every asset and answers analyze requests', async () => {
    // 4h candles: two years of daily candles cannot supply the minimum number of trades on every asset.
    const universe = ['BTC-USD', 'ETH-USD', 'SOL-USD'].map((symbol) => dataset(symbol, '4h', '2y'));
    const settings = settingsFor('long-only');
    const messages = await runOptimizer(universe, settings, { seed: 21, maxEvaluations: 400 });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(done.strategies.length).toBeGreaterThan(0);
    for (const board of ofType(messages, 'leaderboard')) {
      for (const s of board.strategies) {
        expect(s.assets.map((a) => a.symbol)).toEqual(['BTC-USD', 'ETH-USD', 'SOL-USD']);
        for (const asset of s.assets) expect(asset.holdout).toBeNull();
        // The headline metrics are the primary asset's.
        expect(s.search).toEqual(s.assets[0].search);
      }
    }
    for (const s of done.strategies) {
      expect(s.assets).toHaveLength(3);
      for (const asset of s.assets) {
        expect(asset.holdout).toBeNull();
        // Every asset clears the per-year trade gate.
        expect(asset.search.trades).toBeGreaterThan(0);
      }
    }

    // On-demand analysis, with two extra assets supplied for the cross-asset test.
    const target = done.strategies[0];
    const extra = [dataset('XRP-USD', '4h', '2y'), dataset('DOGE-USD', '4h', '2y')];
    const reply = request({ type: 'analyze', strategyId: target.id, genome: target.genome, universe: extra }) as Of<'analysis'>;
    expect(reply.type).toBe('analysis');
    expect(reply.strategyId).toBe(target.id);
    expect(reply.report.crossAsset?.rows.map((row) => row.symbol)).toEqual(['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']);
    expect(reply.report.sensitivity.baseRoi).toBeCloseTo(target.search.roi, 6);
    expect(reply.report.topTrades.netPnlWithout).toBeLessThanOrEqual(reply.report.topTrades.netPnl);
    expect(reply.verdict).toBeNull(); // not locked yet

    // Locking reveals every asset's holdout for that one strategy.
    const locked = lock(target.id) as Of<'locked'>;
    expect(locked.type).toBe('locked');
    for (const asset of locked.strategy.assets) expect(asset.holdout).not.toBeNull();
    expect(locked.strategy.robustness?.crossAsset?.rows.map((row) => row.symbol)).toEqual(['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']);
    const after = request({ type: 'analyze', strategyId: target.id, genome: target.genome }) as Of<'analysis'>;
    expect(['Robust', 'Fragile', 'Overfit']).toContain(after.verdict);

    // Any other strategy can still be analysed, but never gets a verdict.
    const other = request({ type: 'analyze', strategyId: 'S-custom', genome: done.strategies[done.strategies.length - 1].genome }) as Of<'analysis'>;
    expect(other.type).toBe('analysis');
    expect(other.verdict).toBeNull();
  }, 300_000);

  it('keeps the train / validation figures unchanged when the locked strategy\'s holdout is revealed', async () => {
    const eth = dataset('ETH-USD', '4h', '2y');
    const settings = settingsFor('long-short');
    const messages = await runOptimizer([eth], settings, { seed: 99, maxEvaluations: 600 });
    const done = messages[messages.length - 1] as Of<'done'>;
    // The final figures must be exactly what the search ranked on: a backtest that stops at the holdout boundary.
    const split = splitFor(eth.data.c.length);
    const ind = new IndicatorCache(eth.data);
    const ws = new BacktestWorkspace(eth.data.c.length);
    const ppy = barsPerYear(eth.data.t);
    let straddling: (typeof done.strategies)[number] | null = null;
    for (const final of done.strategies) {
      const expected = evaluateSearch(ind, final.genome, settings, ws, split, ppy);
      expect(final.search).toEqual(expected.search);
      expect(final.train).toEqual(expected.train);
      expect(final.validation).toEqual(expected.validation);
      expect(final.holdout).toBeNull();
      runBacktest(ind, final.genome, settings, ws);
      if (straddling === null && ws.inMarket[split.holdoutStart - 1] === 1) straddling = final;
    }
    // Lock a strategy that holds a position across the boundary, so the two backtests can be told apart.
    expect(straddling).not.toBeNull();
    const chosen = straddling as NonNullable<typeof straddling>;
    const reply = lock(chosen.id) as Of<'locked'>;
    expect(reply.type).toBe('locked');
    expect(reply.strategy.holdout).not.toBeNull();
    expect(reply.strategy.search).toEqual(chosen.search);
    expect(reply.strategy.train).toEqual(chosen.train);
    expect(reply.strategy.validation).toEqual(chosen.validation);
  }, 120_000);

  it('gives identical results on an evaluator pool and on a single thread', async () => {
    const universe = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'].map((symbol) => dataset(symbol, '4h', '2y'));
    const settings = settingsFor('long-short');
    const options = { seed: 314, maxEvaluations: 500, batchSize: 75 };

    const single = await runOptimizer(universe, settings, options);

    // Three in-process evaluators on real MessageChannels – the same protocol the browser uses.
    const channels = [0, 1, 2].map(() => new MessageChannel());
    for (const channel of channels) {
      attachEvaluator(channel.port2 as unknown as Parameters<typeof attachEvaluator>[0], {
        datasets: universe,
        settings,
        budgetBytes: 64 * 1024 * 1024,
      });
    }
    const pooled = await runOptimizer(universe, settings, { ...options, ports: channels.map((c) => c.port1) });
    for (const channel of channels) channel.port2.close();

    const a = single[single.length - 1] as Of<'done'>;
    const b = pooled[pooled.length - 1] as Of<'done'>;
    expect(b.type).toBe('done');
    expect(a.strategies.length).toBeGreaterThan(0);
    expect(b.iteration).toBe(a.iteration);
    expect(b.stagnation).toBe(a.stagnation);
    expect(b.strategies.map((s) => s.id)).toEqual(a.strategies.map((s) => s.id));
    expect(b.strategies.map((s) => s.fitness)).toEqual(a.strategies.map((s) => s.fitness));
    expect(b.strategies.map((s) => s.genome)).toEqual(a.strategies.map((s) => s.genome));
    expect(Array.from(b.strategies[0].equity)).toEqual(Array.from(a.strategies[0].equity));
    // Progress is still streamed every 25 iterations even though rounds are 75 candidates long.
    const progress = ofType(pooled, 'progress');
    progress.slice(0, -1).forEach((p, index) => expect(p.iteration).toBe((index + 1) * PROGRESS_EVERY));
  }, 300_000);

  it('runs on 15m and 30m candles', async () => {
    for (const [symbol, interval, range] of [['SOL-USD', '30m', '1y'], ['AAPL', '15m', '1y']] as const) {
      const messages = await runOptimizer([dataset(symbol, interval, range)], settingsFor('long-only'), { seed: 5, maxEvaluations: 300 });
      const done = messages[messages.length - 1] as Of<'done'>;
      expect(done.type, `${symbol} ${interval}`).toBe('done');
      expect(['budget', 'target']).toContain(done.reason);
      for (const s of done.strategies) expect(s.holdout).toBeNull();
      const ready = messages[0] as Of<'ready'>;
      if (interval === '30m') {
        // 30m crypto over a year: 17,520 candles – plenty of trades to clear the gates.
        expect(ready.axis.bars).toBeGreaterThan(17_000);
        expect(done.strategies.length).toBeGreaterThan(0);
      } else {
        // 15m stock: Yahoo's 60-day window. The gates are calibrated to the data – about a fifth of a
        // year needs only the absolute floors – and everything listed clears them.
        expect(ready.axis.bars).toBeLessThan(1_200);
        expect(ready.gates).toMatchObject({ search: 30, validation: 10 });
        for (const s of done.strategies) {
          expect(s.search.trades).toBeGreaterThanOrEqual(ready.gates.search);
          expect(s.validation.trades).toBeGreaterThanOrEqual(ready.gates.validation);
        }
      }
    }
  }, 300_000);

  it('fine-tunes EMA / SMA lengths over every integer when asked to', async () => {
    const settings: BacktestSettings = { ...settingsFor('long-only'), maPeriods: 'integer' };
    const messages = await runOptimizer([dataset('SOL-USD', '30m', '1y')], settings, {
      seed: 21,
      hypothesis: 'TREND_FOLLOWING',
      maxEvaluations: 800,
    });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    const listed = [...ofType(messages, 'leaderboard').flatMap((m) => m.strategies), ...done.strategies];
    expect(listed.length).toBeGreaterThan(0);
    const lengths: number[] = [];
    const types = new Set<string>();
    for (const s of listed) {
      for (const c of s.genome.longEntry.conditions) {
        if (c.ind === 'MA_CROSS') {
          lengths.push(c.fastPeriod, c.slowPeriod);
          types.add(c.fastType);
        }
        if (c.ind === 'PRICE_MA' || c.ind === 'MA_SLOPE') {
          lengths.push(c.period);
          types.add(c.maType);
        }
      }
    }
    for (const n of lengths) expect(Number.isInteger(n)).toBe(true);
    // The search lists lengths the coarse grid could never reach (e.g. 23 or 87).
    const grid = [5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200];
    expect(lengths.filter((n) => !grid.includes(n)).length).toBeGreaterThan(0);
    expect(types.size).toBeGreaterThan(0);
  }, 300_000);

  it('reports an error instead of crashing on unusable data', async () => {
    const tiny = dataset('SPY', '1d', '1y');
    const cut: OHLCV = {
      t: tiny.data.t.slice(0, 50),
      o: tiny.data.o.slice(0, 50),
      h: tiny.data.h.slice(0, 50),
      l: tiny.data.l.slice(0, 50),
      c: tiny.data.c.slice(0, 50),
      v: tiny.data.v.slice(0, 50),
    };
    const messages = await runOptimizer([{ symbol: 'SPY', data: cut }], settingsFor('long-only'), { seed: 1 });
    expect(messages).toHaveLength(1);
    expect(messages[0].type).toBe('error');
  });
});

describe('Monte Carlo stress test and cross-market generalization check', () => {
  it('stress-tests every leaderboard strategy and checks its frozen rules on unseen markets with the holdout sealed', async () => {
    const settings = settingsFor('asymmetric');
    const btc = dataset('BTC-USD', '1d', '5y');
    const messages = await runOptimizer([btc], settings, { seed: 77, maxEvaluations: 400, hypothesis: 'VOL_BREAKOUT' });
    const ready = messages[0] as Of<'ready'>;
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(done.strategies.length).toBeGreaterThan(0);
    for (const s of done.strategies) {
      // Asymmetric L/S: every strategy carries its own short regime filter.
      expect(s.genome.shortRegime).toBeTruthy();
      expect(s.genome.shortEntry?.conditions[1]).toEqual(s.genome.shortRegime);
      // Every leaderboard entry carries the headline Monte Carlo numbers (no chart bands).
      const mc = s.monteCarlo;
      expect(mc).toBeTruthy();
      if (!mc) continue;
      expect(mc.simulations).toBeGreaterThanOrEqual(200);
      expect(mc.probProfit).toBeGreaterThanOrEqual(0);
      expect(mc.probProfit).toBeLessThanOrEqual(1);
      expect(mc.drawdownP95).toBeGreaterThanOrEqual(0);
      expect('bands' in mc).toBe(false);
      expect(mc.trades).toBe(s.search.trades);
    }

    const top = done.strategies[0];
    // The on-demand analysis carries the full report, with the equity-path bands for the chart.
    const analysis = request({ type: 'analyze', strategyId: top.id, genome: top.genome }) as Of<'analysis'>;
    expect(analysis.type).toBe('analysis');
    expect(analysis.report.monteCarlo?.bands.p50.length).toBeGreaterThan(10);
    expect(analysis.report.monteCarlo?.probProfit).toBe(top.monteCarlo?.probProfit);

    // Cross-market check: two unseen assets and one the run was scored on.
    const sol = dataset('SOL-USD', '1d', '5y');
    const bnb = dataset('BNB-USD', '1d', '5y');
    const reply = request({ type: 'cross-market', strategyId: top.id, genome: top.genome, datasets: [sol, bnb, btc] }) as Of<'cross-market'>;
    expect(reply.type).toBe('cross-market');
    const report = reply.report;
    expect(report.rows.map((row) => row.symbol)).toEqual(['SOL-USD', 'BNB-USD', 'BTC-USD']);
    expect(report.rows.map((row) => row.unseen)).toEqual([true, true, false]);
    expect(report.tested).toBe(2);
    // The holdout stays sealed: only candles before the run's holdout boundary are traded.
    const cutoff = ready.axis.time[ready.axis.holdoutPos];
    expect(report.cutoff).toBe(cutoff);
    for (const row of report.rows) {
      expect(row.note).toBeNull();
      expect(row.bars).toBeLessThanOrEqual(ready.axis.holdoutIndex + 1);
      expect(row.bars).toBeGreaterThan(ready.axis.holdoutIndex - 5);
    }
    // Frozen rules: exactly a backtest of the same genome over those candles.
    const ws = new BacktestWorkspace(sol.data.c.length);
    runBacktest(new IndicatorCache(sol.data), top.genome, settings, ws, report.rows[0].bars);
    const m = segmentMetrics(ws.equity, 0, report.rows[0].bars, settings.initialCapital, barsPerYear(sol.data.t), ws);
    expect(report.rows[0].roi).toBeCloseTo(m.roi, 9);
    expect(report.rows[0].trades).toBe(m.trades);
    const unseen = report.rows.filter((row) => row.unseen);
    expect(report.share).toBeCloseTo(unseen.filter((row) => row.profitable).length / unseen.length, 12);
    expect(report.passed).toBe(report.share >= 2 / 3 - 1e-9 && report.medianSharpe > 0);

    // A leg of the run (e.g. of a basket) is never counted as unseen.
    const legs = request({ type: 'cross-market', strategyId: top.id, genome: top.genome, datasets: [sol], runSymbols: ['SOL-USD'] }) as Of<'cross-market'>;
    expect(legs.report.rows[0].unseen).toBe(false);
    expect(legs.report.tested).toBe(0);

    // A history that ends before enough candles exist is reported, not traded.
    const short = { symbol: 'NEW-USD', data: { ...sol.data, t: sol.data.t.map((t) => t + 4 * 365 * 86_400_000) } };
    const late = request({ type: 'cross-market', strategyId: top.id, genome: top.genome, datasets: [short] }) as Of<'cross-market'>;
    expect(late.report.tested).toBe(0);
    expect(late.report.rows[0].note).toMatch(/candles before/);
    expect(late.report.passed).toBe(false);
  }, 300_000);
});

describe('acceptance: pure noise must not look like an edge', () => {
  /*
   * The walks of docs/AUDIT.md §0, reproduced exactly: 1,826 daily candles (≈ 5 years), 3.5% daily
   * volatility, the same seeds. Before any fix, 96 of the 150 strategies on the drift boards
   * (64%) made money on the holdout – the test the verdict used then. These tests judge the WHOLE
   * top-25 board of every run, not just #1, with the holdout conditions the verdict uses now.
   * Revealing every holdout is only done here, by the test acting as an outside oracle; the app
   * itself reveals one locked strategy per run.
   */
  const AUDIT_BARS = 1826;
  const AUDIT_SIGMA = 0.035;
  const AUDIT_DRIFT_PER_BAR = 0.4 / 365;
  const SETTINGS = settingsFor('long-only');

  interface BoardRow {
    dsr: number;
    /** Annualised Sharpe over train + validation – what the search saw. */
    searchSharpe: number;
    holdoutRoi: number;
    holdoutSharpe: number;
    /** Passes every holdout condition of a Robust verdict (`holdoutShortfalls` is empty). */
    holdoutPass: boolean;
    /** Holdout conditions AND a deflated Sharpe of 95% or more: everything Robust needs short of the robustness tests. */
    robustEligible: boolean;
  }

  /** Run on noise, then reveal the holdout of every strategy on the final board (oracle only). */
  async function revealWholeBoard(data: NamedDataset, seed: number, hypothesis: Hypothesis): Promise<{ done: Of<'done'>; rows: BoardRow[] }> {
    const messages = await runOptimizer([data], SETTINGS, { seed, hypothesis });
    const done = messages[messages.length - 1] as Of<'done'>;
    const ready = messages[0] as Of<'ready'>;
    expect(done.type).toBe('done');
    expect(['converged', 'budget']).toContain(done.reason);
    assertConstraints(done.strategies, 'long-only');
    const n = data.data.c.length;
    const split = splitFor(n);
    const ppy = barsPerYear(data.data.t);
    const ind = new IndicatorCache(data.data);
    const ws = new BacktestWorkspace(n);
    const bench = benchmarkEquity(data.data, SETTINGS);
    const benchHoldout = segmentMetrics(bench, split.holdoutStart, n, bench[split.holdoutStart - 1], ppy, null, null);
    const rows = done.strategies.map((s) => {
      expect(s.holdout).toBeNull(); // the app itself never revealed it
      expect(s.validation.trades).toBeGreaterThanOrEqual(ready.gates.validation);
      runBacktest(ind, s.genome, SETTINGS, ws);
      const h = holdoutMetrics(ws.equity, split, ppy, ws);
      const holdoutPass = holdoutShortfalls(h, benchHoldout, ppy, ready.gates.holdout).length === 0;
      return { dsr: s.dsr, searchSharpe: s.search.sharpe, holdoutRoi: h.roi, holdoutSharpe: h.sharpe, holdoutPass, robustEligible: holdoutPass && s.dsr >= 0.95 };
    });
    return { done, rows };
  }

  /** Lock the #1 strategy, as a user would, and return its verdict. */
  function lockTop(done: Of<'done'>): StrategyResult | null {
    if (done.strategies.length === 0) return null;
    const reply = lock(done.strategies[0].id) as Of<'locked'>;
    expect(reply.type).toBe('locked');
    return reply.strategy;
  }

  async function nullBoards(driftPerBar: number, walkSeed: number): Promise<BoardRow[]> {
    const all: BoardRow[] = [];
    const lines: string[] = [];
    for (let k = 0; k < 6; k++) {
      // Every hypothesis gets its turn on the noise.
      const { done, rows } = await revealWholeBoard(
        randomWalk(AUDIT_BARS, walkSeed + k, AUDIT_SIGMA, driftPerBar),
        900 + k,
        HYPOTHESIS_IDS[k % HYPOTHESIS_IDS.length],
      );
      const top = lockTop(done);
      // Nothing on any board may look certified, and the locked #1 is never Robust.
      for (const row of rows) expect(row.dsr).toBeLessThan(0.95);
      // Noise never passes the search-period robustness gates: the scan runs its whole budget.
      expect(done.robustFound).toBe(0);
      expect(done.reason).toBe('budget');
      if (top) expect(top.verdict).not.toBe('Robust');
      all.push(...rows);
      lines.push(
        `run ${k} (${HYPOTHESIS_IDS[k % HYPOTHESIS_IDS.length]}): ${done.evaluations} evaluations (${done.reason}), board ${rows.length} | holdout ROI > 0: ${rows.filter((r) => r.holdoutRoi > 0).length} | ` +
          `passes the holdout conditions: ${rows.filter((r) => r.holdoutPass).length} | robust (search gates) ${done.robustFound} | locked #1 ${top?.verdict ?? 'n/a'}`,
      );
    }
    const share = (f: (r: BoardRow) => boolean): string => `${all.filter(f).length}/${all.length} (${((100 * all.filter(f).length) / Math.max(1, all.length)).toFixed(1)}%)`;
    const summary = (
      `${lines.join('\n')}\nwhole board: holdout ROI > 0 ${share((r) => r.holdoutRoi > 0)} · passes the holdout conditions ${share((r) => r.holdoutPass)} · ` +
        `Robust-eligible ${share((r) => r.robustEligible)} · mean holdout Sharpe ${(all.reduce((sum, r) => sum + r.holdoutSharpe, 0) / Math.max(1, all.length)).toFixed(2)}`
    );
    console.log(summary);
    if (process.env.NULL_LOG) appendFileSync(process.env.NULL_LOG, `${summary}\n`);
    return all;
  }

  it('flags every strategy on the boards of driftless random walks as unproven', async () => {
    const rows = await nullBoards(0, 500);
    // One hypothesis per run: on five years of daily noise the breakout and dip-buy ideas cannot reach
    // 60 validation trades, so the hard gates leave their boards empty and those runs stop early.
    expect(rows.length).toBeGreaterThan(20);
    expect(rows.filter((r) => r.robustEligible)).toHaveLength(0);
    // A false positive is a strategy whose holdout meets the verdict's conditions: rare on noise.
    expect(rows.filter((r) => r.holdoutPass).length / rows.length).toBeLessThanOrEqual(0.1);
    // With no edge to find, the board as a whole does not make risk-adjusted money on the holdout.
    expect(rows.reduce((sum, r) => sum + r.holdoutSharpe, 0) / rows.length).toBeLessThanOrEqual(0);
    // Selection inflation: how much better the board looked in-sample than on the holdout. On these
    // exact walks it was 1.33 Sharpe before any fix and 1.11 after Stage 1; fold + neighbourhood
    // scoring brings it under 1.
    const inflation = rows.reduce((sum, r) => sum + r.searchSharpe - r.holdoutSharpe, 0) / rows.length;
    console.log(`selection inflation (mean in-sample − holdout Sharpe): ${inflation.toFixed(2)}`);
    if (process.env.NULL_LOG) appendFileSync(process.env.NULL_LOG, `inflation ${inflation.toFixed(2)}\n`);
    expect(inflation).toBeLessThan(1);
  }, 900_000);

  it('finds robust strategies when a real edge exists, and stops at its target', async () => {
    // Planted edge: the drift switches sign in persistent regimes (about every 20 candles), so trend
    // following has a genuine, frequently repeated edge. The same gates that reject every random walk
    // must accept it – and the scan then stops at its target instead of spending the budget.
    const rng = mulberry32(3);
    const n = 1826;
    const data: OHLCV = { t: new Float64Array(n), o: new Float64Array(n), h: new Float64Array(n), l: new Float64Array(n), c: new Float64Array(n), v: new Float64Array(n) };
    let close = 100;
    let dir = 1;
    const sigma = 0.02;
    for (let i = 0; i < n; i++) {
      if (rng() < 1 / 20) dir = -dir;
      const open = close;
      close = open * Math.exp(dir * 0.006 + sigma * gaussian(rng) - 0.5 * sigma * sigma);
      data.t[i] = i * 86_400_000;
      data.o[i] = open;
      data.c[i] = close;
      data.h[i] = Math.max(open, close) * (1 + Math.abs(gaussian(rng)) * sigma * 0.5);
      data.l[i] = Math.min(open, close) * (1 - Math.abs(gaussian(rng)) * sigma * 0.5);
      data.v[i] = 1000 * Math.exp(gaussian(rng) * 0.3);
    }
    const messages = await runOptimizer([{ symbol: 'EDGE', data }], { ...settingsFor('long-short'), slippageAtrMult: 0.03 }, { seed: 4, hypothesis: 'TREND_FOLLOWING', robustTarget: 3 });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.reason).toBe('target');
    expect(done.robustFound).toBeGreaterThanOrEqual(3);
    // The target only ends a run after a minimum scan.
    expect(done.evaluations).toBeGreaterThanOrEqual(1000);
    expect(done.evaluations).toBeLessThan(DEFAULT_RUN_EVALUATIONS);
    const robust = done.strategies.filter((s) => s.robust?.passed);
    expect(robust.length).toBe(done.robustFound);
    for (const s of robust) {
      expect(s.validation.sharpe).toBeGreaterThan(0.3);
      expect(s.validation.netPnl).toBeGreaterThan(0);
      expect(s.folds!.filter((f) => f.roi > 0).length).toBeGreaterThanOrEqual(3);
      expect(s.dsr).toBeGreaterThanOrEqual(0.5);
      expect(s.genome.longEntry.conditions.length).toBeLessThanOrEqual(2);
      expect(s.params).toBeLessThanOrEqual(5);
    }
    // A robust strategy can be locked like any other – the holdout is still revealed only once.
    expect(lock(robust[0].id).type).toBe('locked');
  }, 600_000);

  it('pools trades across a multi-asset universe without certifying noise', async () => {
    // Four independent random walks traded with the same rules: the trades are pooled for the gates,
    // the score is the median over the assets, and nothing on the board may look like an edge.
    const walks = [0, 1, 2, 3].map((k) => randomWalk(AUDIT_BARS, 700 + k, AUDIT_SIGMA, AUDIT_DRIFT_PER_BAR));
    const messages = await runOptimizer(walks, { ...SETTINGS, poolAssets: true }, { seed: 77, hypothesis: 'TREND_FOLLOWING' });
    const ready = messages[0] as Of<'ready'>;
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(ready.pooled).toBe(true);
    expect(done.type).toBe('done');
    expect(done.strategies.length).toBeGreaterThan(5);
    let belowSingleAssetGate = 0;
    for (const s of done.strategies) {
      expect(s.sample?.pooled).toBe(true);
      // The gates apply to the pooled sample ...
      const pooledTrades = s.assets.reduce((sum, asset) => sum + asset.search.trades, 0);
      expect(s.sample?.trades).toBe(pooledTrades);
      expect(pooledTrades).toBeGreaterThanOrEqual(ready.gates.search);
      // ... so a strategy no single walk could have qualified on its own can still be judged.
      if (s.search.trades < ready.gates.search) belowSingleAssetGate++;
      expect(s.dsr).toBeLessThan(0.95);
    }
    expect(belowSingleAssetGate).toBeGreaterThan(0);
    const top = lockTop(done);
    expect(top?.verdict).not.toBe('Robust');
    expect(top?.sample?.holdoutTrades).toBe(top?.assets.reduce((sum, asset) => sum + (asset.holdout?.trades ?? 0), 0));
  }, 900_000);

  it('does not certify noise on a trending asset, where most long strategies profit from drift alone', async () => {
    const rows = await nullBoards(AUDIT_DRIFT_PER_BAR, 600);
    expect(rows.length).toBeGreaterThan(20);
    expect(rows.filter((r) => r.robustEligible)).toHaveLength(0);
    // Audit baseline: 64% of these boards passed the old holdout test (a positive holdout return).
    // Under the current holdout conditions the false-positive rate must be a small fraction of that.
    const falsePositiveRate = rows.filter((r) => r.holdoutPass).length / rows.length;
    expect(falsePositiveRate).toBeLessThanOrEqual(0.1);
    expect(falsePositiveRate).toBeLessThan(0.64 / 5);
  }, 900_000);
});
