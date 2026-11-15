import { beforeAll, describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  BacktestWorkspace,
  barsPerYear,
  evaluateSearch,
  MIN_TRADES,
  MIN_VALIDATION_TRADES,
  runBacktest,
  splitFor,
} from '../src/engine/backtester';
import { attachEvaluator } from '../src/engine/evaluation';
import { IndicatorCache } from '../src/engine/indicators';
import { countParameters, gaussian, isCoherent, MAX_PARAMETERS, mirrorGroup, mulberry32 } from '../src/engine/genome';
import {
  DEFAULT_MAX_STAGNATION,
  DEFAULT_SIZING,
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
  options: { seed: number; stopAfterIterations?: number; maxStagnation?: number; ports?: MessagePort[]; batchSize?: number },
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
          maxStagnation: options.maxStagnation,
          ports: options.ports,
          batchSize: options.batchSize,
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
    expect(s.genome.longEntry.conditions.length).toBeLessThanOrEqual(2);
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
    expect(progress.length).toBeGreaterThan(200);
    progress.slice(0, -1).forEach((p, index) => expect(p.iteration).toBe((index + 1) * PROGRESS_EVERY));
    for (const p of progress) expect(p.stagnation).toBeLessThanOrEqual(DEFAULT_MAX_STAGNATION);

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

    // Converged at the default limit (5,000). The holdout is STILL sealed: nothing is revealed for the board.
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(done.reason).toBe('converged');
    expect(DEFAULT_MAX_STAGNATION).toBe(5000);
    expect(done.stagnation).toBe(DEFAULT_MAX_STAGNATION);
    expect(done.benchmark.holdout).toBeNull();
    expect(done.benchmark.equity.length).toBe(axis.holdoutPos);

    const final = done.strategies;
    expect(final.length).toBeGreaterThan(5);
    expect(final.length).toBeLessThanOrEqual(LEADERBOARD_SIZE);
    for (let i = 1; i < final.length; i++) expect(final[i - 1].fitness).toBeGreaterThanOrEqual(final[i].fitness);
    expect(new Set(final.map((s) => s.id)).size).toBe(final.length);
    assertConstraints(final, 'long-short');
    for (const s of final) {
      expect(s.holdout).toBeNull();
      expect(s.verdict).toBeNull();
      expect(s.equity.length).toBe(axis.holdoutPos);
      for (const asset of s.assets) expect(asset.holdout).toBeNull();
      // Hard sample-size gates: nothing below them can be listed.
      expect(s.search.trades).toBeGreaterThanOrEqual(MIN_TRADES);
      expect(s.validation.trades).toBeGreaterThanOrEqual(MIN_VALIDATION_TRADES);
      expect(s.alpha).toBeCloseTo(s.search.roi - done.benchmark.search.roi, 6);
      expect(s.train.netPnl + s.validation.netPnl).toBeCloseTo(s.search.netPnl, 4);
      expect(s.dsr).toBeGreaterThanOrEqual(0);
      expect(s.dsr).toBeLessThanOrEqual(1);
      expect(s.assets).toHaveLength(1);
    }
    expect(done.bestFitness).toBeCloseTo(final[0].fitness, 9);
    expect(done.bestDsr).toBeCloseTo(Math.max(...final.map((s) => s.dsr)), 9);
    expect(done.hurdleSharpe).toBeGreaterThan(0);

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
    expect(done.stagnation).toBeLessThan(DEFAULT_MAX_STAGNATION);
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
        data: { type: 'start', payload: { datasets: [dataset('NVDA', '1h', '1y')], settings: settingsFor('long-only'), seed: 8, maxStagnation: 300 } },
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

  it('honours a custom stagnation limit', async () => {
    const messages = await runOptimizer([dataset('SPY', '1d', '2y')], settingsFor('long-only'), { seed: 3, maxStagnation: 300 });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.reason).toBe('converged');
    expect(done.stagnation).toBe(300);
  }, 120_000);

  it('scores a Robust universe on every asset and answers analyze requests', async () => {
    // 4h candles: two years of daily candles cannot supply the minimum number of trades on every asset.
    const universe = ['BTC-USD', 'ETH-USD', 'SOL-USD'].map((symbol) => dataset(symbol, '4h', '2y'));
    const settings = settingsFor('long-only');
    const messages = await runOptimizer(universe, settings, { seed: 21, maxStagnation: 400 });
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
    const messages = await runOptimizer([eth], settings, { seed: 99, maxStagnation: 600 });
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
    const options = { seed: 314, maxStagnation: 500, batchSize: 75 };

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
      const messages = await runOptimizer([dataset(symbol, interval, range)], settingsFor('long-only'), { seed: 5, maxStagnation: 300 });
      const done = messages[messages.length - 1] as Of<'done'>;
      expect(done.type, `${symbol} ${interval}`).toBe('done');
      expect(done.reason).toBe('converged');
      for (const s of done.strategies) expect(s.holdout).toBeNull();
      const ready = messages[0] as Of<'ready'>;
      if (interval === '30m') {
        // 30m crypto over a year: 17,520 candles – plenty of trades to clear the gates.
        expect(ready.axis.bars).toBeGreaterThan(17_000);
        expect(done.strategies.length).toBeGreaterThan(0);
      } else {
        // 15m stock: Yahoo's 60-day window. About 220 validation candles cannot hold 60 trades, so the
        // hard sample-size gates reject every candidate and the run ends cleanly with an empty board.
        expect(ready.axis.bars).toBeLessThan(1_200);
        expect(done.strategies).toHaveLength(0);
        expect(done.undersampled).toBeGreaterThan(0);
        expect(done.bestDsr).toBeNull();
        expect(lock('S-1').type).toBe('error');
      }
    }
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

describe('acceptance: pure noise must not look like an edge', () => {
  /** Run on noise, lock the #1 strategy (the user's natural pick) and return what the reveal says. */
  async function lockTopOnNoise(data: NamedDataset, seed: number): Promise<{ done: Of<'done'>; top: StrategyResult | null }> {
    // Defaults: 2% risk sizing, 0.1% fee, 0.05% slippage, 5,000-iteration stagnation limit.
    const messages = await runOptimizer([data], settingsFor('long-only'), { seed });
    const done = messages[messages.length - 1] as Of<'done'>;
    expect(done.type).toBe('done');
    expect(done.reason).toBe('converged');
    assertConstraints(done.strategies, 'long-only');
    for (const s of done.strategies) {
      expect(s.holdout).toBeNull();
      expect(s.validation.trades).toBeGreaterThanOrEqual(MIN_VALIDATION_TRADES);
    }
    if (done.strategies.length === 0) return { done, top: null }; // nothing cleared the gates: nothing certified
    const reply = lock(done.strategies[0].id) as Of<'locked'>;
    expect(reply.type).toBe('locked');
    return { done, top: reply.strategy };
  }

  it('flags the best strategy found on driftless random walks as unproven', async () => {
    const holdoutSharpes: number[] = [];
    const lines: string[] = [];
    for (let seed = 0; seed < 6; seed++) {
      const { done, top } = await lockTopOnNoise(randomWalk(3650, 1000 + seed), 77 + seed);
      if (top === null) {
        lines.push(`seed ${seed}: ${done.iteration} iterations | no strategy cleared the trade gates`);
        continue;
      }
      // The deflated Sharpe must never certify noise as real, and neither may the verdict.
      expect(top.dsr).toBeLessThan(0.95);
      expect(top.verdict).not.toBe('Robust');
      holdoutSharpes.push(top.holdout?.sharpe ?? 0);
      lines.push(
        `seed ${seed}: ${done.iteration} iterations | #1 validation Sharpe ${top.validation.sharpe.toFixed(2)} (${top.validation.trades} trades) ` +
          `DSR ${(top.dsr * 100).toFixed(1)}% → holdout Sharpe ${top.holdout?.sharpe.toFixed(2)} (${top.verdict})`,
      );
    }
    const mean = holdoutSharpes.length > 0 ? holdoutSharpes.reduce((a, b) => a + b, 0) / holdoutSharpes.length : 0;
    console.log(`${lines.join('\n')}\nmean holdout Sharpe of the #1 strategies: ${mean.toFixed(2)}`);
    expect(mean).toBeLessThanOrEqual(0.3);
  }, 900_000);

  it('does not certify noise on a trending asset, where any long strategy makes money on the holdout', async () => {
    const lines: string[] = [];
    for (let seed = 0; seed < 4; seed++) {
      // +40% a year of pure drift: no timing skill exists, but long exposure profits.
      const { done, top } = await lockTopOnNoise(randomWalk(3650, 2000 + seed, 0.02, 0.4 / 365), 177 + seed);
      if (top === null) {
        lines.push(`seed ${seed}: ${done.iteration} iterations | no strategy cleared the trade gates`);
        continue;
      }
      expect(top.verdict).not.toBe('Robust');
      lines.push(
        `seed ${seed}: #1 DSR ${(top.dsr * 100).toFixed(1)}% → holdout ROI ${top.holdout?.roi.toFixed(1)}%, Sharpe ${top.holdout?.sharpe.toFixed(2)} ` +
          `(${top.verdict}: ${(top.verdictNotes ?? []).length} unmet conditions)`,
      );
    }
    console.log(lines.join('\n'));
  }, 900_000);
});
