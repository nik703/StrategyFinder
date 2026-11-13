import { beforeAll, describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import {
  LEADERBOARD_SIZE,
  MAX_STAGNATION,
  PROGRESS_EVERY,
  type BacktestSettings,
  type OHLCV,
  type WorkerRequest,
  type WorkerResponse,
} from '../src/types/trading';

/** Minimal stand-in for the dedicated worker global scope. */
interface FakeScope {
  onmessage: ((event: { data: WorkerRequest }) => void) | null;
  postMessage: (message: WorkerResponse) => void;
}

const scope: FakeScope = { onmessage: null, postMessage: () => undefined };

function dataset(symbol: string, interval: '1h' | '4h' | '1d', range: '1y' | '2y' | '5y'): OHLCV {
  const series = generateSyntheticCandles(symbol, interval, range, Date.UTC(2026, 0, 15));
  return buildDataset({ [symbol]: series }, [{ symbol, weight: 1 }]).data;
}

function runOptimizer(
  data: OHLCV,
  settings: BacktestSettings,
  options: { seed: number; stopAfterIterations?: number },
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
    scope.onmessage?.({ data: { type: 'start', payload: { data, settings, seed: options.seed } } });
  });
}

beforeAll(async () => {
  (globalThis as unknown as { self: FakeScope }).self = scope;
  await import('../src/workers/optimizer.worker');
});

describe('optimizer worker', () => {
  it('streams progress every 25 iterations and converges after 2,000 stagnant iterations', async () => {
    const settings: BacktestSettings = { initialCapital: 10_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-short' };
    const messages = await runOptimizer(dataset('BTC-USD', '1d', '5y'), settings, { seed: 12345 });

    expect(messages[0].type).toBe('ready');
    const ready = messages[0] as Extract<WorkerResponse, { type: 'ready' }>;
    expect(ready.axis.bars).toBeGreaterThan(1800);
    expect(ready.axis.splitIndex).toBe(Math.floor(ready.axis.bars * 0.7));
    expect(ready.axis.barIndex[ready.axis.splitPos]).toBe(ready.axis.splitIndex);
    expect(ready.benchmark.equity.length).toBe(ready.axis.barIndex.length);
    expect(ready.benchmark.inSample.netPnl + ready.benchmark.outOfSample.netPnl).toBeCloseTo(ready.benchmark.full.netPnl, 4);

    const progress = messages.filter((m): m is Extract<WorkerResponse, { type: 'progress' }> => m.type === 'progress');
    expect(progress.length).toBeGreaterThan(80);
    progress.slice(0, -1).forEach((p, index) => expect(p.iteration).toBe((index + 1) * PROGRESS_EVERY));
    for (const p of progress) expect(p.stagnation).toBeLessThanOrEqual(MAX_STAGNATION);

    const done = messages[messages.length - 1] as Extract<WorkerResponse, { type: 'done' }>;
    expect(done.type).toBe('done');
    expect(done.reason).toBe('converged');
    expect(done.stagnation).toBe(MAX_STAGNATION);
    expect(done.iteration).toBeGreaterThanOrEqual(MAX_STAGNATION);

    const boards = messages.filter((m): m is Extract<WorkerResponse, { type: 'leaderboard' }> => m.type === 'leaderboard');
    const final = boards[boards.length - 1].strategies;
    expect(final.length).toBeGreaterThan(5);
    expect(final.length).toBeLessThanOrEqual(LEADERBOARD_SIZE);
    for (let i = 1; i < final.length; i++) expect(final[i - 1].fitness).toBeGreaterThanOrEqual(final[i].fitness);
    expect(new Set(final.map((s) => s.id)).size).toBe(final.length);
    for (const s of final) {
      expect(s.equity.length).toBe(ready.axis.barIndex.length);
      expect(s.full.trades).toBeGreaterThan(0);
      expect(s.alpha).toBeCloseTo(s.full.roi - ready.benchmark.full.roi, 6);
      expect(s.inSample.netPnl + s.outOfSample.netPnl).toBeCloseTo(s.full.netPnl, 4);
      expect(s.genome.shortEntry).not.toBeNull();
    }
    expect(done.bestFitness).toBeCloseTo(final[0].fitness, 9);
    // The champion's fitness never decreases over the run.
    let best = -Infinity;
    for (const p of progress) {
      if (p.bestFitness === null) continue;
      expect(p.bestFitness).toBeGreaterThanOrEqual(best - 1e-9);
      best = p.bestFitness;
    }

    const top = final[0];
    console.log(
      `converged after ${done.iteration} iterations in ${(done.elapsedMs / 1000).toFixed(1)}s | ` +
        `best fitness ${top.fitness.toFixed(2)} ROI ${top.full.roi.toFixed(0)}% (B&H ${ready.benchmark.full.roi.toFixed(0)}%) ` +
        `OOS ROI ${top.outOfSample.roi.toFixed(0)}% (B&H ${ready.benchmark.outOfSample.roi.toFixed(0)}%) ` +
        `DD ${top.full.maxDrawdown.toFixed(0)}% Sharpe ${top.full.sharpe.toFixed(2)} trades ${top.full.trades}`,
    );
  }, 120_000);

  it('stops promptly when the user presses Stop, and never shorts in Long-Only mode', async () => {
    const settings: BacktestSettings = { initialCapital: 25_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-only' };
    const messages = await runOptimizer(dataset('NVDA', '1h', '1y'), settings, { seed: 7, stopAfterIterations: 300 });
    const done = messages[messages.length - 1] as Extract<WorkerResponse, { type: 'done' }>;
    expect(done.type).toBe('done');
    expect(done.reason).toBe('stopped');
    expect(done.iteration).toBeGreaterThanOrEqual(300);
    expect(done.iteration).toBeLessThan(300 + 4 * PROGRESS_EVERY);
    expect(done.stagnation).toBeLessThan(MAX_STAGNATION);
    const boards = messages.filter((m): m is Extract<WorkerResponse, { type: 'leaderboard' }> => m.type === 'leaderboard');
    const final = boards[boards.length - 1].strategies;
    expect(final.length).toBeGreaterThan(0);
    for (const s of final) {
      expect(s.genome.shortEntry).toBeNull();
      if (s.genome.exit.mode === 'dynamic') expect(s.genome.exit.shortExit).toBeNull();
    }
  }, 120_000);

  it('reports an error instead of crashing on unusable data', async () => {
    const tiny = dataset('SPY', '1d', '1y');
    const cut: OHLCV = { t: tiny.t.slice(0, 50), o: tiny.o.slice(0, 50), h: tiny.h.slice(0, 50), l: tiny.l.slice(0, 50), c: tiny.c.slice(0, 50), v: tiny.v.slice(0, 50) };
    const settings: BacktestSettings = { initialCapital: 10_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-only' };
    const messages = await runOptimizer(cut, settings, { seed: 1 });
    expect(messages).toHaveLength(1);
    expect(messages[0].type).toBe('error');
  });
});
