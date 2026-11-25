/**
 * Worker pool: sizing, concurrent dispatch with ordered reassembly, and a clean shutdown that leaves
 * no worker running and no request pending ("Reset Runs & Clear Cache", Stop, a new run).
 */

import { describe, expect, it } from 'vitest';
import type { CandidateResult, EvaluatorBatch, EvaluatorInit, PortLike } from '../src/engine/evaluation';
import { BatchDispatcher, evaluatorBudgetBytes, MAX_POOL_SIZE, POOL_CLOSED, poolSize, WorkerPool, type ChannelLike, type WorkerLike } from '../src/workers/workerPool';
import type { StrategyGenome } from '../src/types/trading';

/** A fake evaluator port: records batches and answers them when told to. */
class FakePort implements PortLike {
  onmessage: ((event: MessageEvent) => void) | null = null;
  received: EvaluatorBatch[] = [];
  closed = false;
  postMessage(message: unknown): void {
    this.received.push(message as EvaluatorBatch);
  }
  close(): void {
    this.closed = true;
  }
  /** Answer the oldest unanswered batch with one result per genome (tagged with this port's name). */
  answer(tag: string): void {
    const batch = this.received.shift();
    if (!batch) throw new Error('nothing to answer');
    const results = batch.genomes.map((g) => ({ tag, genome: g }) as unknown as CandidateResult);
    this.onmessage?.({ data: { type: 'results', id: batch.id, results } } as MessageEvent);
  }
  fail(message: string): void {
    const batch = this.received.shift() as EvaluatorBatch;
    this.onmessage?.({ data: { type: 'error', id: batch.id, message } } as MessageEvent);
  }
}

const genomes = (n: number): StrategyGenome[] => Array.from({ length: n }, (_, i) => ({ id: i }) as unknown as StrategyGenome);

describe('pool sizing', () => {
  it('uses every core but one, at least one and at most eight', () => {
    expect(MAX_POOL_SIZE).toBe(8);
    expect(poolSize(1)).toBe(1);
    expect(poolSize(2)).toBe(1);
    expect(poolSize(4)).toBe(3);
    expect(poolSize(8)).toBe(7);
    expect(poolSize(9)).toBe(8);
    expect(poolSize(64)).toBe(8);
    // Unknown core count: treated as 4.
    expect(poolSize(undefined)).toBe(3);
    expect(poolSize(0)).toBe(3);
  });

  it('shares the indicator-cache budget between the evaluators', () => {
    expect(evaluatorBudgetBytes(1)).toBe(512 * 1024 * 1024);
    expect(evaluatorBudgetBytes(4)).toBe(128 * 1024 * 1024);
    expect(evaluatorBudgetBytes(8)).toBe(64 * 1024 * 1024);
    expect(evaluatorBudgetBytes(16)).toBe(64 * 1024 * 1024);
  });
});

describe('BatchDispatcher', () => {
  it('sends one chunk to every evaluator at once and reassembles the results in proposal order', async () => {
    const ports = [new FakePort(), new FakePort(), new FakePort()];
    const dispatcher = new BatchDispatcher(ports);
    const batch = genomes(10);
    const pending = dispatcher.evaluate(batch, 0.5, -1, ['h1']);
    // Concurrency: all three chunks are out before any is answered.
    expect(ports.map((p) => p.received.length)).toEqual([1, 1, 1]);
    expect(ports.map((p) => p.received[0].genomes.length)).toEqual([4, 4, 2]);
    expect(dispatcher.queued).toBe(3);
    for (const p of ports) expect(p.received[0]).toMatchObject({ type: 'batch', hurdle: 0.5, floor: -1, knownTrades: ['h1'] });
    // Answered out of order ...
    ports[2].answer('c');
    ports[0].answer('a');
    ports[1].answer('b');
    const results = (await pending) as unknown as { tag: string; genome: { id: number } }[];
    // ... reassembled in order.
    expect(results.map((r) => r.genome.id)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(results.map((r) => r.tag).join('')).toBe('aaaabbbbcc');
    expect(dispatcher.queued).toBe(0);
  });

  it('rejects a batch when an evaluator reports an error', async () => {
    const ports = [new FakePort(), new FakePort()];
    const dispatcher = new BatchDispatcher(ports);
    const pending = dispatcher.evaluate(genomes(4), 0, 0, []);
    ports[0].answer('a');
    ports[1].fail('out of memory');
    await expect(pending).rejects.toThrow('out of memory');
  });

  it('close() rejects every pending request, clears the queue, detaches and closes the ports', async () => {
    const ports = [new FakePort(), new FakePort()];
    const dispatcher = new BatchDispatcher(ports);
    const first = dispatcher.evaluate(genomes(6), 0, 0, []);
    expect(dispatcher.queued).toBe(2);
    dispatcher.close();
    await expect(first).rejects.toThrow(POOL_CLOSED);
    expect(dispatcher.queued).toBe(0);
    expect(dispatcher.size).toBe(0);
    for (const p of ports) {
      expect(p.closed).toBe(true);
      expect(p.onmessage).toBeNull();
    }
    // A late reply from a terminated evaluator is ignored, and new work is refused.
    await expect(dispatcher.evaluate(genomes(2), 0, 0, [])).rejects.toThrow(POOL_CLOSED);
  });
});

describe('WorkerPool', () => {
  class FakeWorker implements WorkerLike {
    static all: FakeWorker[] = [];
    onerror: ((event: ErrorEvent) => void) | null = null;
    messages: unknown[] = [];
    terminated = false;
    constructor() {
      FakeWorker.all.push(this);
    }
    postMessage(message: unknown): void {
      this.messages.push(message);
    }
    terminate(): void {
      this.terminated = true;
    }
  }
  const fakeChannel = (): ChannelLike => {
    const port = () => ({ close() {}, postMessage() {}, onmessage: null }) as unknown as MessagePort;
    return { port1: port(), port2: port() };
  };
  const init = { datasets: [], settings: {}, budgetBytes: 1 } as unknown as EvaluatorInit;

  it('starts `size` evaluators, each connected to its own channel', () => {
    FakeWorker.all = [];
    const pool = new WorkerPool({ size: 3, createWorker: () => new FakeWorker(), createChannel: fakeChannel });
    const ports = pool.start(init);
    expect(ports).toHaveLength(3);
    expect(pool.alive).toBe(3);
    for (const w of FakeWorker.all) expect(w.messages[0]).toMatchObject({ type: 'connect', budgetBytes: 1 });
  });

  it('terminates every evaluator on terminate() and before a new run starts', () => {
    FakeWorker.all = [];
    const pool = new WorkerPool({ size: 4, createWorker: () => new FakeWorker(), createChannel: fakeChannel });
    pool.start(init);
    const firstRun = [...FakeWorker.all];
    pool.start(init); // a new run: the previous evaluators are gone first
    expect(firstRun.every((w) => w.terminated)).toBe(true);
    expect(pool.alive).toBe(4);
    pool.terminate();
    expect(FakeWorker.all.every((w) => w.terminated)).toBe(true);
    expect(pool.alive).toBe(0);
    pool.terminate(); // idempotent
    expect(pool.alive).toBe(0);
  });

  it('releases the whole pool once when an evaluator crashes', () => {
    FakeWorker.all = [];
    let failures = 0;
    const pool = new WorkerPool({ size: 3, createWorker: () => new FakeWorker(), createChannel: fakeChannel, onFailure: () => failures++ });
    pool.start(init);
    const [a, b] = FakeWorker.all;
    a.onerror?.({} as ErrorEvent);
    b.onerror?.({} as ErrorEvent); // handlers were detached: no second report
    expect(failures).toBe(1);
    expect(pool.alive).toBe(0);
    expect(FakeWorker.all.every((w) => w.terminated)).toBe(true);
  });
});
