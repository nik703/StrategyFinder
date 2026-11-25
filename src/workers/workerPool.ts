/**
 * Multi-core worker pool.
 *
 * Two halves, one per side of the MessageChannels that connect them:
 *
 *   • `WorkerPool` (main thread) starts the evaluator workers – one per spare core, sized by
 *     `poolSize` – hands each its copy of the market data and one end of a MessageChannel, and
 *     terminates them all at once (Stop, a new run, "Reset Runs & Clear Cache").
 *   • `BatchDispatcher` (optimiser worker) holds the other ends. It splits every population batch
 *     into one contiguous chunk per evaluator, sends all chunks at once so the evaluators score
 *     them concurrently, and reassembles the results in proposal order – so a seeded search gives
 *     the same answer whatever the pool size. `close()` rejects and drops every pending request.
 *
 * The evaluators run the FIRST evaluation tier only (single-pass multi-fold scoring – see
 * `evaluateCandidate`); the second tier (Monte Carlo + parameter-neighbourhood grids) runs on the
 * optimiser for the top contenders.
 */

import type { CandidateResult, EvaluatorBatch, EvaluatorConnect, EvaluatorInit, EvaluatorReply, PortLike } from '../engine/evaluation';
import type { StrategyGenome } from '../types/trading';

/** Most evaluator workers a run starts (each holds its own copy of the data and an indicator cache). */
export const MAX_POOL_SIZE = 8;

/**
 * Evaluator workers for a device with `hardwareConcurrency` logical cores: every core but one (left
 * for the page and the optimiser thread), at least one, at most `MAX_POOL_SIZE`. An unknown core
 * count is treated as 4.
 */
export function poolSize(hardwareConcurrency: number | undefined = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : undefined): number {
  const cores = hardwareConcurrency && hardwareConcurrency > 0 ? hardwareConcurrency : 4;
  return Math.max(1, Math.min(cores - 1, MAX_POOL_SIZE));
}

/** Memory budget of each evaluator's indicator cache: 512 MB shared out, at least 64 MB each. */
export function evaluatorBudgetBytes(size: number): number {
  return Math.max(64, Math.floor(512 / Math.max(1, size))) * 1024 * 1024;
}

/* -------------------------------------------------------------------------- */
/*  Main thread: the workers                                                  */
/* -------------------------------------------------------------------------- */

/** The part of a Worker the pool uses (a browser Worker, or a test double). */
export interface WorkerLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
  onerror: ((event: ErrorEvent) => void) | null;
}

/** The part of a MessageChannel the pool uses. */
export interface ChannelLike {
  port1: MessagePort;
  port2: MessagePort;
}

export interface WorkerPoolOptions {
  size?: number;
  /** Starts one evaluator worker. */
  createWorker: () => WorkerLike;
  /** Creates the channel between an evaluator and the optimiser (default: `new MessageChannel()`). */
  createChannel?: () => ChannelLike;
  /** Called once when any evaluator fails: the optimiser should finish the run single-threaded. */
  onFailure?: () => void;
}

export class WorkerPool {
  private workers: WorkerLike[] = [];
  private ports: MessagePort[] = [];
  private failed = false;
  readonly size: number;

  constructor(private readonly options: WorkerPoolOptions) {
    this.size = options.size ?? poolSize();
  }

  /** Evaluator workers currently running. */
  get alive(): number {
    return this.workers.length;
  }

  /**
   * Start `size` evaluators on this run's data. Returns the optimiser's ends of their channels, to be
   * transferred to the optimiser worker with the `start` request.
   */
  start(init: EvaluatorInit): MessagePort[] {
    this.terminate();
    this.failed = false;
    const ports: MessagePort[] = [];
    for (let w = 0; w < this.size; w++) {
      const worker = this.options.createWorker();
      const channel = this.options.createChannel?.() ?? new MessageChannel();
      worker.onerror = () => {
        // A dead evaluator would never answer: release the pool once and let the optimiser carry on alone.
        if (this.failed) return;
        this.failed = true;
        this.terminate();
        this.options.onFailure?.();
      };
      const connect: EvaluatorConnect = { type: 'connect', port: channel.port2, ...init };
      worker.postMessage(connect, [channel.port2]);
      this.workers.push(worker);
      ports.push(channel.port1);
    }
    this.ports = ports;
    return ports;
  }

  /** Terminate every evaluator immediately and close the channel ends still held here. */
  terminate(): void {
    for (const worker of this.workers) {
      worker.onerror = null;
      worker.terminate();
    }
    for (const port of this.ports) {
      try {
        port.close();
      } catch {
        // Already transferred to the optimiser: nothing to close on this side.
      }
    }
    this.workers = [];
    this.ports = [];
  }
}

/* -------------------------------------------------------------------------- */
/*  Optimiser worker: the batches                                             */
/* -------------------------------------------------------------------------- */

type Pending = { resolve: (results: CandidateResult[]) => void; reject: (error: Error) => void };

/** Error raised for every request still waiting when the dispatcher closes. */
export const POOL_CLOSED = 'Evaluator pool closed.';

export class BatchDispatcher {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(private ports: PortLike[]) {
    for (const port of ports) {
      port.onmessage = (event: MessageEvent) => {
        const reply = event.data as EvaluatorReply;
        const waiting = this.pending.get(reply.id);
        if (!waiting) return; // a reply to a request that was cancelled by close()
        this.pending.delete(reply.id);
        if (reply.type === 'results') waiting.resolve(reply.results);
        else waiting.reject(new Error(reply.message));
      };
    }
  }

  get size(): number {
    return this.ports.length;
  }

  /** Requests sent and not yet answered (0 after `close()`). */
  get queued(): number {
    return this.pending.size;
  }

  /**
   * Score `genomes` on every evaluator at once: one contiguous chunk each, all sent before any is
   * awaited, results reassembled in the original order.
   */
  async evaluate(genomes: StrategyGenome[], hurdle: number, floor: number, knownTrades: string[]): Promise<CandidateResult[]> {
    if (this.ports.length === 0) throw new Error(POOL_CLOSED);
    const chunk = Math.ceil(genomes.length / this.ports.length);
    const requests: Promise<CandidateResult[]>[] = [];
    // Every evaluator keeps a full replica of the known trade logs, so each gets the delta.
    for (let w = 0; w < this.ports.length; w++) {
      const id = this.nextId++;
      const message: EvaluatorBatch = { type: 'batch', id, hurdle, floor, knownTrades, genomes: genomes.slice(w * chunk, (w + 1) * chunk) };
      requests.push(
        new Promise((resolve, reject) => {
          this.pending.set(id, { resolve, reject });
          this.ports[w].postMessage(message);
        }),
      );
    }
    return (await Promise.all(requests)).flat();
  }

  /** Detach from every port, reject every pending request and clear the queue. */
  close(): void {
    for (const port of this.ports) {
      port.onmessage = null;
      port.close?.();
    }
    this.ports = [];
    const waiting = [...this.pending.values()];
    this.pending.clear();
    for (const request of waiting) request.reject(new Error(POOL_CLOSED));
  }
}
