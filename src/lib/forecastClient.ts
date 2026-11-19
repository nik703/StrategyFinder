/**
 * Client for the forecast worker. One worker is shared by the whole page and
 * created on first use; requests are matched to replies by id.
 */

import type { NextEntryForecast } from '../engine/nextEntrySolver';
import type { BacktestSettings, OHLCV, StrategyGenome } from '../types/trading';

export interface ForecastJob {
  key: string;
  data: OHLCV;
  genome: StrategyGenome;
  settings: BacktestSettings;
}

export interface ForecastRequest {
  id: number;
  jobs: ForecastJob[];
}

export interface ForecastResult {
  key: string;
  forecast: NextEntryForecast | null;
  error: string | null;
}

export interface ForecastReply {
  id: number;
  results: ForecastResult[];
}

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (results: ForecastResult[]) => void; reject: (error: Error) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  const created = new Worker(new URL('../workers/forecast.worker.ts', import.meta.url), { type: 'module' });
  created.onmessage = (event: MessageEvent<ForecastReply>) => {
    const waiting = pending.get(event.data.id);
    if (!waiting) return;
    pending.delete(event.data.id);
    waiting.resolve(event.data.results);
  };
  created.onerror = (event) => {
    for (const waiting of pending.values()) waiting.reject(new Error(event.message || 'The forecast worker crashed.'));
    pending.clear();
    created.terminate();
    if (worker === created) worker = null;
  };
  worker = created;
  return created;
}

/** Solve the next entry for several assets; resolves in the order given. */
export function requestForecasts(jobs: ForecastJob[]): Promise<ForecastResult[]> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const request: ForecastRequest = { id, jobs };
    getWorker().postMessage(request);
  });
}

/** Terminate the shared forecast worker (part of "Reset Runs & Clear Cache"); the next request starts a fresh one. */
export function resetForecastWorker(): void {
  for (const waiting of pending.values()) waiting.reject(new Error('Forecast cancelled by a reset.'));
  pending.clear();
  worker?.terminate();
  worker = null;
}
