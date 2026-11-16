/**
 * Forecast worker: runs the next-entry solver off the main thread, so ranking a
 * basket or a universe of assets never freezes the page.
 */

import { solveNextEntry } from '../engine/nextEntrySolver';
import type { ForecastReply, ForecastRequest } from '../lib/forecastClient';

interface ForecastScope {
  onmessage: ((event: MessageEvent<ForecastRequest>) => void) | null;
  postMessage(message: ForecastReply): void;
}

const scope = self as unknown as ForecastScope;

scope.onmessage = (event) => {
  const { id, jobs } = event.data;
  const results = jobs.map((job) => {
    try {
      return { key: job.key, forecast: solveNextEntry({ data: job.data, genome: job.genome, settings: job.settings }), error: null };
    } catch (error) {
      return { key: job.key, forecast: null, error: error instanceof Error ? error.message : String(error) };
    }
  });
  scope.postMessage({ id, results });
};
