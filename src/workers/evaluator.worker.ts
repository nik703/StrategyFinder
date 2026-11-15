/**
 * Evaluator worker – one member of the optimiser's evaluation pool.
 *
 * The main thread creates it, hands it the market data and a MessagePort, and
 * the optimiser worker then sends batches of candidate strategies straight
 * over that port. Nothing goes through the main thread while a run is live.
 */

import { attachEvaluator, type EvaluatorConnect } from '../engine/evaluation';

interface EvaluatorScope {
  onmessage: ((event: MessageEvent<EvaluatorConnect>) => void) | null;
}

const scope = self as unknown as EvaluatorScope;

scope.onmessage = (event) => {
  const { type, port, datasets, settings, budgetBytes } = event.data;
  if (type !== 'connect') return;
  attachEvaluator(port, { datasets, settings, budgetBytes });
};
