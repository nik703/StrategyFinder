/**
 * Monte Carlo execution stress test.
 *
 * A backtest is ONE path through the trades a strategy took. Institutional platforms (StrategyQuant X
 * "Monte Carlo robustness", AmiBroker's Monte Carlo, VectorBT's resampling helpers) replay the same
 * trades many times under small execution perturbations to see how much of the result is the luck of
 * that one path. Each simulated path here:
 *
 *   1. skips every trade with probability `skipRate` (10% – missed fills, downtime, a signal you
 *      did not take),
 *   2. shuffles the order of the remaining trades (the same trades in another order: drawdowns
 *      depend on the order, the total does not),
 *   3. charges every fill an extra adverse slippage shock drawn from an exponential distribution
 *      whose mean is the run's whole per-side cost (fee + slippage floor, 5 bp at least) – on
 *      average the trading costs double – with a 5% chance of a 5× spike (gaps, thin books), on the
 *      trade's notional,
 *
 * and compounds the trades' returns from the starting capital, marking each trade's worst intrabar
 * excursion so the drawdown is the intrabar worst case like the backtest's. The report gives the
 * probability that a path ends with a profit and the 95th-percentile worst-case drawdown.
 *
 * Pure and deterministic: the same trades and seed always give the same report.
 */

import type { BacktestSettings, MonteCarloBands, MonteCarloReport, MonteCarloSummary } from '../types/trading';
import type { BacktestWorkspace } from './backtester';
import { mulberry32 } from './genome';

/** Default number of simulated paths (well above the usual 200 minimum). */
export const MC_SIMULATIONS = 500;
/** Share of trades each path skips at random. */
export const MC_SKIP_RATE = 0.1;
/** Smallest mean slippage shock per fill, as a fraction (5 basis points). */
export const MC_MIN_SHOCK = 0.0005;
/** Probability that a fill's shock is a spike, and the spike's size in mean shocks. */
export const MC_SPIKE_RATE = 0.05;
export const MC_SPIKE_SIZE = 5;
/** Pass criteria: probability of profit and 95th-percentile drawdown. */
export const MC_MIN_PROB_PROFIT = 0.8;
export const MC_MAX_DRAWDOWN_P95 = 50;
/** Points per percentile band of the equity-path chart. */
const BAND_POINTS = 60;

/** One trade as the Monte Carlo test sees it: everything relative to the equity at its entry. */
export interface McTrade {
  /** Net return of the trade on the equity at entry (after every modelled cost). */
  ret: number;
  /** Worst intrabar excursion while the trade was open, on the same base (≤ 0). */
  mae: number;
  /** Notional at entry as a share of that equity (0–1). */
  exposure: number;
}

/** The trades a backtest opened before bar `bars`, in the form the stress test needs. */
export function tradesFrom(ws: BacktestWorkspace, bars: number = ws.bars): McTrade[] {
  const out: McTrade[] = [];
  for (let k = 0; k < ws.tradeCount; k++) {
    if (ws.tradeEntryBar[k] >= bars) break;
    const base = ws.tradeEquity[k];
    if (!(base > 0)) continue;
    out.push({ ret: ws.tradePnl[k] / base, mae: Math.min(0, ws.tradeMae[k]), exposure: ws.tradeExposure[k] });
  }
  return out;
}

export interface MonteCarloOptions {
  simulations?: number;
  skipRate?: number;
  seed?: number;
  /** Mean slippage shock per fill, as a fraction (default: the run's per-side cost, at least 5 bp; 0 = no shocks). */
  meanShock?: number;
}

/** Linear-interpolation percentile of an ascending array. */
function percentile(sorted: Float64Array | number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = (n - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Max drawdown (fraction) of a trade sequence replayed in the given order, intrabar excursions included. */
function pathDrawdown(trades: readonly McTrade[], order: ArrayLike<number>, count: number, costs: ArrayLike<number> | null): { dd: number; growth: number } {
  let equity = 1;
  let peak = 1;
  let maxDd = 0;
  for (let j = 0; j < count; j++) {
    const t = trades[order[j]];
    const low = equity * (1 + t.mae);
    const ddLow = 1 - low / peak;
    if (ddLow > maxDd) maxDd = ddLow;
    equity *= Math.max(0, 1 + t.ret - (costs ? costs[j] : 0));
    const dd = 1 - equity / peak;
    if (dd > maxDd) maxDd = dd;
    if (equity > peak) peak = equity;
    if (equity <= 0) {
      maxDd = 1;
      break;
    }
  }
  return { dd: Math.min(1, maxDd), growth: equity };
}

/** Run the stress test. Returns null without trades. */
export function monteCarloStress(
  trades: readonly McTrade[],
  settings: Pick<BacktestSettings, 'slippagePct' | 'feePct'>,
  options: MonteCarloOptions = {},
): MonteCarloReport | null {
  const n = trades.length;
  if (n === 0) return null;
  const simulations = Math.max(1, Math.round(options.simulations ?? MC_SIMULATIONS));
  const skipRate = options.skipRate ?? MC_SKIP_RATE;
  const rng = mulberry32((options.seed ?? 0x5eed) >>> 0);
  // On average the shocks double the modelled cost of every fill: a thin edge that only just pays
  // its costs does not survive, a real one does.
  const meanShock = options.meanShock ?? Math.max(MC_MIN_SHOCK, (settings.feePct + settings.slippagePct) / 100);

  // The original sequence, measured the same way, for comparison.
  const identity = Int32Array.from({ length: n }, (_, k) => k);
  const base = pathDrawdown(trades, identity, n, null);

  const finals = new Float64Array(simulations);
  const drawdowns = new Float64Array(simulations);
  const xs: number[] = [];
  for (let p = 0; p <= BAND_POINTS; p++) {
    const x = Math.round((p * n) / BAND_POINTS);
    if (xs.length === 0 || xs[xs.length - 1] !== x) xs.push(x);
  }
  /** Equity of every path at each band point (a skipped path holds its last value). */
  const atPoints = xs.map(() => new Float64Array(simulations));

  const order = new Int32Array(n);
  const costs = new Float64Array(n);
  for (let sim = 0; sim < simulations; sim++) {
    // 1. Skip ~10% of the trades.
    let count = 0;
    for (let k = 0; k < n; k++) if (rng() >= skipRate) order[count++] = k;
    // 2. Shuffle the survivors (Fisher–Yates).
    for (let k = count - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      const tmp = order[k];
      order[k] = order[j];
      order[j] = tmp;
    }
    // 3. Slippage shocks on the entry and the exit fill of every trade, charged on its notional.
    for (let j = 0; j < count; j++) {
      let shock = 0;
      for (let fill = 0; fill < 2; fill++) {
        const u = rng();
        let s = -Math.log(1 - u) * meanShock; // exponential, mean = meanShock
        if (rng() < MC_SPIKE_RATE) s *= MC_SPIKE_SIZE;
        shock += s;
      }
      costs[j] = trades[order[j]].exposure * shock;
    }
    // Replay, recording the equity at the band points.
    let equity = 1;
    let peak = 1;
    let maxDd = 0;
    let point = 0;
    while (point < xs.length && xs[point] === 0) atPoints[point++][sim] = 1;
    for (let j = 0; j < count; j++) {
      const t = trades[order[j]];
      const ddLow = 1 - (equity * (1 + t.mae)) / peak;
      if (ddLow > maxDd) maxDd = ddLow;
      equity *= Math.max(0, 1 + t.ret - costs[j]);
      const dd = 1 - equity / peak;
      if (dd > maxDd) maxDd = dd;
      if (equity > peak) peak = equity;
      // Trade j + 1 of the path maps onto the original trade axis by its share of the path.
      const progress = count > 0 ? ((j + 1) / count) * n : n;
      while (point < xs.length && xs[point] <= progress + 1e-9) atPoints[point++][sim] = equity;
      if (equity <= 0) {
        maxDd = 1;
        break;
      }
    }
    while (point < xs.length) atPoints[point++][sim] = equity;
    finals[sim] = equity;
    drawdowns[sim] = Math.min(1, maxDd);
  }

  const sortedFinals = Float64Array.from(finals).sort();
  const sortedDd = Float64Array.from(drawdowns).sort();
  let profitable = 0;
  for (let sim = 0; sim < simulations; sim++) if (finals[sim] > 1) profitable++;
  const probProfit = profitable / simulations;
  const drawdownP95 = percentile(sortedDd, 0.95) * 100;

  const bands: MonteCarloBands = { x: xs, p5: [], p25: [], p50: [], p75: [], p95: [] };
  for (const values of atPoints) {
    const sorted = Float64Array.from(values).sort();
    bands.p5.push(percentile(sorted, 0.05));
    bands.p25.push(percentile(sorted, 0.25));
    bands.p50.push(percentile(sorted, 0.5));
    bands.p75.push(percentile(sorted, 0.75));
    bands.p95.push(percentile(sorted, 0.95));
  }

  return {
    simulations,
    trades: n,
    probProfit,
    drawdownP95,
    medianReturn: (percentile(sortedFinals, 0.5) - 1) * 100,
    p5Return: (percentile(sortedFinals, 0.05) - 1) * 100,
    baseDrawdown: base.dd * 100,
    passed: probProfit >= MC_MIN_PROB_PROFIT && drawdownP95 < MC_MAX_DRAWDOWN_P95,
    bands,
  };
}

/** The headline numbers of a report, without the chart bands (small enough for every leaderboard row). */
export function summarise(report: MonteCarloReport | null): MonteCarloSummary | null {
  if (report === null) return null;
  const { bands: _bands, ...summary } = report;
  return summary;
}
