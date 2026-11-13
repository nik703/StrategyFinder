/**
 * Selection-bias statistics.
 *
 * An optimiser that evaluates thousands of strategies will find impressive
 * backtests even in pure noise. The Deflated Sharpe Ratio (Bailey & López de
 * Prado, 2014) asks how likely a Sharpe ratio is to be real once the number of
 * trials, the spread of their results, the sample length and the shape of the
 * return distribution are taken into account.
 */

import type { PerformanceMetrics } from '../types/trading';

/** Inverse of the standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9). */
export function inverseNormalCdf(p: number): number {
  if (!(p > 0 && p < 1)) return p <= 0 ? -Infinity : Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  if (p < low) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - low) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
export function normalCdf(x: number): number {
  if (x === Infinity) return 1;
  if (x === -Infinity) return 0;
  const t = 1 / (1 + 0.3275911 * Math.abs(x) * Math.SQRT1_2);
  const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(x * x) / 2);
  return x >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

const EULER_MASCHERONI = 0.5772156649015329;

/**
 * Expected maximum per-bar Sharpe ratio among `trials` skill-less strategies whose
 * Sharpe ratios have variance `trialVariance` – the hurdle a real edge must clear.
 */
export function expectedMaxSharpe(trials: number, trialVariance: number): number {
  if (!(trials > 1) || !(trialVariance > 0)) return 0;
  const z1 = inverseNormalCdf(1 - 1 / trials);
  const z2 = inverseNormalCdf(1 - 1 / (trials * Math.E));
  return Math.sqrt(trialVariance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2);
}

/** Standard error of a per-bar Sharpe ratio (Mertens / Lo), allowing for skewness and fat tails. */
export function sharpeStandardError(sharpeRaw: number, bars: number, skew: number, kurtosis: number): number {
  if (!(bars > 1)) return Infinity;
  const variance = 1 - skew * sharpeRaw + ((kurtosis - 1) / 4) * sharpeRaw * sharpeRaw;
  return Math.sqrt(Math.max(variance, 1e-9) / (bars - 1));
}

export interface DeflatedSharpe {
  /** z-score of the observed Sharpe against the selection-bias hurdle. */
  z: number;
  /** Deflated Sharpe Ratio: probability that the true Sharpe exceeds the hurdle (0–1). */
  probability: number;
  /** The hurdle itself (per-bar Sharpe expected from the luckiest of the trials). */
  hurdle: number;
}

/**
 * Deflated Sharpe Ratio of a segment.
 * @param m             Metrics of the segment (uses its per-bar Sharpe, bars, skewness and kurtosis).
 * @param trials        Number of strategies evaluated so far.
 * @param trialVariance Variance of the per-bar Sharpe ratio across those trials.
 */
export function deflatedSharpe(
  m: Pick<PerformanceMetrics, 'sharpeRaw' | 'bars' | 'skew' | 'kurtosis'>,
  trials: number,
  trialVariance: number,
): DeflatedSharpe {
  const hurdle = expectedMaxSharpe(trials, trialVariance);
  const se = sharpeStandardError(m.sharpeRaw, m.bars, m.skew, m.kurtosis);
  if (!Number.isFinite(se) || se <= 0) return { z: -4, probability: 0, hurdle };
  const z = (m.sharpeRaw - hurdle) / se;
  return { z, probability: normalCdf(z), hurdle };
}

/** 95% confidence interval for an annualised Sharpe ratio. */
export function sharpeConfidenceInterval(
  m: Pick<PerformanceMetrics, 'sharpeRaw' | 'bars' | 'skew' | 'kurtosis'>,
  periodsPerYear: number,
): [number, number] {
  const se = sharpeStandardError(m.sharpeRaw, m.bars, m.skew, m.kurtosis);
  const scale = Math.sqrt(periodsPerYear);
  if (!Number.isFinite(se)) return [-Infinity, Infinity];
  return [(m.sharpeRaw - 1.96 * se) * scale, (m.sharpeRaw + 1.96 * se) * scale];
}

/** Incremental mean / variance (Welford's algorithm). */
export class RunningStats {
  count = 0;
  mean = 0;
  private m2 = 0;

  push(value: number): void {
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    this.m2 += delta * (value - this.mean);
  }

  /** Sample variance (0 with fewer than two observations). */
  get variance(): number {
    return this.count > 1 ? this.m2 / (this.count - 1) : 0;
  }
}
