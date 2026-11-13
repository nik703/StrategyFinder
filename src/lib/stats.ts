/**
 * Selection-bias helpers.
 *
 * An optimiser that evaluates thousands of strategies will find impressive
 * backtests even in pure noise. `expectedMaxSharpe` estimates how good the
 * BEST of N skill-less strategies would look over a given window, which gives
 * the Strategy Breakdown view an honest yard-stick for out-of-sample results.
 */

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

const EULER_MASCHERONI = 0.5772156649015329;

/**
 * Expected annualised Sharpe ratio of the best of `trials` independent,
 * zero-skill strategies observed over `years` (Bailey & López de Prado's
 * expected-maximum approximation). Real optimiser trials are correlated, so
 * this is an upper estimate of the luck threshold.
 */
export function expectedMaxSharpe(trials: number, years: number): number {
  if (!(trials > 1) || !(years > 0)) return 0;
  const sigma = Math.sqrt(1 / years);
  const z1 = inverseNormalCdf(1 - 1 / trials);
  const z2 = inverseNormalCdf(1 - 1 / (trials * Math.E));
  return sigma * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2);
}
