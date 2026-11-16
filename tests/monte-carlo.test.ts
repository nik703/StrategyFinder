import { describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { BacktestWorkspace, runBacktest } from '../src/engine/backtester';
import { DEFAULT_LIMITS, mulberry32, randomGenome } from '../src/engine/genome';
import { IndicatorCache } from '../src/engine/indicators';
import {
  MC_MAX_DRAWDOWN_P95,
  MC_MIN_PROB_PROFIT,
  MC_SIMULATIONS,
  monteCarloStress,
  summarise,
  tradesFrom,
  type McTrade,
} from '../src/engine/monteCarlo';
import { HYPOTHESIS_IDS, type BacktestSettings, type OHLCV } from '../src/types/trading';

const SETTINGS: BacktestSettings = {
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  tradeMode: 'long-only',
  sizing: { mode: 'full', riskPct: 2, compounding: true },
  fundingPctPerBar: 0,
};

const btc = (): OHLCV =>
  buildDataset({ 'BTC-USD': generateSyntheticCandles('BTC-USD', '1d', '5y', Date.UTC(2026, 0, 15)) }, [{ symbol: 'BTC-USD', weight: 1 }]).data;

/** n trades alternating between a win and a loss of the given sizes. */
const alternating = (n: number, win: number, loss: number): McTrade[] =>
  Array.from({ length: n }, (_, k) => (k % 2 === 0 ? { ret: win, mae: -0.01, exposure: 1 } : { ret: -loss, mae: -loss, exposure: 1 }));

describe('Monte Carlo execution stress test', () => {
  it('extracts per-trade returns that compound exactly to the backtest result', () => {
    const data = btc();
    const ind = new IndicatorCache(data);
    const ws = new BacktestWorkspace(data.c.length);
    const rng = mulberry32(31);
    let checked = 0;
    for (let i = 0; i < 40; i++) {
      const g = randomGenome(rng, 'long-only', DEFAULT_LIMITS, HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length]);
      runBacktest(ind, g, SETTINGS, ws);
      const trades = tradesFrom(ws);
      expect(trades).toHaveLength(ws.tradeCount);
      // Flat between trades and no funding: the account is the product of (1 + trade return).
      const growth = trades.reduce((acc, t) => acc * (1 + t.ret), 1);
      expect(growth).toBeCloseTo(ws.equity[data.c.length - 1] / SETTINGS.initialCapital, 9);
      for (const t of trades) {
        expect(t.mae).toBeLessThanOrEqual(0);
        expect(t.mae).toBeLessThanOrEqual(Math.min(0, t.ret) + 1e-12); // a losing trade was at least that far down
        expect(t.exposure).toBeGreaterThan(0);
        expect(t.exposure).toBeLessThanOrEqual(1);
      }
      if (trades.length > 0) checked++;
    }
    expect(checked).toBeGreaterThan(30);
  });

  it('reports the probability of profit and the 95th-percentile drawdown over 200+ perturbed paths', () => {
    expect(MC_SIMULATIONS).toBeGreaterThanOrEqual(200);
    const strong = monteCarloStress(alternating(120, 0.03, 0.01), SETTINGS);
    expect(strong).not.toBeNull();
    if (!strong) return;
    expect(strong.simulations).toBe(MC_SIMULATIONS);
    expect(strong.trades).toBe(120);
    expect(strong.probProfit).toBe(1);
    expect(strong.drawdownP95).toBeGreaterThanOrEqual(strong.baseDrawdown - 1e-9); // shuffling finds worse orders
    expect(strong.passed).toBe(true);
    expect(strong.p5Return).toBeLessThanOrEqual(strong.medianReturn);

    // A losing edge: almost no path ends in profit.
    const weak = monteCarloStress(alternating(120, 0.01, 0.0125), SETTINGS);
    expect(weak?.probProfit).toBeLessThan(0.05);
    expect(weak?.passed).toBe(false);

    // A thin edge (≈ 0.3% a trade, about what the shocks cost: on average twice the 0.15% per-side
    // cost of every fill): a coin flip, so it fails.
    const marginal = monteCarloStress(alternating(120, 0.03, 0.0233), SETTINGS);
    expect(marginal?.probProfit).toBeGreaterThan(0.05);
    expect(marginal?.probProfit).toBeLessThan(MC_MIN_PROB_PROFIT);

    expect(monteCarloStress([], SETTINGS)).toBeNull();
    expect(MC_MAX_DRAWDOWN_P95).toBe(50);
  });

  it('keeps the total when only the order changes, and only the costs and skips move it', () => {
    const trades = alternating(60, 0.02, 0.012);
    const total = trades.reduce((acc, t) => acc * (1 + t.ret), 1);
    const shuffleOnly = monteCarloStress(trades, SETTINGS, { skipRate: 0, meanShock: 0, simulations: 300 });
    expect(shuffleOnly?.medianReturn).toBeCloseTo((total - 1) * 100, 9);
    expect(shuffleOnly?.p5Return).toBeCloseTo((total - 1) * 100, 9);
    // Shocks always cost money.
    const shocked = monteCarloStress(trades, SETTINGS, { skipRate: 0, simulations: 300 });
    expect(shocked?.medianReturn).toBeLessThan((total - 1) * 100);
    // Skipping ~10% of the trades spreads the outcomes.
    const skipped = monteCarloStress(trades, SETTINGS, { meanShock: 0, simulations: 300 });
    expect((skipped?.medianReturn ?? 0) - (skipped?.p5Return ?? 0)).toBeGreaterThan(0);
  });

  it('is deterministic per seed and draws equity-path percentile bands', () => {
    const trades = alternating(80, 0.025, 0.012);
    const a = monteCarloStress(trades, SETTINGS, { seed: 9 });
    const b = monteCarloStress(trades, SETTINGS, { seed: 9 });
    const c = monteCarloStress(trades, SETTINGS, { seed: 10 });
    expect(a).toEqual(b);
    expect(a?.drawdownP95).not.toBe(c?.drawdownP95);
    const bands = a?.bands;
    expect(bands?.x[0]).toBe(0);
    expect(bands?.x[(bands?.x.length ?? 1) - 1]).toBe(80);
    for (let k = 0; k < (bands?.x.length ?? 0); k++) {
      const [p5, p25, p50, p75, p95] = [bands!.p5[k], bands!.p25[k], bands!.p50[k], bands!.p75[k], bands!.p95[k]];
      expect(p5).toBeLessThanOrEqual(p25 + 1e-12);
      expect(p25).toBeLessThanOrEqual(p50 + 1e-12);
      expect(p50).toBeLessThanOrEqual(p75 + 1e-12);
      expect(p75).toBeLessThanOrEqual(p95 + 1e-12);
    }
    expect(bands?.p50[0]).toBe(1);
    // The leaderboard copy drops the bands.
    const summary = summarise(a ?? null);
    expect(summary && 'bands' in summary).toBe(false);
    expect(summary?.probProfit).toBe(a?.probProfit);
  });

  it('charges intrabar excursions: a deep adverse excursion raises the drawdown even when the trade wins', () => {
    const calm = Array.from({ length: 50 }, (): McTrade => ({ ret: 0.01, mae: -0.002, exposure: 1 }));
    const wild = calm.map((t) => ({ ...t, mae: -0.2 }));
    const a = monteCarloStress(calm, SETTINGS, { meanShock: 0 });
    const b = monteCarloStress(wild, SETTINGS, { meanShock: 0 });
    expect(a?.drawdownP95).toBeLessThan(1);
    expect(b?.drawdownP95).toBeGreaterThanOrEqual(20 - 1e-9);
  });
});
