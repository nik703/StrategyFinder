import { describe, expect, it } from 'vitest';
import { emptyMetrics } from '../src/engine/backtester';
import { diagnosticBadges, strategyVerdict } from '../src/lib/diagnostics';
import { activePreset, PRESETS } from '../src/lib/presets';
import { DEFAULT_CONFIG } from '../src/lib/config';
import type { CrossMarketReport, MonteCarloSummary, PerformanceMetrics, StrategyResult } from '../src/types/trading';

const fold = (roi: number, trades = 8): PerformanceMetrics => ({ ...emptyMetrics(), roi, trades, exposure: trades > 0 ? 30 : 0 });

const mc = (probProfit: number, drawdownP95: number): MonteCarloSummary => ({
  simulations: 500,
  trades: 120,
  probProfit,
  drawdownP95,
  medianReturn: 20,
  p5Return: 5,
  baseDrawdown: 12,
  passed: probProfit >= 0.8 && drawdownP95 < 50,
});

const strategy = (over: Partial<StrategyResult>): StrategyResult =>
  ({
    id: 'S-1',
    genome: { archetype: 'TREND', longEntry: { logic: 'AND', conditions: [] }, shortEntry: null, exit: { mode: 'static', stopLossPct: 2, takeProfitPct: 6 }, maxBarsInTrade: null },
    fitness: 2,
    pointFitness: 2.2,
    neighbourMedian: 1.9,
    neighbourCount: 6,
    params: 3,
    dsr: 0.97,
    search: { ...emptyMetrics(), trades: 120 },
    train: emptyMetrics(),
    validation: { ...emptyMetrics(), trades: 40 },
    folds: [fold(3), fold(2), fold(-1), fold(4), fold(1)],
    holdout: null,
    alpha: 0,
    equity: new Float32Array(0),
    foundAt: 1,
    assets: [],
    robustness: null,
    verdict: null,
    robust: { validation: true, folds: true, neighbours: true, significance: true, passed: true },
    monteCarlo: mc(0.94, 22),
    ...over,
  }) as StrategyResult;

const cross = (share: number, tested = 3, medianSharpe = 0.6): CrossMarketReport => ({
  rows: [],
  tested,
  share,
  medianSharpe,
  passed: tested >= 2 && share >= 2 / 3 - 1e-9 && medianSharpe > 0,
  cutoff: 0,
});

describe('plain-English diagnostics', () => {
  it('gives every check a badge with a status and an explanation', () => {
    const badges = diagnosticBadges({ strategy: strategy({}), report: null, crossMarket: cross(1), trials: 5000 });
    expect(badges.map((b) => b.key)).toEqual(['dsr', 'folds', 'monteCarlo', 'neighbours', 'crossMarket', 'holdout']);
    expect(badges.map((b) => b.status)).toEqual(['pass', 'pass', 'pass', 'pass', 'pass', 'pending']);
    for (const badge of badges) expect(badge.tooltip.length).toBeGreaterThan(40);
    expect(badges[0].tooltip).toContain('5,000 strategies');
    expect(badges[1].value).toBe('4 of 5 periods');
  });

  it('flags luck, uneven folds, stress failures, spikes and a weak cross-market result', () => {
    const weak = strategy({
      dsr: 0.3,
      folds: [fold(3), fold(-2), fold(-1), fold(-4), fold(-1)],
      robust: { validation: true, folds: false, neighbours: false, significance: false, passed: false },
      monteCarlo: mc(0.55, 61),
    });
    const badges = diagnosticBadges({ strategy: weak, report: null, crossMarket: cross(1 / 3), trials: 20000 });
    expect(badges.find((b) => b.key === 'dsr')?.status).toBe('fail');
    expect(badges.find((b) => b.key === 'folds')?.status).toBe('fail');
    expect(badges.find((b) => b.key === 'monteCarlo')?.status).toBe('fail');
    expect(badges.find((b) => b.key === 'neighbours')?.status).toBe('fail');
    expect(badges.find((b) => b.key === 'crossMarket')?.status).toBe('fail');
    const verdict = strategyVerdict({ strategy: weak, report: null, crossMarket: cross(1 / 3), trials: 20000 });
    expect(verdict.tone).toBe('weak');
    expect(verdict.risks.join(' ')).toMatch(/luck/);
    expect(verdict.risks.join(' ')).toMatch(/61% drawdown/);
    // One profitable period short of the bar is a caution, not a failure.
    const close = diagnosticBadges({ strategy: { ...weak, folds: [fold(3), fold(-2), fold(-1), fold(-4), fold(1)] }, report: null, crossMarket: null, trials: 1 });
    expect(close.find((b) => b.key === 'folds')?.status).toBe('warn');
  });

  it('calls a strategy promising only when every check run so far passes, and asks for the cross-market check', () => {
    const promising = strategyVerdict({ strategy: strategy({}), report: null, crossMarket: cross(1), trials: 5000 });
    expect(promising.tone).toBe('promising');
    expect(promising.headline).toMatch(/passes all 5 checks/);
    expect(promising.strengths.length).toBeGreaterThanOrEqual(5);
    const untested = strategyVerdict({ strategy: strategy({}), report: null, crossMarket: null, trials: 5000 });
    expect(untested.risks.join(' ')).toMatch(/cross-market check/);
    // A revealed holdout decides the tone.
    const locked = strategyVerdict({
      strategy: strategy({ holdout: { ...emptyMetrics(), sharpe: -0.4, roi: -3, trades: 20 }, verdict: 'Overfit', verdictNotes: ['Lost money on the holdout (-3.0%).'] }),
      report: null,
      crossMarket: null,
      trials: 5000,
    });
    expect(locked.tone).toBe('weak');
    expect(locked.headline).toMatch(/did not hold up/);
    expect(locked.risks[0]).toContain('Lost money on the holdout');
    // The small-sample warning counts the sample the gates counted: pooled over the assets of a pool.
    const pooledSample = { trades: 160, validationTrades: 30, holdoutTrades: null, gates: { search: 60, validation: 15, perYear: 15, holdout: 15 }, pooled: true };
    const thinPrimary = strategy({ validation: { ...emptyMetrics(), trades: 4 }, sample: pooledSample });
    expect(strategyVerdict({ strategy: thinPrimary, report: null, crossMarket: null, trials: 5000 }).risks.join(' ')).not.toMatch(/small sample/);
    const thinPool = strategy({ validation: { ...emptyMetrics(), trades: 4 }, sample: { ...pooledSample, validationTrades: 12 } });
    expect(strategyVerdict({ strategy: thinPool, report: null, crossMarket: null, trials: 5000 }).risks.join(' ')).toMatch(/Only 12 trades in the validation period \(pooled over the assets\)/);
    // Overfit only because of the deflated Sharpe: the holdout made money, and the verdict says so.
    const lucky = strategy({
      dsr: 0.04,
      holdout: { ...emptyMetrics(), sharpe: 1.1, roi: 4.6, trades: 60 },
      verdict: 'Overfit',
      verdictNotes: ['Deflated Sharpe 4.0% is under 50%: the validation result is most likely luck.'],
    });
    const unproven = strategyVerdict({ strategy: lucky, report: null, crossMarket: null, trials: 25000 });
    expect(unproven.tone).toBe('mixed');
    expect(unproven.headline).toMatch(/^Unproven: profitable on the holdout/);
    expect(unproven.strengths[0]).toContain('Made money on the sealed holdout');
    expect(unproven.risks[0]).toContain('most likely luck');
    expect(diagnosticBadges({ strategy: lucky, report: null, crossMarket: null, trials: 25000 }).find((b) => b.key === 'holdout')?.status).toBe('warn');
  });
});

describe('presets', () => {
  it('cover the four quick-start templates with realistic costs and one direction mode each', () => {
    expect(PRESETS.map((p) => p.title)).toEqual(['Crypto Multi-Asset Swing (4h)', 'Trend Breakout (1d)', 'Stock Momentum (1d)', 'Intraday Scalp (15m)']);
    for (const preset of PRESETS) {
      const config = { ...DEFAULT_CONFIG, ...preset.patch };
      expect(activePreset(config)?.id).toBe(preset.id);
      expect(config.feePct).toBeGreaterThan(0);
      expect(config.slippagePct).toBeGreaterThan(0);
    }
    expect(PRESETS.map((p) => p.patch.tradeMode)).toEqual(['long-only', 'asymmetric', 'long-only', 'long-short']);
    // The app opens on the first preset: Crypto Multi-Asset Swing (4h).
    expect(activePreset(DEFAULT_CONFIG)?.id).toBe('crypto-swing-4h');
    expect(DEFAULT_CONFIG).toMatchObject({ assetMode: 'pool', interval: '4h', range: '2y', tradeMode: 'long-only', hypothesis: 'VOL_BREAKOUT' });
  });
});
