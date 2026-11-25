/**
 * AST → Pine parity: the TypeScript engine and the generated Pine Script v5 compile from the same
 * strategy AST (`src/engine/strategyAst.ts`). These tests RUN the generated Pine (with the subset
 * interpreter in `helpers/pineInterpreter.ts`) on the same candles as the engine and require:
 *
 *   • identical entry and opposite-signal exit series on every candle, for every hypothesis and
 *     building block, in Long-Only, Asymmetric L/S and Mirrored L/S;
 *   • identical stop and target levels after every close of every trade the engine held – including
 *     ATR trailing stops and breakeven locks.
 */

import { describe, expect, it } from 'vitest';
import { buildDataset } from '../src/data/basket';
import { generateSyntheticCandles } from '../src/data/synthetic';
import { BacktestWorkspace, barsPerYear, evaluateGroup, evaluateSearch, runBacktest, splitFor } from '../src/engine/backtester';
import { countParameters, DEFAULT_LIMITS, mulberry32, normalizeGenome, randomGenome } from '../src/engine/genome';
import { IndicatorCache } from '../src/engine/indicators';
import { generatePineScript } from '../src/engine/pine';
import { compileStrategy } from '../src/engine/strategyAst';
import { HYPOTHESIS_IDS, TRADE_MODES, type BacktestSettings, type OHLCV, type RuleGroup, type StrategyGenome, type StrategyResult, type TradeMode } from '../src/types/trading';
import { PineProgram } from './helpers/pineInterpreter';

const NOW = Date.UTC(2026, 0, 15);
const data: OHLCV = buildDataset({ ETH: generateSyntheticCandles('ETH-USD', '4h', '1y', NOW) }, [{ symbol: 'ETH', weight: 1 }]).data;
const n = data.c.length;
const split = splitFor(n);
const ppy = barsPerYear(data.t);

const settingsFor = (tradeMode: TradeMode): BacktestSettings => ({
  initialCapital: 10_000,
  feePct: 0.1,
  slippagePct: 0.05,
  slippageAtrMult: 0.03,
  tradeMode,
  sizing: { mode: 'risk', riskPct: 1, compounding: false },
  fundingPctPerBar: 0,
});

function resultFor(genome: StrategyGenome, settings: BacktestSettings): StrategyResult {
  const m = evaluateSearch(new IndicatorCache(data), genome, settings, new BacktestWorkspace(n), split, ppy);
  return { id: 'S-1', genome, fitness: 1, dsr: 0.3, params: countParameters(genome), ...m, holdout: null, alpha: 0, equity: new Float32Array(0), foundAt: 1, assets: [], robustness: null, verdict: null };
}

const ctx = (settings: BacktestSettings) => ({
  label: 'ETH-USD',
  assets: ['ETH-USD'],
  interval: '4h' as const,
  settings,
  components: [{ symbol: 'ETH-USD', weight: 1 }],
  units: [],
  windowStart: data.t[0],
  windowEnd: data.t[n - 1],
  bars: n,
  benchmarkRoi: 0,
  venue: 'test',
  simulatedData: true,
});

/** The engine's signal for a rule group on every candle. */
function engineSignal(group: RuleGroup, ind: IndicatorCache): Uint8Array {
  const out = new Uint8Array(n);
  evaluateGroup(group, ind, out, new Uint8Array(n));
  return out;
}

/** Candles where the engine's 0/1 signal and Pine's series disagree. */
function mismatches(engine: Uint8Array, pine: Float64Array): number[] {
  const bad: number[] = [];
  for (let i = 0; i < n; i++) if (engine[i] !== (pine[i] === pine[i] && pine[i] !== 0 ? 1 : 0)) bad.push(i);
  return bad;
}

const close = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

describe('strategy AST → Pine Script v5 parity', () => {
  it('compiles every genome into one AST that both the engine and the exporter use', () => {
    const rng = mulberry32(4);
    for (const mode of TRADE_MODES) {
      const g = randomGenome(rng, mode, DEFAULT_LIMITS, 'TREND_FOLLOWING');
      const ast = compileStrategy(g, mode);
      expect(ast.mode).toBe(mode);
      expect(ast.longEntry).toHaveLength(g.longEntry.conditions.length);
      expect(ast.shortEntry === null).toBe(mode === 'long-only');
      if (ast.shortEntry && g.shortEntry) expect(ast.shortEntry).toHaveLength(g.shortEntry.conditions.length);
    }
  });

  it('produces identical entry and exit signals in TypeScript and the generated Pine, in every direction mode', () => {
    const rng = mulberry32(2026);
    const ind = new IndicatorCache(data);
    let compared = 0;
    let firing = 0;
    for (const mode of TRADE_MODES) {
      for (const hypothesis of HYPOTHESIS_IDS) {
        for (let k = 0; k < 12; k++) {
          const genome = randomGenome(rng, mode, DEFAULT_LIMITS, hypothesis);
          const settings = settingsFor(mode);
          const program = new PineProgram(generatePineScript(resultFor(genome, settings), ctx(settings)), data);
          const groups: [string, RuleGroup | null][] = [
            ['longEntry', genome.longEntry],
            ['shortEntry', mode === 'long-only' ? null : genome.shortEntry],
            ['longExitSignal', genome.exit.mode === 'dynamic' ? genome.exit.longExit : null],
            ['shortExitSignal', genome.exit.mode === 'dynamic' && mode !== 'long-only' ? genome.exit.shortExit : null],
          ];
          for (const [name, group] of groups) {
            if (!group) continue;
            const engine = engineSignal(group, ind);
            const bad = mismatches(engine, program.series(name));
            expect(bad, `${mode} ${hypothesis} ${name}: ${JSON.stringify(group.conditions)}`).toEqual([]);
            compared++;
            firing += engine.reduce((sum, x) => sum + x, 0);
          }
        }
      }
    }
    expect(compared).toBeGreaterThan(200);
    expect(firing).toBeGreaterThan(1000); // the signals actually fire: parity is not vacuous
  });

  it('places identical stops and targets – trailing ATR stops and breakeven locks included – after every close', () => {
    const rng = mulberry32(77);
    const cases: { genome: StrategyGenome; mode: TradeMode }[] = [];
    // Hand-picked exits on a trend trigger: trailing, breakeven, both, static bracket, ATR bracket.
    const trigger: StrategyGenome['longEntry'] = {
      logic: 'AND',
      conditions: [{ ind: 'MA_CROSS', fastType: 'EMA', fastPeriod: 10, slowType: 'EMA', slowPeriod: 28, op: 'cross_above' }],
    };
    const dynamic = (trailing: boolean, target: number | null, breakeven: boolean): StrategyGenome['exit'] => ({
      mode: 'dynamic',
      atrPeriod: 14,
      stopAtrMult: 2,
      trailing,
      takeProfitAtrMult: target,
      reversalExit: false,
      longExit: null,
      shortExit: null,
      breakeven,
    });
    for (const mode of TRADE_MODES) {
      for (const exit of [dynamic(true, null, false), dynamic(false, 4, true), dynamic(true, 6, true), dynamic(false, 3, false)]) {
        cases.push({ mode, genome: normalizeGenome({ archetype: 'TREND', hypothesis: 'TREND_FOLLOWING', longEntry: trigger, shortEntry: null, exit, maxBarsInTrade: null }, mode, DEFAULT_LIMITS) });
      }
      cases.push({
        mode,
        genome: normalizeGenome(
          { archetype: 'TREND', hypothesis: 'TREND_FOLLOWING', longEntry: trigger, shortEntry: null, exit: { mode: 'static', stopLossPct: 3, takeProfitPct: 7.5, breakeven: true }, maxBarsInTrade: null },
          mode,
          DEFAULT_LIMITS,
        ),
      });
      for (let k = 0; k < 8; k++) cases.push({ mode, genome: randomGenome(rng, mode, DEFAULT_LIMITS, HYPOTHESIS_IDS[k % HYPOTHESIS_IDS.length]) });
    }

    let barsChecked = 0;
    let trailingMoves = 0;
    let breakevens = 0;
    for (const { genome, mode } of cases) {
      const settings = settingsFor(mode);
      const ws = new BacktestWorkspace(n).recordLevels();
      runBacktest(new IndicatorCache(data), genome, settings, ws);
      const levels = ws.levels as NonNullable<typeof ws.levels>;
      const env = {
        'strategy.position_size': Float64Array.from(levels.dir),
        'strategy.position_avg_price': Float64Array.from(levels.entry),
      };
      const program = new PineProgram(generatePineScript(resultFor(genome, settings), ctx(settings)), data, env);
      const orders = program.runExitManagement();
      for (let i = 0; i < n; i++) {
        const dir = levels.dir[i];
        if (dir === 0) continue;
        const order = orders[dir === 1 ? 'Long exit' : 'Short exit'];
        expect(order, 'Pine never sent an exit order for an open position').toBeDefined();
        const pineStop = order.stop[i];
        const pineTarget = order.limit[i];
        const tsStop = levels.stop[i];
        const tsTarget = levels.target[i];
        expect(close(pineStop, tsStop), `bar ${i}: stop ${pineStop} vs ${tsStop} – ${JSON.stringify(genome.exit)}`).toBe(true);
        if (Number.isFinite(tsTarget)) expect(close(pineTarget, tsTarget), `bar ${i}: target ${pineTarget} vs ${tsTarget}`).toBe(true);
        else expect(Number.isNaN(pineTarget)).toBe(true);
        barsChecked++;
        if (i > 0 && levels.dir[i - 1] === dir && !close(levels.stop[i - 1], tsStop)) {
          if (close(tsStop, levels.entry[i])) breakevens++;
          else trailingMoves++;
        }
      }
    }
    expect(barsChecked).toBeGreaterThan(1000);
    expect(trailingMoves).toBeGreaterThan(50); // the trailing stops really moved
    expect(breakevens).toBeGreaterThan(5); // and breakeven locks really fired
  });
});
