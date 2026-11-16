import { describe, expect, it } from 'vitest';
import {
  conformsTo,
  DEFAULT_LIMITS,
  FINE_STEP,
  LOOKBACK_GRID,
  mulberry32,
  mutateGenome,
  normalizeGenome,
  parameterAxes,
  parameterVariants,
  randomGenome,
  stepNeighbours,
  type GenomeLimits,
} from '../src/engine/genome';
import { emptyMetrics } from '../src/engine/backtester';
import { generatePineScript } from '../src/engine/pine';
import { HYPOTHESIS_IDS, type Condition, type StrategyGenome } from '../src/types/trading';

const INTEGER: GenomeLimits = { ...DEFAULT_LIMITS, maPeriods: 'integer' };
const GRID: GenomeLimits = { ...DEFAULT_LIMITS, maPeriods: 'grid' };

/** Every EMA / SMA length of a condition (and the Bollinger basis, an SMA). */
function maLengths(c: Condition): number[] {
  if (c.ind === 'MA_CROSS') return [c.fastPeriod, c.slowPeriod];
  if (c.ind === 'PRICE_MA' || c.ind === 'MA_SLOPE' || c.ind === 'BB') return [c.period];
  return [];
}

const maTypes = (c: Condition): string[] =>
  c.ind === 'MA_CROSS' ? [c.fastType, c.slowType] : c.ind === 'PRICE_MA' || c.ind === 'MA_SLOPE' ? [c.maType] : [];

const lengthsOf = (g: StrategyGenome): number[] => g.longEntry.conditions.flatMap(maLengths);

const trend = (fast: number, slow: number, type: 'EMA' | 'SMA' = 'EMA'): StrategyGenome =>
  normalizeGenome(
    {
      archetype: 'TREND',
      hypothesis: 'TREND_FOLLOWING',
      longEntry: { logic: 'AND', conditions: [{ ind: 'MA_CROSS', fastType: type, fastPeriod: fast, slowType: type, slowPeriod: slow, op: 'cross_above' }] },
      shortEntry: null,
      exit: { mode: 'static', stopLossPct: 3, takeProfitPct: 7.5 },
      maxBarsInTrade: null,
    } as StrategyGenome,
    'long-only',
    INTEGER,
  );

describe('EMA / SMA lengths', () => {
  it('draws every integer length in integer mode, and only grid lengths in grid mode', () => {
    const rng = mulberry32(11);
    const offGrid = new Set<number>();
    const types = new Set<string>();
    const triggers = new Set<string>();
    for (let i = 0; i < 1500; i++) {
      const h = HYPOTHESIS_IDS[i % HYPOTHESIS_IDS.length];
      const g = randomGenome(rng, 'long-only', INTEGER, h);
      expect(conformsTo(g, INTEGER)).toBe(true);
      for (const c of g.longEntry.conditions) {
        for (const n of maLengths(c)) {
          expect(Number.isInteger(n)).toBe(true);
          expect(n).toBeGreaterThanOrEqual(5);
          expect(n).toBeLessThanOrEqual(200);
          if (!LOOKBACK_GRID.includes(n)) offGrid.add(n);
        }
        maTypes(c).forEach((t) => types.add(t));
        if (c.ind === 'MA_CROSS') expect(c.fastPeriod).toBeLessThan(c.slowPeriod);
        if (c.ind === 'MA_CROSS') expect(c.fastType).toBe(c.slowType); // one idea: EMA×EMA or SMA×SMA
      }
      triggers.add(g.longEntry.conditions[0].ind);
      // Grid mode is unchanged: every length is a grid value.
      const coarse = randomGenome(rng, 'long-only', GRID, h);
      for (const n of lengthsOf(coarse)) expect(LOOKBACK_GRID).toContain(n);
    }
    expect(offGrid.size).toBeGreaterThan(100); // far more than the 12 grid values
    expect([...types].sort()).toEqual(['EMA', 'SMA']);
    // Trend following now also triggers on the close crossing above an EMA / SMA.
    expect(triggers.has('PRICE_MA')).toBe(true);
    expect(triggers.has('MA_CROSS')).toBe(true);
    expect(triggers.has('BB')).toBe(true);
  });

  it('mutation fine-tunes the integer: small ±1–3 refinements and larger relative jumps', () => {
    const rng = mulberry32(5);
    const base = trend(20, 56);
    const fastMoves = new Map<number, number>();
    let typeSwitches = 0;
    for (let i = 0; i < 3000; i++) {
      const child = mutateGenome(rng, base, 'long-only', INTEGER);
      const c = child.longEntry.conditions[0];
      if (c.ind !== 'MA_CROSS') continue;
      if (c.fastType === 'SMA') typeSwitches++;
      if (c.fastPeriod !== 20 && c.slowPeriod === 56) fastMoves.set(c.fastPeriod - 20, (fastMoves.get(c.fastPeriod - 20) ?? 0) + 1);
    }
    // Single-candle refinements happen (impossible on the old grid, where 20's neighbours are 14 and 28).
    expect(fastMoves.get(1)).toBeGreaterThan(0);
    expect(fastMoves.get(-1)).toBeGreaterThan(0);
    expect(fastMoves.get(2)).toBeGreaterThan(0);
    // …and bigger jumps too.
    expect([...fastMoves.keys()].some((d) => Math.abs(d) >= 4)).toBe(true);
    expect(typeSwitches).toBeGreaterThan(0); // EMA → SMA
  });

  it('keeps the plateau check a ×√2 step away, not ±1 candle', () => {
    const g = trend(20, 56);
    const neighbours = stepNeighbours(g, 'long-only', INTEGER).map((n) => n.longEntry.conditions[0] as Extract<Condition, { ind: 'MA_CROSS' }>);
    // Neighbours that move the fast (or slow) length; the others vary the stop or the target.
    const fast = neighbours.filter((c) => c.slowPeriod === 56 && c.fastPeriod !== 20).map((c) => c.fastPeriod).sort((a, b) => a - b);
    const slow = neighbours.filter((c) => c.fastPeriod === 20 && c.slowPeriod !== 56).map((c) => c.slowPeriod).sort((a, b) => a - b);
    expect(fast).toEqual([Math.round(20 / FINE_STEP), Math.round(20 * FINE_STEP)]); // 14 and 28
    expect(slow).toEqual([Math.round(56 / FINE_STEP), Math.round(56 * FINE_STEP)]); // 40 and 79
    // The same distance as one step of the coarse grid.
    const coarse = stepNeighbours(trend(20, 56), 'long-only', GRID).map((n) => n.longEntry.conditions[0] as Extract<Condition, { ind: 'MA_CROSS' }>);
    expect(coarse.filter((c) => c.slowPeriod === 56 && c.fastPeriod !== 20).map((c) => c.fastPeriod).sort((a, b) => a - b)).toEqual([14, 28]);
    // ±20% sensitivity checks move to the nearest integers.
    const pct = parameterVariants(trend(23, 87), 'long-only', INTEGER).map((n) => n.longEntry.conditions[0] as Extract<Condition, { ind: 'MA_CROSS' }>);
    expect(pct.filter((c) => c.slowPeriod === 87 && c.fastPeriod !== 23).map((c) => c.fastPeriod).sort((a, b) => a - b)).toEqual([18, 28]);
  });

  it('gives the stability map a ×√2 ladder through the exact length', () => {
    const axes = parameterAxes(trend(23, 87), INTEGER);
    const fast = axes.find((a) => a.label === 'MA fast period');
    const slow = axes.find((a) => a.label === 'MA slow period');
    expect(fast?.fine).toBe(true);
    expect(fast?.value).toBe(23);
    expect(fast?.grid).toContain(23);
    expect(slow?.grid).toContain(87);
    const ladder = slow?.grid ?? [];
    for (let i = 1; i < ladder.length; i++) expect(ladder[i] / ladder[i - 1]).toBeGreaterThan(1.25);
    expect(ladder.length).toBeLessThanOrEqual(13);
    // Exit parameters keep their own grids.
    expect(axes.find((a) => a.label === 'Stop loss %')?.fine).toBe(false);
  });

  it('exports SMA and fine-tuned lengths to Pine Script', () => {
    const g = trend(23, 87, 'SMA');
    const script = generatePineScript(
      { id: 'S-1', genome: g, dsr: 0, holdout: null, train: emptyMetrics(), validation: emptyMetrics(), search: emptyMetrics() } as never,
      {
        label: 'BTC-USD',
        interval: '1d',
        settings: { initialCapital: 10_000, feePct: 0.1, slippagePct: 0.05, tradeMode: 'long-only', sizing: { mode: 'full', riskPct: 2, compounding: true }, fundingPctPerBar: 0 },
        components: [{ symbol: 'BTC-USD', weight: 1 }],
        units: [],
        windowStart: 0,
        windowEnd: 86_400_000 * 400,
        bars: 400,
        benchmarkRoi: 0,
        venue: 'BTCUSDT on Binance',
        simulatedData: true,
      } as never,
    );
    expect(script).toContain('sma_23 = ta.sma(close, 23)');
    expect(script).toContain('sma_87 = ta.sma(close, 87)');
    expect(script).toContain('ta.crossover(sma_23, sma_87)');
  });
});
