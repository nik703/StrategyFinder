/**
 * Monte Carlo stress curve: the spread of hundreds of perturbed replays of a strategy's trades
 * (skipped trades, shuffled order, slippage shocks). The outer band holds 90% of the paths, the
 * inner band the middle half, the line is the median path; the dashed line is break-even.
 */

import type { MonteCarloReport } from '../types/trading';
import { fmtPct } from '../lib/format';

const W = 560;
const H = 220;
const PAD = { left: 48, right: 12, top: 12, bottom: 26 };

export default function MonteCarloChart({ report }: { report: MonteCarloReport }): JSX.Element {
  const { bands } = report;
  const n = bands.x.length;
  const maxX = bands.x[n - 1] || 1;
  let lo = Infinity;
  let hi = -Infinity;
  for (let k = 0; k < n; k++) {
    lo = Math.min(lo, bands.p5[k], 1);
    hi = Math.max(hi, bands.p95[k], 1);
  }
  const span = hi - lo || 1;
  lo -= span * 0.05;
  hi += span * 0.05;
  const x = (trade: number): number => PAD.left + (trade / maxX) * (W - PAD.left - PAD.right);
  const y = (equity: number): number => PAD.top + (1 - (equity - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);
  const area = (upper: number[], lower: number[]): string => {
    const top = bands.x.map((t, k) => `${k === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(upper[k]).toFixed(1)}`).join(' ');
    const bottom = bands.x
      .map((t, k) => ({ t, v: lower[k] }))
      .reverse()
      .map(({ t, v }) => `L${x(t).toFixed(1)},${y(v).toFixed(1)}`)
      .join(' ');
    return `${top} ${bottom} Z`;
  };
  const median = bands.x.map((t, k) => `${k === 0 ? 'M' : 'L'}${x(t).toFixed(1)},${y(bands.p50[k]).toFixed(1)}`).join(' ');
  const ticks = [lo + (hi - lo) * 0.1, (lo + hi) / 2, hi - (hi - lo) * 0.1];

  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Monte Carlo equity paths: 5th to 95th percentile band, interquartile band and median">
        {ticks.map((value) => (
          <g key={value}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(value)} y2={y(value)} stroke="var(--color-line)" strokeWidth="1" />
            <text x={PAD.left - 6} y={y(value) + 3.5} textAnchor="end" fontSize="10" fill="var(--color-ink-3)" className="num">
              {fmtPct((value - 1) * 100, true, 0)}
            </text>
          </g>
        ))}
        <path d={area(bands.p95, bands.p5)} fill="var(--color-accent)" fillOpacity="0.14" />
        <path d={area(bands.p75, bands.p25)} fill="var(--color-accent)" fillOpacity="0.28" />
        <line x1={PAD.left} x2={W - PAD.right} y1={y(1)} y2={y(1)} stroke="var(--color-ink-3)" strokeDasharray="4 4" strokeWidth="1" />
        <path d={median} fill="none" stroke="var(--color-accent-strong)" strokeWidth="2" strokeLinejoin="round" />
        <text x={PAD.left} y={H - 8} fontSize="10" fill="var(--color-ink-3)">
          trade 0
        </text>
        <text x={W - PAD.right} y={H - 8} fontSize="10" fill="var(--color-ink-3)" textAnchor="end" className="num">
          trade {maxX}
        </text>
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-3">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-4 rounded-sm bg-accent/15" /> 90% of paths
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-4 rounded-sm bg-accent/30" /> middle 50%
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t-2 border-accent-strong" /> median path
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t border-dashed border-ink-3" /> break-even
        </span>
      </figcaption>
    </figure>
  );
}
