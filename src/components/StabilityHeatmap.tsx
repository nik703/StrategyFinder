/**
 * Parameter-stability heatmap: search-period Sharpe over the two most sensitive parameters of a
 * strategy (picked from its one-step neighbourhood), centred on the chosen values.
 *
 * Diverging colour: blue for positive Sharpe, red for negative, a neutral grey at zero, saturated
 * at ±2. Every cell prints its value in ink (colour is never the only carrier) and has a tooltip;
 * combinations that fail the minimum-trade gates are hatched; the chosen cell is outlined.
 */

import { fmtNum } from '../lib/format';
import type { StabilityGrid } from '../types/trading';
import { cx } from './ui';

const POSITIVE = '#3987e5';
const NEGATIVE = '#e45a5a';
const NEUTRAL = '#383835';
/** Sharpe at which the colour saturates. */
const SATURATION = 2;

function cellColor(sharpe: number): string {
  const share = Math.min(1, Math.abs(sharpe) / SATURATION);
  const pole = sharpe >= 0 ? POSITIVE : NEGATIVE;
  return `color-mix(in oklab, ${pole} ${Math.round(share * 85)}%, ${NEUTRAL})`;
}

const HATCH = 'repeating-linear-gradient(45deg, rgba(232,235,243,0.18) 0 2px, transparent 2px 7px)';

/** Share of valid cells whose Sharpe keeps at least half of the chosen cell's. */
export function plateauShare(grid: StabilityGrid): number | null {
  const ci = grid.xValues.indexOf(grid.baseX);
  const ri = grid.yLabel === null ? 0 : grid.yValues.indexOf(grid.baseY);
  const base = grid.cells[ri]?.[ci];
  if (base === null || base === undefined || !(base > 0)) return null;
  let valid = 0;
  let holding = 0;
  grid.cells.forEach((row, r) =>
    row.forEach((value, c) => {
      if (value === null || (r === ri && c === ci)) return;
      valid++;
      if (value >= 0.5 * base && !grid.undersampled[r][c]) holding++;
    }),
  );
  return valid > 0 ? holding / valid : null;
}

export default function StabilityHeatmap({ grid }: { grid: StabilityGrid }): JSX.Element {
  const rows = grid.yLabel === null ? [Number.NaN] : grid.yValues;
  const share = plateauShare(grid);
  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs text-ink-2">
          Search-period Sharpe as <span className="font-semibold text-ink">{grid.xLabel}</span>
          {grid.yLabel !== null && (
            <>
              {' '}
              and <span className="font-semibold text-ink">{grid.yLabel}</span>
            </>
          )}{' '}
          move around the chosen values (outlined), up to two grid steps either way – fewer where a grid ends.
        </p>
        {share !== null && (
          <span className="num text-xs text-ink-3">
            {Math.round(share * 100)}% of neighbouring cells keep half of the Sharpe or more
            {share >= 0.6 ? ' – a plateau' : share <= 0.3 ? ' – a spike' : ''}
          </span>
        )}
      </div>
      <div className="mt-2 overflow-x-auto">
        <table className="num border-separate text-xs" style={{ borderSpacing: 2 }} aria-label="Parameter stability heatmap">
          <thead>
            <tr>
              <th scope="col" className="px-1.5 text-right text-[10px] font-normal uppercase tracking-wide text-ink-3">
                {grid.yLabel ?? ''}
              </th>
              {grid.xValues.map((x) => (
                <th key={x} scope="col" className={cx('min-w-[52px] px-1 pb-1 text-center font-medium', x === grid.baseX ? 'text-ink' : 'text-ink-3')}>
                  {x}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((y, r) => (
              <tr key={Number.isNaN(y) ? 'only' : y}>
                <th scope="row" className={cx('px-1.5 text-right font-medium', y === grid.baseY ? 'text-ink' : 'text-ink-3')}>
                  {Number.isNaN(y) ? '' : y}
                </th>
                {grid.xValues.map((x, c) => {
                  const value = grid.cells[r][c];
                  const under = grid.undersampled[r][c];
                  const chosen = x === grid.baseX && (grid.yLabel === null || y === grid.baseY);
                  const where = `${grid.xLabel} ${x}${grid.yLabel !== null ? `, ${grid.yLabel} ${y}` : ''}`;
                  const tip =
                    value === null
                      ? `${where}: not a valid strategy`
                      : `${where}: Sharpe ${fmtNum(value)}${under ? ' (fails the minimum-trade gates)' : ''}${chosen ? ' – the chosen values' : ''}`;
                  return (
                    <td
                      key={x}
                      title={tip}
                      aria-label={tip}
                      className={cx(
                        'h-9 rounded-[4px] px-1 text-center font-medium',
                        value === null ? 'text-ink-3' : 'text-ink',
                        chosen && 'outline outline-2 outline-offset-[-2px] outline-[var(--color-ink)]',
                      )}
                      style={{
                        background: value === null ? 'transparent' : cellColor(value),
                        backgroundImage: under && value !== null ? HATCH : undefined,
                        border: value === null ? '1px dashed var(--color-line)' : undefined,
                      }}
                    >
                      {value === null ? '–' : fmtNum(value, 1)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {grid.yLabel === null && <p className="mt-1 text-[11px] text-ink-3">The strategy has a single tunable parameter, so the map is one row.</p>}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-3">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-16 rounded-sm" style={{ background: `linear-gradient(90deg, ${NEGATIVE}, ${NEUTRAL}, ${POSITIVE})` }} />
          Sharpe −{SATURATION} … 0 … +{SATURATION}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm" style={{ background: NEUTRAL, backgroundImage: HATCH }} />
          too few trades
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm outline outline-2 outline-offset-[-2px] outline-[var(--color-ink)]" />
          chosen values
        </span>
      </div>
    </div>
  );
}
