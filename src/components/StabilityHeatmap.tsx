/**
 * Parameter-stability heatmap: search-period Sharpe over the two most sensitive parameters of a
 * strategy (picked from its one-step neighbourhood), centred on the chosen values.
 *
 * Diverging colour: blue for positive Sharpe, red for negative, a neutral grey at zero, saturated
 * at ±2. Every cell prints its value in ink (colour is never the only carrier) and has a tooltip;
 * combinations that fail the minimum-trade gates are hatched; the chosen cell is outlined.
 */

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';
import { fmtNum } from '../lib/format';
import type { StabilityGrid } from '../types/trading';

/** Theme tokens (see `index.css`) – the heatmap is DOM, so it can use the CSS variables directly. */
const POSITIVE = 'var(--chart-1)';
const NEGATIVE = 'var(--down)';
const NEUTRAL = 'var(--input)';
/** Sharpe at which the colour saturates. */
const SATURATION = 2;

function cellColor(sharpe: number): string {
  const share = Math.min(1, Math.abs(sharpe) / SATURATION);
  const pole = sharpe >= 0 ? POSITIVE : NEGATIVE;
  return `color-mix(in oklab, ${pole} ${Math.round(share * 85)}%, ${NEUTRAL})`;
}

const HATCH = 'repeating-linear-gradient(45deg, color-mix(in oklab, var(--foreground) 18%, transparent) 0 2px, transparent 2px 7px)';

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
        <p className="text-xs text-secondary-foreground">
          Search-period Sharpe as <span className="font-semibold text-foreground">{grid.xLabel}</span>
          {grid.yLabel !== null && (
            <>
              {' '}
              and <span className="font-semibold text-foreground">{grid.yLabel}</span>
            </>
          )}{' '}
          move around the chosen values (outlined), up to two grid steps either way – fewer where a grid ends.
        </p>
        {share !== null && (
          <span className="num text-xs text-muted-foreground">
            {Math.round(share * 100)}% of neighbouring cells keep half of the Sharpe or more
            {share >= 0.6 ? ' – a plateau' : share <= 0.3 ? ' – a spike' : ''}
          </span>
        )}
      </div>
      <div className="mt-2">
        <Table className="num w-auto border-separate text-xs" style={{ borderSpacing: 2 }} aria-label="Parameter stability heatmap">
          <TableHeader>
            <TableRow className="border-0 hover:bg-transparent">
              <TableHead scope="col" className="h-auto px-1.5 text-right text-[10px] font-normal uppercase tracking-wide text-muted-foreground">
                {grid.yLabel ?? ''}
              </TableHead>
              {grid.xValues.map((x) => (
                <TableHead key={x} scope="col" className={cn('h-auto min-w-[52px] px-1 pb-1 text-center font-medium', x === grid.baseX ? 'text-foreground' : 'text-muted-foreground')}>
                  {x}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((y, r) => (
              <TableRow className="border-0 hover:bg-transparent" key={Number.isNaN(y) ? 'only' : y}>
                <TableHead scope="row" className={cn('h-auto px-1.5 text-right font-medium', y === grid.baseY ? 'text-foreground' : 'text-muted-foreground')}>
                  {Number.isNaN(y) ? '' : y}
                </TableHead>
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
                    <TableCell
                      key={x}
                      title={tip}
                      aria-label={tip}
                      className={cn(
                        'h-9 rounded-[4px] p-0 px-1 text-center font-medium',
                        value === null ? 'text-muted-foreground' : 'text-foreground',
                        chosen && 'outline outline-2 outline-offset-[-2px] outline-[var(--foreground)]',
                      )}
                      style={{
                        background: value === null ? 'transparent' : cellColor(value),
                        backgroundImage: under && value !== null ? HATCH : undefined,
                        border: value === null ? '1px dashed var(--border)' : undefined,
                      }}
                    >
                      {value === null ? '–' : fmtNum(value, 1)}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {grid.yLabel === null && <p className="mt-1 text-[11px] text-muted-foreground">The strategy has a single tunable parameter, so the map is one row.</p>}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-16 rounded-sm" style={{ background: `linear-gradient(90deg, ${NEGATIVE}, ${NEUTRAL}, ${POSITIVE})` }} />
          Sharpe −{SATURATION} … 0 … +{SATURATION}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm" style={{ background: NEUTRAL, backgroundImage: HATCH }} />
          too few trades
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm outline outline-2 outline-offset-[-2px] outline-[var(--foreground)]" />
          chosen values
        </span>
      </div>
    </div>
  );
}
