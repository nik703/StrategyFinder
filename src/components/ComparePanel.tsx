/**
 * Side-by-side comparison of up to three leaderboard strategies (a shadcn Card): their equity curves
 * on one chart and the key risk / return figures and robustness checks in one shadcn Table.
 */

import { GitCompareArrows, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { exitModeLabel, exitTags, indicatorBadges, shortRuleLabel } from '@/engine/describe';
import { HYPOTHESES } from '@/engine/genome';
import { fmtInt, fmtNum, fmtPct } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { StrategyResult } from '@/types/trading';
import { Panel, Pill } from './common';

/** Series colours of the compared strategies: the theme's chart tokens. */
export const COMPARE_COLORS = ['var(--chart-1)', 'var(--chart-4)', 'var(--chart-5)'];

interface Row {
  label: string;
  title?: string;
  value: (s: StrategyResult) => string;
  /** Index of the best value, for highlighting (higher is better unless `lower`). */
  score?: (s: StrategyResult) => number;
  lower?: boolean;
}

const ROWS: Row[] = [
  { label: 'Beats luck (DSR)', title: 'Deflated Sharpe: chance the validation result is real after every strategy tried', value: (s) => fmtPct(s.dsr * 100, false, 0), score: (s) => s.dsr },
  {
    label: 'Robustness checks',
    title: 'Gates passed: 1 effective sample + median-fold return + fold drawdown cap · 2 deflated Sharpe · 3 Monte Carlo stress + neighbour stability',
    value: (s) => (s.robust ? `${[s.robust.sample, s.robust.dsr, s.robust.stress === true].filter(Boolean).length} / 3` : '–'),
    score: (s) => (s.robust ? [s.robust.sample, s.robust.dsr, s.robust.stress === true].filter(Boolean).length : 0),
  },
  {
    label: 'Stress test: P(profit)',
    title: 'Share of 500 Monte Carlo replays (10% of trades skipped, shuffled order, slippage shocks) that end in profit',
    value: (s) => (s.monteCarlo ? fmtPct(s.monteCarlo.probProfit * 100, false, 0) : '–'),
    score: (s) => s.monteCarlo?.probProfit ?? -1,
  },
  {
    label: 'Stress test: 95% worst DD',
    title: '95th-percentile max drawdown of the Monte Carlo replays',
    value: (s) => (s.monteCarlo ? fmtPct(s.monteCarlo.drawdownP95) : '–'),
    score: (s) => s.monteCarlo?.drawdownP95 ?? Infinity,
    lower: true,
  },
  { label: 'Validation Sharpe', value: (s) => fmtNum(s.validation.sharpe), score: (s) => s.validation.sharpe },
  { label: 'Sharpe (in-sample)', value: (s) => fmtNum(s.search.sharpe), score: (s) => s.search.sharpe },
  { label: 'Sortino (in-sample)', value: (s) => fmtNum(s.search.sortino), score: (s) => s.search.sortino },
  { label: 'CAGR (in-sample)', value: (s) => fmtPct(s.search.cagr, true), score: (s) => s.search.cagr },
  { label: 'Max drawdown', value: (s) => fmtPct(s.search.maxDrawdown), score: (s) => s.search.maxDrawdown, lower: true },
  { label: 'Trades (validation)', value: (s) => `${fmtInt(s.search.trades)} (${fmtInt(s.validation.trades)})`, score: (s) => s.search.trades },
  { label: 'Win rate', value: (s) => fmtPct(s.search.winRate) },
  { label: 'Profit factor', value: (s) => fmtNum(s.search.profitFactor), score: (s) => s.search.profitFactor },
  { label: 'Parameters', value: (s) => String(s.params), score: (s) => s.params, lower: true },
];

/** Equity curves of the compared strategies, rebased to 100 at the start. */
function OverlayChart({ strategies }: { strategies: StrategyResult[] }): JSX.Element {
  const W = 560;
  const H = 150;
  const series = strategies.map((s) => {
    const first = s.equity[0] || 1;
    return Array.from(s.equity, (v) => (v / first) * 100);
  });
  let lo = Infinity;
  let hi = -Infinity;
  for (const values of series) for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;
  const path = (values: number[]): string => {
    const step = Math.max(1, Math.floor(values.length / 200));
    const pts: string[] = [];
    for (let i = 0; i < values.length; i += step) pts.push(`${((i / (values.length - 1)) * (W - 4) + 2).toFixed(1)},${(H - 4 - ((values[i] - lo) / span) * (H - 8)).toFixed(1)}`);
    return `M${pts.join(' L')}`;
  };
  const base = H - 4 - ((100 - lo) / span) * (H - 8);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Equity curves of the compared strategies, rebased to 100">
      {base > 0 && base < H && <line x1="0" x2={W} y1={base} y2={base} stroke="var(--muted-foreground)" strokeDasharray="4 4" strokeWidth="1" />}
      {series.map((values, k) => (
        <path key={strategies[k].id} d={path(values)} fill="none" stroke={COMPARE_COLORS[k]} strokeWidth="1.8" strokeLinejoin="round" />
      ))}
    </svg>
  );
}

export default function ComparePanel({
  strategies,
  onRemove,
  onOpen,
  onClear,
}: {
  strategies: StrategyResult[];
  onRemove: (id: string) => void;
  onOpen: (strategy: StrategyResult) => void;
  onClear: () => void;
}): JSX.Element {
  const best = (row: Row): number | null => {
    if (!row.score || strategies.length < 2) return null;
    const values = strategies.map(row.score);
    let at = 0;
    for (let k = 1; k < values.length; k++) if (row.lower ? values[k] < values[at] : values[k] > values[at]) at = k;
    return values.every((v) => v === values[at]) ? null : at;
  };
  return (
    <Panel>
      <CardHeader className="border-b px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <GitCompareArrows className="size-4 text-muted-foreground" />
            <CardTitle className="text-sm font-semibold">Compare strategies</CardTitle>
            <CardDescription className="text-xs">Search period (train + validation) · the best value of each row is highlighted</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={onClear}>
            Clear
          </Button>
        </div>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
        <div>
          <OverlayChart strategies={strategies} />
          <p className="mt-1 text-[11px] text-muted-foreground">Equity rebased to 100 at the start; the dashed line is the starting capital.</p>
        </div>
        <div className="rounded-lg border border-border">
          <Table className="num">
            <TableHeader>
              <TableRow className="bg-muted/50 text-[11px] uppercase tracking-wider hover:bg-muted/50">
                <TableHead className="text-muted-foreground">Metric</TableHead>
                {strategies.map((s, k) => (
                  <TableHead key={s.id} className="text-right">
                    <span className="inline-flex items-center gap-1">
                      <span className="inline-block h-0 w-3 border-t-2" style={{ borderColor: COMPARE_COLORS[k] }} />
                      <Button variant="link" size="xs" onClick={() => onOpen(s)} className="h-auto px-0 normal-case text-foreground" title="Open in the workbench">
                        {s.id}
                      </Button>
                      <Button variant="ghost" size="icon-xs" onClick={() => onRemove(s.id)} aria-label={`Remove ${s.id} from the comparison`}>
                        <X />
                      </Button>
                    </span>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableHead scope="row" className="font-normal text-secondary-foreground">
                  Idea
                </TableHead>
                {strategies.map((s) => (
                  <TableCell key={s.id} className="whitespace-normal text-right">
                    <div className="flex flex-wrap justify-end gap-1">
                      {indicatorBadges(s.genome).map((badge) => (
                        <Pill key={badge}>{badge}</Pill>
                      ))}
                    </div>
                    <div className="mt-0.5 text-[11px] text-muted-foreground">
                      {s.genome.hypothesis ? HYPOTHESES[s.genome.hypothesis].short : s.genome.archetype} · {shortRuleLabel(s.genome)} · {exitModeLabel(s.genome)}
                      {exitTags(s.genome).length > 0 ? ` · ${exitTags(s.genome).join(' · ')}` : ''}
                    </div>
                  </TableCell>
                ))}
              </TableRow>
              {ROWS.map((row) => {
                const winner = best(row);
                return (
                  <TableRow key={row.label}>
                    <TableHead scope="row" title={row.title} className="font-normal text-secondary-foreground">
                      {row.label}
                    </TableHead>
                    {strategies.map((s, k) => (
                      <TableCell key={s.id} className={cn('text-right', winner === k ? 'font-semibold text-up' : 'text-foreground')}>
                        {row.value(s)}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Panel>
  );
}
