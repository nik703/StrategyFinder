/**
 * Next Entry Forecast (a shadcn Card): Long vs Short readiness Progress gauges, the directional
 * verdict Badge, the exact next-candle close that would fire each entry rule with the resulting
 * fill / stop / target summary Cards, a rule-by-rule checklist, a Refresh Latest Price Button and
 * (for baskets and asset pools) a ranked Table of every asset. Forecasts are computed by the
 * forecast worker; this file only renders them.
 */

import { ArrowDownRight, ArrowUpRight, Crosshair, Loader2, RefreshCw, TriangleAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { proximityRank, type CheckState, type NextEntryForecast, type SideForecast, type SideStatus, type Verdict } from '@/engine/nextEntrySolver';
import { requestForecasts, type ForecastResult } from '@/lib/forecastClient';
import { fmtDateTime, fmtPct, fmtPrice } from '@/lib/format';
import type { MarketSnapshot } from '@/lib/marketSnapshot';
import { cn } from '@/lib/utils';
import type { BacktestSettings, StrategyGenome } from '@/types/trading';
import { Pill } from './common';

interface NextEntryForecastProps {
  /** Extra classes for the card (the margin around it). */
  className?: string;
  genome: StrategyGenome;
  settings: BacktestSettings;
  snapshot: MarketSnapshot | null;
  /** Fetch the latest candles; resolves when the snapshot has been replaced. */
  onRefresh: () => Promise<void>;
}

const VERDICT_TONE: Record<Verdict, 'up' | 'down' | 'neutral' | 'accent'> = {
  'Strong Long': 'up',
  'Leaning Long': 'up',
  'Neutral / Waiting': 'neutral',
  'Leaning Short': 'down',
  'Strong Short': 'down',
  'In Position': 'accent',
};

const STATUS_LABEL: Record<SideStatus, string> = {
  pending: 'Entry at next open',
  now: 'Fires at current price',
  trigger: 'Waiting for trigger',
  blocked: 'Blocked by a filter',
  unreachable: 'Out of range',
  holding: 'Position open',
  disabled: 'Not traded',
};

const STATUS_TONE: Record<SideStatus, 'up' | 'down' | 'warn' | 'neutral' | 'accent'> = {
  pending: 'up',
  now: 'up',
  trigger: 'accent',
  blocked: 'warn',
  unreachable: 'neutral',
  holding: 'accent',
  disabled: 'neutral',
};

const CHECK_LABEL: Record<CheckState, string> = { met: 'Met', pending: 'Pending', blocked: 'Blocked', reset: 'Needs reset' };
const CHECK_TONE: Record<CheckState, 'up' | 'warn' | 'down' | 'neutral'> = { met: 'up', pending: 'warn', blocked: 'down', reset: 'neutral' };

/** The checklist text without its trailing "[Status]" tag (the pill shows it). */
const stripTag = (text: string): string => text.replace(/\s*\[[^\]]+\]$/, '');

function ReadinessMeter({ side, forecast }: { side: 'Long' | 'Short'; forecast: SideForecast }): JSX.Element {
  const disabled = forecast.status === 'disabled';
  const value = disabled ? 0 : forecast.readiness;
  const long = side === 'Long';
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="font-medium text-secondary-foreground">{side} readiness</span>
        <span className="num font-semibold text-foreground">
          {disabled ? 'Not traded' : forecast.status === 'holding' ? 'Position open' : `${value.toFixed(0)}%`}
        </span>
      </div>
      <Progress
        value={value}
        aria-label={`${side} readiness`}
        className={cn('h-2', long ? 'bg-up/15 *:data-[slot=progress-indicator]:bg-up' : 'bg-down/15 *:data-[slot=progress-indicator]:bg-down')}
      />
    </div>
  );
}

function PriceRow({ label, value, hint }: { label: string; value: string; hint?: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1" title={hint}>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="num text-sm text-foreground">{value}</dd>
    </div>
  );
}

function SideCard({ forecast, lastClose }: { forecast: SideForecast; lastClose: number }): JSX.Element {
  const long = forecast.side === 'long';
  const { status, triggerPrice, movePct } = forecast;
  const showPrices = triggerPrice !== null && status !== 'holding' && status !== 'disabled';
  const blockedNote = forecast.blockers.length > 0 ? forecast.blockers.map((b) => b.detail).join('; ') : null;
  const Arrow = movePct !== null && movePct < 0 ? ArrowDownRight : ArrowUpRight;

  return (
    <Card size="sm" className="gap-2">
      <CardHeader>
        <CardTitle className={cn('text-sm font-semibold', long ? 'text-up' : 'text-down')}>Next {long ? 'Long' : 'Short'} Entry</CardTitle>
        <CardAction>
          <Pill tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Pill>
        </CardAction>
      </CardHeader>
      <CardContent>

      {showPrices ? (
        <>
          <div className="num text-xl font-semibold text-foreground">
            {fmtPrice(triggerPrice)}
            {status === 'blocked' && blockedNote && (
              <span className="ml-1.5 align-middle text-xs font-medium text-warn">(Blocked: {blockedNote})</span>
            )}
          </div>
          <p className="num mt-0.5 flex items-center gap-1 text-xs text-muted-foreground">
            {status === 'pending' ? (
              'Signal fired on the last close: fills at the next open'
            ) : status === 'now' ? (
              <>
                Fires if the next candle closes
                {forecast.activeRange?.[0] != null && <> above {fmtPrice(forecast.activeRange[0])}</>}
                {forecast.activeRange?.[0] != null && forecast.activeRange?.[1] != null && ' and'}
                {forecast.activeRange?.[1] != null && <> below {fmtPrice(forecast.activeRange[1])}</>}
                {forecast.activeRange?.[0] == null && forecast.activeRange?.[1] == null && ' anywhere near the current price'}
              </>
            ) : (
              <>
                <Arrow className={cn('size-3.5 shrink-0', movePct !== null && movePct < 0 ? 'text-down' : 'text-up')} />
                {fmtPct(movePct, true, 2)} from {fmtPrice(lastClose)} · next candle must close {movePct !== null && movePct < 0 ? '≤' : '≥'}{' '}
                {fmtPrice(triggerPrice)}
              </>
            )}
          </p>
          <dl className="mt-2 divide-y divide-border/60 border-t border-border/60">
            <PriceRow label="Signal trigger price" value={fmtPrice(triggerPrice)} hint="Close of the next candle needed to fire the entry rule" />
            <PriceRow
              label="Estimated fill"
              value={fmtPrice(forecast.fillPrice)}
              hint="Next-bar open execution (assumed at the trigger close) plus the configured slippage"
            />
            <PriceRow
              label={forecast.stopMode === 'trailing' ? 'Stop loss (initial, trailing)' : 'Stop loss'}
              value={fmtPrice(forecast.stopLoss)}
              hint={
                forecast.stopMode === 'trailing'
                  ? `Ratchets ${forecast.trailAtrMult} × ATR behind the best price after every close, and never moves back`
                  : 'Fixed for the life of the trade (unless the breakeven lock moves it to the entry)'
              }
            />
            <PriceRow label="Take profit" value={forecast.takeProfit === null ? 'None – trailing / signal exit' : fmtPrice(forecast.takeProfit)} />
            {forecast.breakevenAt !== null && (
              <PriceRow label="Breakeven lock at" value={fmtPrice(forecast.breakevenAt)} hint="At +1R the stop moves to the entry price, from the next candle on" />
            )}
          </dl>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">{forecast.reason ?? 'No trigger price on the next candle.'}</p>
      )}
      {showPrices && forecast.reason && <p className="mt-2 text-xs text-muted-foreground">{forecast.reason}</p>}
      {forecast.conflict && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-warn">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
          {forecast.conflict}
        </p>
      )}
      </CardContent>
    </Card>
  );
}

function Checklist({ forecast }: { forecast: SideForecast }): JSX.Element | null {
  if (forecast.conditions.length === 0) return null;
  return (
    <div>
      <div className={cn('mb-1.5 text-xs font-semibold', forecast.side === 'long' ? 'text-up' : 'text-down')}>
        {forecast.side === 'long' ? 'Long' : 'Short'} rules
      </div>
      <ul className="space-y-1.5">
        {forecast.conditions.map((check) => (
          <li key={check.formula} className="flex items-start gap-2 text-sm">
            <Pill tone={CHECK_TONE[check.state]} className="mt-px w-[84px] shrink-0 justify-center">
              {CHECK_LABEL[check.state]}
            </Pill>
            <span className="min-w-0 text-secondary-foreground">
              <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{check.job} · </span>
              <span className="num">{stripTag(check.text)}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Readiness cell: blank when the side is not traded, "Holding" when its position is already open. */
function tableReadiness(side: SideForecast): string {
  if (side.status === 'disabled') return '—';
  if (side.status === 'holding') return 'Holding';
  return `${side.readiness.toFixed(0)}%`;
}

function nearest(f: NextEntryForecast): string {
  const sides = [f.long, f.short].filter((s) => s.status !== 'disabled' && s.status !== 'holding');
  const live = sides.find((s) => s.status === 'pending' || s.status === 'now');
  if (live) return `${live.side === 'long' ? 'Long' : 'Short'} · ${STATUS_LABEL[live.status].toLowerCase()}`;
  const priced = sides
    .filter((s) => s.triggerPrice !== null && s.movePct !== null)
    .sort((a, b) => Math.abs(a.movePct as number) - Math.abs(b.movePct as number))[0];
  if (!priced) return '—';
  return `${priced.side === 'long' ? 'Long' : 'Short'} ${fmtPrice(priced.triggerPrice)} (${fmtPct(priced.movePct, true, 2)})${priced.status === 'blocked' ? ' · blocked' : ''}`;
}

export default function NextEntryForecast({ className, genome, settings, snapshot, onRefresh }: NextEntryForecastProps): JSX.Element {
  const [results, setResults] = useState<ForecastResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [computing, setComputing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (!snapshot) return;
    let cancelled = false;
    setComputing(true);
    setError(null);
    requestForecasts(snapshot.assets.map((asset) => ({ key: asset.key, data: asset.data, genome, settings })))
      .then((next) => {
        if (!cancelled) setResults(next);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : 'The forecast could not be computed.');
      })
      .finally(() => {
        if (!cancelled) setComputing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [snapshot, genome, settings]);

  const primaryKey = snapshot?.assets.find((asset) => asset.primary)?.key;
  const primary = results?.find((result) => result.key === primaryKey) ?? null;
  const forecast = primary?.forecast ?? null;
  const ranked = useMemo(() => {
    if (!results || !snapshot || snapshot.assets.length < 2) return [];
    return results
      .filter((r): r is ForecastResult & { forecast: NextEntryForecast } => r.forecast !== null)
      .sort((a, b) => proximityRank(a.forecast) - proximityRank(b.forecast));
  }, [results, snapshot]);

  const refresh = async (): Promise<void> => {
    setRefreshing(true);
    try {
      await onRefresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not fetch the latest candles.');
    } finally {
      setRefreshing(false);
    }
  };

  return (
    <Card className={cn('gap-0 py-4 ring-primary/30', className)} aria-labelledby="next-entry-title">
      <CardHeader className="px-4">
        <CardTitle id="next-entry-title" className="flex flex-wrap items-center gap-2 text-sm font-semibold">
          <Crosshair className="size-4 text-primary" />
          Next Entry Forecast
          {forecast && <Pill tone={VERDICT_TONE[forecast.verdict]}>{forecast.verdict}</Pill>}
          {(computing || refreshing) && <Loader2 className="size-3.5 animate-spin text-muted-foreground" aria-label="Updating" />}
        </CardTitle>
        {forecast && snapshot && (
          <CardDescription className="num text-xs">
            Last close {fmtPrice(forecast.lastClose)} · candle of {fmtDateTime(forecast.lastTime)} · ATR(14) {fmtPrice(forecast.atr)} ·{' '}
            {snapshot.simulated ? 'simulated prices' : snapshot.venue} · fetched {fmtDateTime(snapshot.fetchedAt)}
          </CardDescription>
        )}
        <CardAction>
          <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={refreshing || !snapshot}>
            <RefreshCw className={cn(refreshing && 'animate-spin')} />
            {refreshing ? 'Fetching…' : 'Refresh Latest Price'}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="px-4">

      {snapshot?.simulated && (
        <Alert className="mt-3 border-warn/40 bg-warn/10 text-warn">
          <TriangleAlert />
          <AlertDescription className="text-current">Live candles were unavailable, so these levels come from simulated prices and are not tradable.</AlertDescription>
        </Alert>
      )}
      {(error || primary?.error) && (
        <Alert variant="destructive" className="mt-3">
          <TriangleAlert />
          <AlertDescription>{error ?? primary?.error}</AlertDescription>
        </Alert>
      )}

      {!forecast ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
          {snapshot ? (
            <>
              <Loader2 className="size-3.5 animate-spin" /> Solving the next-candle trigger prices…
            </>
          ) : (
            'Latest candles are not available for this run.'
          )}
        </p>
      ) : (
        <div className="mt-3 space-y-4">
          {forecast.position && (
            <Alert className="block border-primary/40 bg-primary/10 p-3 text-sm">
              <div className="font-semibold text-foreground">
                In a {forecast.position.dir} position since {fmtDateTime(forecast.position.entryTime)} at {fmtPrice(forecast.position.entryPrice)}
                <span className={cn('num ml-2', forecast.position.unrealisedPct >= 0 ? 'text-up' : 'text-down')}>
                  {fmtPct(forecast.position.unrealisedPct, true, 2)}
                </span>
              </div>
              <dl className="num mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-secondary-foreground sm:grid-cols-4">
                <div>
                  <dt className="text-muted-foreground">Stop for next candle</dt>
                  <dd className="text-foreground">{fmtPrice(forecast.position.stopLoss)}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Take profit</dt>
                  <dd className="text-foreground">{forecast.position.takeProfit === null ? 'None' : fmtPrice(forecast.position.takeProfit)}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Signal exit / reversal at close</dt>
                  <dd className="text-foreground">
                    {forecast.position.signalExitPrice === null ? 'None in range' : fmtPrice(forecast.position.signalExitPrice)}
                    {forecast.position.signalExitSource && <span className="text-muted-foreground"> ({forecast.position.signalExitSource})</span>}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Time stop</dt>
                  <dd className="text-foreground">{forecast.position.timeStopIn === null ? 'None' : `${forecast.position.timeStopIn} candles left`}</dd>
                </div>
              </dl>
              {(forecast.position.breakevenLocked || forecast.position.breakevenAt !== null) && (
                <p className="num mt-1.5 text-xs text-secondary-foreground">
                  Breakeven lock:{' '}
                  {forecast.position.breakevenLocked ? (
                    <span className="font-medium text-up">active – the stop sits at the entry price, this trade can no longer become a full loss</span>
                  ) : (
                    <>
                      moves the stop to the entry once price reaches <span className="font-medium text-foreground">{fmtPrice(forecast.position.breakevenAt)}</span> (+1R)
                    </>
                  )}
                </p>
              )}
              {forecast.position.exitAtNextOpen && <p className="mt-1.5 text-xs font-medium text-warn">{forecast.position.exitAtNextOpen}</p>}
            </Alert>
          )}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <ReadinessMeter side="Long" forecast={forecast.long} />
            <ReadinessMeter side="Short" forecast={forecast.short} />
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <SideCard forecast={forecast.long} lastClose={forecast.lastClose} />
            <SideCard forecast={forecast.short} lastClose={forecast.lastClose} />
          </div>

          <div className="space-y-3">
            <Checklist forecast={forecast.long} />
            <Checklist forecast={forecast.short} />
          </div>

          {ranked.length > 1 && (
            <div>
              <div className="mb-1.5 text-xs font-semibold text-secondary-foreground">All assets, closest to an entry first</div>
              <div className="rounded-lg border border-border">
                <Table className="num">
                  <TableHeader>
                    <TableRow className="bg-muted/50 text-[11px] uppercase tracking-wider hover:bg-muted/50">
                      <TableHead>Asset</TableHead>
                      <TableHead className="text-right">Last close</TableHead>
                      <TableHead>Verdict</TableHead>
                      <TableHead className="text-right">Long</TableHead>
                      <TableHead className="text-right">Short</TableHead>
                      <TableHead>Nearest trigger</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ranked.map(({ key, forecast: f }) => {
                      const asset = snapshot?.assets.find((a) => a.key === key);
                      return (
                        <TableRow key={key}>
                          <TableHead scope="row" className="font-normal text-foreground">
                            {asset?.label ?? key}
                            {asset?.primary && <span className="ml-1.5 text-[11px] text-muted-foreground">(traded)</span>}
                          </TableHead>
                          <TableCell className="text-right text-secondary-foreground">{fmtPrice(f.lastClose)}</TableCell>
                          <TableCell>
                            <Pill tone={VERDICT_TONE[f.verdict]}>{f.verdict}</Pill>
                          </TableCell>
                          <TableCell className="text-right text-secondary-foreground">{tableReadiness(f.long)}</TableCell>
                          <TableCell className="text-right text-secondary-foreground">{tableReadiness(f.short)}</TableCell>
                          <TableCell className="text-secondary-foreground">{nearest(f)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Trigger prices are solved on the strategy's own rules: the next candle is simulated opening at the last close and closing at
            each candidate price, then bisected to the exact close where the rule turns true. Fills assume the following open matches that
            close, slipped by the backtester's own rule ({settings.slippagePct}% fixed
            {settings.slippageAtrMult ? `, or ${settings.slippageAtrMult} × ATR(14)% when that is larger` : ''}). {forecast.notes.join(' ')}
          </p>
        </div>
      )}
      </CardContent>
    </Card>
  );
}
