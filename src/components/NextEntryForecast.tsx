/**
 * Next Entry Forecast card: directional readiness, the exact next-candle close
 * that would fire each entry rule, the resulting fill / stop / target, a
 * rule-by-rule checklist, and (for baskets and universes) a ranked table of
 * every asset. Forecasts are computed by the forecast worker.
 */

import { ArrowDownRight, ArrowUpRight, Crosshair, Loader2, RefreshCw, TriangleAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { proximityRank, type CheckState, type NextEntryForecast, type SideForecast, type SideStatus, type Verdict } from '../engine/nextEntrySolver';
import { requestForecasts, type ForecastResult } from '../lib/forecastClient';
import { fmtDateTime, fmtPct, fmtPrice } from '../lib/format';
import type { MarketSnapshot } from '../lib/marketSnapshot';
import type { BacktestSettings, StrategyGenome } from '../types/trading';
import { cx, Pill } from './ui';

interface NextEntryForecastProps {
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
        <span className="font-medium text-ink-2">{side} readiness</span>
        <span className="num font-semibold text-ink">
          {disabled ? 'Not traded' : forecast.status === 'holding' ? 'Position open' : `${value.toFixed(0)}%`}
        </span>
      </div>
      <div
        className={cx('h-2 overflow-hidden rounded-full', long ? 'bg-up/15' : 'bg-down/15')}
        role="meter"
        aria-label={`${side} readiness`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value}
      >
        <div className={cx('h-full rounded-full transition-[width] duration-300', long ? 'bg-up' : 'bg-down')} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function PriceRow({ label, value, hint }: { label: string; value: string; hint?: string }): JSX.Element {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1" title={hint}>
      <dt className="text-xs text-ink-3">{label}</dt>
      <dd className="num text-sm text-ink">{value}</dd>
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
    <div className="rounded-lg border border-line bg-panel p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h4 className={cx('text-sm font-semibold', long ? 'text-up' : 'text-down')}>Next {long ? 'Long' : 'Short'} Entry</h4>
        <Pill tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Pill>
      </div>

      {showPrices ? (
        <>
          <div className="num text-xl font-semibold text-ink">
            {fmtPrice(triggerPrice)}
            {status === 'blocked' && blockedNote && (
              <span className="ml-1.5 align-middle text-xs font-medium text-warn">(Blocked: {blockedNote})</span>
            )}
          </div>
          <p className="num mt-0.5 flex items-center gap-1 text-xs text-ink-3">
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
                <Arrow size={13} className={cx('shrink-0', movePct !== null && movePct < 0 ? 'text-down' : 'text-up')} />
                {fmtPct(movePct, true, 2)} from {fmtPrice(lastClose)} · next candle must close {movePct !== null && movePct < 0 ? '≤' : '≥'}{' '}
                {fmtPrice(triggerPrice)}
              </>
            )}
          </p>
          <dl className="mt-2 divide-y divide-line/60 border-t border-line/60">
            <PriceRow label="Signal trigger price" value={fmtPrice(triggerPrice)} hint="Close of the next candle needed to fire the entry rule" />
            <PriceRow
              label="Estimated fill"
              value={fmtPrice(forecast.fillPrice)}
              hint="Next-bar open execution (assumed at the trigger close) plus the configured slippage"
            />
            <PriceRow label="Stop loss" value={fmtPrice(forecast.stopLoss)} />
            <PriceRow label="Take profit" value={forecast.takeProfit === null ? 'None – trailing / signal exit' : fmtPrice(forecast.takeProfit)} />
          </dl>
        </>
      ) : (
        <p className="text-sm text-ink-3">{forecast.reason ?? 'No trigger price on the next candle.'}</p>
      )}
      {showPrices && forecast.reason && <p className="mt-2 text-xs text-ink-3">{forecast.reason}</p>}
      {forecast.conflict && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-warn">
          <TriangleAlert size={13} className="mt-0.5 shrink-0" />
          {forecast.conflict}
        </p>
      )}
    </div>
  );
}

function Checklist({ forecast }: { forecast: SideForecast }): JSX.Element | null {
  if (forecast.conditions.length === 0) return null;
  return (
    <div>
      <div className={cx('mb-1.5 text-xs font-semibold', forecast.side === 'long' ? 'text-up' : 'text-down')}>
        {forecast.side === 'long' ? 'Long' : 'Short'} rules
      </div>
      <ul className="space-y-1.5">
        {forecast.conditions.map((check) => (
          <li key={check.formula} className="flex items-start gap-2 text-sm">
            <Pill tone={CHECK_TONE[check.state]} className="mt-px w-[84px] shrink-0 justify-center">
              {CHECK_LABEL[check.state]}
            </Pill>
            <span className="min-w-0 text-ink-2">
              <span className="text-[11px] uppercase tracking-wider text-ink-3">{check.job} · </span>
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

export default function NextEntryForecast({ genome, settings, snapshot, onRefresh }: NextEntryForecastProps): JSX.Element {
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
    <section className="mx-5 mb-5 rounded-xl border border-accent/30 bg-raised/40 p-4" aria-labelledby="next-entry-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Crosshair size={16} className="text-accent-strong" />
            <h3 id="next-entry-title" className="text-sm font-semibold text-ink">
              Next Entry Forecast
            </h3>
            {forecast && <Pill tone={VERDICT_TONE[forecast.verdict]}>{forecast.verdict}</Pill>}
            {(computing || refreshing) && <Loader2 size={14} className="animate-spin text-ink-3" aria-label="Updating" />}
          </div>
          {forecast && snapshot && (
            <p className="num mt-1 text-xs text-ink-3">
              Last close {fmtPrice(forecast.lastClose)} · candle of {fmtDateTime(forecast.lastTime)} · ATR(14) {fmtPrice(forecast.atr)} ·{' '}
              {snapshot.simulated ? 'simulated prices' : snapshot.venue} · fetched {fmtDateTime(snapshot.fetchedAt)}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={refreshing || !snapshot}
          className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-panel px-3 text-xs font-medium text-ink-2 transition-colors hover:bg-hover hover:text-ink disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw size={13} className={cx(refreshing && 'animate-spin')} />
          {refreshing ? 'Fetching…' : 'Refresh Latest Price'}
        </button>
      </div>

      {snapshot?.simulated && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-warn">
          <TriangleAlert size={13} className="mt-0.5 shrink-0" />
          Live candles were unavailable, so these levels come from simulated prices and are not tradable.
        </p>
      )}
      {error && (
        <p role="alert" className="mt-2 text-xs text-down">
          {error}
        </p>
      )}
      {primary?.error && <p className="mt-2 text-xs text-down">{primary.error}</p>}

      {!forecast ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-ink-3">
          {snapshot ? (
            <>
              <Loader2 size={14} className="animate-spin" /> Solving the next-candle trigger prices…
            </>
          ) : (
            'Latest candles are not available for this run.'
          )}
        </p>
      ) : (
        <div className="mt-3 space-y-4">
          {forecast.position && (
            <div className="rounded-lg border border-accent/40 bg-accent/10 p-3 text-sm">
              <div className="font-semibold text-ink">
                In a {forecast.position.dir} position since {fmtDateTime(forecast.position.entryTime)} at {fmtPrice(forecast.position.entryPrice)}
                <span className={cx('num ml-2', forecast.position.unrealisedPct >= 0 ? 'text-up' : 'text-down')}>
                  {fmtPct(forecast.position.unrealisedPct, true, 2)}
                </span>
              </div>
              <dl className="num mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-ink-2 sm:grid-cols-4">
                <div>
                  <dt className="text-ink-3">Stop for next candle</dt>
                  <dd className="text-ink">{fmtPrice(forecast.position.stopLoss)}</dd>
                </div>
                <div>
                  <dt className="text-ink-3">Take profit</dt>
                  <dd className="text-ink">{forecast.position.takeProfit === null ? 'None' : fmtPrice(forecast.position.takeProfit)}</dd>
                </div>
                <div>
                  <dt className="text-ink-3">Signal exit / reversal at close</dt>
                  <dd className="text-ink">
                    {forecast.position.signalExitPrice === null ? 'None in range' : fmtPrice(forecast.position.signalExitPrice)}
                    {forecast.position.signalExitSource && <span className="text-ink-3"> ({forecast.position.signalExitSource})</span>}
                  </dd>
                </div>
                <div>
                  <dt className="text-ink-3">Time stop</dt>
                  <dd className="text-ink">{forecast.position.timeStopIn === null ? 'None' : `${forecast.position.timeStopIn} candles left`}</dd>
                </div>
              </dl>
              {forecast.position.exitAtNextOpen && <p className="mt-1.5 text-xs font-medium text-warn">{forecast.position.exitAtNextOpen}</p>}
            </div>
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
              <div className="mb-1.5 text-xs font-semibold text-ink-2">All assets, closest to an entry first</div>
              <div className="overflow-x-auto rounded-lg border border-line">
                <table className="num w-full text-sm">
                  <thead>
                    <tr className="border-b border-line bg-raised/50 text-[11px] uppercase tracking-wider text-ink-3">
                      <th scope="col" className="px-3 py-2 text-left font-medium">Asset</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Last close</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Verdict</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Long</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium">Short</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Nearest trigger</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ranked.map(({ key, forecast: f }) => {
                      const asset = snapshot?.assets.find((a) => a.key === key);
                      return (
                        <tr key={key} className="border-b border-line/70 last:border-b-0">
                          <th scope="row" className="whitespace-nowrap px-3 py-1.5 text-left font-normal text-ink">
                            {asset?.label ?? key}
                            {asset?.primary && <span className="ml-1.5 text-[11px] text-ink-3">(traded)</span>}
                          </th>
                          <td className="px-3 py-1.5 text-right text-ink-2">{fmtPrice(f.lastClose)}</td>
                          <td className="px-3 py-1.5">
                            <Pill tone={VERDICT_TONE[f.verdict]}>{f.verdict}</Pill>
                          </td>
                          <td className="px-3 py-1.5 text-right text-ink-2">{tableReadiness(f.long)}</td>
                          <td className="px-3 py-1.5 text-right text-ink-2">{tableReadiness(f.short)}</td>
                          <td className="whitespace-nowrap px-3 py-1.5 text-ink-2">{nearest(f)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <p className="text-[11px] leading-relaxed text-ink-3">
            Trigger prices are solved on the strategy's own rules: the next candle is simulated opening at the last close and closing at
            each candidate price, then bisected to the exact close where the rule turns true. Fills assume the following open matches that
            close, slipped by the backtester's own rule ({settings.slippagePct}% fixed
            {settings.slippageAtrMult ? `, or ${settings.slippageAtrMult} × ATR(14)% when that is larger` : ''}). {forecast.notes.join(' ')}
          </p>
        </div>
      )}
    </section>
  );
}
