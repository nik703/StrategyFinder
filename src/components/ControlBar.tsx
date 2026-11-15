/**
 * Top control bar: asset / basket builder, timeframe, history range, capital,
 * costs, trade mode, position sizing, funding, the Robust universe and the
 * Start / Stop buttons.
 */

import { Info, Loader2, Play, Plus, Square, X } from 'lucide-react';
import { useId } from 'react';
import { formatWeight, normalizeWeights, TICKER_PATTERN } from '../data/basket';
import {
  DEFAULT_FUNDING_PCT_8H,
  isCrypto,
  MAX_BASKET_ASSETS,
  MAX_UNIVERSE_ASSETS,
  parseUniverse,
  PRESET_TICKERS,
  type BasketRow,
  type RunConfig,
} from '../lib/config';
import {
  HISTORY_RANGES,
  INTERVALS,
  MAX_MAX_STAGNATION,
  MAX_RANGE,
  MIN_MAX_STAGNATION,
  rangeAllowed,
  YAHOO_MAX_DAYS,
  type EngineStatus,
  type HistoryRange,
  type Interval,
  type TradeMode,
} from '../types/trading';
import { cx, FieldLabel, NumberField, Panel, Segmented } from './ui';

interface ControlBarProps {
  config: RunConfig;
  onChange: (patch: Partial<RunConfig>) => void;
  /** Custom tickers remembered from earlier runs. */
  savedTickers: string[];
  onSaveTickers: (symbols: string[]) => void;
  onRemoveTicker: (symbol: string) => void;
  status: EngineStatus;
  onStart: () => void;
  onStop: () => void;
}

/** Colours for the allocation bar (identity only – not a chart series). */
const ALLOCATION_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];

const INTERVAL_OPTIONS: { value: Interval; label: string }[] = INTERVALS.map((value) => ({ value, label: value }));
function rangeOptions(interval: Interval): { value: HistoryRange; label: string; disabled: boolean; title?: string }[] {
  return HISTORY_RANGES.map((value) => {
    const allowed = rangeAllowed(interval, value);
    return {
      value,
      label: value,
      disabled: !allowed,
      title: allowed ? undefined : `${interval} candles are limited to ${MAX_RANGE[interval]} of history`,
    };
  });
}
const MODE_OPTIONS: { value: TradeMode; label: string }[] = [
  { value: 'long-only', label: 'Long-Only' },
  { value: 'long-short', label: 'Long/Short' },
];
const SIZING_OPTIONS: { value: RunConfig['sizingMode']; label: string }[] = [
  { value: 'fixed', label: 'Risk % · fixed' },
  { value: 'compound', label: 'Risk % · compounding' },
  { value: 'full', label: '100% of equity' },
];
const ASSET_MODE_OPTIONS: { value: RunConfig['assetMode']; label: string }[] = [
  { value: 'single', label: 'Single asset' },
  { value: 'basket', label: 'Weighted basket' },
];

function TickerInput({
  value,
  onChange,
  disabled,
  listId,
  ariaLabel,
  className,
  placeholder = 'Ticker',
  onCommit,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  listId: string;
  ariaLabel: string;
  className?: string;
  placeholder?: string;
  /** Called with a valid ticker when the user presses Enter. */
  onCommit?: (ticker: string) => void;
}): JSX.Element {
  const invalid = value.trim() !== '' && !TICKER_PATTERN.test(value.trim().toUpperCase());
  return (
    <input
      type="text"
      value={value}
      list={listId}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-invalid={invalid}
      spellCheck={false}
      autoCapitalize="characters"
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value.toUpperCase().replace(/\s+/g, ''))}
      onKeyDown={(event) => {
        const ticker = value.trim().toUpperCase();
        if (event.key === 'Enter' && onCommit && TICKER_PATTERN.test(ticker)) {
          event.preventDefault();
          onCommit(ticker);
        }
      }}
      title={onCommit ? 'Press Enter to save this ticker for next time' : undefined}
      className={cx(
        'num h-9 rounded-lg border bg-base px-2.5 text-sm text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-accent disabled:cursor-not-allowed disabled:opacity-60',
        invalid ? 'border-down/70' : 'border-line',
        className,
      )}
    />
  );
}

function BasketBuilder({
  rows,
  onChange,
  disabled,
  listId,
  onCommit,
}: {
  rows: BasketRow[];
  onChange: (rows: BasketRow[]) => void;
  disabled: boolean;
  listId: string;
  onCommit: (ticker: string) => void;
}): JSX.Element {
  const normalized = normalizeWeights(rows);
  const weightOf = (symbol: string): number | null =>
    normalized.find((component) => component.symbol === symbol.trim().toUpperCase())?.weight ?? null;
  const rawTotal = rows.reduce((sum, row) => sum + (Number.isFinite(row.weight) && row.weight > 0 ? row.weight : 0), 0);

  const update = (id: number, patch: Partial<BasketRow>): void =>
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));

  const add = (): void => {
    const used = new Set(rows.map((row) => row.symbol));
    const next = PRESET_TICKERS.find((ticker) => !used.has(ticker)) ?? '';
    const id = rows.reduce((max, row) => Math.max(max, row.id), 0) + 1;
    onChange([...rows, { id, symbol: next, weight: 10 }]);
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {rows.map((row) => {
          const share = weightOf(row.symbol);
          return (
            <div key={row.id} className="flex items-center gap-1.5 rounded-lg border border-line bg-raised/60 p-1.5">
              <TickerInput
                value={row.symbol}
                onChange={(symbol) => update(row.id, { symbol })}
                disabled={disabled}
                listId={listId}
                onCommit={onCommit}
                ariaLabel="Basket asset ticker"
                className="w-[104px]"
              />
              <div className="flex h-9 items-center rounded-lg border border-line bg-base px-2 text-sm focus-within:border-accent">
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={Number.isFinite(row.weight) ? row.weight : ''}
                  disabled={disabled}
                  aria-label={`Weight for ${row.symbol || 'asset'}`}
                  onChange={(event) => update(row.id, { weight: event.target.value === '' ? NaN : Number(event.target.value) })}
                  className="num w-12 bg-transparent text-right text-ink outline-none disabled:cursor-not-allowed disabled:opacity-60"
                />
                <span className="ml-1 text-ink-3">%</span>
              </div>
              <span
                className="num w-[68px] whitespace-nowrap text-right text-xs text-ink-2"
                title="Normalised allocation (weights are rescaled to sum to 100%)"
              >
                {share === null ? '–' : `→ ${formatWeight(share)}`}
              </span>
              <button
                type="button"
                onClick={() => onChange(rows.filter((r) => r.id !== row.id))}
                disabled={disabled || rows.length <= 1}
                aria-label={`Remove ${row.symbol || 'asset'} from basket`}
                className="grid h-7 w-7 place-items-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
              >
                <X size={14} />
              </button>
            </div>
          );
        })}
        <button
          type="button"
          onClick={add}
          disabled={disabled || rows.length >= MAX_BASKET_ASSETS}
          className="inline-flex h-[50px] items-center gap-1.5 rounded-lg border border-dashed border-line-strong px-3 text-sm text-ink-2 transition-colors hover:border-accent hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Plus size={14} /> Add asset
        </button>
      </div>

      {normalized.length > 0 && (
        <div>
          <div className="flex h-2 gap-0.5 overflow-hidden rounded-full" role="img" aria-label="Normalised basket allocation">
            {normalized.map((component, index) => (
              <div
                key={component.symbol}
                style={{ width: `${component.weight * 100}%`, backgroundColor: ALLOCATION_COLORS[index % ALLOCATION_COLORS.length] }}
                title={`${component.symbol} ${formatWeight(component.weight)}`}
              />
            ))}
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2">
            {normalized.map((component, index) => (
              <span key={component.symbol} className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="h-2 w-2 rounded-sm"
                  style={{ backgroundColor: ALLOCATION_COLORS[index % ALLOCATION_COLORS.length] }}
                />
                <span className="num">
                  {formatWeight(component.weight)} {component.symbol}
                </span>
              </span>
            ))}
            {Math.abs(rawTotal - 100) > 0.01 && (
              <span className="text-ink-3">
                · entered weights sum to <span className="num">{Number(rawTotal.toFixed(2))}%</span>, auto-normalised to 100%
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** Compact numeric input used next to a toggle. */
function NumberFieldInline({
  ariaLabel,
  value,
  onChange,
  disabled,
}: {
  ariaLabel: string;
  value: number;
  onChange: (value: number) => void;
  disabled: boolean;
}): JSX.Element {
  return (
    <div className="flex h-9 items-center rounded-lg border border-line bg-base px-2.5 text-sm focus-within:border-accent">
      <input
        type="number"
        min={0.001}
        max={1}
        step={0.005}
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (event.target.value !== '' && Number.isFinite(next) && next > 0 && next <= 1) onChange(next);
        }}
        className="num w-14 bg-transparent text-ink outline-none disabled:cursor-not-allowed disabled:opacity-60"
      />
      <span className="ml-1 whitespace-nowrap text-ink-3">% / 8h</span>
    </div>
  );
}

export default function ControlBar({
  config,
  onChange,
  savedTickers,
  onSaveTickers,
  onRemoveTicker,
  status,
  onStart,
  onStop,
}: ControlBarProps): JSX.Element {
  const listId = useId();
  const busy = status === 'loading' || status === 'optimizing';
  const isPreset = (PRESET_TICKERS as readonly string[]).includes(config.symbol);
  // A ticker with its own chip (preset or saved) is not repeated in the free-text field.
  const hasChip = isPreset || savedTickers.includes(config.symbol);
  const subHour = config.interval === '15m' || config.interval === '30m';
  const tradesStocks =
    config.assetMode === 'single'
      ? !isCrypto(config.symbol)
      : config.basket.some((row) => row.symbol.trim() !== '' && !isCrypto(row.symbol));
  const commit = (ticker: string): void => onSaveTickers([ticker]);
  const universeOn = config.robustUniverse && config.assetMode === 'single';
  const universeList = parseUniverse(config.universe);
  const universeInvalid =
    universeOn && (universeList.some((symbol) => !TICKER_PATTERN.test(symbol)) || universeList.length > MAX_UNIVERSE_ASSETS);

  return (
    <Panel className="p-4">
      <datalist id={listId}>
        {[...PRESET_TICKERS, ...savedTickers].map((ticker) => (
          <option key={ticker} value={ticker} />
        ))}
      </datalist>

      {/* Row 1 – what to trade */}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
        <div className="shrink-0">
          <FieldLabel>Asset</FieldLabel>
          <Segmented
            ariaLabel="Asset selection mode"
            options={ASSET_MODE_OPTIONS}
            value={config.assetMode}
            onChange={(assetMode) => onChange({ assetMode })}
            disabled={busy}
          />
        </div>

        <div className="min-w-0 flex-1">
          <FieldLabel>{config.assetMode === 'single' ? 'Ticker' : 'Basket allocation'}</FieldLabel>
          {config.assetMode === 'single' ? (
            <div className="flex flex-wrap items-center gap-1.5">
              {PRESET_TICKERS.map((ticker) => {
                const active = config.symbol === ticker;
                return (
                  <button
                    key={ticker}
                    type="button"
                    disabled={busy}
                    aria-pressed={active}
                    onClick={() => onChange({ symbol: ticker })}
                    className={cx(
                      'num h-9 rounded-lg border px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60',
                      active
                        ? 'border-accent bg-accent/15 text-ink'
                        : 'border-line bg-base text-ink-2 hover:border-line-strong hover:text-ink',
                    )}
                  >
                    {ticker}
                  </button>
                );
              })}
              {savedTickers.map((ticker) => {
                const active = config.symbol === ticker;
                return (
                  <span
                    key={ticker}
                    className={cx(
                      'inline-flex h-9 items-center rounded-lg border text-sm font-medium transition-colors',
                      active ? 'border-accent bg-accent/15 text-ink' : 'border-line bg-base text-ink-2 hover:border-line-strong',
                      busy && 'opacity-60',
                    )}
                  >
                    <button
                      type="button"
                      disabled={busy}
                      aria-pressed={active}
                      onClick={() => onChange({ symbol: ticker })}
                      title="Saved ticker"
                      className="num h-full rounded-l-lg pl-3 pr-1.5 hover:text-ink disabled:cursor-not-allowed"
                    >
                      {ticker}
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => onRemoveTicker(ticker)}
                      aria-label={`Forget saved ticker ${ticker}`}
                      title="Forget this ticker"
                      className="grid h-full place-items-center rounded-r-lg pl-0.5 pr-2 text-ink-3 hover:text-ink disabled:cursor-not-allowed"
                    >
                      <X size={12} />
                    </button>
                  </span>
                );
              })}
              <TickerInput
                value={hasChip ? '' : config.symbol}
                onChange={(symbol) => onChange({ symbol: symbol || 'BTC-USD' })}
                onCommit={commit}
                disabled={busy}
                listId={listId}
                ariaLabel="Custom ticker"
                placeholder="Custom…"
                className={cx('w-[120px]', !hasChip && 'border-accent')}
              />
            </div>
          ) : (
            <BasketBuilder
              rows={config.basket}
              onChange={(basket) => onChange({ basket })}
              disabled={busy}
              listId={listId}
              onCommit={commit}
            />
          )}
        </div>
      </div>

      {/* Row 2 – how to test it */}
      <div className="mt-4 flex flex-wrap items-end gap-x-4 gap-y-3 border-t border-line pt-4">
        <div>
          <FieldLabel>Timeframe</FieldLabel>
          <Segmented
            ariaLabel="Candle timeframe"
            options={INTERVAL_OPTIONS}
            value={config.interval}
            onChange={(interval) => onChange({ interval })}
            disabled={busy}
          />
        </div>
        <div>
          <FieldLabel>History</FieldLabel>
          <Segmented
            ariaLabel="History range"
            options={rangeOptions(config.interval)}
            value={config.range}
            onChange={(range) => onChange({ range })}
            disabled={busy}
          />
        </div>
        <NumberField
          label="Starting capital"
          prefix="$"
          value={config.capital}
          min={100}
          max={1_000_000_000}
          step={1000}
          onChange={(capital) => onChange({ capital })}
          disabled={busy}
          className="w-[136px]"
        />
        <NumberField
          label="Fee / side"
          suffix="%"
          value={config.feePct}
          min={0}
          max={5}
          step={0.01}
          onChange={(feePct) => onChange({ feePct })}
          disabled={busy}
          className="w-[92px]"
        />
        <NumberField
          label="Slippage"
          suffix="%"
          value={config.slippagePct}
          min={0}
          max={5}
          step={0.01}
          onChange={(slippagePct) => onChange({ slippagePct })}
          disabled={busy}
          className="w-[92px]"
        />
        <NumberField
          label="Stagnation limit"
          value={config.maxStagnation}
          min={MIN_MAX_STAGNATION}
          max={MAX_MAX_STAGNATION}
          step={500}
          onChange={(maxStagnation) => onChange({ maxStagnation: Math.round(maxStagnation) })}
          disabled={busy}
          className="w-[124px]"
        />
        <div>
          <FieldLabel>Mode</FieldLabel>
          <Segmented
            ariaLabel="Trade direction mode"
            options={MODE_OPTIONS}
            value={config.tradeMode}
            onChange={(tradeMode) => onChange({ tradeMode })}
            disabled={busy}
          />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={onStart}
            disabled={busy}
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-semibold text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60"
          >
            {status === 'loading' ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Play size={15} fill="currentColor" />
            )}
            {status === 'loading' ? 'Loading data…' : status === 'optimizing' ? 'Optimizing…' : 'Start Optimization'}
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!busy}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-down/50 bg-down/10 px-4 text-sm font-semibold text-down transition-colors hover:bg-down/20 disabled:cursor-not-allowed disabled:border-line disabled:bg-transparent disabled:text-ink-3"
          >
            <Square size={13} fill="currentColor" />
            Stop
          </button>
        </div>
        {subHour && tradesStocks && (
          <p className="flex basis-full items-center gap-1.5 text-xs text-ink-3">
            <Info size={13} className="shrink-0" />
            Yahoo Finance only serves {config.interval} stock candles for the last {YAHOO_MAX_DAYS[config.interval] + 1} days, so stock
            runs on this timeframe are short. Crypto loads the full {config.range} from Binance.
          </p>
        )}
      </div>
      {/* Row 3 – risk and robustness (run settings the optimiser cannot tune) */}
      <div className="mt-4 flex flex-wrap items-end gap-x-4 gap-y-3 border-t border-line pt-4">
        <div>
          <FieldLabel>Position sizing</FieldLabel>
          <Segmented
            ariaLabel="Position sizing"
            options={SIZING_OPTIONS}
            value={config.sizingMode}
            onChange={(sizingMode) => onChange({ sizingMode })}
            disabled={busy}
          />
        </div>
        <NumberField
          label="Risk / trade"
          suffix="%"
          value={config.riskPct}
          min={0.1}
          max={10}
          step={0.25}
          onChange={(riskPct) => onChange({ riskPct })}
          disabled={busy || config.sizingMode === 'full'}
          className="w-[100px]"
        />
        <div>
          <FieldLabel>Perp funding</FieldLabel>
          <div className="flex items-center gap-2">
            <label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-line bg-base px-2.5 text-sm text-ink-2">
              <input
                type="checkbox"
                checked={config.fundingPct8h > 0}
                disabled={busy}
                onChange={(event) => onChange({ fundingPct8h: event.target.checked ? DEFAULT_FUNDING_PCT_8H : 0 })}
                className="h-3.5 w-3.5 accent-[var(--color-accent)]"
              />
              Charge funding
            </label>
            {config.fundingPct8h > 0 && (
              <NumberFieldInline
                ariaLabel="Funding rate per 8 hours"
                value={config.fundingPct8h}
                onChange={(fundingPct8h) => onChange({ fundingPct8h })}
                disabled={busy}
              />
            )}
          </div>
        </div>
        <div className="min-w-[260px] flex-1">
          <FieldLabel>Robust universe</FieldLabel>
          <div className="flex items-center gap-2">
            <label
              className={cx(
                'inline-flex h-9 shrink-0 cursor-pointer items-center gap-2 rounded-lg border bg-base px-2.5 text-sm',
                universeOn ? 'border-accent text-ink' : 'border-line text-ink-2',
                config.assetMode !== 'single' && 'cursor-not-allowed opacity-60',
              )}
              title="Score every strategy on all of these assets, untuned, and rank by the median. Not available for baskets."
            >
              <input
                type="checkbox"
                checked={universeOn}
                disabled={busy || config.assetMode !== 'single'}
                onChange={(event) => onChange({ robustUniverse: event.target.checked })}
                className="h-3.5 w-3.5 accent-[var(--color-accent)]"
              />
              Same rules on every asset
            </label>
            <input
              type="text"
              value={config.universe}
              disabled={busy || !universeOn}
              aria-label="Robust universe tickers"
              aria-invalid={universeInvalid}
              spellCheck={false}
              placeholder="BTC-USD, ETH-USD, …"
              onChange={(event) => onChange({ universe: event.target.value.toUpperCase() })}
              className={cx(
                'num h-9 min-w-0 flex-1 rounded-lg border bg-base px-2.5 text-sm text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-accent disabled:cursor-not-allowed disabled:opacity-50',
                universeInvalid ? 'border-down/70' : 'border-line',
              )}
            />
          </div>
        </div>
      </div>
    </Panel>
  );
}
