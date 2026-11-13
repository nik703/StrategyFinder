/**
 * Top control bar: asset / basket builder, timeframe, history range, capital,
 * costs, trade mode and the Start / Stop buttons.
 */

import { Loader2, Play, Plus, Square, X } from 'lucide-react';
import { useId } from 'react';
import { formatWeight, normalizeWeights, TICKER_PATTERN } from '../data/basket';
import { MAX_BASKET_ASSETS, PRESET_TICKERS, type BasketRow, type RunConfig } from '../lib/config';
import { HISTORY_RANGES, INTERVALS, MAX_MAX_STAGNATION, MIN_MAX_STAGNATION, type EngineStatus, type HistoryRange, type Interval, type TradeMode } from '../types/trading';
import { cx, FieldLabel, NumberField, Panel, Segmented } from './ui';

interface ControlBarProps {
  config: RunConfig;
  onChange: (patch: Partial<RunConfig>) => void;
  status: EngineStatus;
  onStart: () => void;
  onStop: () => void;
}

/** Colours for the allocation bar (identity only – not a chart series). */
const ALLOCATION_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];

const INTERVAL_OPTIONS: { value: Interval; label: string }[] = INTERVALS.map((value) => ({ value, label: value }));
const RANGE_OPTIONS: { value: HistoryRange; label: string }[] = HISTORY_RANGES.map((value) => ({ value, label: value }));
const MODE_OPTIONS: { value: TradeMode; label: string }[] = [
  { value: 'long-only', label: 'Long-Only' },
  { value: 'long-short', label: 'Long/Short' },
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
}: {
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  listId: string;
  ariaLabel: string;
  className?: string;
  placeholder?: string;
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
}: {
  rows: BasketRow[];
  onChange: (rows: BasketRow[]) => void;
  disabled: boolean;
  listId: string;
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

export default function ControlBar({ config, onChange, status, onStart, onStop }: ControlBarProps): JSX.Element {
  const listId = useId();
  const busy = status === 'loading' || status === 'optimizing';
  const isPreset = (PRESET_TICKERS as readonly string[]).includes(config.symbol);

  return (
    <Panel className="p-4">
      <datalist id={listId}>
        {PRESET_TICKERS.map((ticker) => (
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
              <TickerInput
                value={isPreset ? '' : config.symbol}
                onChange={(symbol) => onChange({ symbol: symbol || 'BTC-USD' })}
                disabled={busy}
                listId={listId}
                ariaLabel="Custom ticker"
                placeholder="Custom…"
                className={cx('w-[120px]', !isPreset && 'border-accent')}
              />
            </div>
          ) : (
            <BasketBuilder rows={config.basket} onChange={(basket) => onChange({ basket })} disabled={busy} listId={listId} />
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
            options={RANGE_OPTIONS}
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
      </div>
    </Panel>
  );
}
