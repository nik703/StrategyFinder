/**
 * Step 1 of the guided workflow – presets. One-click preset cards set the assets, timeframe,
 * history, realistic costs, direction mode and market hypothesis; a one-line summary shows the
 * active setup; every granular knob (market picker, timeframe, direction, hypothesis, costs,
 * sizing, financing, search budget) sits in the collapsible Advanced Settings drawer. Runs are
 * started from the top action bar.
 */

import { ChevronDown, ChevronRight, Info, Plus, SlidersHorizontal, X } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { describeAllocation, formatWeight, normalizeWeights, TICKER_PATTERN } from '../data/basket';
import {
  DEFAULT_BORROW_PCT_YEAR,
  DEFAULT_FUNDING_PCT_8H,
  financingFor,
  isCrypto,
  MAX_BASKET_ASSETS,
  MAX_POOL_ASSETS,
  parseUniverse,
  PRESET_TICKERS,
  type BasketRow,
  type RunConfig,
} from '../lib/config';
import {
  HISTORY_RANGES,
  HYPOTHESIS_IDS,
  INTERVALS,
  MAX_RANGE,
  MIN_RUN_EVALUATIONS,
  rangeAllowed,
  MAX_RUN_EVALUATIONS,
  TRADE_MODE_LABEL,
  TRADE_MODES,
  WINDOW_TRIAL_BUDGET,
  YAHOO_MAX_DAYS,
  type EngineStatus,
  type HistoryRange,
  type Hypothesis,
  type Interval,
  type TradeMode,
  type MaPeriodSearch,
} from '../types/trading';
import { HYPOTHESES } from '../engine/genome';
import PresetCards from './PresetCards';
import { cx, FieldLabel, NumberField, Panel, Segmented, StepHeader } from './ui';

interface ControlBarProps {
  config: RunConfig;
  onChange: (patch: Partial<RunConfig>) => void;
  /** Custom tickers remembered from earlier runs. */
  savedTickers: string[];
  onSaveTickers: (symbols: string[]) => void;
  onRemoveTicker: (symbol: string) => void;
  status: EngineStatus;
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
const MA_PERIOD_OPTIONS: { value: MaPeriodSearch; label: string; title: string }[] = [
  {
    value: 'integer',
    label: 'Every integer',
    title: 'The search fine-tunes each EMA / SMA length (and the Bollinger period) one candle at a time, e.g. EMA 23 or SMA 87',
  },
  {
    value: 'grid',
    label: 'Coarse grid',
    title: 'Only 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200 – fewer combinations, less room to overfit',
  },
];
const HYPOTHESIS_OPTIONS: { value: Hypothesis; label: string; title: string }[] = HYPOTHESIS_IDS.map((id) => ({
  value: id,
  label: HYPOTHESES[id].label,
  title: `${HYPOTHESES[id].rationale}\nTrigger: ${HYPOTHESES[id].trigger}\n${HYPOTHESES[id].filter}`,
}));
const MODE_TITLE: Record<TradeMode, string> = {
  'long-only': 'Never sells short – the default for equities and spot crypto, which drift upwards over time and charge for shorting.',
  asymmetric:
    'Shorts use the mirrored entry trigger, but only inside their own bear-market filter (close below a slow EMA/SMA, or a falling one), tuned independently of the long rules.',
  'long-short': 'Shorts are the exact mirror image of the long rules, in every market regime.',
};
const MODE_OPTIONS: { value: TradeMode; label: string; title: string }[] = TRADE_MODES.map((value) => ({
  value,
  label: TRADE_MODE_LABEL[value],
  title: MODE_TITLE[value],
}));
const MODE_NOTE: Record<TradeMode, string> = {
  'long-only': 'Long-Only: buys only, treated as spot (no funding or borrow).',
  asymmetric: 'Asymmetric L/S: shorts need their own bear-regime filter, so they sit out bull markets; shorts pay funding / borrow.',
  'long-short': 'Mirrored L/S: shorts mirror the long rules in every regime; shorts pay funding / borrow.',
};
const SIZING_OPTIONS: { value: RunConfig['sizingMode']; label: string }[] = [
  { value: 'fixed', label: 'Risk % · fixed' },
  { value: 'compound', label: 'Risk % · compounding' },
  { value: 'full', label: '100% of equity' },
];
type RobustTargetOption = '1' | '3' | '5' | '0';
const ROBUST_TARGET_OPTIONS: { value: RobustTargetOption; label: string }[] = [
  { value: '1', label: '1' },
  { value: '3', label: '3' },
  { value: '5', label: '5' },
  { value: '0', label: 'Run Full Budget' },
];
const ASSET_MODE_OPTIONS: { value: RunConfig['assetMode']; label: string }[] = [
  { value: 'single', label: 'Single asset' },
  { value: 'basket', label: 'Weighted basket' },
  { value: 'pool', label: 'Asset pool' },
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

/** One titled group of settings inside the Advanced Settings drawer. */
function Group({ title, children }: { title: string; children: ReactNode }): JSX.Element {
  return (
    <fieldset className="border-t border-line pt-3 first:border-t-0 first:pt-0">
      <legend className="float-left mb-2 w-full text-[11px] font-semibold uppercase tracking-wider text-ink-2">{title}</legend>
      <div className="clear-both flex flex-wrap items-end gap-x-4 gap-y-3">{children}</div>
    </fieldset>
  );
}

/** The active setup in one line: what will be traded, how, and what the run looks for. */
function setupSummary(config: RunConfig): string[] {
  const assets =
    config.assetMode === 'pool'
      ? `Pool ${parseUniverse(config.pool).map((symbol) => symbol.replace(/-USD$/, '')).join(' · ')}`
      : config.assetMode === 'single'
        ? config.symbol
        : `Basket ${describeAllocation(normalizeWeights(config.basket.filter((row) => row.symbol.trim() !== '')))}`;
  const sizing = config.sizingMode === 'full' ? '100% of equity' : `${config.riskPct}% risk / trade`;
  return [
    assets,
    `${config.interval} · ${config.range}`,
    TRADE_MODE_LABEL[config.tradeMode],
    HYPOTHESES[config.hypothesis].label,
    `${config.feePct}% fee + ${config.slippagePct}% / ${config.slippageAtrMult}×ATR slippage`,
    sizing,
    `${config.maxEvaluations.toLocaleString('en-US')} evals · stop after ${config.robustTarget > 0 ? `${config.robustTarget} robust` : 'the full budget'}`,
  ];
}

export default function ControlBar({ config, onChange, savedTickers, onSaveTickers, onRemoveTicker, status }: ControlBarProps): JSX.Element {
  const listId = useId();
  const busy = status === 'loading' || status === 'optimizing';
  const isPreset = (PRESET_TICKERS as readonly string[]).includes(config.symbol);
  // A ticker with its own chip (preset or saved) is not repeated in the free-text field.
  const hasChip = isPreset || savedTickers.includes(config.symbol);
  const subHour = config.interval === '15m' || config.interval === '30m';
  const financing = financingFor(config);
  const tradesStocks =
    config.assetMode === 'pool'
      ? parseUniverse(config.pool).some((symbol) => !isCrypto(symbol))
      : config.assetMode === 'single'
        ? !isCrypto(config.symbol)
        : config.basket.some((row) => row.symbol.trim() !== '' && !isCrypto(row.symbol));
  const commit = (ticker: string): void => onSaveTickers([ticker]);
  const poolList = parseUniverse(config.pool);
  const poolInvalid =
    config.assetMode === 'pool' &&
    (poolList.length < 2 || poolList.length > MAX_POOL_ASSETS || poolList.some((symbol) => !TICKER_PATTERN.test(symbol)));

  const [advanced, setAdvanced] = useState(false);
  const hypothesis = HYPOTHESES[config.hypothesis];

  return (
    <Panel className="p-4">
      <datalist id={listId}>
        {[...PRESET_TICKERS, ...savedTickers].map((ticker) => (
          <option key={ticker} value={ticker} />
        ))}
      </datalist>

      <StepHeader
        step={1}
        title="Presets"
        hint="One click sets the assets, timeframe, costs, direction and stop / target ranges. Press 1-Click Run in the top bar to scan."
      />
      <div className="mt-3">
        <PresetCards config={config} onApply={(preset) => onChange(preset.patch)} disabled={busy} />
      </div>

      {/* Active setup + Advanced settings drawer */}
      <div className="mt-4 border-t border-line pt-3">
        <button
          type="button"
          onClick={() => setAdvanced((open) => !open)}
          aria-expanded={advanced}
          aria-controls={`${listId}-advanced`}
          className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 rounded-lg px-1 py-1 text-left transition-colors hover:bg-hover/50"
        >
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-ink-2">
            {advanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <SlidersHorizontal size={13} /> Advanced settings
          </span>
          <span className="num min-w-0 flex-1 truncate text-xs text-ink-3">{setupSummary(config).join(' · ')}</span>
        </button>

        {advanced && (
          <div id={`${listId}-advanced`} className="mt-3 space-y-4">
            <Group title="Market">
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
              <div className="min-w-[280px] flex-1">
                <FieldLabel>{config.assetMode === 'single' ? 'Ticker' : config.assetMode === 'pool' ? 'Pooled assets' : 'Basket allocation'}</FieldLabel>
                {config.assetMode === 'pool' ? (
                  <div>
                    <input
                      type="text"
                      value={config.pool}
                      disabled={busy}
                      aria-label="Pooled asset tickers"
                      aria-invalid={poolInvalid}
                      spellCheck={false}
                      onChange={(event) => onChange({ pool: event.target.value.toUpperCase() })}
                      className={cx(
                        'num h-9 w-full max-w-[520px] rounded-lg border bg-base px-3 text-sm text-ink outline-none focus:border-accent disabled:cursor-not-allowed disabled:opacity-60',
                        poolInvalid ? 'border-down/70' : 'border-line',
                      )}
                    />
                    <p className="mt-1.5 text-xs text-ink-3">
                      Every asset is traded separately with the same rules. Trades are pooled for the sample-size gates and the score is
                      the median over the assets – one strategy that must work on all of them. 2–{MAX_POOL_ASSETS} tickers.
                    </p>
                  </div>
                ) : config.assetMode === 'single' ? (
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
                            active ? 'border-accent bg-accent/15 text-ink' : 'border-line bg-base text-ink-2 hover:border-line-strong hover:text-ink',
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
                  <BasketBuilder rows={config.basket} onChange={(basket) => onChange({ basket })} disabled={busy} listId={listId} onCommit={commit} />
                )}
              </div>
              <div>
                <FieldLabel>Timeframe</FieldLabel>
                <Segmented ariaLabel="Candle timeframe" options={INTERVAL_OPTIONS} value={config.interval} onChange={(interval) => onChange({ interval })} disabled={busy} />
              </div>
              <div>
                <FieldLabel>History</FieldLabel>
                <Segmented ariaLabel="History range" options={rangeOptions(config.interval)} value={config.range} onChange={(range) => onChange({ range })} disabled={busy} />
              </div>
              <div>
                <FieldLabel>Direction</FieldLabel>
                <Segmented ariaLabel="Trade direction mode" options={MODE_OPTIONS} value={config.tradeMode} onChange={(tradeMode) => onChange({ tradeMode })} disabled={busy} />
              </div>
              {subHour && tradesStocks && (
                <p className="flex basis-full items-center gap-1.5 text-xs text-ink-3">
                  <Info size={13} className="shrink-0" />
                  Yahoo Finance only serves {config.interval} stock candles for the last {YAHOO_MAX_DAYS[config.interval] + 1} days, so stock runs on
                  this timeframe are short. Crypto loads the full {config.range} from Binance.
                </p>
              )}
            </Group>

            <Group title="Market hypothesis">
              <div>
                <Segmented
                  ariaLabel="Market hypothesis"
                  options={HYPOTHESIS_OPTIONS}
                  value={config.hypothesis}
                  onChange={(hypothesis) => onChange({ hypothesis })}
                  disabled={busy}
                />
              </div>
              <p className="min-w-[260px] flex-1 text-xs leading-relaxed text-ink-3">
                <span className="font-semibold text-ink-2">{hypothesis.label}.</span> {hypothesis.rationale}{' '}
                <span className="text-ink-2">Entry trigger:</span> {hypothesis.trigger} <span className="text-ink-2">Filter:</span> {hypothesis.filter}{' '}
                <span className="text-ink-2">{MODE_NOTE[config.tradeMode]}</span>
              </p>
            </Group>

            <Group title="Costs & position sizing">
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
              <NumberField label="Fee / side" suffix="%" value={config.feePct} min={0} max={5} step={0.01} onChange={(feePct) => onChange({ feePct })} disabled={busy} className="w-[92px]" />
              <NumberField
                label="Slippage floor"
                suffix="%"
                value={config.slippagePct}
                min={0}
                max={5}
                step={0.01}
                onChange={(slippagePct) => onChange({ slippagePct })}
                disabled={busy}
                className="w-[100px]"
              />
              <div title="Each fill slips by the larger of the floor and this multiple of ATR(14) as a % of price, so volatile candles cost more. 0 = fixed slippage only.">
                <NumberField
                  label="Vol. slippage"
                  suffix="× ATR"
                  value={config.slippageAtrMult}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(slippageAtrMult) => onChange({ slippageAtrMult })}
                  disabled={busy}
                  className="w-[112px]"
                />
              </div>
              <div>
                <FieldLabel>Position sizing</FieldLabel>
                <Segmented ariaLabel="Position sizing" options={SIZING_OPTIONS} value={config.sizingMode} onChange={(sizingMode) => onChange({ sizingMode })} disabled={busy} />
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
              {financing === 'none' ? (
                <div>
                  <FieldLabel>Financing</FieldLabel>
                  <p className="flex h-9 items-center text-xs text-ink-3" title="Long-Only runs are treated as spot positions">
                    Long-Only spot: none
                  </p>
                </div>
              ) : financing === 'funding' ? (
                <div>
                  <FieldLabel>Perp funding (L/S crypto)</FieldLabel>
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
              ) : (
                <div title={`Borrow fee on short stock / ETF positions, per year (default ${DEFAULT_BORROW_PCT_YEAR}%).`}>
                  <NumberField
                    label="Short borrow / yr"
                    suffix="%"
                    value={config.borrowPctYear}
                    min={0}
                    max={100}
                    step={0.25}
                    onChange={(borrowPctYear) => onChange({ borrowPctYear })}
                    disabled={busy}
                    className="w-[120px]"
                  />
                </div>
              )}
            </Group>

            <Group title="Search">
              <div title={`Strategies this run may evaluate. Every evaluation raises the luck hurdle; a data window allows ${WINDOW_TRIAL_BUDGET.toLocaleString('en-US')} across all runs by default (extendable).`}>
                <NumberField
                  label="Evaluations / run"
                  value={config.maxEvaluations}
                  min={MIN_RUN_EVALUATIONS}
                  max={MAX_RUN_EVALUATIONS}
                  step={1000}
                  onChange={(maxEvaluations) => onChange({ maxEvaluations: Math.round(maxEvaluations) })}
                  disabled={busy}
                  className="w-[124px]"
                />
              </div>
              <div title="Keep scanning – with diversity restarts when the search stalls – until this many strategies pass every robustness gate, or the budget runs out.">
                <FieldLabel>Stop after N robust</FieldLabel>
                <Segmented
                  ariaLabel="Stop after N robust strategies"
                  options={ROBUST_TARGET_OPTIONS}
                  value={String(config.robustTarget) as RobustTargetOption}
                  onChange={(value) => onChange({ robustTarget: Number(value) })}
                  disabled={busy}
                />
              </div>
              <div>
                <FieldLabel>EMA / SMA lengths</FieldLabel>
                <Segmented ariaLabel="EMA / SMA length search" options={MA_PERIOD_OPTIONS} value={config.maPeriods} onChange={(maPeriods) => onChange({ maPeriods })} disabled={busy} />
              </div>
            </Group>
          </div>
        )}
      </div>
    </Panel>
  );
}
