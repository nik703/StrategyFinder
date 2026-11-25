/**
 * Step 1 of the guided workflow – presets. One-click preset Cards set the assets, timeframe,
 * history, realistic costs, direction mode and market hypothesis; a one-line summary shows the
 * active setup; every granular knob sits in the Collapsible Advanced Settings drawer – Selects for
 * choices, Sliders for risk and volatility slippage, a Switch for funding, Inputs for numbers. Runs
 * are started from the top action bar.
 */

import { ChevronDown, Info, Plus, SlidersHorizontal, X } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { describeAllocation, formatWeight, normalizeWeights, TICKER_PATTERN } from '@/data/basket';
import { HYPOTHESES } from '@/engine/genome';
import {
  DEFAULT_BORROW_PCT_YEAR,
  financingFor,
  isCrypto,
  MAX_BASKET_ASSETS,
  MAX_POOL_ASSETS,
  parseUniverse,
  PRESET_TICKERS,
  type BasketRow,
  type RunConfig,
} from '@/lib/config';
import { cn } from '@/lib/utils';
import {
  MIN_BORROW_PCT_YEAR,
  MIN_FUNDING_PCT_8H,
  MIN_SLIPPAGE_ATR_MULT,
  HISTORY_RANGES,
  HYPOTHESIS_IDS,
  INTERVALS,
  MAX_RANGE,
  MAX_RUN_EVALUATIONS,
  MIN_RUN_EVALUATIONS,
  rangeAllowed,
  TRADE_MODE_LABEL,
  TRADE_MODES,
  WINDOW_TRIAL_BUDGET,
  YAHOO_MAX_DAYS,
  type EngineStatus,
  type HistoryRange,
  type Hypothesis,
  type Interval,
  type MaPeriodSearch,
  type TradeMode,
} from '@/types/trading';
import { FieldLabel, NumberField, Panel, Segmented, SelectField, SliderField, StepHeader, type ChoiceOption } from './common';
import PresetCards from './PresetCards';

interface ControlBarProps {
  config: RunConfig;
  onChange: (patch: Partial<RunConfig>) => void;
  /** Custom tickers remembered from earlier runs. */
  savedTickers: string[];
  onSaveTickers: (symbols: string[]) => void;
  onRemoveTicker: (symbol: string) => void;
  status: EngineStatus;
}

/** Identity colours of the basket legs: the theme's chart tokens, cycled. */
const ALLOCATION_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)', 'var(--chart-5)'];

const INTERVAL_OPTIONS: ChoiceOption<Interval>[] = INTERVALS.map((value) => ({ value, label: `${value} candles` }));
function rangeOptions(interval: Interval): ChoiceOption<HistoryRange>[] {
  return HISTORY_RANGES.map((value) => {
    const allowed = rangeAllowed(interval, value);
    return { value, label: `${value} of history`, disabled: !allowed, title: allowed ? undefined : `${interval} candles are limited to ${MAX_RANGE[interval]} of history` };
  });
}
const MA_PERIOD_OPTIONS: ChoiceOption<MaPeriodSearch>[] = [
  { value: 'integer', label: 'Every integer', title: 'The search fine-tunes each EMA / SMA length (and the Bollinger period) one candle at a time, e.g. EMA 23 or SMA 87' },
  { value: 'grid', label: 'Coarse grid', title: 'Only 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160, 200 – fewer combinations, less room to overfit' },
];
const HYPOTHESIS_OPTIONS: ChoiceOption<Hypothesis>[] = HYPOTHESIS_IDS.map((id) => ({ value: id, label: HYPOTHESES[id].label, title: HYPOTHESES[id].rationale }));
const MODE_TITLE: Record<TradeMode, string> = {
  'long-only': 'Never sells short – the default for equities and spot crypto, which drift upwards over time and charge for shorting.',
  asymmetric:
    'Shorts use the mirrored entry trigger, but only inside their own bear-market filter (close below a slow EMA/SMA, or a falling one), tuned independently of the long rules.',
  'long-short': 'Shorts are the exact mirror image of the long rules, in every market regime.',
};
const MODE_OPTIONS: ChoiceOption<TradeMode>[] = TRADE_MODES.map((value) => ({ value, label: TRADE_MODE_LABEL[value], title: MODE_TITLE[value] }));
const MODE_NOTE: Record<TradeMode, string> = {
  'long-only': 'Long-Only: buys only, treated as spot (no funding or borrow).',
  asymmetric: 'Asymmetric L/S: shorts need their own bear-regime filter, so they sit out bull markets; shorts pay funding / borrow.',
  'long-short': 'Mirrored L/S: shorts mirror the long rules in every regime; shorts pay funding / borrow.',
};
const SIZING_OPTIONS: ChoiceOption<RunConfig['sizingMode']>[] = [
  { value: 'fixed', label: 'Risk % · fixed' },
  { value: 'compound', label: 'Risk % · compounding' },
  { value: 'full', label: '100% of equity' },
];
type RobustTargetOption = '1' | '3' | '5' | '0';
const ROBUST_TARGET_OPTIONS: ChoiceOption<RobustTargetOption>[] = [
  { value: '1', label: '1 robust strategy' },
  { value: '3', label: '3 robust strategies' },
  { value: '5', label: '5 robust strategies' },
  { value: '0', label: 'Run full budget' },
];
const ASSET_MODE_OPTIONS: ChoiceOption<RunConfig['assetMode']>[] = [
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
    <Input
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
      className={cn('num h-9', className)}
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
  const weightOf = (symbol: string): number | null => normalized.find((component) => component.symbol === symbol.trim().toUpperCase())?.weight ?? null;
  const rawTotal = rows.reduce((sum, row) => sum + (Number.isFinite(row.weight) && row.weight > 0 ? row.weight : 0), 0);
  const update = (id: number, patch: Partial<BasketRow>): void => onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
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
            <div key={row.id} className="flex items-center gap-1.5 rounded-lg border border-border bg-muted/60 p-1.5">
              <TickerInput
                value={row.symbol}
                onChange={(symbol) => update(row.id, { symbol })}
                disabled={disabled}
                listId={listId}
                onCommit={onCommit}
                ariaLabel="Basket asset ticker"
                className="w-[104px]"
              />
              <Input
                type="number"
                min={0}
                step={1}
                value={Number.isFinite(row.weight) ? row.weight : ''}
                disabled={disabled}
                aria-label={`Weight for ${row.symbol || 'asset'} (%)`}
                onChange={(event) => update(row.id, { weight: event.target.value === '' ? NaN : Number(event.target.value) })}
                className="num h-9 w-16 text-right"
              />
              <span className="text-sm text-muted-foreground">%</span>
              <span className="num w-[68px] whitespace-nowrap text-right text-xs text-secondary-foreground" title="Normalised allocation (weights are rescaled to sum to 100%)">
                {share === null ? '–' : `→ ${formatWeight(share)}`}
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onChange(rows.filter((r) => r.id !== row.id))}
                disabled={disabled || rows.length <= 1}
                aria-label={`Remove ${row.symbol || 'asset'} from basket`}
              >
                <X />
              </Button>
            </div>
          );
        })}
        <Button variant="outline" onClick={add} disabled={disabled || rows.length >= MAX_BASKET_ASSETS} className="h-[50px] border-dashed">
          <Plus /> Add asset
        </Button>
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
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-secondary-foreground">
            {normalized.map((component, index) => (
              <span key={component.symbol} className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="size-2 rounded-sm" style={{ backgroundColor: ALLOCATION_COLORS[index % ALLOCATION_COLORS.length] }} />
                <span className="num">
                  {formatWeight(component.weight)} {component.symbol}
                </span>
              </span>
            ))}
            {Math.abs(rawTotal - 100) > 0.01 && (
              <span className="text-muted-foreground">
                · entered weights sum to <span className="num">{Number(rawTotal.toFixed(2))}%</span>, auto-normalised to 100%
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** One titled group of settings inside the Advanced Settings drawer. */
function Group({ title, children }: { title: string; children: ReactNode }): JSX.Element {
  return (
    <section>
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-secondary-foreground">{title}</h3>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">{children}</div>
    </section>
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
  const fundingId = useId();
  const refreshId = useId();
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
    config.assetMode === 'pool' && (poolList.length < 2 || poolList.length > MAX_POOL_ASSETS || poolList.some((symbol) => !TICKER_PATTERN.test(symbol)));

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

      <Separator className="mt-4" />
      <Collapsible open={advanced} onOpenChange={setAdvanced} className="pt-3">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" className="h-auto w-full flex-wrap justify-start gap-x-3 gap-y-1 px-1 py-1 text-left">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-secondary-foreground">
              <ChevronDown className={cn('transition-transform', !advanced && '-rotate-90')} />
              <SlidersHorizontal className="size-3.5" /> Advanced settings
            </span>
            <span className="num min-w-0 flex-1 truncate text-xs font-normal text-muted-foreground">{setupSummary(config).join(' · ')}</span>
          </Button>
        </CollapsibleTrigger>

        <CollapsibleContent className="mt-3 space-y-4">
          <Group title="Market">
            <div className="shrink-0">
              <FieldLabel>Asset</FieldLabel>
              <Segmented ariaLabel="Asset selection mode" options={ASSET_MODE_OPTIONS} value={config.assetMode} onChange={(assetMode) => onChange({ assetMode })} disabled={busy} />
            </div>
            <div className="min-w-[280px] flex-1">
              <FieldLabel>{config.assetMode === 'single' ? 'Ticker' : config.assetMode === 'pool' ? 'Pooled assets' : 'Basket allocation'}</FieldLabel>
              {config.assetMode === 'pool' ? (
                <div>
                  <Input
                    type="text"
                    value={config.pool}
                    disabled={busy}
                    aria-label="Pooled asset tickers"
                    aria-invalid={poolInvalid}
                    spellCheck={false}
                    onChange={(event) => onChange({ pool: event.target.value.toUpperCase() })}
                    className="num h-9 max-w-[520px]"
                  />
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    Every asset is traded separately with the same rules. Trades are pooled for the sample-size gates and the score is the median over
                    the assets – one strategy that must work on all of them. 2–{MAX_POOL_ASSETS} tickers.
                  </p>
                </div>
              ) : config.assetMode === 'single' ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  {PRESET_TICKERS.map((ticker) => (
                    <Button
                      key={ticker}
                      variant="outline"
                      disabled={busy}
                      aria-pressed={config.symbol === ticker}
                      onClick={() => onChange({ symbol: ticker })}
                      className={cn('num h-9', config.symbol === ticker && 'border-primary bg-primary/15 text-foreground')}
                    >
                      {ticker}
                    </Button>
                  ))}
                  {savedTickers.map((ticker) => (
                    <span key={ticker} className="inline-flex items-center">
                      <Button
                        variant="outline"
                        disabled={busy}
                        aria-pressed={config.symbol === ticker}
                        onClick={() => onChange({ symbol: ticker })}
                        title="Saved ticker"
                        className={cn('num h-9 rounded-r-none border-r-0', config.symbol === ticker && 'border-primary bg-primary/15 text-foreground')}
                      >
                        {ticker}
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        disabled={busy}
                        onClick={() => onRemoveTicker(ticker)}
                        aria-label={`Forget saved ticker ${ticker}`}
                        title="Forget this ticker"
                        className={cn('h-9 w-7 rounded-l-none', config.symbol === ticker && 'border-primary')}
                      >
                        <X className="size-3" />
                      </Button>
                    </span>
                  ))}
                  <TickerInput
                    value={hasChip ? '' : config.symbol}
                    onChange={(symbol) => onChange({ symbol: symbol || 'BTC-USD' })}
                    onCommit={commit}
                    disabled={busy}
                    listId={listId}
                    ariaLabel="Custom ticker"
                    placeholder="Custom…"
                    className={cn('w-[120px]', !hasChip && 'border-primary')}
                  />
                </div>
              ) : (
                <BasketBuilder rows={config.basket} onChange={(basket) => onChange({ basket })} disabled={busy} listId={listId} onCommit={commit} />
              )}
            </div>
            <SelectField label="Timeframe" value={config.interval} options={INTERVAL_OPTIONS} onChange={(interval) => onChange({ interval })} disabled={busy} />
            <SelectField label="History" value={config.range} options={rangeOptions(config.interval)} onChange={(range) => onChange({ range })} disabled={busy} />
            <SelectField label="Direction" value={config.tradeMode} options={MODE_OPTIONS} onChange={(tradeMode) => onChange({ tradeMode })} disabled={busy} />
            {subHour && tradesStocks && (
              <p className="flex basis-full items-center gap-1.5 text-xs text-muted-foreground">
                <Info className="size-3.5 shrink-0" />
                Yahoo Finance only serves {config.interval} stock candles for the last {YAHOO_MAX_DAYS[config.interval] + 1} days, so stock runs on this
                timeframe are short. Crypto loads the full {config.range} from Binance.
              </p>
            )}
          </Group>

          <Separator />
          <Group title="Market hypothesis">
            <SelectField
              label="Hypothesis"
              value={config.hypothesis}
              options={HYPOTHESIS_OPTIONS}
              onChange={(next) => onChange({ hypothesis: next })}
              disabled={busy}
              triggerClassName="min-w-[200px]"
            />
            <p className="min-w-[260px] flex-1 text-xs leading-relaxed text-muted-foreground">
              <span className="font-semibold text-secondary-foreground">{hypothesis.label}.</span> {hypothesis.rationale}{' '}
              <span className="text-secondary-foreground">Entry trigger:</span> {hypothesis.trigger} <span className="text-secondary-foreground">Filter:</span>{' '}
              {hypothesis.filter} <span className="text-secondary-foreground">{MODE_NOTE[config.tradeMode]}</span>
            </p>
          </Group>

          <Separator />
          <Group title="Costs & position sizing">
            <NumberField label="Starting capital" prefix="$" value={config.capital} min={100} max={1_000_000_000} step={1000} onChange={(capital) => onChange({ capital })} disabled={busy} className="w-[150px]" />
            <NumberField label="Fee / side" suffix="%" value={config.feePct} min={0} max={5} step={0.01} onChange={(feePct) => onChange({ feePct })} disabled={busy} className="w-[110px]" />
            <NumberField label="Slippage floor" suffix="%" value={config.slippagePct} min={0} max={5} step={0.01} onChange={(slippagePct) => onChange({ slippagePct })} disabled={busy} className="w-[120px]" />
            <SliderField
              label="Vol. slippage"
              value={config.slippageAtrMult}
              min={MIN_SLIPPAGE_ATR_MULT}
              max={0.2}
              step={0.01}
              format={(v) => `${v.toFixed(2)} × ATR`}
              onChange={(slippageAtrMult) => onChange({ slippageAtrMult: Math.max(MIN_SLIPPAGE_ATR_MULT, slippageAtrMult) })}
              disabled={busy}
              title={`Each fill slips by the larger of the floor and this multiple of ATR(14) as a % of price, so volatile candles cost more. Always on: at least ${MIN_SLIPPAGE_ATR_MULT} × ATR.`}
            />
            <SelectField label="Position sizing" value={config.sizingMode} options={SIZING_OPTIONS} onChange={(sizingMode) => onChange({ sizingMode })} disabled={busy} triggerClassName="min-w-[180px]" />
            <SliderField
              label="Risk / trade"
              value={config.riskPct}
              min={0.25}
              max={5}
              step={0.25}
              format={(v) => `${v}%`}
              onChange={(riskPct) => onChange({ riskPct })}
              disabled={busy || config.sizingMode === 'full'}
              title="Share of capital lost when a trade is stopped out"
            />
            {financing === 'none' ? (
              <div>
                <FieldLabel>Financing</FieldLabel>
                <p className="flex h-9 items-center text-xs text-muted-foreground" title="Long-Only runs are treated as spot positions">
                  Long-Only spot: none
                </p>
              </div>
            ) : financing === 'funding' ? (
              <div title={`Long/Short crypto trades a perpetual: every open position pays funding (always charged, at least ${MIN_FUNDING_PCT_8H}% per 8 h).`}>
                <FieldLabel htmlFor={fundingId}>Perp funding (L/S crypto)</FieldLabel>
                <div className="flex h-9 items-center gap-2">
                  <Input
                    id={fundingId}
                    type="number"
                    min={MIN_FUNDING_PCT_8H}
                    max={1}
                    step={0.005}
                    value={Math.max(MIN_FUNDING_PCT_8H, config.fundingPct8h)}
                    disabled={busy}
                    aria-label="Funding rate per 8 hours (%)"
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      if (event.target.value !== '' && Number.isFinite(next) && next <= 1) onChange({ fundingPct8h: Math.max(MIN_FUNDING_PCT_8H, next) });
                    }}
                    className="num h-9 w-20"
                  />
                  <span className="whitespace-nowrap text-sm text-muted-foreground">% / 8h</span>
                </div>
              </div>
            ) : (
              <NumberField
                label="Short borrow / yr"
                suffix="%"
                value={Math.max(MIN_BORROW_PCT_YEAR, config.borrowPctYear)}
                min={MIN_BORROW_PCT_YEAR}
                max={100}
                step={0.25}
                onChange={(borrowPctYear) => onChange({ borrowPctYear: Math.max(MIN_BORROW_PCT_YEAR, borrowPctYear) })}
                disabled={busy}
                className="w-[130px]"
              />
            )}
            {financing === 'borrow' && (
              <p className="basis-full text-xs text-muted-foreground">
                Borrow fee on short stock / ETF positions (default {DEFAULT_BORROW_PCT_YEAR}% a year, never below {MIN_BORROW_PCT_YEAR}%).
              </p>
            )}
            <p className="basis-full text-xs text-muted-foreground">
              Always enforced by the engine: fills at the next candle&apos;s open, volatility-scaled slippage, short financing and the single-strategy
              holdout lock.
            </p>
          </Group>

          <Separator />
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
                className="w-[140px]"
              />
            </div>
            <div title="Keep scanning – with diversity restarts when the search stalls – until this many strategies pass every robustness gate, or the budget runs out.">
              <SelectField
                label="Stop after"
                value={String(config.robustTarget) as RobustTargetOption}
                options={ROBUST_TARGET_OPTIONS}
                onChange={(value) => onChange({ robustTarget: Number(value) })}
                disabled={busy}
                triggerClassName="min-w-[180px]"
              />
            </div>
            <SelectField label="EMA / SMA lengths" value={config.maPeriods} options={MA_PERIOD_OPTIONS} onChange={(maPeriods) => onChange({ maPeriods })} disabled={busy} />
          </Group>

          <Separator />
          <Group title="Data">
            <div
              className="flex items-center gap-2"
              title="Closed candles are cached in this browser per exchange, symbol and timeframe, and later runs download only the new ones. Force a full download, e.g. after a provider revised its history."
            >
              <Switch id={refreshId} checked={config.forceRefreshData} disabled={busy} onCheckedChange={(checked) => onChange({ forceRefreshData: checked })} />
              <Label htmlFor={refreshId} className="text-sm font-normal text-secondary-foreground">
                Force Refresh Data (ignore the candle cache on the next run)
              </Label>
            </div>
          </Group>
        </CollapsibleContent>
      </Collapsible>
    </Panel>
  );
}
