/**
 * App-level building blocks composed from shadcn/ui primitives (`@/components/ui/*`): section
 * panels, field labels, single-choice toggle groups, numeric inputs, status badges, step headers,
 * info tooltips, line keys and sparklines. Nothing here draws a raw control: every interactive
 * element is a shadcn/ui primitive styled with the theme's semantic tokens.
 */

import { Info } from 'lucide-react';
import { useEffect, useId, useState, type ReactNode } from 'react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

/* -------------------------------------------------------------------------- */

/** A section of the page: a shadcn Card whose children manage their own padding. */
export function Panel({ children, className, id }: { children: ReactNode; className?: string; id?: string }): JSX.Element {
  return (
    <Card id={id} className={cn('gap-0 py-0', className)}>
      {children}
    </Card>
  );
}

export function FieldLabel({ children, htmlFor }: { children: ReactNode; htmlFor?: string }): JSX.Element {
  return (
    <Label htmlFor={htmlFor} className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
      {children}
    </Label>
  );
}

/* -------------------------------------------------------------------------- */

interface SegmentedProps<T extends string> {
  options: readonly { value: T; label: string; disabled?: boolean; title?: string }[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
  disabled?: boolean;
  size?: 'sm' | 'md';
}

/** Single-choice control: a shadcn ToggleGroup that never lets the selection become empty. */
export function Segmented<T extends string>({ options, value, onChange, ariaLabel, disabled = false, size = 'md' }: SegmentedProps<T>): JSX.Element {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size={size === 'sm' ? 'sm' : 'default'}
      value={value}
      onValueChange={(next) => {
        if (next) onChange(next as T);
      }}
      aria-label={ariaLabel}
      disabled={disabled}
      className="flex-wrap"
    >
      {options.map((option) => (
        <ToggleGroupItem
          key={option.value}
          value={option.value}
          title={option.title}
          disabled={option.disabled}
          className="data-[state=on]:border-primary/60 data-[state=on]:bg-primary/15 data-[state=on]:text-foreground"
        >
          {option.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

/* -------------------------------------------------------------------------- */

interface NumberFieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  prefix?: string;
  suffix?: string;
  disabled?: boolean;
  className?: string;
}

/**
 * Numeric shadcn Input that tolerates intermediate typing states ("", "0.", "-") and only reports
 * valid, in-range numbers to its parent.
 */
export function NumberField({ label, value, onChange, min, max, step, prefix, suffix, disabled = false, className }: NumberFieldProps): JSX.Element {
  const id = useId();
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft((current) => (Number(current) === value ? current : String(value)));
  }, [value]);

  const parsed = Number(draft);
  const invalid = draft.trim() === '' || !Number.isFinite(parsed) || parsed < min || parsed > max;

  return (
    <div className={className}>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <div className="flex items-center gap-1.5">
        {prefix && <span className="text-sm text-muted-foreground">{prefix}</span>}
        <Input
          id={id}
          type="number"
          inputMode="decimal"
          value={draft}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          aria-invalid={invalid}
          onChange={(event) => {
            const text = event.target.value;
            setDraft(text);
            const next = Number(text);
            if (text.trim() !== '' && Number.isFinite(next) && next >= min && next <= max) onChange(next);
          }}
          onBlur={() => {
            if (invalid) setDraft(String(value));
          }}
          className="num h-9 min-w-0"
        />
        {suffix && <span className="whitespace-nowrap text-sm text-muted-foreground">{suffix}</span>}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
  title?: string;
}

/** Labelled single choice from a list: a shadcn Select. */
export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled = false,
  className,
  triggerClassName,
}: {
  label: string;
  value: T;
  options: readonly ChoiceOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
  triggerClassName?: string;
}): JSX.Element {
  const id = useId();
  return (
    <div className={className}>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Select value={value} onValueChange={(next) => onChange(next as T)} disabled={disabled}>
        <SelectTrigger id={id} aria-label={label} className={cn('h-9 min-w-[140px]', triggerClassName)}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value} disabled={option.disabled} title={option.title}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

/** Labelled numeric range control: a shadcn Slider with its current value shown next to the label. */
export function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
  disabled = false,
  className,
  title,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  disabled?: boolean;
  className?: string;
  title?: string;
}): JSX.Element {
  const id = useId();
  return (
    <div className={cn('min-w-[180px]', className)} title={title}>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <Label htmlFor={id} className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </Label>
        <span className="num text-xs font-semibold text-foreground">{format(value)}</span>
      </div>
      <Slider
        id={id}
        aria-label={label}
        value={[Math.min(max, Math.max(min, value))]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([next]) => {
          if (next !== undefined) onChange(Number(next.toFixed(6)));
        }}
        className="py-2.5"
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export type PillTone = 'neutral' | 'accent' | 'up' | 'down' | 'warn' | 'bench';

const PILL_TONES: Record<PillTone, string> = {
  neutral: 'border-border bg-muted text-secondary-foreground',
  accent: 'border-primary/40 bg-primary/10 text-primary',
  up: 'border-up/35 bg-up/10 text-up',
  down: 'border-down/35 bg-down/10 text-down',
  warn: 'border-warn/35 bg-warn/10 text-warn',
  bench: 'border-bench/45 bg-bench/10 text-bench',
};

/** Status badge: a shadcn Badge in one of the app's semantic tones. */
export function Pill({ children, tone = 'neutral', className, title }: { children: ReactNode; tone?: PillTone; className?: string; title?: string }): JSX.Element {
  return (
    <Badge variant="outline" title={title} className={cn('rounded-md px-1.5 text-[11px] leading-4', PILL_TONES[tone], className)}>
      {children}
    </Badge>
  );
}

/** A label with an (i) icon whose Tooltip explains it – used for column headers and metric names. */
export function InfoTip({ children, tip, className }: { children: ReactNode; tip: ReactNode; className?: string }): JSX.Element {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className={cn('inline-flex cursor-help items-center gap-1 outline-none', className)}>
          {children}
          <Info className="size-3 shrink-0 opacity-60" aria-hidden="true" />
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs leading-relaxed">{tip}</TooltipContent>
    </Tooltip>
  );
}

/** Short stroke in a series colour – the legend / tooltip key for a line. */
export function LineKey({ color, dashed = false }: { color: string; dashed?: boolean }): JSX.Element {
  return (
    <span
      aria-hidden="true"
      className="inline-block h-0 w-4 shrink-0 rounded-full border-t-2"
      style={{ borderColor: color, borderTopStyle: dashed ? 'dashed' : 'solid' }}
    />
  );
}

/* -------------------------------------------------------------------------- */

/** Numbered heading of one step of the guided workflow (1 Presets → 2 Discover → 3 Leaderboard → 4 Workbench). */
export function StepHeader({ step, title, hint, children, id }: { step: number; title: string; hint?: string; children?: ReactNode; id?: string }): JSX.Element {
  return (
    <div id={id} className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
      <div className="flex min-w-0 items-center gap-2.5">
        <Badge className="num size-6 shrink-0 rounded-full p-0 text-xs font-semibold">{step}</Badge>
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

/** Tiny equity curve: a single line, green when it ends above its start, red otherwise. */
export function Sparkline({ values, width = 96, height = 28, className }: { values: ArrayLike<number> | null; width?: number; height?: number; className?: string }): JSX.Element {
  const n = values?.length ?? 0;
  if (!values || n < 2) return <span className="text-muted-foreground">–</span>;
  // At most 80 points: plenty for a sparkline.
  const step = Math.max(1, Math.floor(n / 80));
  const points: number[] = [];
  for (let i = 0; i < n; i += step) points.push(values[i]);
  if ((n - 1) % step !== 0) points.push(values[n - 1]);
  let min = Infinity;
  let max = -Infinity;
  for (const v of points) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min || 1;
  const path = points
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${((i / (points.length - 1)) * (width - 2) + 1).toFixed(1)},${(height - 1 - ((v - min) / span) * (height - 2)).toFixed(1)}`)
    .join(' ');
  const up = points[points.length - 1] >= points[0];
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={cn('block', className)} aria-hidden="true">
      <path d={path} fill="none" stroke={up ? 'var(--up)' : 'var(--down)'} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}
