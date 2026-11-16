/** Small, dependency-free UI primitives shared by the dashboard. */

import { useEffect, useId, useState, type ReactNode } from 'react';

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}

/* -------------------------------------------------------------------------- */

export function Panel({ children, className }: { children: ReactNode; className?: string }): JSX.Element {
  return <section className={cx('rounded-xl border border-line bg-panel', className)}>{children}</section>;
}

export function FieldLabel({ children, htmlFor }: { children: ReactNode; htmlFor?: string }): JSX.Element {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-ink-3">
      {children}
    </label>
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

/** Segmented control (single choice). */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  disabled = false,
  size = 'md',
}: SegmentedProps<T>): JSX.Element {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cx('inline-flex rounded-lg border border-line bg-base p-0.5', disabled && 'opacity-60')}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.title}
            disabled={disabled || option.disabled}
            onClick={() => onChange(option.value)}
            className={cx(
              'rounded-md font-medium transition-colors disabled:cursor-not-allowed',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
              active ? 'bg-raised text-ink shadow-sm ring-1 ring-line-strong' : 'text-ink-3 hover:text-ink-2',
              // An option that is unavailable on its own (not just while the whole control is locked).
              option.disabled && !active && 'opacity-35 hover:text-ink-3',
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
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
 * Numeric input that tolerates intermediate typing states ("", "0.", "-")
 * and only reports valid, in-range numbers to its parent.
 */
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  prefix,
  suffix,
  disabled = false,
  className,
}: NumberFieldProps): JSX.Element {
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
      <div
        className={cx(
          'flex h-9 items-center rounded-lg border bg-base px-2.5 text-sm transition-colors focus-within:border-accent',
          invalid ? 'border-down/70' : 'border-line',
          disabled && 'opacity-60',
        )}
      >
        {prefix && <span className="mr-1 text-ink-3">{prefix}</span>}
        <input
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
          className="num w-full min-w-0 bg-transparent text-ink outline-none placeholder:text-ink-3 disabled:cursor-not-allowed"
        />
        {suffix && <span className="ml-1 text-ink-3">{suffix}</span>}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

type PillTone = 'neutral' | 'accent' | 'up' | 'down' | 'warn' | 'bench';

const PILL_TONES: Record<PillTone, string> = {
  neutral: 'border-line-strong bg-raised text-ink-2',
  accent: 'border-accent/40 bg-accent/10 text-accent-strong',
  up: 'border-up/35 bg-up/10 text-up',
  down: 'border-down/35 bg-down/10 text-down',
  warn: 'border-warn/35 bg-warn/10 text-warn',
  bench: 'border-bench/45 bg-bench/10 text-[#f0895d]',
};

export function Pill({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode;
  tone?: PillTone;
  className?: string;
}): JSX.Element {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4',
        PILL_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
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

/** Numbered heading of one step of the guided workflow (1 Setup → 2 Scan → 3 Rank → 4 Workbench). */
export function StepHeader({
  step,
  title,
  hint,
  children,
  id,
}: {
  step: number;
  title: string;
  hint?: string;
  children?: ReactNode;
  id?: string;
}): JSX.Element {
  return (
    <div id={id} className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="num grid h-6 w-6 shrink-0 place-items-center rounded-full border border-accent/50 bg-accent/15 text-xs font-semibold text-accent-strong">
          {step}
        </span>
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {hint && <p className="text-xs text-ink-3">{hint}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

/** Tiny equity curve: a single line, green when it ends above its start, red otherwise. */
export function Sparkline({
  values,
  width = 96,
  height = 28,
  className,
}: {
  values: ArrayLike<number> | null;
  width?: number;
  height?: number;
  className?: string;
}): JSX.Element {
  const n = values?.length ?? 0;
  if (!values || n < 2) return <span className="text-ink-3">–</span>;
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
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={cx('block', className)} aria-hidden="true">
      <path d={path} fill="none" stroke={up ? 'var(--color-up)' : 'var(--color-down)'} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}
