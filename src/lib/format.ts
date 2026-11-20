/** Number, currency, percentage and date formatting shared by every view. */

const usd0 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const usdCompact = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1,
});
const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

const DASH = '–';

const sign = (value: number, signed: boolean): string => (signed && value > 0 ? '+' : '');

/** $12,345 – cents are shown only for small amounts. */
export function fmtUsd(value: number | null | undefined, signed = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const formatter = Math.abs(value) < 100 ? usd2 : usd0;
  return `${sign(value, signed)}${formatter.format(value)}`;
}

/**
 * A market price with as many decimals as its size needs:
 * $96,412.50 · $185.42 · $12.3456 · $0.6234 · $0.00001235.
 */
export function fmtPrice(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const abs = Math.abs(value);
  let decimals = 2;
  if (abs > 0 && abs < 100) decimals = 4;
  if (abs > 0 && abs < 1) decimals = Math.min(10, 3 - Math.floor(Math.log10(abs)));
  const text = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: decimals }).format(abs);
  return `${value < 0 ? '-' : ''}$${text}`;
}

/** $12.5K / $1.2M – for axis ticks. */
export function fmtUsdCompact(value: number): string {
  if (!Number.isFinite(value)) return DASH;
  return Math.abs(value) < 1000 ? usd0.format(value) : usdCompact.format(value);
}

/** +12.3% – large values drop their decimals so columns stay compact. */
export function fmtPct(value: number | null | undefined, signed = false, decimals = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 0 : decimals;
  const text = new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
  return `${sign(value, signed)}${text}%`;
}

export function fmtNum(value: number | null | undefined, decimals = 2, signed = false): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  if (value === Infinity) return '∞';
  if (value === -Infinity) return '−∞';
  return `${sign(value, signed)}${value.toFixed(decimals)}`;
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return int.format(Math.round(value));
}

const dateFmt = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
const dateTimeFmt = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
  timeZone: 'UTC',
});
const monthFmt = new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const dayFmt = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });

/** 14 Nov 2026 (UTC). */
export function fmtDate(ms: number): string {
  return Number.isFinite(ms) ? dateFmt.format(ms) : DASH;
}

/** 14 Nov 2026, 13:00 (UTC). */
export function fmtDateTime(ms: number): string {
  return Number.isFinite(ms) ? `${dateTimeFmt.format(ms)} UTC` : DASH;
}

/** Axis label: "Nov 2026" for long spans, "14 Nov" for short ones. */
export function fmtAxisDate(ms: number, spanMs: number): string {
  if (!Number.isFinite(ms)) return '';
  return spanMs > 200 * 86_400_000 ? monthFmt.format(ms) : dayFmt.format(ms);
}

/** 83 s → "1m 23s". */
export function fmtDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return DASH;
  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds < 60) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

/** Tailwind text colour for a signed value (profit / loss / neutral). */
export function toneClass(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return 'text-secondary-foreground';
  return value > 0 ? 'text-up' : 'text-down';
}
