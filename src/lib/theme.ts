/**
 * Theme tokens for canvas drawing. Chart.js paints on a <canvas>, which cannot read CSS variables,
 * so charts resolve the shadcn/ui theme tokens (`--primary`, `--muted-foreground`, …) defined in
 * `index.css` to concrete colours at render time. DOM elements should use the Tailwind classes
 * (`text-muted-foreground`, `bg-card`, …) or `var(--token)` directly instead.
 */

/** Current value of a theme token (without the leading `--`), or `fallback` outside a browser. */
export function themeColor(token: string, fallback: string): string {
  if (typeof window === 'undefined' || typeof document === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim();
  return value || fallback;
}

/** The palette the canvas charts draw with, read from the active theme. */
export interface ChartPalette {
  strategy: string;
  benchmark: string;
  foreground: string;
  muted: string;
  secondary: string;
  axis: string;
  surface: string;
  popover: string;
}

export function chartPalette(): ChartPalette {
  return {
    strategy: themeColor('chart-1', '#3987e5'),
    benchmark: themeColor('bench', '#d95926'),
    foreground: themeColor('foreground', '#e8ebf3'),
    muted: themeColor('muted-foreground', '#7f8aa5'),
    secondary: themeColor('secondary-foreground', '#aab3c8'),
    axis: themeColor('input', '#2c3856'),
    surface: themeColor('card', '#0f1524'),
    popover: themeColor('popover', '#121a2c'),
  };
}
