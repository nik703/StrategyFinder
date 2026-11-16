/**
 * One-click quick-start templates (see `lib/presets.ts`): each card configures the assets,
 * timeframe, costs, direction mode and hypothesis of a common way of trading.
 */

import { Check, Coins, LineChart, TrendingUp, Zap } from 'lucide-react';
import type { RunConfig } from '../lib/config';
import { activePreset, PRESETS, type Preset, type PresetIcon } from '../lib/presets';
import { cx } from './ui';

const ICONS: Record<PresetIcon, typeof Coins> = { coins: Coins, trend: TrendingUp, stock: LineChart, zap: Zap };

export default function PresetCards({
  config,
  onApply,
  disabled,
}: {
  config: RunConfig;
  onApply: (preset: Preset) => void;
  disabled: boolean;
}): JSX.Element {
  const active = activePreset(config);
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4" role="group" aria-label="Quick-start presets">
      {PRESETS.map((preset) => {
        const Icon = ICONS[preset.icon];
        const on = active?.id === preset.id;
        return (
          <button
            key={preset.id}
            type="button"
            disabled={disabled}
            aria-pressed={on}
            onClick={() => onApply(preset)}
            className={cx(
              'group flex min-w-0 flex-col gap-2 rounded-lg border p-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60',
              on ? 'border-accent bg-accent/10' : 'border-line bg-base hover:border-line-strong hover:bg-hover/50',
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="flex min-w-0 items-center gap-2">
                <span
                  className={cx(
                    'grid h-7 w-7 shrink-0 place-items-center rounded-md border',
                    on ? 'border-accent/50 bg-accent/20 text-accent-strong' : 'border-line bg-raised text-ink-2',
                  )}
                >
                  <Icon size={14} />
                </span>
                <span className="text-sm font-semibold leading-tight text-ink">{preset.title}</span>
              </span>
              {on && <Check size={15} className="mt-1 shrink-0 text-accent-strong" aria-label="Active preset" />}
            </div>
            <p className="text-xs leading-snug text-ink-2">{preset.tagline}</p>
            <ul className="num mt-auto flex flex-wrap gap-1">
              {preset.facts.map((fact) => (
                <li key={fact} className="rounded border border-line bg-raised/60 px-1.5 py-0.5 text-[10.5px] text-ink-3">
                  {fact}
                </li>
              ))}
            </ul>
          </button>
        );
      })}
    </div>
  );
}
