/**
 * One-click quick-start templates (see `lib/presets.ts`) as selectable shadcn Cards: each card
 * configures the assets, timeframe, costs, direction mode and hypothesis of a common way of trading.
 */

import { Check, Coins, LineChart, TrendingUp, Zap } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { RunConfig } from '@/lib/config';
import { activePreset, PRESETS, type Preset, type PresetIcon } from '@/lib/presets';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { cn } from '@/lib/utils';

const ICONS: Record<PresetIcon, typeof Coins> = { coins: Coins, trend: TrendingUp, stock: LineChart, zap: Zap };

export default function PresetCards({ config, onApply, disabled }: { config: RunConfig; onApply: (preset: Preset) => void; disabled: boolean }): JSX.Element {
  const active = activePreset(config);
  return (
    <ToggleGroup
      type="single"
      value={active?.id ?? ''}
      onValueChange={(id) => {
        const preset = PRESETS.find((entry) => entry.id === id);
        if (preset) onApply(preset);
      }}
      disabled={disabled}
      spacing={2}
      aria-label="Quick-start presets"
      className="grid w-full grid-cols-1 items-stretch sm:grid-cols-2 xl:grid-cols-4"
    >
      {PRESETS.map((preset) => {
        const Icon = ICONS[preset.icon];
        const on = active?.id === preset.id;
        return (
          <ToggleGroupItem
            key={preset.id}
            value={preset.id}
            aria-label={preset.title}
            className="group block h-auto w-full self-stretch rounded-xl p-0 text-left font-normal whitespace-normal hover:bg-transparent disabled:cursor-not-allowed data-[state=on]:bg-transparent"
          >
            <Card
              size="sm"
              className={cn(
                'h-full transition-colors',
                on ? 'bg-primary/10 ring-2 ring-primary' : 'group-hover:bg-accent/50 group-hover:ring-input',
              )}
            >
              <CardHeader className="grid-cols-[1fr_auto]">
                <CardTitle className="flex min-w-0 items-center gap-2 text-sm font-semibold">
                  <span
                    className={cn(
                      'grid size-7 shrink-0 place-items-center rounded-md border',
                      on ? 'border-primary/50 bg-primary/20 text-primary' : 'border-border bg-muted text-secondary-foreground',
                    )}
                  >
                    <Icon className="size-3.5" />
                  </span>
                  <span className="leading-tight">{preset.title}</span>
                </CardTitle>
                {on && <Check className="mt-1 size-4 text-primary" aria-label="Active preset" />}
                <CardDescription className="col-span-2 text-xs leading-snug">{preset.tagline}</CardDescription>
              </CardHeader>
              <CardContent className="mt-auto flex flex-wrap gap-1">
                {preset.facts.map((fact) => (
                  <Badge key={fact} variant="outline" className="num rounded-md px-1.5 text-[10.5px] font-normal text-muted-foreground">
                    {fact}
                  </Badge>
                ))}
              </CardContent>
            </Card>
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}
