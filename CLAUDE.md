# CLAUDE.md – StrategyFinder conventions

StrategyFinder's UI is built on **shadcn/ui** (Radix primitives, `radix-nova` style) with
**Tailwind CSS v4** and **lucide-react** icons. Follow these rules for every change to the frontend.

## 1. Use shadcn/ui primitives and lucide-react icons

- Build UI from the primitives in `@/components/ui/*` (`Button`, `Card`, `Badge`, `Table`, `Tabs`,
  `Progress`, `Slider`, `Switch`, `Select`, `Dialog`, `Tooltip`, `Alert`, `Separator`, `Collapsible`,
  `Input`, `Label`, `Checkbox`, `ToggleGroup`, …), imported through the `@/` alias:

  ```tsx
  import { Button } from '@/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
  ```

- Use icons from `lucide-react` only. Do not add other icon sets or inline SVG icons.
- App-level building blocks composed from those primitives (`Panel`, `SelectField`, `SliderField`,
  `NumberField`, `Segmented`, `Pill`, `InfoTip`, `StepHeader`, …) live in `src/components/common.tsx`.
  Reuse them before writing a new wrapper.

## 2. Check `src/components/ui/` first, then add missing primitives with the CLI

- Before building a control, check whether `src/components/ui/` already has it.
- If it does not, install it with the shadcn CLI instead of hand-writing one:

  ```bash
  npx shadcn@latest add <component>
  ```

- Do **not** use raw `<button>`, `<input>`, `<select>`, `<textarea>`, `<table>` or `<dialog>`
  elements, and do not write custom tab, modal, dropdown, tooltip or toggle wrappers. Use `Button`,
  `Input`, `Select`, `Table`, `Dialog`, `Tabs`, `Tooltip`, `ToggleGroup`, etc.
- Files in `src/components/ui/` are generated. Prefer styling through `className` at the call site
  over editing them. If a primitive must change, keep the change small and note why.
- The project runs **React 18**, but the shadcn CLI generates React 19 style components (no
  `forwardRef`). A primitive that is used as an `asChild` target (`Button`) or receives a ref from Radix
  (`DialogOverlay`) has been wrapped in `React.forwardRef` locally. Re-apply that wrapper if the CLI
  overwrites one of them, and add it to any new primitive that shows React's "Function components
  cannot be given refs" warning in the console.

## 3. Use semantic Tailwind tokens, not hard-coded colours

- Colours come from the theme tokens defined in `src/index.css`: `bg-background`, `bg-card`,
  `bg-popover`, `bg-muted`, `bg-primary`, `text-foreground`, `text-muted-foreground`,
  `text-secondary-foreground`, `text-primary`, `border-border`, `border-input`, `ring-ring`, the
  `chart-1` … `chart-5` series colours, and the app-specific `up` (profit), `down` (loss), `warn` and
  `bench` (Buy & Hold) tokens.
- Do not use hex/rgb values or raw Tailwind palette classes (`bg-slate-900`, `text-gray-400`, …) in
  components. Add a new token to `index.css` (both `:root` and `.dark`, plus the `@theme inline`
  mapping) if one is genuinely missing.
- In SVG and inline styles use `var(--token)`. Canvas charts (Chart.js) cannot read CSS variables:
  resolve tokens with `themeColor` / `chartPalette` from `@/lib/theme`.

## Scope boundaries

The UI layer must not change trading logic. Backtesting math (`src/engine/backtester.ts`,
`src/lib/stats.ts`), the genetic optimiser and worker protocols (`src/workers/*`,
`src/engine/evaluation.ts`, `src/types/trading.ts`), `src/engine/nextEntrySolver.ts` and the Pine
Script v5 exporter (`src/engine/pine.ts`) are owned by the engine. UI changes only read their
outputs.

## Checks before committing

```bash
npm run typecheck   # tsc --noEmit for the app and the tests
npm run build
npm test
```
