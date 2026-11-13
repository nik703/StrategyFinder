/**
 * Interactive equity-curve chart: selected strategy vs. the Buy & Hold
 * benchmark, with a dashed vertical divider at the 70% in-sample / 30%
 * out-of-sample boundary, a crosshair and a shared tooltip.
 */

import {
  Chart as ChartJS,
  Filler,
  LinearScale,
  LineController,
  LineElement,
  LogarithmicScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin,
} from 'chart.js';
import { LineChart } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { fmtAxisDate, fmtDate, fmtDateTime, fmtPct, fmtUsd, fmtUsdCompact } from '../lib/format';
import type { ChartAxis, Interval } from '../types/trading';
import { LineKey, Segmented } from './ui';

ChartJS.register(LineController, LineElement, PointElement, LinearScale, LogarithmicScale, Tooltip, Filler);

/** Series colours – validated for CVD separation and ≥3:1 contrast on the panel surface. */
export const STRATEGY_COLOR = '#3987e5';
export const BENCHMARK_COLOR = '#d95926';

const INK = '#e8ebf3';
const INK_MUTED = '#7f8aa5';
const GRID = 'rgba(148, 163, 184, 0.09)';
const AXIS = '#2c3856';
const FONT = "'Inter Variable', ui-sans-serif, system-ui, sans-serif";

interface EquityChartProps {
  axis: ChartAxis | null;
  benchmark: Float32Array | null;
  strategy: Float32Array | null;
  /** Legend name of the plotted strategy, e.g. "Strategy S-4821". */
  strategyName: string | null;
  initialCapital: number;
  interval: Interval;
  /** Shown instead of the chart before the first run. */
  placeholder: string;
}

interface SplitOptions {
  splitPos: number;
}

/** Shades the out-of-sample region and draws the dashed in-sample / out-of-sample divider. */
const splitDividerPlugin: Plugin<'line', SplitOptions> = {
  id: 'splitDivider',
  beforeDatasetsDraw(chart, _args, options) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.x || !(options.splitPos > 0)) return;
    const x = scales.x.getPixelForValue(options.splitPos);
    if (x <= chartArea.left || x >= chartArea.right) return;

    ctx.save();
    // Out-of-sample region: a barely-there wash so the boundary reads at a glance.
    ctx.fillStyle = 'rgba(148, 163, 184, 0.045)';
    ctx.fillRect(x, chartArea.top, chartArea.right - x, chartArea.bottom - chartArea.top);

    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = 'rgba(232, 235, 243, 0.55)';
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = `600 10px ${FONT}`;
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(170, 179, 200, 0.95)';
    const y = chartArea.top + 6;
    const left = 'IN-SAMPLE · 70%';
    const right = 'OUT-OF-SAMPLE · 30%';
    if (x - chartArea.left > ctx.measureText(left).width + 16) {
      ctx.textAlign = 'right';
      ctx.fillText(left, x - 8, y);
    }
    if (chartArea.right - x > ctx.measureText(right).width + 16) {
      ctx.textAlign = 'left';
      ctx.fillText(right, x + 8, y);
    }
    ctx.restore();
  },
};

/** Vertical hairline that follows the hovered data position. */
const crosshairPlugin: Plugin<'line'> = {
  id: 'crosshair',
  afterDatasetsDraw(chart) {
    const active = chart.tooltip?.getActiveElements() ?? [];
    if (active.length === 0) return;
    const { ctx, chartArea } = chart;
    const x = active[0].element.x;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(232, 235, 243, 0.35)';
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

const SCALE_OPTIONS: { value: 'linear' | 'log'; label: string }[] = [
  { value: 'linear', label: 'Linear' },
  { value: 'log', label: 'Log' },
];

export default function EquityChart({
  axis,
  benchmark,
  strategy,
  strategyName,
  initialCapital,
  interval,
  placeholder,
}: EquityChartProps): JSX.Element {
  const [scale, setScale] = useState<'linear' | 'log'>('linear');

  const data = useMemo<ChartData<'line', { x: number; y: number }[]> | null>(() => {
    if (!axis || !benchmark) return null;
    const toPoints = (curve: Float32Array): { x: number; y: number }[] => {
      const points = new Array<{ x: number; y: number }>(curve.length);
      for (let i = 0; i < curve.length; i++) points[i] = { x: i, y: curve[i] };
      return points;
    };
    const datasets: ChartData<'line', { x: number; y: number }[]>['datasets'] = [];
    if (strategy && strategy.length === axis.time.length) {
      datasets.push({
        label: strategyName ?? 'Strategy',
        data: toPoints(strategy),
        borderColor: STRATEGY_COLOR,
        backgroundColor: 'rgba(57, 135, 229, 0.10)',
        pointHoverBackgroundColor: STRATEGY_COLOR,
        fill: scale === 'linear' ? 'origin' : false,
        order: 1,
      });
    }
    datasets.push({
      label: 'Buy & Hold',
      data: toPoints(benchmark),
      borderColor: BENCHMARK_COLOR,
      backgroundColor: BENCHMARK_COLOR,
      pointHoverBackgroundColor: BENCHMARK_COLOR,
      fill: false,
      order: 2,
    });
    return { datasets };
  }, [axis, benchmark, strategy, strategyName, scale]);

  const options = useMemo<ChartOptions<'line'>>(() => {
    const time = axis?.time ?? new Float64Array(0);
    const last = Math.max(1, time.length - 1);
    const span = time.length > 1 ? time[time.length - 1] - time[0] : 0;
    const splitPos = axis?.splitPos ?? 0;
    const at = (value: number): number => time[Math.min(time.length - 1, Math.max(0, Math.round(value)))];
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      normalized: true,
      interaction: { mode: 'index', intersect: false, axis: 'x' },
      layout: { padding: { top: 4, right: 4 } },
      elements: {
        line: { borderWidth: 2, tension: 0, borderJoinStyle: 'round', borderCapStyle: 'round' },
        point: { radius: 0, hoverRadius: 4, hitRadius: 12, hoverBorderWidth: 2, hoverBorderColor: '#0f1524' },
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: last,
          grid: { display: false },
          border: { color: AXIS },
          ticks: {
            color: INK_MUTED,
            font: { family: FONT, size: 11 },
            maxTicksLimit: 8,
            maxRotation: 0,
            autoSkipPadding: 24,
            callback: (value) => fmtAxisDate(at(Number(value)), span),
          },
        },
        y: {
          type: scale === 'log' ? 'logarithmic' : 'linear',
          position: 'right',
          grid: { color: GRID, drawTicks: false },
          border: { display: false },
          ticks: {
            color: INK_MUTED,
            font: { family: FONT, size: 11 },
            padding: 8,
            maxTicksLimit: 7,
            callback: (value) => fmtUsdCompact(Number(value)),
          },
        },
      },
      plugins: {
        legend: { display: false },
        splitDivider: { splitPos },
        tooltip: {
          backgroundColor: '#151c2f',
          borderColor: AXIS,
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          caretSize: 0,
          titleColor: '#aab3c8',
          titleFont: { family: FONT, size: 11, weight: 500 },
          titleMarginBottom: 6,
          bodyColor: INK,
          bodyFont: { family: FONT, size: 12, weight: 600 },
          bodySpacing: 4,
          footerColor: INK_MUTED,
          footerFont: { family: FONT, size: 10, weight: 500 },
          footerMarginTop: 6,
          boxWidth: 14,
          boxHeight: 2,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            title: (items) => {
              if (items.length === 0) return '';
              const ms = at(items[0].parsed.x ?? 0);
              return interval === '1d' ? fmtDate(ms) : fmtDateTime(ms);
            },
            // Values lead, series names follow.
            label: (item) => {
              const value = item.parsed.y ?? 0;
              const roi = (value / initialCapital - 1) * 100;
              return ` ${fmtUsd(value)}  ${fmtPct(roi, true)}  ·  ${item.dataset.label ?? ''}`;
            },
            labelPointStyle: () => ({ pointStyle: 'line', rotation: 0 }),
            labelColor: (item) => {
              const color = String(item.dataset.borderColor);
              return { borderColor: color, backgroundColor: color, borderWidth: 2 };
            },
            footer: (items) =>
              items.length > 0 && (items[0].parsed.x ?? 0) >= splitPos ? 'Out-of-sample segment' : 'In-sample segment',
          },
        },
      },
    } as ChartOptions<'line'>;
  }, [axis, scale, interval, initialCapital]);

  const lastOf = (curve: Float32Array | null): number | null => (curve && curve.length > 0 ? curve[curve.length - 1] : null);
  const strategyEnd = lastOf(strategy);
  const benchmarkEnd = lastOf(benchmark);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 pb-2">
        {/* Legend: line keys + final equity, always visible for the two series. */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
          <span className="inline-flex items-center gap-2">
            <LineKey color={STRATEGY_COLOR} />
            <span className="text-ink-2">{strategyName ?? 'Selected strategy'}</span>
            <span className="num font-semibold text-ink">{fmtUsd(strategyEnd)}</span>
          </span>
          <span className="inline-flex items-center gap-2">
            <LineKey color={BENCHMARK_COLOR} />
            <span className="text-ink-2">Buy &amp; Hold</span>
            <span className="num font-semibold text-ink">{fmtUsd(benchmarkEnd)}</span>
          </span>
          <span className="inline-flex items-center gap-2 text-ink-3">
            <LineKey color="rgba(232, 235, 243, 0.55)" dashed />
            70% in-sample / 30% out-of-sample
          </span>
        </div>
        <Segmented ariaLabel="Y-axis scale" size="sm" options={SCALE_OPTIONS} value={scale} onChange={setScale} />
      </div>

      <div className="relative min-h-0 flex-1 px-2 pb-2">
        {data ? (
          <Line
            data={data}
            options={options}
            plugins={[splitDividerPlugin, crosshairPlugin]}
            aria-label="Equity curve of the selected strategy compared with Buy & Hold"
            role="img"
          />
        ) : (
          <div className="grid h-full place-items-center">
            <div className="max-w-sm text-center">
              <div className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-xl border border-line bg-raised text-ink-3">
                <LineChart size={20} />
              </div>
              <p className="text-sm text-ink-2">{placeholder}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
