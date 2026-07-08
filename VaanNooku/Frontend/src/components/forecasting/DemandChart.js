"use client";

import { useMemo, useState } from "react";
import { Table2, LineChart as LineChartIcon } from "lucide-react";

const WIDTH = 600;
const HEIGHT = 260;
const PAD_LEFT = 44;
const PAD_RIGHT = 20;
const PAD_TOP = 16;
const PAD_BOTTOM = 32;

const SERIES = [
  { key: "raw", label: "Raw ML signal", color: "var(--chart-1)", dash: null },
  { key: "adjusted", label: "Decision-adjusted output", color: "var(--chart-2)", dash: null },
  { key: "actual", label: "Actual sales (historical)", color: "var(--chart-muted)", dash: "4 3" },
];

function niceMax(value) {
  const step = 250;
  return Math.ceil((value * 1.15) / step) * step || step;
}

export function DemandChart({ days, raw, adjusted, actual, capacity, isBreaching }) {
  const [hoverIndex, setHoverIndex] = useState(null);
  const [showTable, setShowTable] = useState(false);

  const maxVal = useMemo(() => {
    const values = [...raw, ...adjusted, ...actual];
    if (Number.isFinite(capacity)) values.push(capacity);
    return niceMax(Math.max(...values));
  }, [raw, adjusted, actual, capacity]);

  const plotWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;

  const xFor = (i) => PAD_LEFT + (i / (days.length - 1)) * plotWidth;
  const yFor = (v) => PAD_TOP + plotHeight - (v / maxVal) * plotHeight;

  const buildPath = (series) =>
    series.map((v, i) => `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(v)}`).join(" ");

  const ySteps = 4;
  const yTicks = Array.from({ length: ySteps + 1 }, (_, i) => Math.round((maxVal / ySteps) * i));

  const handleMove = (evt) => {
    const svg = evt.currentTarget;
    const rect = svg.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * WIDTH;
    const ratio = Math.max(0, Math.min(1, (x - PAD_LEFT) / plotWidth));
    const idx = Math.round(ratio * (days.length - 1));
    setHoverIndex(idx);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <ul className="flex flex-wrap items-center gap-4 text-[10px] text-zinc-600 dark:text-zinc-400">
          {SERIES.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-0.5 rounded-full"
                style={{ background: s.color, opacity: s.dash ? 0.7 : 1 }}
              />
              <span>{s.label}</span>
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={() => setShowTable((v) => !v)}
          className="flex items-center gap-1.5 text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-emerald-600 dark:hover:text-emerald-400 border border-zinc-200 dark:border-zinc-800 rounded-lg px-2 py-1"
        >
          {showTable ? <LineChartIcon className="w-3.5 h-3.5" /> : <Table2 className="w-3.5 h-3.5" />}
          {showTable ? "View chart" : "View table"}
        </button>
      </div>

      {showTable ? (
        <div className="overflow-x-auto border border-zinc-200 dark:border-zinc-800 rounded-xl">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="bg-zinc-50 dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400 text-[9px] uppercase tracking-wider">
                <th className="p-2.5">Day</th>
                <th className="p-2.5 text-right">Raw ML signal</th>
                <th className="p-2.5 text-right">Decision-adjusted</th>
                <th className="p-2.5 text-right">Actual sales</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-900">
              {days.map((day, i) => (
                <tr key={day}>
                  <td className="p-2.5 font-semibold text-zinc-800 dark:text-zinc-200">{day}</td>
                  <td className="p-2.5 text-right tabular-nums">{raw[i]}</td>
                  <td className="p-2.5 text-right tabular-nums font-semibold text-emerald-600 dark:text-emerald-400">
                    {adjusted[i]}
                  </td>
                  <td className="p-2.5 text-right tabular-nums text-zinc-500">{actual[i]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="relative bg-zinc-50 dark:bg-zinc-950/60 border border-zinc-200 dark:border-zinc-800/80 rounded-xl p-3">
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className="w-full h-[220px]"
            onMouseMove={handleMove}
            onMouseLeave={() => setHoverIndex(null)}
            role="img"
            aria-label="Demand forecast chart comparing raw ML signal, decision-adjusted output and actual sales"
          >
            {yTicks.map((tick) => (
              <g key={tick}>
                <line
                  x1={PAD_LEFT}
                  x2={WIDTH - PAD_RIGHT}
                  y1={yFor(tick)}
                  y2={yFor(tick)}
                  className="stroke-zinc-200 dark:stroke-zinc-800"
                  strokeWidth="1"
                />
                <text x={PAD_LEFT - 8} y={yFor(tick) + 3} textAnchor="end" className="fill-zinc-400 dark:fill-zinc-600 text-[8px]">
                  {tick}
                </text>
              </g>
            ))}

            {days.map((day, i) => (
              <text
                key={day}
                x={xFor(i)}
                y={HEIGHT - 10}
                textAnchor="middle"
                className="fill-zinc-500 dark:fill-zinc-400 text-[9px] font-semibold"
              >
                {day}
              </text>
            ))}

            {Number.isFinite(capacity) && capacity <= maxVal && (
              <g>
                <line
                  x1={PAD_LEFT}
                  x2={WIDTH - PAD_RIGHT}
                  y1={yFor(capacity)}
                  y2={yFor(capacity)}
                  stroke={isBreaching ? "var(--chart-critical)" : "var(--chart-warning)"}
                  strokeWidth="1.5"
                  strokeDasharray="5 3"
                />
                <text
                  x={WIDTH - PAD_RIGHT}
                  y={yFor(capacity) - 5}
                  textAnchor="end"
                  fill={isBreaching ? "var(--chart-critical)" : "var(--chart-warning)"}
                  className="text-[8px] font-bold"
                >
                  CAPACITY CAP · {capacity}/day
                </text>
              </g>
            )}

            <path d={buildPath(actual)} fill="none" stroke="var(--chart-muted)" strokeWidth="2" strokeDasharray="4 3" opacity="0.75" strokeLinecap="round" />
            <path d={buildPath(raw)} fill="none" stroke="var(--chart-1)" strokeWidth="2" strokeLinecap="round" />
            <path d={buildPath(adjusted)} fill="none" stroke="var(--chart-2)" strokeWidth="2.5" strokeLinecap="round" />

            {raw.map((v, i) => (
              <circle key={`raw-${i}`} cx={xFor(i)} cy={yFor(v)} r="4" fill="var(--chart-1)" stroke="var(--background,#fff)" className="stroke-white dark:stroke-zinc-950" strokeWidth="2" />
            ))}
            {adjusted.map((v, i) => (
              <circle key={`adj-${i}`} cx={xFor(i)} cy={yFor(v)} r="4.5" fill="var(--chart-2)" stroke="var(--background,#fff)" className="stroke-white dark:stroke-zinc-950" strokeWidth="2" />
            ))}

            {hoverIndex !== null && (
              <line
                x1={xFor(hoverIndex)}
                x2={xFor(hoverIndex)}
                y1={PAD_TOP}
                y2={HEIGHT - PAD_BOTTOM}
                className="stroke-zinc-400 dark:stroke-zinc-600"
                strokeWidth="1"
              />
            )}
          </svg>

          {hoverIndex !== null && (
            <div
              className="absolute z-10 bg-white dark:bg-zinc-950 border border-zinc-250 dark:border-zinc-800 p-2.5 rounded-lg text-[10px] space-y-1 shadow-xl pointer-events-none min-w-[150px]"
              style={{
                left: `${(xFor(hoverIndex) / WIDTH) * 100}%`,
                top: `${(yFor(Math.max(raw[hoverIndex], adjusted[hoverIndex])) / HEIGHT) * 100}%`,
                transform: "translate(-50%, -115%)",
              }}
            >
              <div className="font-bold text-zinc-800 dark:text-zinc-100 border-b border-zinc-150 dark:border-zinc-850 pb-1 mb-1">
                {days[hoverIndex]}
              </div>
              {SERIES.map((s) => {
                const value = s.key === "raw" ? raw[hoverIndex] : s.key === "adjusted" ? adjusted[hoverIndex] : actual[hoverIndex];
                return (
                  <div key={s.key} className="flex items-center justify-between gap-4">
                    <span className="flex items-center gap-1.5 text-zinc-500 dark:text-zinc-400">
                      <span className="inline-block w-2 h-0.5 rounded-full" style={{ background: s.color }} />
                      {s.label}
                    </span>
                    <span className="font-bold text-zinc-900 dark:text-white tabular-nums">{value}u</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
