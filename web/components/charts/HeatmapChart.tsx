"use client";

import { scoreToColor, fmtPct } from "@/lib/utils";
import { MONTH_NAMES } from "@/types";

interface HeatmapRow {
  airport: string;
  scores: (number | null)[];
}

interface HeatmapChartProps {
  rows: HeatmapRow[];
}

export function HeatmapChart({ rows }: HeatmapChartProps) {
  const months = MONTH_NAMES.slice(1);

  return (
    <div className="overflow-x-auto hide-scrollbar">
      <table className="w-full text-xs border-separate border-spacing-0.5">
        <thead>
          <tr>
            <th className="text-left text-[10px] text-white/30 font-medium tracking-wider uppercase px-2 py-1.5 w-16">
              Origin
            </th>
            {months.map((m) => (
              <th key={m} className="text-center text-[10px] text-white/30 font-medium w-10 py-1.5">
                {m}
              </th>
            ))}
            <th className="text-right text-[10px] text-white/30 font-medium px-2 py-1.5">Avg</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const valid = row.scores.filter((s): s is number => s !== null);
            const avg = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
            return (
              <tr key={row.airport} className="group">
                <td className="px-2 py-1 font-mono font-semibold text-white/70 text-[11px] group-hover:text-white transition-colors">
                  {row.airport}
                </td>
                {row.scores.map((score, mi) => (
                  <td key={mi} className="p-0">
                    <div
                      className="w-full h-8 rounded flex items-center justify-center text-[9px] font-semibold cursor-default transition-transform hover:scale-105 hover:z-10 relative"
                      style={{
                        background: score !== null ? scoreToColor(score) : "rgba(255,255,255,0.04)",
                        color: score !== null && score > 0.15 ? "white" : "rgba(255,255,255,0.5)",
                      }}
                      title={score !== null ? `${row.airport} · ${months[mi]}: ${fmtPct(score)}` : "No data"}
                    >
                      {score !== null ? fmtPct(score, 0) : "–"}
                    </div>
                  </td>
                ))}
                <td className="px-2 text-right">
                  {avg !== null ? (
                    <span
                      className="text-[11px] font-semibold"
                      style={{ color: scoreToColor(avg) }}
                    >
                      {fmtPct(avg, 1)}
                    </span>
                  ) : (
                    <span className="text-white/20">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
