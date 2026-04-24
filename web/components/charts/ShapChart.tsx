"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine, ResponsiveContainer } from "recharts";
import type { ShapFeature } from "@/types";
import { cn } from "@/lib/utils";

interface ShapChartProps {
  features: ShapFeature[];
}

interface TooltipPayload {
  payload?: {
    label: string;
    shap_value: number;
    feature_value: number;
    imputed: boolean;
  };
}

function CustomTooltip({ payload }: TooltipPayload) {
  if (!payload) return null;
  const d = payload;
  return (
    <div className="glass rounded-xl p-3 text-xs max-w-[240px] shadow-glass-lg">
      <div className="text-white font-medium mb-1.5 leading-snug">{d.label}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-white/40">SHAP</span>
        <span className={cn("font-mono font-semibold", d.shap_value > 0 ? "text-red-400" : "text-emerald-400")}>
          {d.shap_value > 0 ? "+" : ""}{d.shap_value.toFixed(4)}
        </span>
        <span className="text-white/40">Value</span>
        <span className="text-white/70 font-mono">{d.feature_value.toFixed(3)}</span>
        {d.imputed && (
          <>
            <span className="text-amber-400 col-span-2 mt-1">★ GSOM median imputed</span>
          </>
        )}
      </div>
    </div>
  );
}

export function ShapChart({ features }: ShapChartProps) {
  const sorted = [...features].sort((a, b) => a.shap_value - b.shap_value);

  return (
    <div className="w-full">
      <div className="flex items-center gap-4 mb-4 text-[10px] text-white/40 uppercase tracking-wider">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-sm bg-red-400" />
          Increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-sm bg-emerald-400" />
          Decreases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="text-amber-400">★</span>
          GSOM imputed
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(300, sorted.length * 30)}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 0, right: 60, bottom: 0, left: 8 }}
        >
          <XAxis
            type="number"
            tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={200}
            tick={({ y, payload }: { y: number; payload: { value: string } }) => {
              const feat = sorted.find((f) => f.label === payload.value);
              return (
                <text y={y} x={-4} dy={4} textAnchor="end" fontSize={10} fill={feat?.imputed ? "#fbbf24" : "rgba(255,255,255,0.55)"}>
                  {feat?.imputed ? `★ ${payload.value}` : payload.value}
                </text>
              );
            }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={({ payload }) => {
              const d = payload?.[0]?.payload as ShapFeature | undefined;
              if (!d) return null;
              return <CustomTooltip payload={d as unknown as TooltipPayload["payload"]} />;
            }}
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
          <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
            {sorted.map((entry, i) => (
              <Cell
                key={i}
                fill={
                  entry.shap_value > 0
                    ? entry.imputed ? "rgba(251,191,36,0.7)" : "rgba(239,68,68,0.75)"
                    : "rgba(34,197,94,0.75)"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
