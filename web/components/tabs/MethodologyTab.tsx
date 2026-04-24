"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, CartesianGrid,
  ScatterChart, Scatter, LineChart, Line, ReferenceLine,
} from "recharts";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/common/MetricCard";
import type { EvalData, FeatureImportance } from "@/types";

const GROUP_COLORS: Record<string, string> = {
  "Origin BTS": "#0d73b1",
  "Dest BTS": "#0088CC",
  "Pair BTS": "#14acdc",
  "Temporal": "#7B2D8B",
  "Origin GSOM": "#1a7a4a",
  "Dest GSOM": "#2ca02c",
  "Pair GSOM": "#5cb85c",
  "DFW Hub": "#b61f23",
  "Tail-Chain / Duty": "#8B4513",
  "Airport Cascade": "#f97316",
  "Multi-Hop Cascade": "#B8860B",
  "Other": "#6b7280",
};

const EVAL_METRICS = [
  { label: "Val AUC", value: "0.825", sub: "sequence-level 2024", accent: "red" as const },
  { label: "Val AP", value: "0.445", sub: "sequence-level 2024", accent: "blue" as const },
  { label: "Pair AUC", value: "0.803", sub: "pair-level aggregation", accent: "blue" as const },
  { label: "Pair AP", value: "0.349", sub: "pair-level aggregation", accent: "default" as const },
  { label: "F1 @ 0.11", value: "0.388", sub: "optimized threshold", accent: "default" as const },
];

export function MethodologyTab() {
  const [evalData, setEvalData] = useState<EvalData | null>(null);
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.eval(), api.features()])
      .then(([ev, fi]) => {
        setEvalData(ev);
        setFeatures(fi);
      })
      .finally(() => setLoading(false));
  }, []);

  const top25 = features.slice(0, 25).map((f) => ({
    ...f,
    color: GROUP_COLORS[f.group] ?? "#6b7280",
  }));

  const groupSums = features.reduce<Record<string, number>>((acc, f) => {
    acc[f.group] = (acc[f.group] ?? 0) + f.importance;
    return acc;
  }, {});
  const groupData = Object.entries(groupSums)
    .sort(([, a], [, b]) => b - a)
    .map(([group, importance]) => ({ group, importance, color: GROUP_COLORS[group] ?? "#6b7280" }));

  const rocData = evalData
    ? evalData.fpr.map((fpr, i) => ({ fpr, tpr: evalData.tpr[i] }))
    : [];
  const prData = evalData
    ? evalData.recall.map((rec, i) => ({ rec, prec: evalData.precision[i] }))
    : [];

  return (
    <div className="space-y-6">
      {/* Model card */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        {[
          { label: "Algorithm", value: "XGBoost v7" },
          { label: "Val AUC", value: "0.825" },
          { label: "Val AP", value: "0.445" },
          { label: "Features", value: "70" },
          { label: "Train rows", value: "~398k" },
          { label: "Val split", value: "Time-based" },
        ].map((m, i) => (
          <MetricCard key={m.label} label={m.label} value={m.value} delay={i * 0.05} />
        ))}
      </div>

      {/* Pipeline */}
      <Card>
        <CardHeader><CardTitle>End-to-End Pipeline</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-0 overflow-x-auto hide-scrollbar">
            {[
              { step: "01", label: "BTS", sub: "18M rows · 2015–2024", color: "#0d73b1" },
              { step: "02", label: "NOAA GSOM", sub: "Wind · Precip · Gust", color: "#1a7a4a" },
              { step: "03", label: "Tail-Chain", sub: "FAA Part 117 duty proxy", color: "#8B4513" },
              { step: "04", label: "Feature Eng", sub: "70 features · 8 groups", color: "#7B2D8B" },
              { step: "05", label: "XGBoost v7", sub: "Optuna · hist · time split", color: "#b61f23" },
              { step: "06", label: "Calibration", sub: "Isotonic · 309 points", color: "#2ca02c" },
              { step: "07", label: "Dashboard", sub: "SHAP · Optimizer · Live", color: "#0d73b1" },
            ].map((s, i) => (
              <div key={s.step} className="flex items-center">
                <div className="flex flex-col items-center min-w-[100px] px-3">
                  <div className="text-[9px] font-medium tracking-widest uppercase mb-1.5" style={{ color: s.color }}>
                    {s.step}
                  </div>
                  <div
                    className="text-[11px] font-semibold text-white/80 border px-3 py-1.5 rounded-lg mb-1.5 whitespace-nowrap"
                    style={{ borderColor: `${s.color}40`, background: `${s.color}12` }}
                  >
                    {s.label}
                  </div>
                  <div className="text-[9px] text-white/30 text-center leading-tight">{s.sub}</div>
                </div>
                {i < 6 && <div className="w-6 h-px bg-white/15 shrink-0" />}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Eval metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {EVAL_METRICS.map((m, i) => (
          <MetricCard key={m.label} {...m} delay={i * 0.06} />
        ))}
      </div>

      {/* ROC + PR curves */}
      {!loading && evalData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader><CardTitle>ROC Curve (pair-level)</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rocData} margin={{ top: 0, right: 16, bottom: 16, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} label={{ value: "FPR", position: "insideBottom", offset: -8, fill: "rgba(255,255,255,0.3)", fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} />
                  <ReferenceLine stroke="rgba(255,255,255,0.15)" strokeDasharray="4 3" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                  <Line type="monotone" dataKey="tpr" stroke="#0d73b1" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-center text-xs text-aa-blue-light mt-1">AUC = {evalData.auc.toFixed(3)}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Precision-Recall Curve</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={prData} margin={{ top: 0, right: 16, bottom: 16, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="rec" type="number" domain={[0, 1]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} label={{ value: "Recall", position: "insideBottom", offset: -8, fill: "rgba(255,255,255,0.3)", fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} />
                  <Line type="monotone" dataKey="prec" stroke="#b61f23" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="text-center text-xs text-aa-red mt-1">AP = {evalData?.ap.toFixed(3)}</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Calibration chart */}
      {!loading && evalData && (
        <Card>
          <CardHeader><CardTitle>Calibration by Score Decile — Predicted vs Observed</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={evalData.calibration} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="decile" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `D${v + 1}`} />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  content={({ payload }) => {
                    const d = payload?.[0]?.payload as { decile: number; mean_score: number; mean_obs: number; n: number } | undefined;
                    if (!d) return null;
                    return (
                      <div className="glass rounded-xl p-3 text-xs">
                        <div className="text-white font-medium mb-1">Decile {d.decile + 1}</div>
                        <div className="text-white/50">Predicted: <span className="text-aa-blue-light">{(d.mean_score * 100).toFixed(1)}%</span></div>
                        <div className="text-white/50">Observed: <span className="text-aa-red">{(d.mean_obs * 100).toFixed(1)}%</span></div>
                        <div className="text-white/30">n = {d.n.toLocaleString()}</div>
                      </div>
                    );
                  }}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                />
                <Bar dataKey="mean_score" name="Predicted" fill="#0d73b1" opacity={0.8} radius={[3, 3, 0, 0]} />
                <Bar dataKey="mean_obs" name="Observed" fill="#14acdc" opacity={0.6} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Feature importance */}
      {features.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="md:col-span-2">
            <CardHeader><CardTitle>Top 25 Features by Gain</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={580}>
                <BarChart
                  data={[...top25].reverse()}
                  layout="vertical"
                  margin={{ top: 0, right: 56, bottom: 0, left: 8 }}
                >
                  <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="label" width={190} tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    content={({ payload }) => {
                      const d = payload?.[0]?.payload as FeatureImportance & { color: string } | undefined;
                      if (!d) return null;
                      return (
                        <div className="glass rounded-xl p-3 text-xs">
                          <div className="text-white font-medium">{d.label}</div>
                          <div className="text-white/50 mt-0.5">Group: <span style={{ color: d.color }}>{d.group}</span></div>
                          <div className="text-white/50">Gain: {d.importance.toFixed(5)}</div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {[...top25].reverse().map((f, i) => (
                      <Cell key={i} fill={f.color} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Importance by Group</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={580}>
                <BarChart
                  data={[...groupData].reverse()}
                  layout="vertical"
                  margin={{ top: 0, right: 56, bottom: 0, left: 8 }}
                >
                  <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="group" width={130} tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    content={({ payload }) => {
                      const d = payload?.[0]?.payload as { group: string; importance: number } | undefined;
                      if (!d) return null;
                      return (
                        <div className="glass rounded-xl p-3 text-xs">
                          <div className="text-white font-medium">{d.group}</div>
                          <div className="text-white/50">Total gain: {d.importance.toFixed(4)}</div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {[...groupData].reverse().map((f, i) => (
                      <Cell key={i} fill={f.color} fillOpacity={0.8} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Key findings */}
      <Card>
        <CardHeader><CardTitle>Key Findings</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              title: "Seasonality dominates (26% of importance)",
              body: "Spring/summer, not winter, is riskiest. DFW is a convective storm hub — afternoon thunderstorms peak June–August, generating rapid-onset ground stops.",
              color: "#7B2D8B",
            },
            {
              title: "Destination-side weather > origin-side",
              body: "B_avg_wind_speed (11.1%) outranks A_avg_wind_speed (2.5%). Outbound leg is more constrained — crew has absorbed inbound delays, shorter buffer, FDP limits approaching.",
              color: "#0d73b1",
            },
            {
              title: "Cascade amplification is 4th most important",
              body: "tc_cascade_amplif_mean — ratio of outbound to inbound delay minutes — captures sequences where a 20-min inbound routinely becomes a 45-min outbound.",
              color: "#8B4513",
            },
            {
              title: "Multi-hop depth > multi-hop rate",
              body: "mhc_n_hops_mean (3.5%) ranks above mhc_cascade_hop_rate (0.7%). High-degree nodes in the DFW rotation network carry systemic risk even in good weather.",
              color: "#B8860B",
            },
            {
              title: "FDP overrun is a leading indicator",
              body: "tc_fdp_overrun_rate (1.2%) predicts disruption before it happens. Crews flying near FAA Part 117 FDP limits have elevated bad rates.",
              color: "#2ca02c",
            },
            {
              title: "Optimizer saves 15–25% total risk",
              body: "Hungarian algorithm on a representative daily schedule reduces total risk vs random assignment by 15–25%, concentrating gains in the moderate-risk band.",
              color: "#b61f23",
            },
          ].map((f) => (
            <motion.div
              key={f.title}
              whileHover={{ scale: 1.01 }}
              className="rounded-xl p-4 border-l-2"
              style={{ borderLeftColor: f.color, background: `${f.color}0a` }}
            >
              <div className="text-sm font-semibold text-white mb-1.5">{f.title}</div>
              <div className="text-xs text-white/45 leading-relaxed">{f.body}</div>
            </motion.div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
