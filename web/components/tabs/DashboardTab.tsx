"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { api } from "@/lib/api";
import { scoreToColor, fmtPct } from "@/lib/utils";
import { HeatmapChart } from "@/components/charts/HeatmapChart";
import { MetricCard } from "@/components/common/MetricCard";
import { RiskBadge } from "@/components/common/RiskBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TopPair, PairScore } from "@/types";
import { MONTH_NAMES, MONTH_FULL } from "@/types";

const TOP_ORIGINS = ["ATL", "ORD", "LAX", "MCO", "JFK", "LGA", "MIA", "BOS", "DEN", "SEA", "PHX", "LAS", "SFO", "CLT", "EWR"];

export function DashboardTab() {
  const [month, setMonth] = useState(6);
  const [topN, setTopN] = useState(20);
  const [topPairs, setTopPairs] = useState<TopPair[]>([]);
  const [heatmapRows, setHeatmapRows] = useState<{ airport: string; scores: (number | null)[] }[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, pctHigh: 0, avgRisk: 0, riskiestOrigin: "" });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [top, scores] = await Promise.all([
        api.topPairs(month, topN),
        api.allScores(month),
      ]);
      setTopPairs(top);

      const pctHigh = scores.filter((s: PairScore) => s.avg_risk_score >= 0.3).length / (scores.length || 1);
      const avgRisk = scores.reduce((a: number, s: PairScore) => a + s.avg_risk_score, 0) / (scores.length || 1);
      const byOrigin = scores.reduce((acc: Record<string, number[]>, s: PairScore) => {
        if (!acc[s.airport_A]) acc[s.airport_A] = [];
        acc[s.airport_A].push(s.avg_risk_score);
        return acc;
      }, {});
      const riskiestOrigin = Object.entries(byOrigin).sort(
        ([, a], [, b]) => b.reduce((x, y) => x + y, 0) / b.length - a.reduce((x, y) => x + y, 0) / a.length
      )[0]?.[0] ?? "—";
      setStats({ total: scores.length, pctHigh, avgRisk, riskiestOrigin });

      // Build heatmap data for top origins from all-month scores
      const allScores = await api.allScores();
      const rows = TOP_ORIGINS.map((origin) => {
        const scores12 = Array.from({ length: 12 }, (_, i) => {
          const found = allScores.find(
            (s: PairScore) => s.airport_A === origin && s.Month === i + 1
          );
          return found ? found.avg_risk_score : null;
        });
        return { airport: origin, scores: scores12 };
      }).filter((r) => r.scores.some((s) => s !== null));
      setHeatmapRows(rows);
    } finally {
      setLoading(false);
    }
  }, [month, topN]);

  useEffect(() => { loadData(); }, [loadData]);

  const barData = topPairs.map((p) => ({
    name: `${p.airport_A}→${p.airport_B}`,
    risk: p.avg_risk_score,
    obs: p.observed_bad_rate,
  }));

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3">
          <span className="text-xs text-white/40 uppercase tracking-wider">Month</span>
          <div className="flex gap-1">
            {MONTH_NAMES.slice(1).map((m, i) => (
              <button
                key={m}
                onClick={() => setMonth(i + 1)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors ${
                  month === i + 1
                    ? "bg-aa-blue-light/20 text-aa-blue-light border border-aa-blue-light/30"
                    : "text-white/40 hover:text-white/70 hover:bg-white/5"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-white/40">Top</span>
          {[10, 20, 50].map((n) => (
            <button
              key={n}
              onClick={() => setTopN(n)}
              className={`px-3 py-1 rounded-lg text-xs transition-colors ${
                topN === n ? "bg-white/10 text-white" : "text-white/40 hover:text-white/70"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Pairs Analyzed" value={loading ? "—" : stats.total.toLocaleString()} delay={0} />
        <MetricCard
          label="High Risk ≥30%"
          value={loading ? "—" : fmtPct(stats.pctHigh)}
          accent="red"
          delay={0.05}
        />
        <MetricCard
          label="Avg Calibrated Risk"
          value={loading ? "—" : fmtPct(stats.avgRisk)}
          accent="blue"
          delay={0.1}
        />
        <MetricCard
          label="Riskiest Origin"
          value={loading ? "—" : stats.riskiestOrigin}
          sub={MONTH_FULL[month]}
          delay={0.15}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Bar chart */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>Top {topN} Riskiest Sequences — {MONTH_FULL[month]}</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-64 flex items-center justify-center text-white/30 text-sm">Loading…</div>
            ) : (
              <ResponsiveContainer width="100%" height={Math.max(280, barData.length * 22)}>
                <BarChart data={barData} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 8 }}>
                  <CartesianGrid horizontal={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    type="number"
                    domain={[0, 0.6]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={108}
                    tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10 }}
                    axisLine={false} tickLine={false}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    content={({ payload, label }) => {
                      const d = payload?.[0]?.payload as { risk: number; obs: number } | undefined;
                      if (!d) return null;
                      return (
                        <div className="glass rounded-xl p-3 text-xs shadow-glass-lg">
                          <div className="text-white font-medium mb-1">{label}</div>
                          <div className="text-white/50">Model risk: <span className="text-white">{fmtPct(d.risk)}</span></div>
                          <div className="text-white/50">Observed: <span className="text-white">{fmtPct(d.obs ?? 0)}</span></div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="risk" radius={[0, 4, 4, 0]}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={scoreToColor(entry.risk)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Risk table */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Risk Table</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-auto max-h-[380px] hide-scrollbar">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/6">
                    {["Route", "Risk", "Obs %", "N"].map((h) => (
                      <th key={h} className="text-left text-[10px] text-white/35 font-medium tracking-wider uppercase px-4 py-2.5">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {topPairs.slice(0, 30).map((p, i) => (
                    <tr
                      key={i}
                      className="border-b border-white/4 hover:bg-white/4 transition-colors"
                    >
                      <td className="px-4 py-2 font-mono text-white/70">
                        {p.airport_A}→{p.airport_B}
                      </td>
                      <td className="px-4 py-2">
                        <RiskBadge
                          label={p.avg_risk_score >= 0.3 ? "HIGH RISK" : p.avg_risk_score >= 0.2 ? "MODERATE RISK" : "LOW RISK"}
                          score={p.avg_risk_score}
                          size="sm"
                        />
                      </td>
                      <td className="px-4 py-2 text-white/50">
                        {fmtPct(p.observed_bad_rate ?? 0)}
                      </td>
                      <td className="px-4 py-2 text-white/35">{p.n_sequences ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Monthly heatmap */}
      <Card>
        <CardHeader>
          <CardTitle>Monthly Risk Heatmap — Top Origins (All Months)</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="h-32 flex items-center justify-center text-white/30 text-sm">Building heatmap…</div>
          ) : (
            <HeatmapChart rows={heatmapRows} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
