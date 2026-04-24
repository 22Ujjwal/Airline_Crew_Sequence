"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { Search, ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { fmtPct, scoreToColor } from "@/lib/utils";
import { RiskGauge } from "@/components/charts/RiskGauge";
import { ShapChart } from "@/components/charts/ShapChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import type { PredictResult } from "@/types";
import { MONTH_FULL } from "@/types";

const MONTHS = Array.from({ length: 12 }, (_, i) => ({ value: String(i + 1), label: MONTH_FULL[i + 1] }));

export function QueryTab() {
  const [airports, setAirports] = useState<string[]>([]);
  const [airportA, setAirportA] = useState("MCO");
  const [airportB, setAirportB] = useState("LAX");
  const [month, setMonth] = useState("6");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [timeline, setTimeline] = useState<{ month: number; risk_score: number; label: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.airports().then(setAirports).catch(console.error);
  }, []);

  const handleQuery = async () => {
    setLoading(true);
    setError(null);
    try {
      const [pred, tl] = await Promise.all([
        api.predict(airportA, airportB, Number(month)),
        api.pairTimeline(airportA, airportB),
      ]);
      setResult(pred);
      setTimeline(tl);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const timelineData = timeline.map((t) => ({
    month: MONTH_FULL[t.month].slice(0, 3),
    risk: Number((t.risk_score * 100).toFixed(1)),
    monthNum: t.month,
  }));

  return (
    <div className="space-y-6">
      {/* Query form */}
      <Card>
        <CardContent className="pt-5">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex flex-col gap-1.5 min-w-[140px]">
              <label className="text-[10px] text-white/40 uppercase tracking-wider">Origin (A)</label>
              <Select value={airportA} onValueChange={setAirportA}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {airports.map((a) => (
                    <SelectItem key={a} value={a}>{a}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-end pb-2.5 text-white/20">
              <ArrowRight className="w-4 h-4" />
              <span className="text-xs mx-1 text-aa-blue/60">DFW</span>
              <ArrowRight className="w-4 h-4" />
            </div>

            <div className="flex flex-col gap-1.5 min-w-[140px]">
              <label className="text-[10px] text-white/40 uppercase tracking-wider">Destination (B)</label>
              <Select value={airportB} onValueChange={setAirportB}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {airports.map((a) => (
                    <SelectItem key={a} value={a}>{a}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5 min-w-[160px]">
              <label className="text-[10px] text-white/40 uppercase tracking-wider">Month</label>
              <Select value={month} onValueChange={setMonth}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MONTHS.map((m) => (
                    <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={handleQuery}
              disabled={loading}
              className="mb-0.5"
            >
              <Search className="w-4 h-4" />
              {loading ? "Scoring…" : "Predict Risk"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <div className="glass rounded-xl p-4 border border-red-500/25 text-red-400 text-sm">
          {error} — pair may not exist in the dataset.
        </div>
      )}

      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            key={`${result.airport_A}-${result.airport_B}-${result.month}`}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            {/* Gauge */}
            <Card className="flex flex-col items-center justify-center py-8">
              <div className="text-xs text-white/40 uppercase tracking-wider mb-4 font-medium">
                {result.airport_A} → DFW → {result.airport_B} · {MONTH_FULL[result.month]}
              </div>
              <RiskGauge score={result.risk_score} size={200} />
              <div className="mt-4 grid grid-cols-2 gap-4 w-full px-8 text-center">
                <div>
                  <div className="font-serif text-2xl text-white">{fmtPct(result.risk_score)}</div>
                  <div className="text-[10px] text-white/35 uppercase tracking-wider mt-0.5">Model risk</div>
                </div>
                <div>
                  <div className="font-serif text-2xl text-aa-muted/70">
                    {result.observed_bad_rate != null ? fmtPct(result.observed_bad_rate) : "—"}
                  </div>
                  <div className="text-[10px] text-white/35 uppercase tracking-wider mt-0.5">Observed</div>
                </div>
              </div>
              {result.n_sequences != null && (
                <div className="mt-2 text-[11px] text-white/25">
                  {result.n_sequences.toLocaleString()} rotations in dataset
                </div>
              )}
            </Card>

            {/* SHAP */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Feature Contributions (SHAP)</CardTitle>
              </CardHeader>
              <CardContent>
                <ShapChart features={result.shap} />
              </CardContent>
            </Card>

            {/* Timeline — full width */}
            {timeline.length > 0 && (
              <Card className="md:col-span-3">
                <CardHeader>
                  <CardTitle>12-Month Risk Timeline — {result.airport_A} → DFW → {result.airport_B}</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={timelineData} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis
                        dataKey="month"
                        tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tickFormatter={(v) => `${v}%`}
                        tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        domain={[0, 50]}
                      />
                      <Tooltip
                        cursor={{ stroke: "rgba(255,255,255,0.15)", strokeWidth: 1 }}
                        content={({ payload, label }) => {
                          const d = payload?.[0];
                          if (!d) return null;
                          const val = Number(d.value);
                          return (
                            <div className="glass rounded-xl p-3 text-xs shadow-glass">
                              <div className="text-white/50 mb-1">{label}</div>
                              <div className="text-white font-semibold">{val.toFixed(1)}% risk</div>
                            </div>
                          );
                        }}
                      />
                      <ReferenceLine y={30} stroke="rgba(239,68,68,0.4)" strokeDasharray="4 3" label={{ value: "HIGH", fill: "rgba(239,68,68,0.6)", fontSize: 9 }} />
                      <ReferenceLine y={20} stroke="rgba(249,115,22,0.3)" strokeDasharray="4 3" />
                      <Line
                        type="monotone"
                        dataKey="risk"
                        stroke="#14acdc"
                        strokeWidth={2}
                        dot={({ cx, cy, payload }: { cx: number; cy: number; payload: { monthNum: number; risk: number } }) => (
                          <circle
                            key={payload.monthNum}
                            cx={cx}
                            cy={cy}
                            r={payload.monthNum === Number(month) ? 5 : 3}
                            fill={scoreToColor(payload.risk / 100)}
                            stroke={payload.monthNum === Number(month) ? "white" : "transparent"}
                            strokeWidth={2}
                          />
                        )}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
