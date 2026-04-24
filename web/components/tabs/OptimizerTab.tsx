"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Plus, X, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";
import { fmtPct, scoreToColor } from "@/lib/utils";
import { RiskBadge } from "@/components/common/RiskBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { OptimizeResult, FlightEntry } from "@/types";
import { MONTH_FULL } from "@/types";

const SAMPLE_ARRIVALS: FlightEntry[] = [
  { airport: "ATL", time_str: "07:15", flight: "AA1042" },
  { airport: "ORD", time_str: "07:45", flight: "AA2205" },
  { airport: "LAX", time_str: "08:20", flight: "AA3310" },
  { airport: "MCO", time_str: "08:55", flight: "AA4418" },
  { airport: "JFK", time_str: "09:30", flight: "AA5523" },
];

const SAMPLE_DEPARTURES: FlightEntry[] = [
  { airport: "MIA", time_str: "09:00", flight: "AA1101" },
  { airport: "BOS", time_str: "09:30", flight: "AA2202" },
  { airport: "SFO", time_str: "10:15", flight: "AA3303" },
  { airport: "DEN", time_str: "10:50", flight: "AA4404" },
  { airport: "SEA", time_str: "11:25", flight: "AA5505" },
];

const AIRPORTS = ["ATL", "ORD", "LAX", "MCO", "JFK", "LGA", "MIA", "BOS", "DEN", "SEA", "PHX", "LAS", "SFO", "CLT", "EWR", "MSP", "DTW", "PHL", "FLL", "MDW"];
const MONTHS = Array.from({ length: 12 }, (_, i) => String(i + 1));

function FlightRow({
  entry,
  onRemove,
}: {
  entry: FlightEntry;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-white/4 hover:bg-white/6 group transition-colors">
      <span className="font-mono text-sm text-white/80 w-10">{entry.airport}</span>
      <span className="text-xs text-white/40">{entry.time_str}</span>
      {entry.flight && <span className="text-[10px] text-aa-blue-light/60">{entry.flight}</span>}
      <button
        onClick={onRemove}
        className="ml-auto text-white/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

export function OptimizerTab() {
  const [arrivals, setArrivals] = useState<FlightEntry[]>(SAMPLE_ARRIVALS);
  const [departures, setDepartures] = useState<FlightEntry[]>(SAMPLE_DEPARTURES);
  const [month, setMonth] = useState("6");
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newArr, setNewArr] = useState({ airport: "ATL", time_str: "10:00" });
  const [newDep, setNewDep] = useState({ airport: "BOS", time_str: "11:30" });

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.optimize({ arrivals, departures, month: Number(month) });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Optimization failed");
    } finally {
      setLoading(false);
    }
  };

  const removeArrival = (i: number) => setArrivals((prev) => prev.filter((_, j) => j !== i));
  const removeDep = (i: number) => setDepartures((prev) => prev.filter((_, j) => j !== i));

  return (
    <div className="space-y-6">
      {/* Header info */}
      <Card className="border border-aa-blue-light/15">
        <CardContent className="pt-5 pb-4">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-aa-blue-light mt-0.5 shrink-0" />
            <div>
              <div className="text-sm font-medium text-white mb-1">Hungarian Algorithm Optimizer</div>
              <div className="text-xs text-white/40 leading-relaxed">
                Finds the one-to-one assignment of DFW arrivals to departures that minimizes total weather risk
                across all matched sequences. Enforces 30–240 min turnaround constraint.
                Uses <code className="text-aa-blue-light/70">scipy.optimize.linear_sum_assignment</code> (Jonker-Volgenant, O(n³)).
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Schedule input */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Arrivals */}
        <Card>
          <CardHeader>
            <CardTitle>Arrivals → DFW</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {arrivals.map((a, i) => (
              <FlightRow key={i} entry={a} onRemove={() => removeArrival(i)} />
            ))}
            <div className="flex gap-2 mt-3 pt-3 border-t border-white/6">
              <Select value={newArr.airport} onValueChange={(v) => setNewArr((p) => ({ ...p, airport: v }))}>
                <SelectTrigger className="flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {AIRPORTS.map((a) => <SelectItem key={a} value={a}>{a}</SelectItem>)}
                </SelectContent>
              </Select>
              <input
                value={newArr.time_str}
                onChange={(e) => setNewArr((p) => ({ ...p, time_str: e.target.value }))}
                placeholder="HH:MM"
                className="w-20 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white text-center"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={() => setArrivals((p) => [...p, { ...newArr }])}
              >
                <Plus className="w-3.5 h-3.5" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Departures */}
        <Card>
          <CardHeader>
            <CardTitle>Departures from DFW</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {departures.map((d, i) => (
              <FlightRow key={i} entry={d} onRemove={() => removeDep(i)} />
            ))}
            <div className="flex gap-2 mt-3 pt-3 border-t border-white/6">
              <Select value={newDep.airport} onValueChange={(v) => setNewDep((p) => ({ ...p, airport: v }))}>
                <SelectTrigger className="flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {AIRPORTS.map((a) => <SelectItem key={a} value={a}>{a}</SelectItem>)}
                </SelectContent>
              </Select>
              <input
                value={newDep.time_str}
                onChange={(e) => setNewDep((p) => ({ ...p, time_str: e.target.value }))}
                placeholder="HH:MM"
                className="w-20 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white text-center"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={() => setDepartures((p) => [...p, { ...newDep }])}
              >
                <Plus className="w-3.5 h-3.5" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Run controls */}
      <div className="flex items-center gap-4">
        <Select value={month} onValueChange={setMonth}>
          <SelectTrigger className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MONTHS.map((m) => (
              <SelectItem key={m} value={m}>{MONTH_FULL[Number(m)]}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button onClick={handleOptimize} disabled={loading || arrivals.length === 0 || departures.length === 0}>
          <Zap className="w-4 h-4" />
          {loading ? "Optimizing…" : "Run Optimizer"}
        </Button>
        {result && (
          <span className="text-xs text-white/40">
            {result.stats.n_matched} sequences matched · {result.stats.feasible_pairs} feasible pairs
          </span>
        )}
      </div>

      {error && (
        <div className="glass rounded-xl p-4 border border-red-500/25 text-red-400 text-sm flex gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Stats summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="glass rounded-xl p-4 border-t-2 border-t-emerald-400">
                <div className="font-serif text-3xl text-emerald-400">{fmtPct(result.stats.optimal_avg)}</div>
                <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Avg risk / sequence</div>
              </div>
              <div className="glass rounded-xl p-4 border-t-2 border-t-aa-blue-light">
                <div className="font-serif text-3xl text-aa-blue-light">{fmtPct(result.stats.risk_saved / (result.stats.worst_total || 1))}</div>
                <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Risk reduction vs worst</div>
              </div>
              <div className="glass rounded-xl p-4 border-t-2 border-t-white/20">
                <div className="font-serif text-3xl text-white">{result.stats.n_matched}</div>
                <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">Sequences matched</div>
              </div>
              <div className="glass rounded-xl p-4 border-t-2 border-t-aa-red">
                <div className="font-serif text-3xl text-aa-red">{fmtPct(result.stats.pct_high)}</div>
                <div className="text-[10px] text-white/40 uppercase tracking-wider mt-1">High risk pairs</div>
              </div>
            </div>

            {/* Sequence table */}
            <Card>
              <CardHeader>
                <CardTitle>Optimal Assignment — {MONTH_FULL[Number(month)]}</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/6">
                      {["Sequence", "Inbound", "Arr", "Turn", "Outbound", "Dep", "Risk"].map((h) => (
                        <th key={h} className="text-left text-[10px] text-white/35 font-medium tracking-wider uppercase px-4 py-2.5">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.sequences.map((s, i) => (
                      <tr
                        key={i}
                        className="border-b border-white/4 hover:bg-white/4 transition-colors"
                      >
                        <td className="px-4 py-2.5 font-mono text-white/80 text-[11px]">
                          <span
                            className="inline-block w-2 h-2 rounded-full mr-2"
                            style={{ background: scoreToColor(s.risk_score) }}
                          />
                          {s.sequence}
                        </td>
                        <td className="px-4 py-2.5 text-white/50">{s.flight_in}</td>
                        <td className="px-4 py-2.5 text-white/40 font-mono">{s.arr_time}</td>
                        <td className="px-4 py-2.5 text-white/40">{s.turnaround_min}m</td>
                        <td className="px-4 py-2.5 text-white/50">{s.flight_out}</td>
                        <td className="px-4 py-2.5 text-white/40 font-mono">{s.dep_time}</td>
                        <td className="px-4 py-2.5">
                          <RiskBadge label={s.risk_label === "HIGH" ? "HIGH RISK" : s.risk_label === "MODERATE" ? "MODERATE RISK" : "LOW RISK"} score={s.risk_score} size="sm" />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
