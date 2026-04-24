const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  airports: () => get<string[]>("/airports"),
  topPairs: (month: number, n: number) => get<import("@/types").TopPair[]>(`/scores/top?month=${month}&n=${n}`),
  heatmap: (month: number) => get<import("@/types").HeatmapCell[]>(`/scores/heatmap/${month}`),
  allScores: (month?: number) => get<import("@/types").PairScore[]>(`/scores${month ? `?month=${month}` : ""}`),
  predict: (a: string, b: string, month: number) =>
    get<import("@/types").PredictResult>(`/predict/${a}/${b}/${month}`),
  pairTimeline: (a: string, b: string) =>
    get<{ month: number; risk_score: number; label: string }[]>(`/timeline/${a}/${b}`),
  eval: () => get<import("@/types").EvalData>("/eval"),
  features: () => get<import("@/types").FeatureImportance[]>("/features"),
  optimize: (body: {
    arrivals: { airport: string; time_str: string; flight?: string }[];
    departures: { airport: string; time_str: string; flight?: string }[];
    month: number;
  }) => post<import("@/types").OptimizeResult>("/optimize", body),
  health: () => get<{ status: string }>("/health"),
};
