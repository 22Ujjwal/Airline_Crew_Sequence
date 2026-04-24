export interface PairScore {
  airport_A: string;
  airport_B: string;
  Month: number;
  avg_risk_score: number;
  avg_risk_score_raw: number;
  observed_bad_rate: number;
  n_sequences: number;
}

export interface ShapFeature {
  feature: string;
  label: string;
  shap_value: number;
  feature_value: number;
  imputed: boolean;
  abs_shap: number;
}

export interface PredictResult {
  airport_A: string;
  airport_B: string;
  month: number;
  risk_score: number;
  label: "HIGH RISK" | "MODERATE RISK" | "LOW RISK";
  observed_bad_rate: number | null;
  n_sequences: number | null;
  shap: ShapFeature[];
}

export interface HeatmapCell {
  airport_A: string;
  month: number;
  avg_risk: number;
}

export interface TopPair {
  airport_A: string;
  airport_B: string;
  Month: number;
  avg_risk_score: number;
  observed_bad_rate: number;
  n_sequences: number;
}

export interface FeatureImportance {
  rank: number;
  feature: string;
  label: string;
  group: string;
  importance: number;
}

export interface EvalData {
  auc: number;
  ap: number;
  pair_auc: number;
  pair_ap: number;
  fpr: number[];
  tpr: number[];
  precision: number[];
  recall: number[];
  calibration: CalibrationPoint[];
  score_distribution: ScoreBand[];
}

export interface CalibrationPoint {
  decile: number;
  mean_score: number;
  mean_obs: number;
  n: number;
}

export interface ScoreBand {
  band: string;
  fraction: number;
  count: number;
}

export interface SequenceRow {
  sequence: string;
  airport_A: string;
  airport_B: string;
  flight_in: string;
  arr_time: string;
  flight_out: string;
  dep_time: string;
  turnaround_min: number;
  risk_score: number;
  risk_label: string;
}

export interface OptimizeResult {
  sequences: SequenceRow[];
  stats: {
    n_arrivals: number;
    n_departures: number;
    n_matched: number;
    feasible_pairs: number;
    optimal_total: number;
    optimal_avg: number;
    worst_total: number;
    risk_saved: number;
    pct_high: number;
  };
}

export interface FlightEntry {
  airport: string;
  time_str: string;
  flight?: string;
}

export type RiskLevel = "HIGH RISK" | "MODERATE RISK" | "LOW RISK";

export const MONTH_NAMES = [
  "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export const MONTH_FULL = [
  "", "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];
