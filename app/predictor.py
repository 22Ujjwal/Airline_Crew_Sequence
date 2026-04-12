"""
Model wrapper: loads XGBoost model + all feature data, provides prediction + SHAP explanation.

At startup, builds a merged feature table (same joins as model.py's load_features()),
then keeps only the most-recent-year row per (airport_A, airport_B, Month) to reduce memory.
Cached at the Streamlit session level via @st.cache_resource.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xgboost as xgb

PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RAW       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

RISK_THRESHOLD = 0.25

FEATURE_COLS = [
    # Airport A BTS weather stats
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    # Airport B BTS weather stats
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    # Pair-level BTS features
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    # Temporal
    "Month", "is_spring_summer", "median_turnaround_min",
    # GSOM weather features
    "A_avg_wind_speed", "A_precip_days", "A_extreme_precip",
    "A_total_precip", "A_max_wind_gust",
    "B_avg_wind_speed", "B_precip_days", "B_extreme_precip",
    "B_total_precip", "B_max_wind_gust",
    "pair_max_avg_wind_speed", "pair_max_precip_days",
    "pair_max_extreme_precip", "pair_max_total_precip", "pair_max_max_wind_gust",
    # DFW hub weather
    "DFW_weather_delay_rate", "DFW_weather_cancel_rate",
    "DFW_avg_weather_delay_min", "DFW_p95_weather_delay_min",
    # Tail-chain crew duty features
    "tc_legs_before_mean", "tc_block_before_mean", "tc_duty_start_hour",
    "tc_total_duty_mean", "tc_total_duty_p75",
    "tc_fdp_util_mean", "tc_fdp_util_p75", "tc_fdp_overrun_rate",
    "tc_wocl_rate", "tc_legs_after_mean", "tc_legs_in_day_mean",
    "tc_downstream_rate", "tc_cascade_late_rate",
    "tc_cascade_late_min", "tc_cascade_amplif_mean",
    # Airport-level cascade propagation
    "A_ap_cascade_rate", "A_ap_cascade_given_late",
    "B_ap_cascade_rate", "B_ap_cascade_given_late",
    "pair_cascade_product", "pair_max_cascade_rate",
    # Multi-hop DFW cascade
    "mhc_n_hops_mean", "mhc_n_hops_max",
    "mhc_total_late_min_mean", "mhc_total_late_min_p75",
    "mhc_cascade_hop_rate", "mhc_cascade_depth_mean",
    "mhc_unique_airports_mean", "mhc_recovery_rate",
]

FEATURE_LABELS = {
    "A_weather_delay_rate": "Origin: Weather Delay Rate",
    "A_weather_cancel_rate": "Origin: Weather Cancel Rate",
    "A_avg_weather_delay_min": "Origin: Avg Weather Delay (min)",
    "A_p75_weather_delay_min": "Origin: P75 Weather Delay (min)",
    "A_p95_weather_delay_min": "Origin: P95 Weather Delay (min)",
    "A_nas_delay_rate": "Origin: NAS Delay Rate",
    "A_overall_weather_delay_rate": "Origin: Overall Weather Delay Rate",
    "A_overall_avg_weather_delay_min": "Origin: Overall Avg Weather Delay (min)",
    "B_weather_delay_rate": "Dest: Weather Delay Rate",
    "B_weather_cancel_rate": "Dest: Weather Cancel Rate",
    "B_avg_weather_delay_min": "Dest: Avg Weather Delay (min)",
    "B_p75_weather_delay_min": "Dest: P75 Weather Delay (min)",
    "B_p95_weather_delay_min": "Dest: P95 Weather Delay (min)",
    "B_nas_delay_rate": "Dest: NAS Delay Rate",
    "B_overall_weather_delay_rate": "Dest: Overall Weather Delay Rate",
    "B_overall_avg_weather_delay_min": "Dest: Overall Avg Weather Delay (min)",
    "pair_combined_weather_rate": "Pair: Combined Weather Rate",
    "pair_max_weather_rate": "Pair: Max Weather Rate",
    "pair_min_weather_rate": "Pair: Min Weather Rate",
    "pair_weather_rate_sum": "Pair: Weather Rate Sum",
    "pair_avg_weather_delay_min": "Pair: Avg Weather Delay (min)",
    "both_high_risk": "Both Airports High Risk",
    "Month": "Month",
    "is_spring_summer": "Spring/Summer Season",
    "median_turnaround_min": "Median Turnaround (min)",
    "A_avg_wind_speed": "Origin: Avg Wind Speed",
    "A_precip_days": "Origin: Precipitation Days",
    "A_extreme_precip": "Origin: Extreme Precip Events",
    "A_total_precip": "Origin: Total Precipitation",
    "A_max_wind_gust": "Origin: Max Wind Gust",
    "B_avg_wind_speed": "Dest: Avg Wind Speed",
    "B_precip_days": "Dest: Precipitation Days",
    "B_extreme_precip": "Dest: Extreme Precip Events",
    "B_total_precip": "Dest: Total Precipitation",
    "B_max_wind_gust": "Dest: Max Wind Gust",
    "pair_max_avg_wind_speed": "Pair: Max Avg Wind Speed",
    "pair_max_precip_days": "Pair: Max Precip Days",
    "pair_max_extreme_precip": "Pair: Max Extreme Precip",
    "pair_max_total_precip": "Pair: Max Total Precip",
    "pair_max_max_wind_gust": "Pair: Max Wind Gust",
    "DFW_weather_delay_rate": "DFW Hub: Weather Delay Rate",
    "DFW_weather_cancel_rate": "DFW Hub: Weather Cancel Rate",
    "DFW_avg_weather_delay_min": "DFW Hub: Avg Weather Delay (min)",
    "DFW_p95_weather_delay_min": "DFW Hub: P95 Weather Delay (min)",
    "tc_legs_before_mean": "Crew: Legs Before DFW (avg)",
    "tc_block_before_mean": "Crew: Block Time Before (avg min)",
    "tc_duty_start_hour": "Crew: Duty Start Hour",
    "tc_total_duty_mean": "Crew: Total Duty Time (avg min)",
    "tc_total_duty_p75": "Crew: Total Duty Time (P75 min)",
    "tc_fdp_util_mean": "Crew: FDP Utilization (avg)",
    "tc_fdp_util_p75": "Crew: FDP Utilization (P75)",
    "tc_fdp_overrun_rate": "Crew: FDP Overrun Rate",
    "tc_wocl_rate": "Crew: WOCL Overlap Rate",
    "tc_legs_after_mean": "Crew: Legs After DFW (avg)",
    "tc_legs_in_day_mean": "Crew: Total Legs in Day (avg)",
    "tc_downstream_rate": "Cascade: Downstream Late Rate",
    "tc_cascade_late_rate": "Cascade: B→DFW Late Rate",
    "tc_cascade_late_min": "Cascade: B→DFW Avg Late (min)",
    "tc_cascade_amplif_mean": "Cascade: Delay Amplification",
    "A_ap_cascade_rate": "Origin: Airport Cascade Rate",
    "A_ap_cascade_given_late": "Origin: Cascade Rate Given Late",
    "B_ap_cascade_rate": "Dest: Airport Cascade Rate",
    "B_ap_cascade_given_late": "Dest: Cascade Rate Given Late",
    "pair_cascade_product": "Pair: Cascade Rate Product",
    "pair_max_cascade_rate": "Pair: Max Cascade Rate",
    "mhc_n_hops_mean": "Multi-Hop: Avg Downstream Hops",
    "mhc_n_hops_max": "Multi-Hop: Max Downstream Hops",
    "mhc_total_late_min_mean": "Multi-Hop: Avg Total Late (min)",
    "mhc_total_late_min_p75": "Multi-Hop: P75 Total Late (min)",
    "mhc_cascade_hop_rate": "Multi-Hop: Cascade Rate",
    "mhc_cascade_depth_mean": "Multi-Hop: Avg Cascade Depth",
    "mhc_unique_airports_mean": "Multi-Hop: Avg Airports Affected",
    "mhc_recovery_rate": "Multi-Hop: Recovery Rate",
}


def _get_dfw_weather() -> pd.DataFrame:
    cache = os.path.join(PROCESSED, "dfw_weather_monthly.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    return pd.DataFrame()


_APP_CACHE = os.path.join(PROCESSED, "app_features_cache.parquet")


def build_features_df(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load merged feature table for the app.
    First run: joins all supplementary parquets + collapses to latest year per
    (airport_A, airport_B, Month) → saves to app_features_cache.parquet.
    Subsequent runs: loads cache directly (~1s vs ~60s for full rebuild).
    """
    if not force_rebuild and os.path.exists(_APP_CACHE):
        print("Loading app feature cache...")
        return pd.read_parquet(_APP_CACHE)

    print("Building feature cache (one-time, ~60s)...")
    df = pd.read_parquet(os.path.join(PROCESSED, "sequence_features.parquet"))
    df["target"] = (df["observed_bad_rate"] > RISK_THRESHOLD).astype(int)

    if "pair_max_weather_rate" not in df.columns and "A_weather_delay_rate" in df.columns:
        df["pair_max_weather_rate"] = df[["A_weather_delay_rate", "B_weather_delay_rate"]].max(axis=1)

    dfw = _get_dfw_weather()
    if not dfw.empty:
        df = df.merge(dfw, on="Month", how="left")

    tc_path = os.path.join(PROCESSED, "tail_chain_features.parquet")
    if os.path.exists(tc_path):
        tc = pd.read_parquet(tc_path)
        tc_meta = ["airport_A", "airport_B", "Month", "Year"]
        df = df.merge(tc[tc_meta + [c for c in tc.columns if c not in tc_meta]], on=tc_meta, how="left")

    ap_path = os.path.join(PROCESSED, "airport_cascade_features.parquet")
    if os.path.exists(ap_path):
        ap = pd.read_parquet(ap_path)
        ap_feat = [c for c in ap.columns if c not in ("airport", "Month")]
        for side in ("A", "B"):
            rename = {c: f"{side}_ap_{c}" for c in ap_feat}
            merged = ap.rename(columns={"airport": f"airport_{side}", **rename})
            df = df.merge(merged[[f"airport_{side}", "Month"] + list(rename.values())],
                          on=[f"airport_{side}", "Month"], how="left")
        if "A_ap_cascade_rate" in df.columns and "B_ap_cascade_rate" in df.columns:
            df["pair_cascade_product"] = df["A_ap_cascade_rate"] * df["B_ap_cascade_rate"]
            df["pair_max_cascade_rate"] = df[["A_ap_cascade_rate", "B_ap_cascade_rate"]].max(axis=1)

    mhc_path = os.path.join(PROCESSED, "multihop_cascade_features.parquet")
    if os.path.exists(mhc_path):
        mhc = pd.read_parquet(mhc_path)
        mhc_meta = ["airport_A", "airport_B", "Month", "Year"]
        df = df.merge(mhc[mhc_meta + [c for c in mhc.columns if c not in mhc_meta]], on=mhc_meta, how="left")

    df = (
        df.sort_values("Year")
        .groupby(["airport_A", "airport_B", "Month"], as_index=False)
        .last()
    )
    df.to_parquet(_APP_CACHE, index=False)
    print(f"Feature cache saved → {_APP_CACHE} ({len(df):,} rows)")
    return df


class RiskPredictor:
    def __init__(self, features_df: pd.DataFrame):
        self.df = features_df.set_index(["airport_A", "airport_B", "Month"])
        model_path = os.path.join(PROCESSED, "xgb_model.json")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        # Use exact feature names the model was trained with (authoritative)
        self.feature_cols = self.model.get_booster().feature_names
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def predict_pair(self, airport_a: str, airport_b: str, month: int) -> dict | None:
        """Return prediction dict or None if pair not in dataset."""
        try:
            row = self.df.loc[(airport_a, airport_b, month)]
        except KeyError:
            return None

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # Index consumed airport_A, airport_B, Month — add Month back for model
        row = row.copy()
        row["Month"] = month

        X = row[self.feature_cols].to_frame().T.astype(float)
        prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "risk_score": prob,
            "label": _risk_label(prob),
            "color": _risk_color(prob),
            "observed_bad_rate": float(row.get("observed_bad_rate", np.nan)),
            "n_sequences": int(row.get("n_sequences", 0)),
            "X": X,
            "row": row,
        }

    def explain_pair(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Return DataFrame of feature contributions sorted by |SHAP value|."""
        import shap
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals = shap_vals[0]
        feat_names = X.columns.tolist()
        result = pd.DataFrame({
            "feature": feat_names,
            "shap_value": vals,
            "feature_value": X.iloc[0].values,
            "label": [FEATURE_LABELS.get(f, f) for f in feat_names],
        })
        result["abs_shap"] = result["shap_value"].abs()
        return result.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    def predict_all_months(self, airport_a: str, airport_b: str) -> pd.DataFrame:
        """Risk score for every month for a given pair."""
        rows = []
        for m in range(1, 13):
            res = self.predict_pair(airport_a, airport_b, m)
            rows.append({
                "Month": m,
                "risk_score": res["risk_score"] if res else np.nan,
                "label": res["label"] if res else "No data",
            })
        return pd.DataFrame(rows)

    @property
    def airports_a(self) -> list[str]:
        return sorted(self.df.index.get_level_values("airport_A").unique())

    @property
    def airports_b(self) -> list[str]:
        return sorted(self.df.index.get_level_values("airport_B").unique())


def _risk_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH RISK"
    if score >= 0.40:
        return "MODERATE RISK"
    return "LOW RISK"


def _risk_color(score: float) -> str:
    if score >= 0.70:
        return "#d62728"
    if score >= 0.40:
        return "#ff7f0e"
    return "#2ca02c"
