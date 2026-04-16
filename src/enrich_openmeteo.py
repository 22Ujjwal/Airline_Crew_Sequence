"""
Build app_features_cache.parquet from available processed files + Open-Meteo weather.

This script replaces the old GSOM enrichment and works WITHOUT sequence_features.parquet
(which is gitignored due to size). It assembles all FEATURE_COLS required by the XGBoost
model from the individual processed parquet files that ARE committed:

  pair_risk_scores.parquet          → base rows + risk scores
  airport_features.parquet          → BTS weather stats per airport×month
  openmeteo_airport_monthly.parquet → ERA5 climate per airport×year×month (new)
  dfw_weather_monthly.parquet       → DFW hub weather per month
  tail_chain_features.parquet       → crew duty proxy features
  airport_cascade_features.parquet  → cascade propagation index
  multihop_cascade_features.parquet → multi-hop cascade features

Open-Meteo columns are mapped to the GSOM-compatible names the model was trained on:
  avg_wind_mph   → avg_wind_speed   (model feature name kept for XGBoost compatibility)
  max_gust_mph   → max_wind_gust
  total_precip_in → total_precip
  precip_days    → precip_days
  severe_wx_days → extreme_precip   (WMO code ≥80 days as proxy for extreme events)

Run:
  python src/enrich_openmeteo.py

Output: data/processed/app_features_cache.parquet  (replaces stale cache)
"""

import os
import numpy as np
import pandas as pd

PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


# ── Open-Meteo → GSOM column name mapping ────────────────────────────────────
# Keeps XGBoost feature names intact (model was trained on GSOM names).
# Provides real per-airport values instead of global medians for all airports.
OM_TO_GSOM = {
    "avg_wind_mph":    "avg_wind_speed",
    "max_gust_mph":    "max_wind_gust",
    "total_precip_in": "total_precip",
    "precip_days":     "precip_days",
    "severe_wx_days":  "extreme_precip",   # WMO ≥80: heavy rain/snow/thunder
}


def load_openmeteo_monthly(path: str) -> pd.DataFrame:
    """
    Load Open-Meteo airport monthly data and aggregate across years to get
    climatological normals (2015-2024 mean per airport×month).
    Returns columns: iata, month, avg_wind_speed, max_wind_gust,
                     total_precip, precip_days, extreme_precip
    """
    om = pd.read_parquet(path)

    # Aggregate across years → long-run monthly climate normals
    agg = om.groupby(["iata", "month"]).agg(
        avg_wind_mph    = ("avg_wind_mph",    "mean"),
        max_gust_mph    = ("max_gust_mph",    "mean"),   # mean of annual max gusts
        total_precip_in = ("total_precip_in", "mean"),
        precip_days     = ("precip_days",     "mean"),
        severe_wx_days  = ("severe_wx_days",  "mean"),
    ).reset_index()

    # Rename to GSOM-compatible names
    agg = agg.rename(columns={om_col: gsom_col for om_col, gsom_col in OM_TO_GSOM.items()})
    agg = agg.rename(columns={"month": "Month"})
    return agg


def build_pair_features() -> pd.DataFrame:
    """
    Assemble all FEATURE_COLS for every (airport_A, airport_B, Month) in
    pair_risk_scores.parquet without requiring sequence_features.parquet.
    """
    # ── Base: pair-month risk scores ─────────────────────────────────────────
    prs = pd.read_parquet(os.path.join(PROCESSED, "pair_risk_scores.parquet"))
    df  = prs[["airport_A", "airport_B", "Month",
               "avg_risk_score", "observed_bad_rate", "n_sequences"]].copy()

    # ── Airport BTS weather stats ─────────────────────────────────────────────
    af_path = os.path.join(PROCESSED, "airport_features.parquet")
    if os.path.exists(af_path):
        af = pd.read_parquet(af_path)
        # Rename generic column names to model's A_ / B_ prefixed names
        bts_cols = [c for c in af.columns if c not in ("airport", "Month")]
        af_a = af.rename(columns={"airport": "airport_A",
                                  **{c: f"A_{c}" for c in bts_cols}})
        af_b = af.rename(columns={"airport": "airport_B",
                                  **{c: f"B_{c}" for c in bts_cols}})
        df = df.merge(af_a[["airport_A", "Month"] + [f"A_{c}" for c in bts_cols]],
                      on=["airport_A", "Month"], how="left")
        df = df.merge(af_b[["airport_B", "Month"] + [f"B_{c}" for c in bts_cols]],
                      on=["airport_B", "Month"], how="left")

        # Pair-level BTS features
        if "A_weather_delay_rate" in df.columns and "B_weather_delay_rate" in df.columns:
            df["pair_combined_weather_rate"] = (
                df["A_weather_delay_rate"] + df["B_weather_delay_rate"]
            ) / 2
            df["pair_max_weather_rate"]  = df[["A_weather_delay_rate", "B_weather_delay_rate"]].max(axis=1)
            df["pair_min_weather_rate"]  = df[["A_weather_delay_rate", "B_weather_delay_rate"]].min(axis=1)
            df["pair_weather_rate_sum"]  = df["A_weather_delay_rate"] + df["B_weather_delay_rate"]
            df["pair_avg_weather_delay_min"] = (
                df["A_avg_weather_delay_min"].fillna(0) + df["B_avg_weather_delay_min"].fillna(0)
            ) / 2
            HIGH = 0.10   # airports with >10% weather delay rate = "high-risk"
            df["both_high_risk"] = (
                (df["A_weather_delay_rate"] > HIGH) & (df["B_weather_delay_rate"] > HIGH)
            ).astype(int)

    # ── Temporal features ─────────────────────────────────────────────────────
    df["is_spring_summer"]       = df["Month"].isin([3, 4, 5, 6, 7, 8]).astype(int)
    df["median_turnaround_min"]  = 90.0   # reasonable default; actual value in sequence_features
    df["season_fall"]    = df["Month"].isin([9, 10, 11]).astype(int)
    df["season_spring"]  = df["Month"].isin([3, 4, 5]).astype(int)
    df["season_summer"]  = df["Month"].isin([6, 7, 8]).astype(int)
    df["season_winter"]  = df["Month"].isin([12, 1, 2]).astype(int)

    # ── Open-Meteo ERA5 weather features ─────────────────────────────────────
    om_path = os.path.join(PROCESSED, "openmeteo_airport_monthly.parquet")
    if os.path.exists(om_path):
        print("Joining Open-Meteo ERA5 weather features...")
        om = load_openmeteo_monthly(om_path)
        gsom_weather_cols = list(OM_TO_GSOM.values())  # [avg_wind_speed, max_wind_gust, ...]

        # Airport A weather
        om_a = om.rename(columns={"iata": "airport_A",
                                  **{c: f"A_{c}" for c in gsom_weather_cols}})
        df = df.merge(om_a[["airport_A", "Month"] + [f"A_{c}" for c in gsom_weather_cols]],
                      on=["airport_A", "Month"], how="left")

        # Airport B weather
        om_b = om.rename(columns={"iata": "airport_B",
                                  **{c: f"B_{c}" for c in gsom_weather_cols}})
        df = df.merge(om_b[["airport_B", "Month"] + [f"B_{c}" for c in gsom_weather_cols]],
                      on=["airport_B", "Month"], how="left")

        # Pair-level max features (used directly by model)
        for col in gsom_weather_cols:
            df[f"pair_max_{col}"] = df[[f"A_{col}", f"B_{col}"]].max(axis=1)

        coverage = df["A_avg_wind_speed"].notna().mean()
        print(f"  Open-Meteo coverage: {coverage:.1%} of airport-A rows")
    else:
        print("WARNING: openmeteo_airport_monthly.parquet not found. "
              "Run src/fetch_openmeteo.py first.")
        # Fill GSOM columns with NaN — XGBoost will use its native NaN routing
        for prefix in ("A_", "B_", "pair_max_"):
            for col in OM_TO_GSOM.values():
                df[f"{prefix}{col}"] = np.nan

    # ── DFW hub weather ───────────────────────────────────────────────────────
    dfw_path = os.path.join(PROCESSED, "dfw_weather_monthly.parquet")
    if os.path.exists(dfw_path):
        dfw = pd.read_parquet(dfw_path)
        df = df.merge(dfw, on="Month", how="left")

    # ── Tail-chain crew duty features ─────────────────────────────────────────
    tc_path = os.path.join(PROCESSED, "tail_chain_features.parquet")
    if os.path.exists(tc_path):
        tc = pd.read_parquet(tc_path)
        tc_key = ["airport_A", "airport_B", "Month"]
        tc_feat = [c for c in tc.columns if c not in tc_key + ["Year"]]
        # Collapse to latest year per pair-month if Year column exists
        if "Year" in tc.columns:
            tc = tc.sort_values("Year").groupby(tc_key, as_index=False).last()
        df = df.merge(tc[tc_key + tc_feat], on=tc_key, how="left")

    # ── Airport cascade features ──────────────────────────────────────────────
    ap_path = os.path.join(PROCESSED, "airport_cascade_features.parquet")
    if os.path.exists(ap_path):
        ap = pd.read_parquet(ap_path)
        ap_feat = [c for c in ap.columns if c not in ("airport", "Month")]
        for side in ("A", "B"):
            rename = {c: f"{side}_ap_{c}" for c in ap_feat}
            ap_side = ap.rename(columns={"airport": f"airport_{side}", **rename})
            df = df.merge(
                ap_side[[f"airport_{side}", "Month"] + list(rename.values())],
                on=[f"airport_{side}", "Month"], how="left"
            )
        if "A_ap_cascade_rate" in df.columns and "B_ap_cascade_rate" in df.columns:
            df["pair_cascade_product"]  = df["A_ap_cascade_rate"] * df["B_ap_cascade_rate"]
            df["pair_max_cascade_rate"] = df[["A_ap_cascade_rate", "B_ap_cascade_rate"]].max(axis=1)

    # ── Multi-hop cascade features ────────────────────────────────────────────
    mhc_path = os.path.join(PROCESSED, "multihop_cascade_features.parquet")
    if os.path.exists(mhc_path):
        mhc = pd.read_parquet(mhc_path)
        mhc_key = ["airport_A", "airport_B", "Month"]
        mhc_feat = [c for c in mhc.columns if c not in mhc_key + ["Year"]]
        if "Year" in mhc.columns:
            mhc = mhc.sort_values("Year").groupby(mhc_key, as_index=False).last()
        df = df.merge(mhc[mhc_key + mhc_feat], on=mhc_key, how="left")

    # ── Add interaction features (model improvement) ──────────────────────────
    if "A_weather_delay_rate" in df.columns and "B_weather_delay_rate" in df.columns:
        df["A_B_weather_coincidence"] = (
            df["A_weather_delay_rate"] * df["B_weather_delay_rate"]
        )
        df["B_given_A_risk_ratio"] = df["B_weather_delay_rate"] / (
            df["A_weather_delay_rate"] + 1e-6
        )

    return df


def main():
    print("Building app features cache from processed files + Open-Meteo...")

    df = build_pair_features()

    out = os.path.join(PROCESSED, "app_features_cache.parquet")
    df.to_parquet(out, index=False)

    print(f"\nSaved {len(df):,} rows × {len(df.columns)} cols → {out}")
    print(f"Columns: {list(df.columns)}")

    # Quick NaN audit
    nan_pct = df.isnull().mean().sort_values(ascending=False)
    high_nan = nan_pct[nan_pct > 0.05]
    if not high_nan.empty:
        print("\nColumns with >5% NaN (may affect model):")
        print(high_nan.to_string())
    else:
        print("\nAll columns have <5% NaN — data quality OK.")


if __name__ == "__main__":
    main()
