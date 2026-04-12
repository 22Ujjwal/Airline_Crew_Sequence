"""
Duty-aware feature engineering for the alt-model-duty-aware branch.

Extends the base features with:
  - Time-of-day signals (departure hour, red-eye flags, late-night rates)
  - Block time (proxy for crew fatigue accumulation)
  - Turnaround tightness (connection risk)
  - Late aircraft delay rate (cascade propagation proxy)
  - Day-of-week effects

These directly address three PDF objectives not covered by the base model:
  - Duty time violations
  - Missed connections due to tight turnarounds
  - Increased fatigue and operational risk
"""

import os
import numpy as np
import pandas as pd

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def _dep_hour(crs_time_series: pd.Series) -> pd.Series:
    """HHMM integer → hour of day (0-23)."""
    return (crs_time_series // 100).clip(0, 23)


def _is_late_night(hour: pd.Series) -> pd.Series:
    return (hour >= 21).astype(float)


def _is_early_morning(hour: pd.Series) -> pd.Series:
    return (hour <= 6).astype(float)


def build_duty_airport_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-airport×month duty/fatigue features from raw BTS flights.

    Inbound to DFW  (Origin=A, Dest=DFW):
      late_dep_rate, avg_block_min, late_aircraft_delay_rate,
      avg_late_aircraft_min, dep_hour_median

    Outbound from DFW (Origin=DFW, Dest=B):
      same schema prefixed appropriately
    """
    df = raw_df.copy()
    df["dep_hour"] = _dep_hour(df["CRSDepTime"].fillna(1200))
    df["arr_hour"] = _dep_hour(df["CRSArrTime"].fillna(1200))
    df["is_late_dep"]        = _is_late_night(df["dep_hour"])
    df["is_early_dep"]       = _is_early_morning(df["dep_hour"])
    df["late_aircraft_flag"] = (df["LateAircraftDelay"].fillna(0) > 0).astype(float)
    df["block_min"]          = df["CRSElapsedTime"].fillna(df["CRSElapsedTime"].median())

    inbound = df[df["Dest"] == "DFW"].copy()
    outbound = df[df["Origin"] == "DFW"].copy()

    def agg_duty(flights: pd.DataFrame, airport_col: str) -> pd.DataFrame:
        return (
            flights.groupby([airport_col, "Month"])
            .agg(
                dep_hour_median         = ("dep_hour",           "median"),
                late_dep_rate           = ("is_late_dep",        "mean"),
                early_dep_rate          = ("is_early_dep",       "mean"),
                avg_block_min           = ("block_min",          "mean"),
                late_aircraft_delay_rate= ("late_aircraft_flag", "mean"),
                avg_late_aircraft_min   = ("LateAircraftDelay",  lambda x: x.fillna(0).mean()),
                n_flights               = ("block_min",          "count"),
            )
            .reset_index()
            .rename(columns={airport_col: "airport"})
        )

    in_feat  = agg_duty(inbound,  "Origin")
    out_feat = agg_duty(outbound, "Dest")
    return in_feat, out_feat


def build_turnaround_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate tight-connection risk by aggregating inbound/outbound to
    airport×month level first, then computing turnaround proxy from
    median scheduled arrival/departure times. Avoids date-level cross-join OOM.

    Returns per-(A, B, Month) turnaround stats.
    """
    df = raw_df.copy()

    inbound  = df[df["Dest"] == "DFW"].copy()
    outbound = df[df["Origin"] == "DFW"].copy()

    inbound["arr_min"]  = (inbound["CRSArrTime"].fillna(1200) // 100) * 60 + inbound["CRSArrTime"].fillna(1200) % 100
    outbound["dep_min"] = (outbound["CRSDepTime"].fillna(1200) // 100) * 60 + outbound["CRSDepTime"].fillna(1200) % 100

    # Airport×month median arrival / departure time at DFW
    in_agg = (
        inbound.groupby(["Origin", "Month"])["arr_min"]
        .agg(["median", "std"]).reset_index()
        .rename(columns={"Origin": "airport_A", "median": "median_arr_min", "std": "std_arr_min"})
    )
    out_agg = (
        outbound.groupby(["Dest", "Month"])["dep_min"]
        .agg(["median", "std"]).reset_index()
        .rename(columns={"Dest": "airport_B", "median": "median_dep_min", "std": "std_dep_min"})
    )

    # Cross-join at airport×month level (manageable: ~N_airports^2 × 12 rows)
    merged = in_agg.merge(out_agg, on="Month")
    merged["turnaround_proxy_min"] = (merged["median_dep_min"] - merged["median_arr_min"]).clip(0)
    merged = merged[(merged["turnaround_proxy_min"] >= 30) & (merged["turnaround_proxy_min"] <= 240)]
    merged["tight_connection_rate"] = (merged["turnaround_proxy_min"] < 60).astype(float)
    merged["very_tight_rate"]       = (merged["turnaround_proxy_min"] < 45).astype(float)

    ta = merged[["airport_A","airport_B","Month","turnaround_proxy_min",
                 "tight_connection_rate","very_tight_rate"]].copy()
    return ta


def build_duty_sequence_features() -> pd.DataFrame:
    """
    Load all raw BTS years, compute duty/fatigue/cascade features,
    join with existing sequence_features, return enriched DataFrame.
    """
    import glob
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "bts_all_dfw_*.parquet")))
    aa_files  = sorted(glob.glob(os.path.join(RAW_DIR, "bts_aa_dfw_*.parquet")))
    year_file = {}
    for f in aa_files:
        yr = f.split("_")[-1].replace(".parquet","")
        year_file[yr] = f
    for f in all_files:
        yr = f.split("_")[-1].replace(".parquet","")
        year_file[yr] = f
    files = sorted(year_file.values())
    print(f"Loading {len(files)} raw BTS files...")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Total rows: {len(raw):,}")

    print("Building duty airport features...")
    in_feat, out_feat = build_duty_airport_features(raw)

    # Rename to A_ / B_ prefix
    in_rename  = {c: f"A_{c}" for c in in_feat.columns if c not in ("airport","Month")}
    out_rename  = {c: f"B_{c}" for c in out_feat.columns if c not in ("airport","Month")}
    in_feat  = in_feat.rename(columns=in_rename)
    out_feat = out_feat.rename(columns=out_rename)

    print("Building turnaround features...")
    ta_feat = build_turnaround_features(raw)

    # Load base sequence features
    base = pd.read_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"))
    print(f"Base features: {base.shape}")

    # Join inbound features (A airport)
    enriched = base.merge(
        in_feat.rename(columns={"airport": "airport_A"}),
        on=["airport_A", "Month"], how="left"
    )
    # Join outbound features (B airport)
    enriched = enriched.merge(
        out_feat.rename(columns={"airport": "airport_B"}),
        on=["airport_B", "Month"], how="left"
    )
    # Join turnaround features (replace existing median_turnaround_min with richer stats)
    enriched = enriched.merge(
        ta_feat[["airport_A","airport_B","Month",
                 "tight_connection_rate","very_tight_rate"]],
        on=["airport_A","airport_B","Month"], how="left"
    )

    # Pair-level derived duty features
    enriched["cascade_risk"] = (
        enriched.get("A_late_aircraft_delay_rate", pd.Series(0, index=enriched.index)).fillna(0) *
        enriched.get("B_late_aircraft_delay_rate", pd.Series(0, index=enriched.index)).fillna(0)
    )
    enriched["total_duty_block_min"] = (
        enriched.get("A_avg_block_min", pd.Series(90, index=enriched.index)).fillna(90) +
        enriched["median_turnaround_min"].fillna(90) +
        enriched.get("B_avg_block_min", pd.Series(90, index=enriched.index)).fillna(90)
    )
    enriched["duty_overrun_risk"] = (enriched["total_duty_block_min"] > 480).astype(float)
    enriched["late_dep_sequence"] = (
        enriched.get("A_late_dep_rate", pd.Series(0, index=enriched.index)).fillna(0) +
        enriched.get("B_late_dep_rate", pd.Series(0, index=enriched.index)).fillna(0)
    ).clip(0, 1)

    out_path = os.path.join(PROC_DIR, "sequence_features_duty.parquet")
    enriched.to_parquet(out_path, index=False)
    print(f"Duty-enriched features saved → {out_path}  shape={enriched.shape}")

    new_cols = [c for c in enriched.columns if c not in base.columns]
    print(f"\nNew duty features added ({len(new_cols)}):")
    for c in new_cols:
        print(f"  {c}")

    return enriched


if __name__ == "__main__":
    build_duty_sequence_features()
