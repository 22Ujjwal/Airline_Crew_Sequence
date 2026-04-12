"""
Feature engineering for the AA Crew Sequences Bad Weather XGBoost model.

Pipeline:
  1. Load raw BTS parquet files
  2. Build airport-level weather risk features (by airport x month)
  3. Construct sequence pairs: inbound (A→DFW) + outbound (DFW→B) same day/pilot window
  4. Build pair-level features
  5. Label pairs as high-risk (target)
  6. Save feature matrix for model training
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Thresholds
WEATHER_DELAY_THRESHOLD_MIN = 15   # minutes to count as a weather delay event
TURNAROUND_MAX_HRS = 4             # max hours between inbound arrival and outbound departure
TURNAROUND_MIN_HRS = 0.5           # min turnaround (shorter = invalid / not a real sequence)


# ---------------------------------------------------------------------------
# Step 1: Load all raw data
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    # Prefer all-carrier files over AA-only; fall back to AA-only if not present
    all_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.startswith("bts_all_dfw_") and f.endswith(".parquet")
    ])
    aa_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.startswith("bts_aa_dfw_") and f.endswith(".parquet")
    ])

    # For each year use all-carrier if available, else AA-only
    year_file = {}
    for f in aa_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f
    for f in all_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f  # overrides AA-only

    files = list(year_file.values())
    if not files:
        raise FileNotFoundError(f"No BTS parquet files found in {RAW_DIR}. Run download_bts.py first.")
    print(f"Using files: {[os.path.basename(f) for f in sorted(files)]}")

    # Load year-by-year and aggregate airport features to avoid OOM
    frames = []
    for f in sorted(files):
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Month"] = df["FlightDate"].dt.month
    df["Year"] = df["FlightDate"].dt.year
    df["Season"] = df["Month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    })
    print(f"Loaded {len(df):,} flights across {df['Year'].nunique()} years")
    return df


# ---------------------------------------------------------------------------
# Step 2: Airport-level weather risk features
# ---------------------------------------------------------------------------

def build_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-airport, per-month aggregates used as features.
    We use ALL flights touching that airport (origin or dest) to get robust stats.
    """
    # For each flight, the "airport of interest" for weather is the non-DFW end
    inbound = df[df["Dest"] == "DFW"].copy()
    inbound["airport"] = inbound["Origin"]
    inbound["weather_delayed"] = (inbound["WeatherDelay"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN).astype(int)
    inbound["weather_cancel"] = ((inbound["Cancelled"] == 1) & (inbound["CancellationCode"] == "B")).astype(int)
    inbound["weather_delay_min"] = inbound["WeatherDelay"].fillna(0)
    inbound["nas_delayed"] = (inbound["NASDelay"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN).astype(int)

    outbound = df[df["Origin"] == "DFW"].copy()
    outbound["airport"] = outbound["Dest"]
    outbound["weather_delayed"] = (outbound["WeatherDelay"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN).astype(int)
    outbound["weather_cancel"] = ((outbound["Cancelled"] == 1) & (outbound["CancellationCode"] == "B")).astype(int)
    outbound["weather_delay_min"] = outbound["WeatherDelay"].fillna(0)
    outbound["nas_delayed"] = (outbound["NASDelay"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN).astype(int)

    combined = pd.concat([inbound, outbound], ignore_index=True)

    agg = combined.groupby(["airport", "Month"]).agg(
        total_flights=("weather_delayed", "count"),
        weather_delay_rate=("weather_delayed", "mean"),
        weather_cancel_rate=("weather_cancel", "mean"),
        avg_weather_delay_min=("weather_delay_min", "mean"),
        p75_weather_delay_min=("weather_delay_min", lambda x: x.quantile(0.75)),
        p95_weather_delay_min=("weather_delay_min", lambda x: x.quantile(0.95)),
        nas_delay_rate=("nas_delayed", "mean"),
    ).reset_index()

    # Overall (across all months) per airport
    overall = combined.groupby("airport").agg(
        overall_weather_delay_rate=("weather_delayed", "mean"),
        overall_weather_cancel_rate=("weather_cancel", "mean"),
        overall_avg_weather_delay_min=("weather_delay_min", "mean"),
    ).reset_index()

    agg = agg.merge(overall, on="airport", how="left")
    print(f"Airport features: {len(agg):,} rows ({agg['airport'].nunique()} airports × months)")
    return agg


# ---------------------------------------------------------------------------
# Step 3: Construct sequence pairs
# ---------------------------------------------------------------------------

def _parse_hhmm(series: pd.Series) -> pd.Series:
    """Convert HHMM integer (e.g. 1435) to minutes since midnight."""
    s = series.fillna(0).astype(int).astype(str).str.zfill(4)
    hours = s.str[:2].astype(int)
    minutes = s.str[2:].astype(int)
    return hours * 60 + minutes


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (airport_A, airport_B, date) aggregates without materializing
    flight-level pairs. For each date, collapse inbound flights to unique
    origins and outbound to unique destinations, then cross-join the airport
    aggregates (~100 unique origins × ~100 unique destinations = ~10k rows/day)
    instead of flight-level pairs (~400 × ~400 = 160k rows/day).
    """
    inbound  = df[(df["Dest"] == "DFW") & (df["Cancelled"] != 1)].copy()
    outbound = df[(df["Origin"] == "DFW") & (df["Cancelled"] != 1)].copy()

    inbound["arr_min"]  = _parse_hhmm(inbound["ArrTime"])
    outbound["dep_min"] = _parse_hhmm(outbound["DepTime"])

    # Aggregate inbound flights to (date, airport_A): worst-case weather outcomes
    ib_agg = (
        inbound.groupby(["FlightDate", "Origin", "Month", "Season", "Year"])
        .agg(
            arr_min_earliest = ("arr_min",          "min"),
            arr_min_latest   = ("arr_min",          "max"),
            weather_delay_A  = ("WeatherDelay",     lambda x: x.fillna(0).max()),
            arr_delay_A      = ("ArrDelay",         lambda x: x.fillna(0).max()),
            nas_delay_A      = ("NASDelay",         lambda x: x.fillna(0).max()),
            late_aircraft_A  = ("LateAircraftDelay",lambda x: x.fillna(0).max()),
        )
        .reset_index()
        .rename(columns={"Origin": "airport_A"})
    )

    # Aggregate outbound flights to (date, airport_B): worst-case weather outcomes
    ob_agg = (
        outbound.groupby(["FlightDate", "Dest"])
        .agg(
            dep_min_earliest = ("dep_min",          "min"),
            dep_min_latest   = ("dep_min",          "max"),
            weather_delay_B  = ("WeatherDelay",     lambda x: x.fillna(0).max()),
            dep_delay_B      = ("DepDelay",         lambda x: x.fillna(0).max()),
            late_aircraft_B  = ("LateAircraftDelay",lambda x: x.fillna(0).max()),
        )
        .reset_index()
        .rename(columns={"Dest": "airport_B"})
    )

    # Merge on date → cross-join of unique airports per day (~10k rows/day vs 160k)
    pairs = ib_agg.merge(ob_agg, on="FlightDate", how="inner")
    pairs = pairs[pairs["airport_A"] != pairs["airport_B"]].copy()

    # Feasibility filter: at least one valid turnaround window exists
    # (earliest outbound departs after earliest inbound arrives + min turnaround,
    #  and latest outbound departs before latest inbound arrives + max turnaround)
    turnaround_mid = pairs["dep_min_earliest"] - pairs["arr_min_latest"]
    feasible = (
        (pairs["dep_min_latest"]   >= pairs["arr_min_earliest"] + TURNAROUND_MIN_HRS * 60) &
        (pairs["dep_min_earliest"] <= pairs["arr_min_latest"]   + TURNAROUND_MAX_HRS * 60)
    )
    pairs = pairs[feasible].copy()
    pairs["turnaround_min"] = turnaround_mid[feasible].clip(lower=0)

    print(f"Constructed {len(pairs):,} airport-pair×date rows from {pairs['FlightDate'].nunique():,} dates")
    return pairs


# ---------------------------------------------------------------------------
# Step 4: Build pair-level feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(pairs: pd.DataFrame, airport_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join airport-level features onto each pair and compute pair-level derived features.
    """
    feat = airport_features.rename(columns=lambda c: f"A_{c}" if c not in ("airport", "Month") else c)
    feat_b = airport_features.rename(columns=lambda c: f"B_{c}" if c not in ("airport", "Month") else c)

    # Join airport A features
    pairs = pairs.merge(
        feat.rename(columns={"airport": "airport_A"}),
        on=["airport_A", "Month"], how="left"
    )
    # Join airport B features
    pairs = pairs.merge(
        feat_b.rename(columns={"airport": "airport_B"}),
        on=["airport_B", "Month"], how="left"
    )

    # Pair-level derived features
    pairs["pair_combined_weather_rate"] = (
        pairs["A_weather_delay_rate"] * pairs["B_weather_delay_rate"]
    )
    pairs["pair_max_weather_rate"] = pairs[["A_weather_delay_rate", "B_weather_delay_rate"]].max(axis=1)
    pairs["pair_min_weather_rate"] = pairs[["A_weather_delay_rate", "B_weather_delay_rate"]].min(axis=1)
    pairs["pair_weather_rate_sum"] = pairs["A_weather_delay_rate"] + pairs["B_weather_delay_rate"]
    pairs["pair_avg_weather_delay_min"] = (
        pairs["A_avg_weather_delay_min"] + pairs["B_avg_weather_delay_min"]
    ) / 2
    pairs["both_high_risk"] = (
        (pairs["A_weather_delay_rate"] > pairs["A_weather_delay_rate"].quantile(0.75)) &
        (pairs["B_weather_delay_rate"] > pairs["B_weather_delay_rate"].quantile(0.75))
    ).astype(int)

    # Season dummies
    pairs = pd.get_dummies(pairs, columns=["Season"], prefix="season", drop_first=False)
    pairs["is_spring_summer"] = pairs.get("season_spring", 0) | pairs.get("season_summer", 0)

    return pairs


# ---------------------------------------------------------------------------
# Step 5: Label sequences as high-risk
# ---------------------------------------------------------------------------

def label_sequences(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    A sequence is labeled 1 (bad) if:
      - Either leg had a significant weather delay (>= threshold), OR
      - Leg 1 arrival delay cascaded into leg 2 departure (late aircraft propagation)
    """
    weather_A = pairs["weather_delay_A"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    weather_B = pairs["weather_delay_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    cascade = (
        (pairs["arr_delay_A"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN) &
        (pairs["late_aircraft_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN)
    )

    pairs["target"] = (weather_A | weather_B | cascade).astype(int)
    print(f"Label distribution: {pairs['target'].mean():.1%} high-risk sequences")
    return pairs


# ---------------------------------------------------------------------------
# Step 6: Final feature selection and save
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Airport A features
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    # Airport B features
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    # Pair features
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    # Temporal
    "Month", "is_spring_summer", "turnaround_min",
    # Season dummies (added dynamically)
]


def save_features(pairs: pd.DataFrame, year: int = None):
    """
    Aggregate flight-level pairs to (airport_A, airport_B, Month, Year).
    This is the right unit of analysis: we want to score pair × month combos,
    not individual sequences. Reduces 21M rows → ~tens of thousands.
    Features are stable airport-level stats; target is observed bad rate binarized.
    """
    season_cols = [c for c in pairs.columns if c.startswith("season_")]

    # Stable feature cols: take first value per group (they're the same within group)
    static_feat_cols = [c for c in FEATURE_COLS + season_cols if c in pairs.columns
                        and c not in ("Month", "turnaround_min")]

    agg_dict = {c: "first" for c in static_feat_cols}
    agg_dict["target"]          = "mean"    # observed bad rate
    agg_dict["turnaround_min"]  = "median"  # median turnaround for the pair×month
    agg_dict["airport_A"]       = "first"
    agg_dict["airport_B"]       = "first"

    grouped = (
        pairs
        .groupby(["airport_A", "airport_B", "Month", "Year"])
        .agg(
            n_sequences=("target", "count"),
            observed_bad_rate=("target", "mean"),
            median_turnaround_min=("turnaround_min", "median"),
            **{c: ("target" if c == "target" else c, "first")
               for c in static_feat_cols}
        )
    )

    # Flatten multi-index columns from named agg
    grouped = (
        pairs
        .assign(n_seq=1)
        .groupby(["airport_A", "airport_B", "Month", "Year"])
        .agg(
            n_sequences          = ("n_seq",           "count"),
            observed_bad_rate    = ("target",           "mean"),
            median_turnaround_min= ("turnaround_min",  "median"),
            **{c: (c, "first") for c in static_feat_cols}
        )
        .reset_index()
    )

    grouped["Month"] = grouped["Month"].astype(int)
    # Provisional per-year threshold (will be re-set globally in main)
    threshold = grouped["observed_bad_rate"].median()
    grouped["target"] = (grouped["observed_bad_rate"] > threshold).astype(int)
    print(f"  {year or ''}: {len(grouped):,} pair×month rows  bad_rate threshold={threshold:.3f}  high-risk={grouped['target'].mean():.1%}")
    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_files() -> list[str]:
    """Return best available file per year (all-carrier preferred over AA-only)."""
    all_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.startswith("bts_all_dfw_") and f.endswith(".parquet")
    ])
    aa_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.startswith("bts_aa_dfw_") and f.endswith(".parquet")
    ])
    year_file = {}
    for f in aa_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f
    for f in all_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f
    files = sorted(year_file.values())
    print(f"Using files: {[os.path.basename(f) for f in files]}")
    return files


def main():
    files = _resolve_files()

    # Build airport features by streaming one year at a time, concat the small aggregates
    print("\nBuilding airport features (streaming per year)...")
    ap_agg_frames = []
    for fpath in files:
        df_yr = pd.read_parquet(fpath)
        df_yr["FlightDate"] = pd.to_datetime(df_yr["FlightDate"])
        df_yr["Month"] = df_yr["FlightDate"].dt.month
        df_yr["Year"]  = df_yr["FlightDate"].dt.year
        ap_agg_frames.append(build_airport_features(df_yr))
        del df_yr

    # Re-aggregate the per-year airport summaries into a single cross-year profile
    ap_combined = pd.concat(ap_agg_frames, ignore_index=True)
    del ap_agg_frames
    airport_features = (
        ap_combined.groupby(["airport", "Month"])
        .mean(numeric_only=True)
        .reset_index()
    )
    airport_features.to_parquet(os.path.join(PROCESSED_DIR, "airport_features.parquet"), index=False)
    print(f"Airport features: {airport_features.shape}")
    del ap_combined

    # Sequences: use all-carrier files (broader training signal; airport risk is carrier-agnostic)
    seq_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.startswith("bts_all_dfw_") and f.endswith(".parquet")
    ])
    print(f"\nBuilding sequences from all-carrier files: {[os.path.basename(f) for f in seq_files]}")

    yearly_aggs = []
    files = seq_files

    for fpath in files:
        year = int(os.path.basename(fpath).split("_")[-1].replace(".parquet",""))
        print(f"\n--- Processing {year} ---")
        df_year = pd.read_parquet(fpath)
        df_year["FlightDate"] = pd.to_datetime(df_year["FlightDate"])
        df_year["Month"] = df_year["FlightDate"].dt.month
        df_year["Year"] = df_year["FlightDate"].dt.year
        df_year["Season"] = df_year["Month"].map({
            12:"winter",1:"winter",2:"winter",
            3:"spring",4:"spring",5:"spring",
            6:"summer",7:"summer",8:"summer",
            9:"fall",10:"fall",11:"fall",
        })

        seqs  = build_sequences(df_year)
        seqs  = build_feature_matrix(seqs, airport_features)
        seqs  = label_sequences(seqs)
        # Aggregate immediately per year to avoid accumulating flight-level DFs
        yearly_aggs.append(save_features(seqs, year=year))
        del df_year, seqs

    # Combine all per-year aggregates and re-label with a consistent threshold
    all_seqs = pd.concat(yearly_aggs, ignore_index=True)
    threshold = all_seqs["observed_bad_rate"].median()
    all_seqs["target"] = (all_seqs["observed_bad_rate"] > threshold).astype(int)
    out_path = os.path.join(PROCESSED_DIR, "sequence_features.parquet")
    all_seqs.to_parquet(out_path, index=False)
    print(f"\nFinal: {len(all_seqs):,} pair×month×year rows saved → {out_path}")
    print(f"Global threshold: {threshold:.3f}  High-risk rate: {all_seqs['target'].mean():.1%}")


if __name__ == "__main__":
    main()
