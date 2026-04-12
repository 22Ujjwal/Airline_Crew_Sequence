"""
Tail-chain feature engineering.

Same aircraft tail on consecutive legs → likely same crew. For each A→DFW→B
sequence where both legs share a tail number (same FlightDate), computes:

  legs_before         — legs flown by this tail before the A→DFW leg today
  block_min_before    — block time (min) accumulated before A→DFW leg
  duty_start_hour     — first dep hour of tail today (proxy for report time)
  total_duty_min      — full window: tail's first dep → DFW→B scheduled arr
  fdp_utilization     — total_duty_min / FAA Part 117 FDP limit (0–1.5 clipped)
  legs_after          — legs on same tail after DFW→B (scheduling pressure)
  crosses_wocl        — 1 if duty window overlaps FAA WOCL (0200–0559)
  legs_in_day         — total legs on this tail today

Aggregated to (airport_A, airport_B, Month, Year) to match sequence_features.parquet.

Note: same-FlightDate matching only. Overnight arrivals (e.g. HNL→DFW dep
evening, arr next morning) are on a different FlightDate and not chained —
affects ~10% of sequences; acceptable for monthly aggregate features.

Run:
  conda run -n aadata python src/tail_chain_features.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TURNAROUND_MIN = 30    # min gap between A arr and B dep at DFW
TURNAROUND_MAX = 240   # max gap (4 hrs)
WOCL_START     = 2 * 60   # 0200 in minutes
WOCL_END       = 6 * 60   # 0600 in minutes


# ---------------------------------------------------------------------------
# FAA Part 117 simplified FDP table
# ---------------------------------------------------------------------------

# Table B (acclimated crew, no augmentation) — max FDP by report hour and
# total segments in duty period. Rounded to nearest 30 min for simplicity.
# Source: 14 CFR Part 117.23 Table B
_FDP_TABLE = {
    # (report_hour_bin) -> [1seg, 2seg, 3seg, 4seg, 5+seg]
    # report 0000-0459: WOCL overlap → shorter limits
    "wocl":    [9.0, 9.0, 9.0, 9.0, 9.0],
    # report 0500-0759: morning starts
    "morning": [9.0, 9.0, 9.0, 9.0, 9.0],
    # report 0800-1359: midday — longest FDP allowed
    "midday":  [9.0, 10.0, 10.0, 10.5, 11.0],
    # report 1400-1759: afternoon
    "aftnoon": [9.0, 9.5, 9.5, 10.0, 10.0],
    # report 1800-2359: evening — WOCL approaches
    "evening": [9.0, 9.0, 9.0, 9.0, 9.0],
}


def fdp_limit_hours(duty_start_hour: int, total_legs: int) -> float:
    if duty_start_hour < 5:
        key = "wocl"
    elif duty_start_hour < 8:
        key = "morning"
    elif duty_start_hour < 14:
        key = "midday"
    elif duty_start_hour < 18:
        key = "aftnoon"
    else:
        key = "evening"
    seg_idx = min(total_legs - 1, 4)
    return _FDP_TABLE[key][seg_idx]


# Vectorised version for DataFrame apply
def _fdp_vec(start_hrs: pd.Series, legs: pd.Series) -> pd.Series:
    conditions = [
        start_hrs < 5,
        start_hrs < 8,
        start_hrs < 14,
        start_hrs < 18,
    ]
    keys = ["wocl", "morning", "midday", "aftnoon"]
    result = pd.Series("evening", index=start_hrs.index)
    for cond, k in zip(reversed(conditions), reversed(keys)):
        result[cond] = k

    seg_idx = (legs - 1).clip(0, 4).astype(int)
    return result.map(_FDP_TABLE).combine(seg_idx, lambda table, idx: table[idx])


# ---------------------------------------------------------------------------
# Core: build tail chains for one year's raw BTS file
# ---------------------------------------------------------------------------

def _parse_hhmm_to_min(series: pd.Series) -> pd.Series:
    """HHMM integer → minutes since midnight. Fills NaN with -1 (invalid)."""
    s = series.fillna(-1).astype(int)
    valid = s >= 0
    h = (s // 100).clip(0, 23)
    m = (s % 100).clip(0, 59)
    result = (h * 60 + m).where(valid, other=np.nan)
    return result


def build_year_chain_features(fpath: str) -> pd.DataFrame:
    year = int(os.path.basename(fpath).split("_")[-1].replace(".parquet", ""))
    print(f"  Processing {year}...")

    df = pd.read_parquet(fpath)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Drop cancelled flights (no actual times) and rows without tail numbers
    df = df[(df["Cancelled"] != 1) & df["Tail_Number"].notna()].copy()
    df = df[df["Tail_Number"].str.strip() != ""].copy()

    # Actual dep/arr in minutes since midnight
    df["dep_min"] = _parse_hhmm_to_min(df["DepTime"])
    df["block"]   = df["ActualElapsedTime"].fillna(df["CRSElapsedTime"]).fillna(0)
    df["arr_min"] = df["dep_min"] + df["block"]  # avoids midnight-crossing issues

    # Drop rows where we can't compute dep_min
    df = df.dropna(subset=["dep_min"])

    # Sort chronologically within each tail×day
    df = df.sort_values(["Tail_Number", "FlightDate", "dep_min"])

    # --- Per-tail×day aggregate features ---
    grp = df.groupby(["Tail_Number", "FlightDate"])

    # Cumulative block time BEFORE each flight (shift within group)
    df["cum_block_before"] = grp["block"].cumsum() - df["block"]

    # Leg rank (0 = first flight of day)
    df["leg_rank"] = grp.cumcount()

    # Total legs in the day
    df["legs_in_day"] = grp["block"].transform("count")

    # Duty start = dep_min of first flight of day on this tail
    df["duty_start_min"] = grp["dep_min"].transform("min")
    df["duty_start_hour"] = (df["duty_start_min"] // 60).astype(int)

    # --- Inbound: A → DFW ---
    ib = df[df["Dest"] == "DFW"].copy()
    ib = ib.rename(columns={
        "Origin":          "airport_A",
        "leg_rank":        "leg_rank_A",
        "cum_block_before": "block_min_before",
        "arr_min":         "arr_min_A",
        "dep_min":         "dep_min_A",
    })
    ib["legs_before"] = ib["leg_rank_A"]  # legs before this inbound

    # --- Outbound: DFW → B ---
    ob = df[df["Origin"] == "DFW"].copy()
    ob = ob.rename(columns={
        "Dest":      "airport_B",
        "dep_min":   "dep_min_B",
        "arr_min":   "arr_min_B",
        "leg_rank":  "leg_rank_B",
        "ArrDelay":  "arr_delay_B",
    })
    ob["arr_delay_B"]  = ob["arr_delay_B"].fillna(0)
    ob["legs_after"]   = ob["legs_in_day"] - ob["leg_rank_B"] - 1

    # --- C leg pool: B → DFW (cascade downstream leg, IS in our DFW-filtered data) ---
    # These are the flights departing from airport B back to DFW.
    # LateAircraftDelay on this leg = cascade FROM the DFW→B arrival.
    c_pool = df[df["Dest"] == "DFW"].copy()
    c_pool = c_pool.rename(columns={
        "Origin":             "airport_C_origin",  # == airport_B in cascade join
        "dep_min":            "dep_min_C",
        "arr_min":            "arr_min_C",
        "LateAircraftDelay":  "c_late_aircraft_min",
        "ArrDelay":           "c_arr_delay",
    })
    c_pool["c_late_aircraft_min"] = c_pool["c_late_aircraft_min"].fillna(0)
    c_pool["c_late_flag"]         = (c_pool["c_late_aircraft_min"] >= 15).astype(int)
    c_pool["c_arr_delay"]         = c_pool["c_arr_delay"].fillna(0)

    # --- Match inbound + outbound by tail × date ---
    merge_keys = ["Tail_Number", "FlightDate", "Month",
                  "duty_start_min", "duty_start_hour", "legs_in_day"]

    ib_cols = merge_keys + ["airport_A", "arr_min_A", "block_min_before", "legs_before"]
    ob_cols = merge_keys + ["airport_B", "dep_min_B", "arr_min_B",
                            "arr_delay_B", "legs_after", "leg_rank_B"]

    pairs = ib[ib_cols].merge(ob[ob_cols], on=merge_keys, how="inner")

    # Turnaround filter: outbound must depart 30–240 min after inbound arrives
    ta = pairs["dep_min_B"] - pairs["arr_min_A"]
    pairs = pairs[(ta >= TURNAROUND_MIN) & (ta <= TURNAROUND_MAX)].copy()

    if pairs.empty:
        print(f"    No matched pairs for {year}")
        return pd.DataFrame()

    # --- Match downstream C leg: B → DFW on same tail, 30–360 min after B arrival ---
    c_join_keys = ["Tail_Number", "FlightDate"]
    c_cols      = c_join_keys + ["airport_C_origin", "dep_min_C",
                                  "c_late_aircraft_min", "c_late_flag", "c_arr_delay"]

    cascade = pairs[c_join_keys + ["airport_B", "arr_min_B"]].merge(
        c_pool[c_cols],
        on=c_join_keys,
        how="left",
    )
    # C leg must depart from airport B (not some other airport on same tail)
    cascade = cascade[cascade["airport_C_origin"] == cascade["airport_B"]].copy()
    # Time window: C departs 30–360 min after B arrives
    c_gap = cascade["dep_min_C"] - cascade["arr_min_B"]
    cascade = cascade[(c_gap >= 30) & (c_gap <= 360)].copy()
    cascade["has_c_leg"] = 1

    # Aggregate C-leg cascade per A→DFW→B pair (take worst-case if multiple C legs)
    c_agg = (
        cascade.groupby(c_join_keys + ["airport_B", "arr_min_B"])
        .agg(
            has_c_leg             = ("has_c_leg",            "max"),
            c_late_flag           = ("c_late_flag",           "max"),
            c_late_aircraft_min   = ("c_late_aircraft_min",  "max"),
            c_arr_delay           = ("c_arr_delay",           "max"),
        )
        .reset_index()
    )

    # Join cascade back to pairs
    pairs = pairs.merge(
        c_agg[c_join_keys + ["airport_B", "arr_min_B",
                              "has_c_leg", "c_late_flag",
                              "c_late_aircraft_min", "c_arr_delay"]],
        on=c_join_keys + ["airport_B", "arr_min_B"],
        how="left",
    )
    pairs["has_c_leg"]           = pairs["has_c_leg"].fillna(0)
    pairs["c_late_flag"]         = pairs["c_late_flag"].fillna(0)
    pairs["c_late_aircraft_min"] = pairs["c_late_aircraft_min"].fillna(0)
    pairs["c_arr_delay"]         = pairs["c_arr_delay"].fillna(0)

    # Cascade amplification: downstream LateAircraftDelay / B arrival delay
    # (how much of B's delay propagates to next leg)
    pairs["cascade_amplification"] = np.where(
        pairs["arr_delay_B"] > 5,
        (pairs["c_late_aircraft_min"] / pairs["arr_delay_B"].clip(lower=1)).clip(0, 3),
        np.nan,
    )

    # --- Duty chain derived features ---
    pairs["total_duty_min"] = (pairs["arr_min_B"] - pairs["duty_start_min"]).clip(lower=0)

    fdp_hours = pairs.apply(
        lambda r: fdp_limit_hours(int(r["duty_start_hour"]), int(r["legs_in_day"])),
        axis=1,
    )
    pairs["fdp_limit_min"]   = fdp_hours * 60
    pairs["fdp_utilization"] = (pairs["total_duty_min"] / pairs["fdp_limit_min"]).clip(0, 1.5)

    pairs["crosses_wocl"] = (
        (pairs["duty_start_min"] < WOCL_END) &
        (pairs["arr_min_B"]      > WOCL_START)
    ).astype(int)

    pairs["fdp_overrun"] = (pairs["fdp_utilization"] > 1.0).astype(int)

    # --- Aggregate to (airport_A, airport_B, Month, Year) ---
    pairs["Year"] = year

    agg = (
        pairs.groupby(["airport_A", "airport_B", "Month", "Year"])
        .agg(
            tc_n_pairs              = ("total_duty_min",        "count"),
            tc_legs_before_mean     = ("legs_before",           "mean"),
            tc_block_before_mean    = ("block_min_before",      "mean"),
            tc_duty_start_hour      = ("duty_start_hour",       "median"),
            tc_total_duty_mean      = ("total_duty_min",        "mean"),
            tc_total_duty_p75       = ("total_duty_min",        lambda x: x.quantile(0.75)),
            tc_fdp_util_mean        = ("fdp_utilization",       "mean"),
            tc_fdp_util_p75         = ("fdp_utilization",       lambda x: x.quantile(0.75)),
            tc_fdp_overrun_rate     = ("fdp_overrun",           "mean"),
            tc_wocl_rate            = ("crosses_wocl",          "mean"),
            tc_legs_after_mean      = ("legs_after",            "mean"),
            tc_legs_in_day_mean     = ("legs_in_day",           "mean"),
            # Downstream cascade: B → DFW third leg
            tc_downstream_rate      = ("has_c_leg",             "mean"),   # fraction with matched C leg
            tc_cascade_late_rate    = ("c_late_flag",           "mean"),   # fraction C late (late aircraft)
            tc_cascade_late_min     = ("c_late_aircraft_min",   "mean"),   # avg downstream late aircraft min
            tc_cascade_amplif_mean  = ("cascade_amplification", "mean"),   # delay amplification ratio
        )
        .reset_index()
    )

    n_with_cascade = (pairs["has_c_leg"] > 0).sum()
    print(f"    {year}: {len(pairs):,} pairs, "
          f"{n_with_cascade:,} ({n_with_cascade/len(pairs):.0%}) with B→DFW cascade leg "
          f"→ {len(agg):,} pair×month rows")
    return agg


# ---------------------------------------------------------------------------
# Airport-level cascade propagation index (no tail matching needed)
# ---------------------------------------------------------------------------

def build_airport_cascade_features(files: list) -> pd.DataFrame:
    """
    For each airport × month: how strongly does it propagate incoming delays
    to its outbound flights?

    Uses LateAircraftDelay on flights departing from each airport (X→DFW legs):
    when an inbound flight arrives late at X, the outbound X→DFW leg often
    carries LateAircraftDelay. This measures cascade amplification at X.

    Features:
      ap_cascade_rate       — fraction of departures from X with LateAircraftDelay > 15 min
      ap_cascade_min_mean   — avg LateAircraftDelay on departures from X
      ap_cascade_min_p75    — p75 LateAircraftDelay on departures from X
      ap_cascade_given_late — conditional: P(late aircraft | inbound was delayed)
    """
    print("\nBuilding airport-level cascade propagation features...")
    ap_frames = []

    for fpath in files:
        year = int(os.path.basename(fpath).split("_")[-1].replace(".parquet", ""))
        df = pd.read_parquet(fpath)
        df = df[df["Cancelled"] != 1].copy()

        # Departures from non-DFW airports heading to DFW (X → DFW)
        depart = df[df["Dest"] == "DFW"].copy()
        depart["late_aircraft_min"] = depart["LateAircraftDelay"].fillna(0)
        depart["late_flag"]         = (depart["late_aircraft_min"] >= 15).astype(int)
        depart["inbound_late"]      = (depart["ArrDelay"].fillna(0) >= 15).astype(int)
        depart["airport"]           = depart["Origin"]

        ap = (
            depart.groupby(["airport", "Month"])
            .agg(
                ap_n_flights          = ("late_flag",           "count"),
                ap_cascade_rate       = ("late_flag",           "mean"),
                ap_cascade_min_mean   = ("late_aircraft_min",   "mean"),
                ap_cascade_min_p75    = ("late_aircraft_min",   lambda x: x.quantile(0.75)),
                # Conditional: P(late aircraft | inbound delayed)
                _late_both            = ("late_flag",           lambda x:
                                         (x & (depart.loc[x.index, "inbound_late"] == 1)).sum()),
                _n_inbound_late       = ("inbound_late",        "sum"),
            )
            .reset_index()
        )
        # Conditional cascade rate (guard against division by zero)
        ap["ap_cascade_given_late"] = np.where(
            ap["_n_inbound_late"] > 0,
            ap["_late_both"] / ap["_n_inbound_late"],
            np.nan,
        )
        ap["Year"] = year
        ap_frames.append(ap.drop(columns=["_late_both", "_n_inbound_late"]))

    ap_all = pd.concat(ap_frames, ignore_index=True)

    # Average across years → stable airport×month profile
    ap_avg = (
        ap_all.groupby(["airport", "Month"])
        .agg(
            ap_cascade_rate       = ("ap_cascade_rate",       "mean"),
            ap_cascade_min_mean   = ("ap_cascade_min_mean",   "mean"),
            ap_cascade_min_p75    = ("ap_cascade_min_p75",    "mean"),
            ap_cascade_given_late = ("ap_cascade_given_late", "mean"),
        )
        .reset_index()
    )

    out = os.path.join(PROC_DIR, "airport_cascade_features.parquet")
    ap_avg.to_parquet(out, index=False)
    print(f"Airport cascade features: {ap_avg.shape}  saved → {out}")
    print(ap_avg[["ap_cascade_rate","ap_cascade_min_mean","ap_cascade_given_late"]]
          .describe().T[["mean","50%","max"]].round(3).to_string())
    return ap_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "bts_all_dfw_*.parquet")))
    aa_files  = sorted(glob.glob(os.path.join(RAW_DIR, "bts_aa_dfw_*.parquet")))

    year_file = {}
    for f in aa_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f
    for f in all_files:
        yr = f.split("_")[-1].replace(".parquet", "")
        year_file[yr] = f
    files = sorted(year_file.values())

    print(f"Processing {len(files)} raw BTS files for tail-chain + cascade features...")

    # --- Tail-chain features ---
    frames = []
    for fpath in files:
        try:
            frames.append(build_year_chain_features(fpath))
        except Exception as e:
            print(f"  ERROR on {fpath}: {e}")

    if frames:
        result = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        out = os.path.join(PROC_DIR, "tail_chain_features.parquet")
        result.to_parquet(out, index=False)
        print(f"\nTail-chain: {len(result):,} pair×month×year rows → {out}")
        tc_cols = [c for c in result.columns if c.startswith("tc_")]
        print(result[tc_cols].describe().T[["mean", "50%", "max"]].round(3).to_string())

    # --- Airport-level cascade features ---
    build_airport_cascade_features(files)


if __name__ == "__main__":
    main()
