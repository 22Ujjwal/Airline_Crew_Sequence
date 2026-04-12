"""
Full downstream cascade chain analysis: A→DFW→B→C→D→...

For each A→DFW→B sequence (tail-matched), follows the SAME aircraft forward
through any subsequent legs on the same day: B→C→D→E→...
Measures how far and how severely the delay cascade propagates downstream.

Requires:
  - data/raw/bts_full_{year}.parquet  (from download_bts_full.py)
  - data/raw/bts_all_dfw_{year}.parquet  (existing, to identify A→DFW→B seed sequences)

Output: data/processed/cascade_chain_features.parquet

Features per (airport_A, airport_B, Month, Year):
  cc_chain_depth_mean       — avg downstream legs observed after DFW→B
  cc_chain_depth_max        — max downstream chain depth observed
  cc_total_delay_mean       — avg total LateAircraftDelay across ALL downstream legs (min)
  cc_total_delay_p75        — p75 total downstream delay
  cc_cascade_rate           — fraction of sequences where ≥1 downstream leg is late-aircraft
  cc_recovery_rate          — fraction where delay fully recovers (downstream leg has 0 delay)
  cc_amplification_mean     — total_downstream_delay / B_arr_delay (>1 = amplified)
  cc_affected_airports_mean — avg unique airports hit downstream per cascade
  cc_max_single_leg_delay   — avg of worst single downstream leg delay

Run:
  conda run -n aadata python src/cascade_chain_features.py
  conda run -n aadata python src/cascade_chain_features.py --max-depth 6
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TURNAROUND_MIN  = 30    # min gap inbound arr → outbound dep (minutes)
TURNAROUND_MAX  = 240   # max gap A→DFW (4 hrs)
LEG_GAP_MIN     = 15    # min gap between any two legs in cascade chain
LEG_GAP_MAX     = 360   # max gap between cascade legs (6 hrs)
LATE_THRESHOLD  = 15    # minutes to count as a late aircraft delay event


def _parse_hhmm_to_min(series: pd.Series) -> pd.Series:
    s = series.fillna(-1).astype(int)
    valid = s >= 0
    h = (s // 100).clip(0, 23)
    m = (s % 100).clip(0, 59)
    return (h * 60 + m).where(valid, other=np.nan)


def _resolve_files() -> dict:
    """Map year → (dfw_file, full_file). Only years with both files are processed."""
    dfw_files  = {}
    full_files = {}

    for f in glob.glob(os.path.join(RAW_DIR, "bts_all_dfw_*.parquet")):
        yr = int(os.path.basename(f).split("_")[-1].replace(".parquet", ""))
        dfw_files[yr] = f
    for f in glob.glob(os.path.join(RAW_DIR, "bts_full_*.parquet")):
        yr = int(os.path.basename(f).split("_")[-1].replace(".parquet", ""))
        full_files[yr] = f

    common = sorted(set(dfw_files) & set(full_files))
    if not common:
        print("No years with both bts_all_dfw_* and bts_full_* files found.")
        print("Run download_bts_full.py first.")
        return {}

    return {yr: (dfw_files[yr], full_files[yr]) for yr in common}


def build_year_cascade(year: int, dfw_path: str, full_path: str,
                       max_depth: int = 8) -> pd.DataFrame:
    print(f"  {year}: loading data...")

    # --- Load DFW data: find A→DFW→B seed sequences (tail-matched) ---
    dfw = pd.read_parquet(dfw_path)
    dfw = dfw[dfw["Cancelled"] != 1].copy()
    dfw["dep_min"] = _parse_hhmm_to_min(dfw["DepTime"])
    dfw["block"]   = dfw["ActualElapsedTime"].fillna(dfw["CRSElapsedTime"]).fillna(0)
    dfw["arr_min"] = dfw["dep_min"] + dfw["block"]
    dfw = dfw.dropna(subset=["dep_min", "Tail_Number"])
    dfw = dfw[dfw["Tail_Number"].str.strip() != ""]

    # Inbound A→DFW
    ib = dfw[dfw["Dest"] == "DFW"][
        ["Tail_Number", "FlightDate", "Month", "Origin", "arr_min", "dep_min"]
    ].rename(columns={"Origin": "airport_A", "arr_min": "arr_min_A", "dep_min": "dep_min_A"})

    # Outbound DFW→B
    ob = dfw[dfw["Origin"] == "DFW"][
        ["Tail_Number", "FlightDate", "ArrDelay", "Dest", "dep_min", "arr_min"]
    ].rename(columns={"Dest": "airport_B", "dep_min": "dep_min_B",
                      "arr_min": "arr_min_B", "ArrDelay": "arr_delay_B"})
    ob["arr_delay_B"] = ob["arr_delay_B"].fillna(0)

    # Match A→DFW + DFW→B by tail × date with turnaround window
    seeds = ib.merge(ob, on=["Tail_Number", "FlightDate"])
    ta = seeds["dep_min_B"] - seeds["arr_min_A"]
    seeds = seeds[(ta >= TURNAROUND_MIN) & (ta <= TURNAROUND_MAX)].copy()

    if seeds.empty:
        print(f"    No seed sequences for {year}")
        return pd.DataFrame()

    print(f"    {year}: {len(seeds):,} seed sequences  |  loading full BTS...")

    # --- Load full national BTS: all flights for cascade tracking ---
    full = pd.read_parquet(full_path)
    full = full[full["Cancelled"] != 1].copy()
    full["dep_min"]          = _parse_hhmm_to_min(full["DepTime"])
    full["block"]            = full["ActualElapsedTime"].fillna(full["CRSElapsedTime"]).fillna(0)
    full["arr_min"]          = full["dep_min"] + full["block"]
    full["late_aircraft_min"] = full["LateAircraftDelay"].fillna(0)
    full["late_flag"]        = (full["late_aircraft_min"] >= LATE_THRESHOLD).astype(int)
    full["arr_delay"]        = full["ArrDelay"].fillna(0)
    full = full.dropna(subset=["dep_min", "Tail_Number"])
    full = full[full["Tail_Number"].str.strip() != ""]

    # Keep only tails that appear in our seed sequences (massive memory saving)
    seed_tails = set(seeds["Tail_Number"].unique())
    full = full[full["Tail_Number"].isin(seed_tails)].copy()
    print(f"    Full BTS filtered to {len(seed_tails):,} seed tails: {len(full):,} flights")

    # Sort for chain building
    full = full.sort_values(["Tail_Number", "FlightDate", "dep_min"])

    # Build per-tail×date lookup: list of (dep_min, arr_min, dest, late_aircraft_min, arr_delay)
    # We'll use this to walk forward from the DFW→B arrival
    full_grp = full.groupby(["Tail_Number", "FlightDate"])

    # --- Walk the cascade chain for each seed sequence ---
    records = []

    for _, seed in seeds.iterrows():
        tail    = seed["Tail_Number"]
        date    = seed["FlightDate"]
        arr_B   = seed["arr_min_B"]
        arr_dly = seed["arr_delay_B"]

        try:
            tail_day = full_grp.get_group((tail, date))
        except KeyError:
            # Tail not in full data for this date (rare)
            continue

        # Find all legs AFTER the DFW→B arrival
        downstream = tail_day[tail_day["dep_min"] >= arr_B + LEG_GAP_MIN].sort_values("dep_min")

        # Walk the chain leg by leg
        chain_airports  = []
        chain_delays    = []   # LateAircraftDelay on each leg
        chain_arr_delays = []  # ArrDelay on each leg
        current_arr     = arr_B

        for _, leg in downstream.iterrows():
            gap = leg["dep_min"] - current_arr
            if gap < LEG_GAP_MIN or gap > LEG_GAP_MAX:
                break  # too short (overlap) or too long (crew rest / new duty)
            if len(chain_delays) >= max_depth:
                break

            chain_airports.append(leg["Dest"])
            chain_delays.append(leg["late_aircraft_min"])
            chain_arr_delays.append(leg["arr_delay"])
            current_arr = leg["arr_min"]

        depth         = len(chain_delays)
        total_delay   = sum(chain_delays)
        any_late      = any(d >= LATE_THRESHOLD for d in chain_delays)
        recovered     = any(d == 0 for d in chain_delays) if chain_delays else False
        n_airports    = len(set(chain_airports))
        max_leg_delay = max(chain_delays) if chain_delays else 0
        amplif        = (total_delay / max(arr_dly, 1)) if arr_dly > 5 else np.nan

        records.append({
            "Tail_Number": tail,
            "FlightDate":  date,
            "airport_A":   seed["airport_A"],
            "airport_B":   seed["airport_B"],
            "Month":       seed["Month"],
            "Year":        year,
            "depth":       depth,
            "total_delay": total_delay,
            "any_late":    int(any_late),
            "recovered":   int(recovered),
            "n_airports":  n_airports,
            "max_leg_delay": max_leg_delay,
            "amplification": amplif,
        })

    if not records:
        print(f"    No cascade chains built for {year}")
        return pd.DataFrame()

    chains = pd.DataFrame(records)
    n_cascaded = chains["any_late"].sum()
    print(f"    {year}: {len(chains):,} chains  |  "
          f"{n_cascaded/len(chains):.1%} with downstream cascade  |  "
          f"avg depth {chains['depth'].mean():.1f}")

    # --- Aggregate to (airport_A, airport_B, Month, Year) ---
    agg = (
        chains.groupby(["airport_A", "airport_B", "Month", "Year"])
        .agg(
            cc_n_chains              = ("depth",         "count"),
            cc_chain_depth_mean      = ("depth",         "mean"),
            cc_chain_depth_max       = ("depth",         "max"),
            cc_total_delay_mean      = ("total_delay",   "mean"),
            cc_total_delay_p75       = ("total_delay",   lambda x: x.quantile(0.75)),
            cc_cascade_rate          = ("any_late",      "mean"),
            cc_recovery_rate         = ("recovered",     "mean"),
            cc_amplification_mean    = ("amplification", "mean"),
            cc_affected_airports_mean= ("n_airports",    "mean"),
            cc_max_single_leg_delay  = ("max_leg_delay", "mean"),
        )
        .reset_index()
    )

    print(f"    → {len(agg):,} pair×month rows")
    return agg


def main(max_depth: int = 8):
    year_files = _resolve_files()
    if not year_files:
        return

    print(f"Building cascade chain features for years: {sorted(year_files)}")
    print(f"Max cascade depth: {max_depth} legs\n")

    frames = []
    for year, (dfw_path, full_path) in year_files.items():
        try:
            agg = build_year_cascade(year, dfw_path, full_path, max_depth)
            if not agg.empty:
                frames.append(agg)
        except Exception as e:
            print(f"  ERROR {year}: {e}")

    if not frames:
        print("No cascade features produced.")
        return

    result = pd.concat(frames, ignore_index=True)
    out = os.path.join(PROC_DIR, "cascade_chain_features.parquet")
    result.to_parquet(out, index=False)

    print(f"\nTotal: {len(result):,} pair×month×year rows → {out}")
    print("\nFeature summary:")
    cc_cols = [c for c in result.columns if c.startswith("cc_")]
    print(result[cc_cols].describe().T[["mean", "50%", "max"]].round(3).to_string())
    print("\nNext: add CC_FEATURES to model_lgbm.py TAIL_CHAIN_FEATURES list and retrain.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=8,
                        help="Max number of downstream legs to track (default: 8)")
    args = parser.parse_args()
    main(args.max_depth)
