"""
Multi-hop DFW cascade chain: A→DFW→B→DFW→C→DFW→D→...

For hub-and-spoke operations (AA at DFW), most aircraft return to DFW between
every spoke stop. Our existing DFW-filtered BTS data captures the FULL chain:
  hop 1: A→DFW→B (seed sequence)
  hop 2: B→DFW→C (already in our data — B→DFW is a DFW-touching flight)
  hop 3: C→DFW→D (same)
  ...

LateAircraftDelay propagates through each hop. This measures total network
disruption caused by the original A→DFW→B sequence.

Does NOT require full national BTS download — uses existing bts_all_dfw_*.parquet.

Output: data/processed/multihop_cascade_features.parquet

Features per (airport_A, airport_B, Month, Year):
  mhc_n_hops_mean           — avg subsequent DFW hops after DFW→B
  mhc_n_hops_max            — max hops observed
  mhc_total_late_min_mean   — avg total LateAircraftDelay across all subsequent hops (min)
  mhc_total_late_min_p75    — p75 total downstream late aircraft delay
  mhc_cascade_hop_rate      — fraction of chains where ≥1 subsequent hop has late aircraft delay
  mhc_cascade_depth_mean    — avg number of hops with LateAircraftDelay > 15 min
  mhc_unique_airports_mean  — avg unique airports touched downstream
  mhc_recovery_rate         — fraction where delay fully clears before end of chain

Run:
  conda run -n aadata python src/multihop_dfw_cascade.py
"""

import os
import glob
import numpy as np
import pandas as pd

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TURNAROUND_MIN = 30
TURNAROUND_MAX = 240
HOP_GAP_MIN    = 30    # min gap between DFW return arrival and next DFW departure
HOP_GAP_MAX    = 360   # max gap (6 hrs — beyond this = new duty period / rest)
LATE_THRESHOLD = 15    # min LateAircraftDelay to count as cascaded


def _parse_hhmm_to_min(series: pd.Series) -> pd.Series:
    s = series.fillna(-1).astype(int)
    valid = s >= 0
    h = (s // 100).clip(0, 23)
    m = (s % 100).clip(0, 59)
    return (h * 60 + m).where(valid, other=np.nan)


def build_year_multihop(fpath: str) -> pd.DataFrame:
    year = int(os.path.basename(fpath).split("_")[-1].replace(".parquet", ""))
    print(f"  {year}: loading...")

    df = pd.read_parquet(fpath)
    df = df[df["Cancelled"] != 1].copy()
    df["dep_min"]           = _parse_hhmm_to_min(df["DepTime"])
    df["block"]             = df["ActualElapsedTime"].fillna(df["CRSElapsedTime"]).fillna(0)
    df["arr_min"]           = df["dep_min"] + df["block"]
    df["late_aircraft_min"] = df["LateAircraftDelay"].fillna(0)
    df["arr_delay"]         = df["ArrDelay"].fillna(0)
    df = df.dropna(subset=["dep_min", "Tail_Number"])
    df = df[df["Tail_Number"].str.strip() != ""]
    df = df.sort_values(["Tail_Number", "FlightDate", "dep_min"])

    # --- Build ordered DFW-hop sequences per tail × date ---
    # Each "hop" = one (inbound_to_DFW, outbound_from_DFW) pair on same tail
    ib = df[df["Dest"] == "DFW"].copy()
    ob = df[df["Origin"] == "DFW"].copy()

    ib = ib.rename(columns={
        "Origin": "stop_airport",
        "arr_min": "arr_dfw_min",
        "arr_delay": "inb_arr_delay",
        "late_aircraft_min": "inb_late_min",
    })
    ob = ob.rename(columns={
        "Dest": "next_airport",
        "dep_min": "dep_dfw_min",
        "arr_min": "arr_next_min",
        "late_aircraft_min": "ob_late_min",
    })

    merge_keys = ["Tail_Number", "FlightDate"]
    ib_cols = merge_keys + ["stop_airport", "arr_dfw_min", "inb_late_min", "inb_arr_delay", "Month"]
    ob_cols = merge_keys + ["next_airport", "dep_dfw_min", "arr_next_min", "ob_late_min"]

    # All DFW hops: inbound × outbound on same tail×date within turnaround window
    hops = ib[ib_cols].merge(ob[ob_cols], on=merge_keys, how="inner")
    ta = hops["dep_dfw_min"] - hops["arr_dfw_min"]
    hops = hops[(ta >= TURNAROUND_MIN) & (ta <= TURNAROUND_MAX)].copy()
    hops["hop_late_flag"] = (
        (hops["inb_late_min"] >= LATE_THRESHOLD) | (hops["ob_late_min"] >= LATE_THRESHOLD)
    ).astype(int)
    hops["hop_max_late"] = hops[["inb_late_min", "ob_late_min"]].max(axis=1)
    hops = hops.sort_values(["Tail_Number", "FlightDate", "arr_dfw_min"])

    # Group hops by tail×date into an ordered list
    hop_grp = hops.groupby(["Tail_Number", "FlightDate"])

    # --- Identify seed sequences: first DFW hop where stop_airport = A, next_airport = B ---
    # (A→DFW→B is the first matching hop; subsequent hops are the cascade)
    records = []

    for (tail, date), group in hop_grp:
        group = group.reset_index(drop=True)
        n = len(group)

        for i, seed in group.iterrows():
            airport_A = seed["stop_airport"]
            airport_B = seed["next_airport"]
            arr_B_min = seed["arr_next_min"]

            # Subsequent hops: same tail×date, arriving at DFW AFTER the B arrival
            # i.e., the chain continues: B→DFW→C, C→DFW→D, ...
            chain_airports  = []
            chain_late_mins = []
            chain_late_flags = []

            prev_arr = arr_B_min
            for j in range(i + 1, n):
                hop = group.iloc[j]
                # B must be the "stop_airport" of next hop (returned from B to DFW)
                # and timing must be feasible
                gap = hop["arr_dfw_min"] - prev_arr
                if gap < HOP_GAP_MIN:
                    continue   # too fast — this hop was already in-progress
                if gap > HOP_GAP_MAX:
                    break      # too long — crew rest / different duty period

                chain_airports.append(hop["next_airport"])
                chain_late_mins.append(hop["hop_max_late"])
                chain_late_flags.append(hop["hop_late_flag"])
                prev_arr = hop["arr_next_min"] if not np.isnan(hop["arr_next_min"]) else hop["dep_dfw_min"] + 90

            n_hops          = len(chain_late_mins)
            total_late      = sum(chain_late_mins)
            cascade_depth   = sum(chain_late_flags)
            any_cascade     = int(any(f > 0 for f in chain_late_flags))
            recovered       = int(any(m == 0 for m in chain_late_mins)) if chain_late_mins else 0
            unique_airports = len(set(chain_airports))

            records.append({
                "Tail_Number": tail,
                "FlightDate":  date,
                "airport_A":   airport_A,
                "airport_B":   airport_B,
                "Month":       seed["Month"],
                "Year":        year,
                "n_hops":          n_hops,
                "total_late_min":  total_late,
                "cascade_depth":   cascade_depth,
                "any_cascade":     any_cascade,
                "recovered":       recovered,
                "unique_airports": unique_airports,
            })

    if not records:
        return pd.DataFrame()

    chains = pd.DataFrame(records)
    n_cascaded = chains["any_cascade"].sum()
    print(f"    {year}: {len(chains):,} seed sequences  |  "
          f"avg {chains['n_hops'].mean():.1f} downstream hops  |  "
          f"{n_cascaded/len(chains):.1%} with downstream cascade")

    agg = (
        chains.groupby(["airport_A", "airport_B", "Month", "Year"])
        .agg(
            mhc_n_chains            = ("n_hops",          "count"),
            mhc_n_hops_mean         = ("n_hops",          "mean"),
            mhc_n_hops_max          = ("n_hops",          "max"),
            mhc_total_late_min_mean = ("total_late_min",  "mean"),
            mhc_total_late_min_p75  = ("total_late_min",  lambda x: x.quantile(0.75)),
            mhc_cascade_hop_rate    = ("any_cascade",     "mean"),
            mhc_cascade_depth_mean  = ("cascade_depth",   "mean"),
            mhc_unique_airports_mean= ("unique_airports", "mean"),
            mhc_recovery_rate       = ("recovered",       "mean"),
        )
        .reset_index()
    )

    print(f"    → {len(agg):,} pair×month rows")
    return agg


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

    print(f"Building multi-hop DFW cascade features from {len(files)} files...")
    print("Chain: A→DFW→B→DFW→C→DFW→D→... (hub-and-spoke cascade)\n")

    frames = []
    for fpath in files:
        try:
            agg = build_year_multihop(fpath)
            if not agg.empty:
                frames.append(agg)
        except Exception as e:
            import traceback
            print(f"  ERROR {fpath}: {e}")
            traceback.print_exc()

    if not frames:
        print("No data produced.")
        return

    result = pd.concat(frames, ignore_index=True)
    out = os.path.join(PROC_DIR, "multihop_cascade_features.parquet")
    result.to_parquet(out, index=False)

    print(f"\nTotal: {len(result):,} pair×month×year rows → {out}")
    print("\nFeature summary:")
    mhc_cols = [c for c in result.columns if c.startswith("mhc_")]
    print(result[mhc_cols].describe().T[["mean", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
