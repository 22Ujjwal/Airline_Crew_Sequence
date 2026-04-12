"""
Download FULL BTS On-Time Performance data — all carriers, all routes, no DFW filter.

Required for downstream cascade chain analysis: A→DFW→B→C→D→...
Without DFW filtering we can follow the aircraft anywhere after the DFW sequence.

Output: data/raw/bts_full_{year}.parquet
Size: ~3-5 GB per year, ~30-40 GB total for 8 years.

Usage:
  conda run -n aadata python src/download_bts_full.py
  conda run -n aadata python src/download_bts_full.py --years 2022 2023 2024
"""

import os
import io
import argparse
import zipfile
import requests
import pandas as pd
import time
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]

BTS_URL = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Same columns as DFW download — keeps downstream join compatible
KEEP_COLS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Tail_Number", "Origin", "Dest",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15",
    "Cancelled", "CancellationCode", "Diverted",
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "DayOfWeek", "Month", "Year",
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_month(year: int, month: int) -> pd.DataFrame | None:
    url = BTS_URL.format(year=year, month=month)
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=300, headers=HEADERS)
            if resp.status_code == 503:
                wait = 2 ** attempt * 5
                print(f"  [RETRY {attempt+1}] {year}-{month:02d}: 503, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 3:
                print(f"  [WARN] {year}-{month:02d}: {e}")
                return None
            time.sleep(2 ** attempt * 3)
    else:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding="latin1", low_memory=False)
    except Exception as e:
        print(f"  [WARN] {year}-{month:02d} parse error: {e}")
        return None

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    df = df[df["Cancelled"] != 1].copy()
    return df if not df.empty else None


def download_year(year: int) -> None:
    out_path = os.path.join(RAW_DIR, f"bts_full_{year}.parquet")
    if os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"{year}: already downloaded ({size_gb:.1f} GB), skipping.")
        return

    print(f"\nDownloading {year} (all carriers, all routes, sequential)...")
    frames = {}
    for month in tqdm(range(1, 13), desc=str(year)):
        df = download_month(year, month)
        if df is not None:
            frames[month] = df

    if frames:
        combined = pd.concat([frames[m] for m in sorted(frames)], ignore_index=True)
        combined.to_parquet(out_path, index=False)
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"  Saved {len(combined):,} rows → {out_path}  ({size_gb:.1f} GB)")
    else:
        print(f"  No data for {year}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    args = parser.parse_args()

    print(f"Downloading full BTS data (all carriers, all routes) for years: {args.years}")
    print(f"Expected size: ~3-5 GB/year, ~{3.5 * len(args.years):.0f} GB total")
    print(f"Output: data/raw/bts_full_{{year}}.parquet\n")

    for year in args.years:
        download_year(year)

    print("\nDone. Run cascade_chain_features.py next.")


if __name__ == "__main__":
    main()
