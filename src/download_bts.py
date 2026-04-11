"""
Download BTS On-Time Performance data for American Airlines flights involving DFW.
Skips 2020-2021 (COVID anomaly). Saves filtered parquet files per year.

BTS zip URL pattern:
  https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{YEAR}_{MONTH}.zip
"""

import os
import io
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Years to pull (skip COVID)
YEARS = [2018, 2019, 2022, 2023, 2024]
MONTHS = list(range(1, 13))

BTS_URL = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Columns we actually need
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


def download_month(year: int, month: int) -> pd.DataFrame | None:
    url = BTS_URL.format(year=year, month=month)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] {year}-{month:02d}: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)

    # Keep only columns we need (some may be missing in older files)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available]

    # Filter: AA flights where DFW is origin or destination
    mask = (df["Reporting_Airline"] == "AA") & (
        (df["Origin"] == "DFW") | (df["Dest"] == "DFW")
    )
    return df[mask].copy()


def download_year(year: int) -> None:
    out_path = os.path.join(RAW_DIR, f"bts_aa_dfw_{year}.parquet")
    if os.path.exists(out_path):
        print(f"{year}: already downloaded, skipping.")
        return

    print(f"\nDownloading {year} (parallel)...")
    frames = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download_month, year, m): m for m in MONTHS}
        for fut in tqdm(as_completed(futures), total=len(MONTHS), desc=str(year)):
            month = futures[fut]
            df = fut.result()
            if df is not None and not df.empty:
                frames[month] = df

    if frames:
        combined = pd.concat([frames[m] for m in sorted(frames)], ignore_index=True)
        combined.to_parquet(out_path, index=False)
        print(f"  Saved {len(combined):,} rows → {out_path}")
    else:
        print(f"  No data for {year}")


def main():
    for year in YEARS:
        download_year(year)


if __name__ == "__main__":
    main()
