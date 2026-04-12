"""
Download BTS On-Time Performance data for flights involving DFW.

Modes:
  --years 2015 2016 2017 2022 2023 2024   (default: 2015-2019 + 2022-2024)
  --all-carriers                          (default: AA only)
  --include-covid                         (include 2020-2021, tagged with covid_year=1)

All-carriers mode downloads full BTS monthly zips (~100-200MB each) and
filters to any flight where Origin=DFW or Dest=DFW. Larger but richer
airport risk profiles from the full industry.

AA-only mode is faster (smaller filtered files) and directly matches the
challenge scope (AA crew sequencing).
"""

import os
import io
import argparse
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
COVID_YEARS   = [2020, 2021]

BTS_URL = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

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


def download_month(year: int, month: int, aa_only: bool, is_covid: bool) -> pd.DataFrame | None:
    url = BTS_URL.format(year=year, month=month)
    try:
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] {year}-{month:02d}: {e}")
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

    # Filter to DFW-connected flights
    dfw_mask = (df["Origin"] == "DFW") | (df["Dest"] == "DFW")
    if aa_only:
        dfw_mask = dfw_mask & (df["Reporting_Airline"] == "AA")
    df = df[dfw_mask].copy()

    if is_covid:
        df["covid_year"] = 1

    return df if not df.empty else None


def download_year(year: int, aa_only: bool, include_covid: bool) -> None:
    is_covid = year in COVID_YEARS
    if is_covid and not include_covid:
        print(f"{year}: COVID year, skipping (use --include-covid to include)")
        return

    suffix   = "aa" if aa_only else "all"
    out_path = os.path.join(RAW_DIR, f"bts_{suffix}_dfw_{year}.parquet")
    if os.path.exists(out_path):
        print(f"{year} ({suffix}): already downloaded, skipping.")
        return

    print(f"\nDownloading {year} ({'AA only' if aa_only else 'all carriers'}"
          f"{', COVID-tagged' if is_covid else ''})...")
    frames = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(download_month, year, m, aa_only, is_covid): m
            for m in range(1, 13)
        }
        for fut in tqdm(as_completed(futures), total=12, desc=str(year)):
            month = futures[fut]
            df = fut.result()
            if df is not None:
                frames[month] = df

    if frames:
        combined = pd.concat([frames[m] for m in sorted(frames)], ignore_index=True)
        combined.to_parquet(out_path, index=False)
        print(f"  Saved {len(combined):,} rows → {out_path}")
    else:
        print(f"  No data for {year}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--all-carriers", action="store_true",
                        help="Include all airlines at DFW (not just AA)")
    parser.add_argument("--include-covid", action="store_true",
                        help="Download 2020-2021 (tagged with covid_year=1)")
    args = parser.parse_args()

    years = args.years
    if args.include_covid:
        years = sorted(set(years) | set(COVID_YEARS))

    aa_only = not args.all_carriers

    print(f"Years: {years}")
    print(f"Mode:  {'AA only' if aa_only else 'All carriers at DFW'}")
    print(f"COVID: {'included' if args.include_covid else 'skipped'}")
    print()

    for year in years:
        download_year(year, aa_only=aa_only, include_covid=args.include_covid)


if __name__ == "__main__":
    main()
