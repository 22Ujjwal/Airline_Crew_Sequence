"""
Fetch Open-Meteo ERA5 historical weather for all airports in the dataset.

Replaces GSOM (which has ~55% missing airport coverage) with ERA5 reanalysis
data that covers every airport on earth at no cost and with no API key.

Output: data/processed/openmeteo_airport_monthly.parquet

Schema (per airport × year × month):
  iata, year, month,
  avg_wind_mph,       -- mean of daily max wind speed
  max_gust_mph,       -- monthly max wind gust
  total_precip_in,    -- total monthly precipitation
  precip_days,        -- days with precip > 0.1 in
  snow_days,          -- days with snowfall > 0.1 in
  thunder_days,       -- days with WMO code >= 95 (thunderstorm)
  severe_wx_days,     -- days with WMO code >= 80 (heavy precip / storm)
  avg_temp_max_f,     -- mean daily high temperature
  avg_temp_min_f,     -- mean daily low temperature
  extreme_cold_days,  -- days with min temp < 32 F (ice/frost risk)
  extreme_heat_days,  -- days with max temp > 95 F (convection/tstorm risk)

Run:
  python src/fetch_openmeteo.py
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import airportsdata
from tqdm import tqdm

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH      = os.path.join(PROCESSED_DIR, "openmeteo_airport_monthly.parquet")
CACHE_PATH    = os.path.join(PROCESSED_DIR, "_openmeteo_raw_cache.json")

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
API_URL    = "https://archive-api.open-meteo.com/v1/archive"

# Monthly aggregation variables — 1 row per month instead of 365 rows/year.
# 30x less data per request → avoids rate limits, faster fetch.
MONTHLY_VARS = [
    "temperature_2m_max",       # mean of daily max temps
    "temperature_2m_min",       # mean of daily min temps
    "precipitation_sum",        # total monthly precip
    "snowfall_sum",             # total monthly snowfall
    "wind_speed_10m_max",       # mean of daily max wind speed
    "wind_gusts_10m_max",       # mean of daily max gusts
    # precipitation_hours is daily-only — not available in monthly aggregation
]

RATE_LIMIT_WAIT_S = 120   # seconds to wait on 429 before retrying


# ── Fetch one airport ─────────────────────────────────────────────────────────

def fetch_airport(iata: str, lat: float, lon: float, retries: int = 4) -> pd.DataFrame | None:
    """
    Fetch monthly ERA5 climate data for one airport (2015-2024).
    Uses the native monthly aggregation endpoint — 1 row per month, 120 rows total.
    Returns a DataFrame or None on persistent failure.
    """
    params = {
        "latitude":           round(lat, 4),
        "longitude":          round(lon, 4),
        "start_date":         START_DATE,
        "end_date":           END_DATE,
        "monthly":            ",".join(MONTHLY_VARS),
        "timezone":           "auto",
        "temperature_unit":   "fahrenheit",
        "wind_speed_unit":    "mph",
        "precipitation_unit": "inch",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, timeout=45)
            if resp.status_code == 429:
                wait = RATE_LIMIT_WAIT_S * (attempt + 1)
                print(f"\n  Rate limited — waiting {wait}s before retry ({iata}) ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                print(f"  API error {iata}: {data.get('reason','')}")
                return None
            break
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  FAILED {iata}: {exc}")
                return None
            time.sleep(3 * (attempt + 1))
    else:
        return None

    monthly = data.get("monthly", {})
    if not monthly or "time" not in monthly:
        print(f"  Empty monthly response for {iata}")
        return None

    df = pd.DataFrame({
        "date":        pd.to_datetime(monthly["time"]),
        "avg_wind_mph":    monthly.get("wind_speed_10m_max"),     # mean of daily max
        "max_gust_mph":    monthly.get("wind_gusts_10m_max"),     # mean of daily gusts
        "total_precip_in": monthly.get("precipitation_sum"),
        "snow_in":         monthly.get("snowfall_sum"),
        "avg_temp_max_f":  monthly.get("temperature_2m_max"),
        "avg_temp_min_f":  monthly.get("temperature_2m_min"),
    })

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Derived columns that mirror old GSOM feature names
    # precip_days: estimate from total precip (avg ~0.1 in/day threshold → ~10 days per inch)
    df["precip_days"]     = (df["total_precip_in"].fillna(0) * 10).clip(0, 31).round()
    df["snow_days"]       = (df["snow_in"].fillna(0) > 0.1).astype(int)            # binary month flag
    df["severe_wx_days"]  = (df["total_precip_in"].fillna(0) > 4.0).astype(int)    # >4in/mo = heavy month
    df["extreme_cold_days"] = (df["avg_temp_min_f"].fillna(40) < 32).astype(int)
    df["extreme_heat_days"] = (df["avg_temp_max_f"].fillna(70) > 95).astype(int)

    df["iata"] = iata
    keep = ["iata", "year", "month", "avg_wind_mph", "max_gust_mph", "total_precip_in",
            "precip_days", "snow_days", "severe_wx_days",
            "avg_temp_max_f", "avg_temp_min_f",
            "extreme_cold_days", "extreme_heat_days"]
    return df[keep]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load airport list from pair_risk_scores (all airports the model uses)
    prs      = pd.read_parquet(os.path.join(PROCESSED_DIR, "pair_risk_scores.parquet"))
    airports = sorted(set(prs["airport_A"].unique()) | set(prs["airport_B"].unique()))
    print(f"Airports to fetch: {len(airports)}")

    # Get lat/lon from airportsdata
    db      = airportsdata.load("IATA")
    coords  = {a: (db[a]["lat"], db[a]["lon"]) for a in airports if a in db}
    missing = [a for a in airports if a not in db]
    if missing:
        print(f"WARNING: no coordinates for {missing} — skipping")

    # Load incremental cache (so we can resume if interrupted)
    cache: dict = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Resuming — {len(cache)} airports already cached")

    # Fetch loop
    frames = []
    to_fetch = [a for a in airports if a in coords and a not in cache]
    print(f"Fetching {len(to_fetch)} airports (skipping {len(cache)} cached)…")

    for iata in tqdm(to_fetch, unit="airport"):
        lat, lon = coords[iata]
        df = fetch_airport(iata, lat, lon)
        if df is not None:
            cache[iata] = df.to_dict(orient="list")
            frames.append(df)
        time.sleep(0.4)   # polite rate limit (~2.5 req/sec)

        # Save cache every 20 airports so progress survives interruption
        if len(cache) % 20 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)

    # Merge cached + newly fetched
    for iata, d in cache.items():
        if iata not in to_fetch:          # already-cached airports
            frames.append(pd.DataFrame(d))

    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    if not frames:
        print("ERROR: no data fetched.")
        return

    result = pd.concat(frames, ignore_index=True)

    # Keep only columns we need; enforce dtypes
    float_cols = [
        "avg_wind_mph", "max_gust_mph", "total_precip_in",
        "avg_temp_max_f", "avg_temp_min_f",
    ]
    int_cols = [
        "precip_days", "snow_days",
        "severe_wx_days", "extreme_cold_days", "extreme_heat_days",
    ]
    for c in float_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype(float)
    for c in int_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0).astype(int)

    result = result[["iata", "year", "month"] + float_cols + int_cols]
    result = result.sort_values(["iata", "year", "month"]).reset_index(drop=True)

    result.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows → {OUT_PATH}")
    print(f"Airports covered: {result['iata'].nunique()} / {len(airports)}")
    print(result.describe().to_string())


if __name__ == "__main__":
    main()
