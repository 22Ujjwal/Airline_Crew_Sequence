"""
Fetch Open-Meteo ERA5 historical daily weather for all airports (2015-2024).

Keeps daily granularity (not monthly aggregates) so weather can be joined
at FlightDate level in feature_engineering. Also fetches weather_code which
gives real thunderstorm / fog / snow event flags (was missing in monthly fetch).

Output: data/processed/openmeteo_daily_weather.parquet
Schema: iata, date, year, month,
        temperature_max_f, temperature_min_f,
        precipitation_in, snowfall_in, wind_speed_mph, wind_gust_mph,
        weather_code,
        is_thunder, is_heavy_snow, is_freezing_rain, is_fog, is_severe_wx

WMO weather_code reference:
  0        Clear sky
  1-3      Mainly clear / partly cloudy / overcast
  45,48    Fog / depositing rime fog
  51-57    Drizzle (light/moderate/dense, freezing)
  61-67    Rain (slight/moderate/heavy, freezing)
  71-77    Snow (slight/moderate/heavy, snow grains, ice crystals)
  80-82    Rain showers (slight/moderate/violent)
  85-86    Snow showers (slight/heavy)
  95       Thunderstorm (slight/moderate)
  96,99    Thunderstorm with hail

Run:
  python src/fetch_openmeteo_daily.py

Resumable: if interrupted, re-run — already-fetched airports are skipped.
Estimated time: ~90-110 min for 204 airports (25s sleep + jitter between calls).
"""

import os
import json
import random
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH      = os.path.join(PROCESSED_DIR, "openmeteo_daily_weather.parquet")
CACHE_PATH    = os.path.join(PROCESSED_DIR, "_openmeteo_daily_cache.json")

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
API_URL    = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",          # WMO codes — thunder (95+), snow (71-77), fog (45-48)
]

SLEEP_BETWEEN_S = 25        # conservative — previous 15s hit rate limits
SLEEP_JITTER_S  = 5         # random 0–5s per request to avoid burst pattern
RATE_LIMIT_WAIT_S = 120     # 120s base on 429 (more aggressive than old 60s)
CACHE_SAVE_EVERY  = 10      # save progress every N airports (was 28, now 10)


def fetch_airport_daily(iata: str, lat: float, lon: float, retries: int = 4) -> pd.DataFrame | None:
    """
    Fetch daily ERA5 data for one airport (2015-2024). Returns daily DataFrame
    (~3650 rows) or None on persistent failure.
    """
    params = {
        "latitude":           round(lat, 4),
        "longitude":          round(lon, 4),
        "start_date":         START_DATE,
        "end_date":           END_DATE,
        "daily":              ",".join(DAILY_VARS),
        "timezone":           "UTC",
        "temperature_unit":   "fahrenheit",
        "wind_speed_unit":    "mph",
        "precipitation_unit": "inch",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, timeout=90)
            if resp.status_code == 429:
                wait = RATE_LIMIT_WAIT_S * (attempt + 1)
                print(f"\n  Rate limited — waiting {wait}s ({iata}, attempt {attempt+1}) ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                print(f"  API error {iata}: {data.get('reason', '')}")
                return None
            break
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  FAILED {iata}: {exc}")
                return None
            time.sleep(5 * (attempt + 1))
    else:
        return None

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        print(f"  Empty response for {iata}")
        return None

    wc = daily.get("weather_code") or [0] * len(daily["time"])

    df = pd.DataFrame({
        "date":          pd.to_datetime(daily["time"]),
        "temp_max_f":    daily.get("temperature_2m_max"),
        "temp_min_f":    daily.get("temperature_2m_min"),
        "precip_in":     daily.get("precipitation_sum"),
        "snow_in":       daily.get("snowfall_sum"),
        "wind_mph":      daily.get("wind_speed_10m_max"),
        "gust_mph":      daily.get("wind_gusts_10m_max"),
        "weather_code":  pd.array(wc, dtype="Int16"),
    })

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["iata"]  = iata

    wc_num = pd.to_numeric(df["weather_code"], errors="coerce").fillna(0)
    df["is_thunder"]       = (wc_num >= 95).astype("int8")
    df["is_heavy_snow"]    = (wc_num.between(71, 77)).astype("int8")
    df["is_freezing_rain"] = (wc_num.between(56, 67)).astype("int8")
    df["is_fog"]           = (wc_num.between(40, 49)).astype("int8")
    df["is_severe_wx"]     = (wc_num >= 80).astype("int8")

    keep = [
        "iata", "date", "year", "month",
        "temp_max_f", "temp_min_f",
        "precip_in", "snow_in", "wind_mph", "gust_mph",
        "weather_code",
        "is_thunder", "is_heavy_snow", "is_freezing_rain", "is_fog", "is_severe_wx",
    ]
    return df[keep]


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Airport list from pair_risk_scores (same 204 airports model uses)
    prs      = pd.read_parquet(os.path.join(PROCESSED_DIR, "pair_risk_scores.parquet"))
    airports = sorted(set(prs["airport_A"].unique()) | set(prs["airport_B"].unique()))
    print(f"Airports to fetch: {len(airports)}")

    # Get lat/lon from airportsdata
    try:
        import airportsdata
        db     = airportsdata.load("IATA")
        coords = {a: (db[a]["lat"], db[a]["lon"]) for a in airports if a in db}
        missing = [a for a in airports if a not in db]
        if missing:
            print(f"WARNING: no coordinates for {missing} — skipping")
    except ImportError:
        print("ERROR: airportsdata not installed.")
        print("Run: pip install airportsdata")
        print("Or on this machine: /Users/ujjwalgupta/anaconda3/bin/python3 -m pip install airportsdata")
        return

    # Load incremental cache
    cache: dict = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Resuming — {len(cache)}/{len(airports)} airports already cached")

    to_fetch = [a for a in airports if a in coords and a not in cache]
    print(f"Fetching {len(to_fetch)} airports (skipping {len(cache)} cached) ...")
    print(f"Estimated time: ~{len(to_fetch) * (SLEEP_BETWEEN_S + SLEEP_JITTER_S/2) / 60:.0f} min")

    frames = []

    for i, iata in enumerate(tqdm(to_fetch, unit="airport")):
        lat, lon = coords[iata]
        df = fetch_airport_daily(iata, lat, lon)

        if df is not None:
            # Store as dict for JSON cache — convert date to string
            df_cache = df.copy()
            df_cache["date"] = df_cache["date"].astype(str)
            cache[iata] = df_cache.to_dict(orient="list")
            frames.append(df)
        else:
            print(f"  Skipping {iata} after failed retries — will retry on next run")

        # Save cache checkpoint
        if (len(cache) % CACHE_SAVE_EVERY == 0) or (i == len(to_fetch) - 1):
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)
            print(f"  Checkpoint saved ({len(cache)} airports)")

        # Rate-limit-safe sleep with jitter
        if i < len(to_fetch) - 1:
            sleep_s = SLEEP_BETWEEN_S + random.uniform(0, SLEEP_JITTER_S)
            time.sleep(sleep_s)

    # Merge cached + newly fetched
    for iata, d in cache.items():
        if iata not in [f.iloc[0]["iata"] for f in frames] if frames else True:
            cached_df = pd.DataFrame(d)
            cached_df["date"] = pd.to_datetime(cached_df["date"])
            for col in ["is_thunder", "is_heavy_snow", "is_freezing_rain", "is_fog", "is_severe_wx"]:
                if col in cached_df.columns:
                    cached_df[col] = cached_df[col].astype("int8")
            frames.append(cached_df)

    if not frames:
        print("ERROR: no data to save.")
        return

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["iata", "date"])
    result = result.sort_values(["iata", "date"]).reset_index(drop=True)

    result.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(result):,} rows → {OUT_PATH}")
    print(f"Airports: {result['iata'].nunique()} / {len(airports)}")
    print(f"Date range: {result['date'].min()} → {result['date'].max()}")
    print(f"Thunder days: {result['is_thunder'].sum():,}")
    print(f"Heavy snow days: {result['is_heavy_snow'].sum():,}")
    print(f"Fog days: {result['is_fog'].sum():,}")
    print(result[["temp_max_f", "precip_in", "wind_mph", "gust_mph"]].describe().to_string())


if __name__ == "__main__":
    main()
