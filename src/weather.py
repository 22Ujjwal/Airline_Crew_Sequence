"""
Weather data integration for the AA Crew Sequence Risk model.

Two modes:
  1. Historical (training enrichment) — Iowa Environmental Mesonet (IEM) ASOS data
     Free, no API key, covers all US airports back to the 1970s.
     Download: daily aggregates per airport per year.

  2. Real-time (inference) — Aviation Weather Center (AWC) METAR API
     Free, no API key, returns current + recent observations.

Output features (same schema for both modes):
  has_thunderstorm      bool   Any hour with TS in wx codes
  has_low_ceiling       bool   Ceiling < 1000 ft AGL
  has_fog               bool   FG or BR in wx codes + visibility < 3mi
  has_snow_ice          bool   SN, FZRA, PL in wx codes
  min_visibility_mi     float  Lowest reported visibility
  max_wind_kt           float  Highest wind speed (knots)
  max_gust_kt           float  Highest wind gust (knots)
  ceiling_ft            float  Lowest ceiling (ft AGL), NaN if clear
  weather_severity      float  Composite 0–1 severity score
"""

import os
import io
import time
import logging
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)

RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather")
os.makedirs(WEATHER_DIR, exist_ok=True)

# Most US airports: ICAO = "K" + IATA. Exceptions listed explicitly.
IATA_ICAO_OVERRIDES = {
    "HNL": "PHNL", "OGG": "PHOG", "KOA": "PHKO", "ITO": "PHTO", "LIH": "PHLI",
    "GUM": "PGUM", "SJU": "TJSJ", "STT": "TIST", "STX": "TISX",
    "ANC": "PANC", "FAI": "PAFA", "JNU": "PAJN", "KTN": "PAKT",
}


def iata_to_icao(iata: str) -> str:
    return IATA_ICAO_OVERRIDES.get(iata.upper(), f"K{iata.upper()}")


# ---------------------------------------------------------------------------
# IEM Historical METAR Download
# ---------------------------------------------------------------------------

IEM_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={icao}&data=vsby,sknt,gust,wxcodes,skyc1,skyc2,skyl1,skyl2,tmpf,dwpf"
    "&tz=UTC&format=comma&latlon=no&report=1"
    "&year1={year}&month1=1&day1=1&year2={year}&month2=12&day2=31"
)


def _download_iem_airport_year(icao: str, year: int) -> pd.DataFrame | None:
    url = IEM_URL.format(icao=icao, year=year)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # IEM prepends comment lines starting with '#'
        lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
        if len(lines) < 2:
            return None
        df = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)
        df["icao"] = icao
        df["iata"] = icao[1:] if icao.startswith("K") else icao
        return df
    except Exception as e:
        logger.warning(f"IEM download failed {icao} {year}: {e}")
        return None


def download_historical_weather(airports: list[str], years: list[int], max_workers: int = 8) -> pd.DataFrame:
    """
    Download hourly IEM METAR for given IATA airport codes and years.
    Results are cached per airport×year in data/weather/.
    """
    tasks = [(iata_to_icao(a), y) for a in airports for y in years]
    all_frames = []

    def fetch(icao, year):
        cache = os.path.join(WEATHER_DIR, f"iem_{icao}_{year}.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)
        df = _download_iem_airport_year(icao, year)
        if df is not None and not df.empty:
            df.to_parquet(cache, index=False)
        return df

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch, icao, year): (icao, year) for icao, year in tasks}
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    return aggregate_daily_weather(raw)


def aggregate_daily_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse hourly METAR observations to daily airport-level weather features.
    """
    if df.empty:
        return df

    # Parse timestamp
    time_col = next((c for c in df.columns if "valid" in c.lower() or "time" in c.lower()), None)
    if time_col:
        df["date"] = pd.to_datetime(df[time_col], errors="coerce").dt.date
    else:
        return pd.DataFrame()

    df = df.copy()

    # Coerce numeric columns
    for col in ["vsby", "sknt", "gust", "skyl1", "skyl2", "tmpf", "dwpf"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Weather code flags
    wx = df.get("wxcodes", pd.Series([""] * len(df), dtype=str)).fillna("")
    df["_ts"]   = wx.str.contains(r"\bTS\b",          regex=True)
    df["_fog"]  = wx.str.contains(r"\b(FG|BR)\b",     regex=True) & (df.get("vsby", pd.Series(10.0)) < 3)
    df["_snow"] = wx.str.contains(r"\b(SN|FZRA|PL)\b",regex=True)

    # Ceiling: lowest BKN or OVC layer
    def ceiling_row(row):
        for sky_col, hgt_col in [("skyc1","skyl1"),("skyc2","skyl2")]:
            if sky_col in row and hgt_col in row:
                if str(row[sky_col]) in ("BKN","OVC") and pd.notna(row[hgt_col]):
                    return row[hgt_col] * 100  # IEM reports in hundreds of feet
        return np.nan

    if "skyc1" in df.columns:
        df["ceiling_ft"] = df.apply(ceiling_row, axis=1)
        df["_low_ceiling"] = df["ceiling_ft"] < 1000
    else:
        df["ceiling_ft"] = np.nan
        df["_low_ceiling"] = False

    daily = (
        df.groupby(["iata", "date"])
        .agg(
            has_thunderstorm  = ("_ts",          "any"),
            has_fog           = ("_fog",          "any"),
            has_snow_ice      = ("_snow",         "any"),
            has_low_ceiling   = ("_low_ceiling",  "any"),
            min_visibility_mi = ("vsby",          "min"),
            max_wind_kt       = ("sknt",          "max"),
            max_gust_kt       = ("gust",          "max"),
            ceiling_ft        = ("ceiling_ft",    "min"),
            avg_temp_f        = ("tmpf",          "mean"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["weather_severity"] = _compute_severity(daily)
    return daily


def _compute_severity(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    s += df.get("has_thunderstorm", False).astype(float) * 0.40
    s += df.get("has_snow_ice",     False).astype(float) * 0.30
    s += df.get("has_low_ceiling",  False).astype(float) * 0.20
    s += df.get("has_fog",          False).astype(float) * 0.15
    vis = df.get("min_visibility_mi", pd.Series(10.0, index=df.index))
    s += (vis < 1).astype(float) * 0.25
    s += ((vis >= 1) & (vis < 3)).astype(float) * 0.10
    wind = df.get("max_wind_kt", pd.Series(0.0, index=df.index)).fillna(0)
    s += (wind > 30).astype(float) * 0.20
    s += ((wind > 20) & (wind <= 30)).astype(float) * 0.08
    return s.clip(0, 1)


# ---------------------------------------------------------------------------
# AWC Real-time METAR
# ---------------------------------------------------------------------------

AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"


def fetch_live_metar(iata_codes: list[str], hours: int = 2) -> dict[str, dict]:
    """
    Fetch recent METARs from AWC for a list of IATA airport codes.
    Returns {iata: weather_features_dict}.
    """
    icao_codes = [iata_to_icao(a) for a in iata_codes]
    ids_param  = ",".join(icao_codes)

    try:
        resp = requests.get(
            AWC_METAR_URL,
            params={"ids": ids_param, "format": "json", "hours": hours},
            timeout=15,
            headers={"User-Agent": "AA-CrewSeq-RiskModel/1.0 research@example.com"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"AWC METAR fetch failed: {e}")
        return {a: _empty_weather() for a in iata_codes}

    # Group by station, take most recent observation
    by_station: dict[str, list] = {}
    for obs in data:
        sid = obs.get("icaoId", "")
        by_station.setdefault(sid, []).append(obs)

    results = {}
    for iata, icao in zip(iata_codes, icao_codes):
        obs_list = by_station.get(icao, [])
        if not obs_list:
            results[iata] = _empty_weather()
            continue
        # Most recent first
        obs = sorted(obs_list, key=lambda x: x.get("reportTime",""), reverse=True)[0]
        results[iata] = _parse_awc_obs(obs)

    return results


def _parse_awc_obs(obs: dict) -> dict:
    wx_string = obs.get("wxString", "") or ""
    sky       = obs.get("sky", []) or []

    # Ceiling: lowest BKN or OVC layer
    ceiling = np.nan
    for layer in sky:
        cover  = layer.get("cover", "")
        height = layer.get("base", None)
        if cover in ("BKN", "OVC") and height is not None:
            ceiling = float(height) * 100  # hundreds of feet → feet
            break

    vis  = obs.get("visib", None)
    wind = obs.get("wspd",  None)
    gust = obs.get("wgst",  None)

    feat = {
        "has_thunderstorm":  "TS"   in wx_string,
        "has_fog":           any(x in wx_string for x in ("FG","BR")),
        "has_snow_ice":      any(x in wx_string for x in ("SN","FZRA","PL")),
        "has_low_ceiling":   (ceiling < 1000) if not np.isnan(ceiling) else False,
        "min_visibility_mi": float(str(vis).replace("+","")) if vis is not None else 10.0,
        "max_wind_kt":       float(wind) if wind is not None else 0.0,
        "max_gust_kt":       float(gust) if gust is not None else 0.0,
        "ceiling_ft":        ceiling,
        "raw":               obs.get("rawOb", ""),
        "report_time":       obs.get("reportTime", ""),
    }
    feat["weather_severity"] = float(_compute_severity(pd.DataFrame([feat]))[0])
    return feat


def _empty_weather() -> dict:
    return {
        "has_thunderstorm": False, "has_fog": False, "has_snow_ice": False,
        "has_low_ceiling": False, "min_visibility_mi": 10.0,
        "max_wind_kt": 0.0, "max_gust_kt": 0.0, "ceiling_ft": np.nan,
        "weather_severity": 0.0, "raw": "", "report_time": "",
    }
