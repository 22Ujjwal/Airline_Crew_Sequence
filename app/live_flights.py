"""
Live/current AA flight schedule at DFW.

Sources (in priority order):
  1. AviationStack API  — if api_key provided; free tier: 100 req/month
  2. BTS 2024 analog    — same month + same day-of-week from 2024 data
  3. OpenSky Network    — fallback, sparse US coverage

All sources return (arrivals_raw, departures_raw, status_msg, source_name).
Callers pass raw data to optimizer.bts_to_arrivals/departures or
optimizer.aviationstack_to_arrivals/departures for formatting.
"""
from __future__ import annotations
import os
import time
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import airportsdata

# ── Airport lookup ────────────────────────────────────────────────────────────
_iata_db    = airportsdata.load("IATA")
_icao2iata: dict[str, str] = {
    info["icao"]: code for code, info in _iata_db.items() if info.get("icao")
}

AVIATIONSTACK_BASE = "http://api.aviationstack.com/v1/flights"
OPENSKY_BASE       = "https://opensky-network.org/api/flights"
TIMEOUT_S          = 12


def icao_to_iata(icao: str | None) -> str | None:
    return _icao2iata.get((icao or "").upper())


# ── AviationStack ─────────────────────────────────────────────────────────────

def fetch_aviationstack(api_key: str) -> tuple[list, list, str]:
    """
    Fetch AA arrivals + departures at DFW via AviationStack.
    2 API calls (counts against 100/month free quota).
    Returns (arr_raw, dep_raw, status).
    """
    error_msg = None

    def _call(airport_param: str) -> list:
        nonlocal error_msg
        try:
            r = requests.get(AVIATIONSTACK_BASE, params={
                "access_key": api_key,
                airport_param: "DFW",
                "limit": 100,
            }, timeout=TIMEOUT_S)
            data = r.json()
            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                return []
            flights = data.get("data", [])
            # Filter AA client-side (more reliable than server-side airline_iata on free tier)
            return [f for f in flights
                    if (f.get("airline") or {}).get("iata") == "AA"]
        except Exception as e:
            error_msg = str(e)
            return []

    arr = _call("arr_iata")
    dep = _call("dep_iata")
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    if error_msg:
        status = f"AviationStack error: {error_msg} — {now}"
    else:
        status = f"AviationStack — {len(arr)} AA arrivals, {len(dep)} AA departures at DFW — {now}"
    return arr, dep, status


# ── BTS 2024 Analog ───────────────────────────────────────────────────────────

def get_bts_analog(bts_df: pd.DataFrame, reference_date: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """
    Return BTS 2024 flights for the same month + day-of-week as reference_date (default: today).
    This provides a realistic AA schedule proxy when live APIs aren't available.
    """
    if reference_date is None:
        reference_date = datetime.now()

    target_month = reference_date.month
    target_dow   = reference_date.weekday()   # 0=Mon, 6=Sun
    dow_names    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    bts_df = bts_df.copy()
    bts_df["_date"] = pd.to_datetime(bts_df["FlightDate"])

    monthly = bts_df[bts_df["_date"].dt.month == target_month]
    if monthly.empty:
        # Fallback: any available month
        monthly = bts_df

    monthly = monthly.copy()
    monthly["_dow"] = monthly["_date"].dt.dayofweek
    same_dow = monthly[monthly["_dow"] == target_dow]

    if same_dow.empty:
        same_dow = monthly   # any day in same month

    # Pick most recent matching date
    best_date = same_dow["FlightDate"].max()
    day_df    = same_dow[same_dow["FlightDate"] == best_date].copy()

    actual_dow  = pd.to_datetime(best_date).strftime("%A")
    import calendar
    month_name  = calendar.month_name[target_month]
    status = (
        f"Schedule proxy: BTS 2024 data for {actual_dow} {best_date} "
        f"(analog for {dow_names[target_dow]}s in {month_name}) — "
        f"{len(day_df):,} AA flights"
    )
    return day_df, status


# ── OpenSky (sparse US fallback) ──────────────────────────────────────────────

def fetch_opensky(hours_back: int = 12) -> tuple[list, list, str]:
    """OpenSky Network — free but sparse US coverage (~20-40% of AA flights)."""
    end   = int(time.time())
    begin = end - hours_back * 3600
    params = {"airport": "KDFW", "begin": begin, "end": end}

    def _get(ep: str) -> list:
        try:
            r = requests.get(f"{OPENSKY_BASE}/{ep}", params=params, timeout=TIMEOUT_S)
            return r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
        except Exception:
            return []

    raw_arr = [f for f in _get("arrival")   if (f.get("callsign") or "").startswith("AAL")]
    raw_dep = [f for f in _get("departure") if (f.get("callsign") or "").startswith("AAL")]
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    status = (
        f"OpenSky (limited US coverage) — {len(raw_arr)} AA arrivals, "
        f"{len(raw_dep)} departures in last {hours_back}h — {now}"
    )
    return raw_arr, raw_dep, status


def score_sequences(seqs: pd.DataFrame, scores_flat: pd.DataFrame) -> pd.DataFrame:
    """
    Add risk_score + risk_label via vectorized merge on (airport_A, airport_B, Month).
    scores_flat must be the raw pair_risk_scores DataFrame (NOT indexed).
    """
    if seqs.empty:
        return seqs
    seqs = seqs.copy()
    seqs["Month"] = seqs["Month"].astype(int)
    lookup = scores_flat[["airport_A", "airport_B", "Month", "avg_risk_score"]].copy()
    seqs = seqs.merge(lookup, on=["airport_A", "airport_B", "Month"], how="left")
    seqs.rename(columns={"avg_risk_score": "risk_score"}, inplace=True)
    seqs["risk_label"] = seqs["risk_score"].apply(
        lambda s: ("HIGH" if s >= 0.70 else "MODERATE" if s >= 0.40 else "LOW")
        if pd.notna(s) else "N/A"
    )
    return seqs.sort_values("risk_score", ascending=False, na_position="last")


def opensky_to_standard(raw: list, ts_key: str, airport_key: str) -> pd.DataFrame:
    """Convert OpenSky records to standard arrivals/departures DataFrame."""
    rows = []
    for r in raw:
        iata = icao_to_iata(r.get(airport_key))
        if not iata or iata == "DFW":
            continue
        ts = r.get(ts_key)
        if not ts:
            continue
        dt   = datetime.fromtimestamp(ts, tz=timezone.utc)
        tmin = dt.hour * 60 + dt.minute
        cs   = (r.get("callsign") or "").strip()
        rows.append({
            "airport":    iata,
            "time_min":   tmin,
            "time_str":   dt.strftime("%H:%M UTC"),
            "flight":     cs.replace("AAL", "AA"),
            "Tail_Number": r.get("icao24", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
