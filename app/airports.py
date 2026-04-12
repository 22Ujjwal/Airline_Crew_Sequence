"""
Airport metadata: IATA → name, city, state, lat, lon.
Uses the `airportsdata` package; falls back to a hardcoded dict for any misses.
"""
from __future__ import annotations
import airportsdata

_RAW = airportsdata.load("IATA")

# Supplement / override for a handful of airports airportsdata might miss
_OVERRIDES: dict[str, dict] = {}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def get(iata: str) -> dict:
    """Return airport info dict with keys: name, city, subd (state), lat, lon."""
    code = iata.upper().strip()
    if code in _OVERRIDES:
        return _OVERRIDES[code]
    return _RAW.get(code, {"name": code, "city": code, "subd": "", "lat": None, "lon": None})


def label(iata: str) -> str:
    """Short display label: 'DFW — Dallas/Fort Worth'."""
    info = get(iata)
    city = info.get("city") or info.get("name") or iata
    return f"{iata} — {city}"


def coords(iata: str) -> tuple[float | None, float | None]:
    """Return (lat, lon) or (None, None)."""
    info = get(iata)
    return info.get("lat"), info.get("lon")


def build_airport_df(iata_codes: list[str]) -> "pd.DataFrame":
    """DataFrame of airport metadata for a list of IATA codes."""
    import pandas as pd
    rows = []
    for code in iata_codes:
        info = get(code)
        lat, lon = info.get("lat"), info.get("lon")
        rows.append({
            "iata": code,
            "name": info.get("name", code),
            "city": info.get("city", code),
            "state": info.get("subd", ""),
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(rows)
