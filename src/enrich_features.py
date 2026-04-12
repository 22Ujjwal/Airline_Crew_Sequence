"""
Enrich airport features with real historical weather from NOAA CDO GSOM
(Global Summary of the Month). Replaces IEM ASOS approach.

New airport×month features from GSOM:
  thunder_days          — WT03: avg days/month with thunderstorms
  fog_days              — WT01: avg days/month with fog/freezing fog
  high_wind_days        — WT11: avg days/month with high/damaging winds
  snow_days             — WT18: avg days/month with snow/ice
  precip_days           — DP01: avg days/month with >= 0.01in precip
  avg_wind_speed        — AWND: average wind speed (m/s)
  max_wind_gust         — WSF2: fastest 2-min wind speed (m/s)
  total_precip          — PRCP: total monthly precipitation (mm)
  extreme_precip        — EMXP: extreme max daily precip (mm)
  weather_disruption_idx — sum of thunder+fog+wind+snow days

Pair-level derived (A×B):
  pair_max_thunder_days, pair_sum_thunder_days,
  pair_max_fog_days, pair_sum_fog_days, etc.

Usage:
  python src/enrich_features.py --token SBRhkMknyZHEHobLChLzPRVSgtXUlmwL
"""

import os
import time
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

DATATYPES = ["WT03", "WT01", "WT11", "WT18", "DP01", "AWND", "WSF2", "PRCP", "EMXP"]

RENAME = {
    "WT03": "thunder_days",
    "WT01": "fog_days",
    "WT11": "high_wind_days",
    "WT18": "snow_days",
    "DP01": "precip_days",
    "AWND": "avg_wind_speed",
    "WSF2": "max_wind_gust",
    "PRCP": "total_precip",
    "EMXP": "extreme_precip",
}

# IATA → NOAA GHCND station ID (major US commercial airports)
AIRPORT_STATIONS = {
    "ABQ": "GHCND:USW00023050", "ATL": "GHCND:USW00013874", "AUS": "GHCND:USW00013958",
    "BDL": "GHCND:USW00014740", "BNA": "GHCND:USW00013897", "BOS": "GHCND:USW00014739",
    "BUF": "GHCND:USW00014733", "BUR": "GHCND:USW00023152", "BWI": "GHCND:USW00093721",
    "CLE": "GHCND:USW00014820", "CLT": "GHCND:USW00013881", "CMH": "GHCND:USW00014821",
    "CVG": "GHCND:USW00093814", "DAL": "GHCND:USW00013960", "DAY": "GHCND:USW00093815",
    "DCA": "GHCND:USW00013743", "DEN": "GHCND:USW00003017", "DFW": "GHCND:USW00003927",
    "DSM": "GHCND:USW00094910", "DTW": "GHCND:USW00094847", "ELP": "GHCND:USW00023044",
    "EWR": "GHCND:USW00014734", "FLL": "GHCND:USW00012843", "GRR": "GHCND:USW00094860",
    "GSO": "GHCND:USW00013723", "HNL": "GHCND:USW00022521", "HOU": "GHCND:USW00012918",
    "IAD": "GHCND:USW00093738", "IAH": "GHCND:USW00012960", "ICT": "GHCND:USW00003928",
    "IND": "GHCND:USW00093819", "JAX": "GHCND:USW00013889", "JFK": "GHCND:USW00094789",
    "LAS": "GHCND:USW00023169", "LAX": "GHCND:USW00023174", "LBB": "GHCND:USW00023042",
    "LGA": "GHCND:USW00014732", "LIT": "GHCND:USW00013963", "MCI": "GHCND:USW00003947",
    "MCO": "GHCND:USW00012815", "MDW": "GHCND:USW00094846", "MEM": "GHCND:USW00013893",
    "MIA": "GHCND:USW00012839", "MKE": "GHCND:USW00014839", "MSP": "GHCND:USW00014922",
    "MSY": "GHCND:USW00012916", "OAK": "GHCND:USW00023230", "OGG": "GHCND:USW00022508",
    "OKC": "GHCND:USW00013967", "OMA": "GHCND:USW00014942", "ONT": "GHCND:USW00023161",
    "ORD": "GHCND:USW00094846", "ORF": "GHCND:USW00013737", "PBI": "GHCND:USW00012844",
    "PDX": "GHCND:USW00024229", "PHL": "GHCND:USW00013739", "PHX": "GHCND:USW00023183",
    "PIT": "GHCND:USW00094823", "PVD": "GHCND:USW00014765", "RDU": "GHCND:USW00013722",
    "RIC": "GHCND:USW00013740", "RNO": "GHCND:USW00023185", "RSW": "GHCND:USW00012894",
    "SAN": "GHCND:USW00023188", "SAT": "GHCND:USW00012921", "SAV": "GHCND:USW00003822",
    "SDF": "GHCND:USW00093821", "SEA": "GHCND:USW00024233", "SFO": "GHCND:USW00023234",
    "SJC": "GHCND:USW00023293", "SJU": "GHCND:USW00011641", "SLC": "GHCND:USW00024127",
    "SMF": "GHCND:USW00023232", "SNA": "GHCND:USW00093184", "STL": "GHCND:USW00013994",
    "SYR": "GHCND:USW00014771", "TPA": "GHCND:USW00012842", "TUL": "GHCND:USW00013968",
    "TUS": "GHCND:USW00023160", "XNA": "GHCND:USW00053951",
}


# ---------------------------------------------------------------------------
# NOAA CDO fetch
# ---------------------------------------------------------------------------

def fetch_gsom_batch(station_ids: list[str], token: str,
                     start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch GSOM data for one batch of stations (max 10). Auto-paginates."""
    session = requests.Session()
    session.headers.update({"token": token})
    records = []
    offset = 1

    while True:
        params = {
            "datasetid":  "GSOM",
            "datatypeid": ",".join(DATATYPES),
            "stationid":  ",".join(station_ids),
            "startdate":  start,
            "enddate":    end,
            "limit":      1000,
            "offset":     offset,
            "units":      "metric",
        }
        for attempt in range(4):
            try:
                resp = session.get(f"{CDO_BASE}/data", params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt); continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3: return pd.DataFrame(records)
                time.sleep(1 + attempt)

        results = resp.json().get("results", [])
        records.extend(results)
        if len(results) < 1000:
            break
        offset += 1000
        time.sleep(0.2)

    return pd.DataFrame(records)


def fetch_all_gsom(station_ids: list[str], token: str) -> pd.DataFrame:
    frames = []
    batches = [station_ids[i:i+10] for i in range(0, len(station_ids), 10)]
    for batch in tqdm(batches, desc="Fetching GSOM"):
        df = fetch_gsom_batch(batch, token)
        if not df.empty:
            frames.append(df)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["date"]  = pd.to_datetime(raw["date"])
    raw["month"] = raw["date"].dt.month
    raw["year"]  = raw["date"].dt.year
    raw["station"] = raw["station"].str.replace("GHCND:", "", regex=False)
    return raw


# ---------------------------------------------------------------------------
# Build weather features
# ---------------------------------------------------------------------------

def build_weather_features(raw: pd.DataFrame, station_to_iata: dict) -> pd.DataFrame:
    """
    Pivot GSOM records → airport×month features averaged across all years.
    WT-type fields = count of days that condition occurred (avg across years).
    """
    raw = raw.copy()
    raw["iata"]  = raw["station"].map(station_to_iata)
    raw = raw.dropna(subset=["iata"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    # Average across years for each airport×month×datatype
    monthly = (
        raw.groupby(["iata", "month", "year", "datatype"])["value"].mean()
        .groupby(level=["iata", "month", "datatype"]).mean()
        .unstack("datatype")
        .reset_index()
    )
    monthly.columns.name = None
    monthly = monthly.rename(columns={k: v for k, v in RENAME.items() if k in monthly.columns})
    monthly = monthly.rename(columns={"iata": "airport", "month": "Month"})

    # Derived: combined disruption index
    disrupt_cols = [v for k, v in RENAME.items()
                    if k in ("WT03","WT01","WT11","WT18") and v in monthly.columns]
    if disrupt_cols:
        monthly["weather_disruption_idx"] = monthly[disrupt_cols].sum(axis=1)

    print(f"Weather features: {monthly.shape}  |  airports: {monthly['airport'].nunique()}")
    return monthly


# ---------------------------------------------------------------------------
# Propagate into sequence_features
# ---------------------------------------------------------------------------

def enrich_sequences(sf: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather_cols = [c for c in weather.columns if c not in ("airport", "Month")]

    # Airport A features
    wx_a = weather.rename(
        columns={c: f"A_{c}" for c in weather_cols} | {"airport": "airport_A"}
    )
    # Airport B features
    wx_b = weather.rename(
        columns={c: f"B_{c}" for c in weather_cols} | {"airport": "airport_B"}
    )

    sf = sf.merge(wx_a, on=["airport_A", "Month"], how="left")
    sf = sf.merge(wx_b, on=["airport_B", "Month"], how="left")

    # Pair-level: max and sum for each weather feature
    for col in weather_cols:
        a_col, b_col = f"A_{col}", f"B_{col}"
        if a_col in sf.columns and b_col in sf.columns:
            sf[f"pair_max_{col}"] = sf[[a_col, b_col]].max(axis=1)
            sf[f"pair_sum_{col}"] = sf[a_col] + sf[b_col]

    new_cols = [c for c in sf.columns if c.startswith(("A_thunder","A_fog","A_high","A_snow",
                                                         "B_thunder","B_fog","B_high","B_snow",
                                                         "pair_max_","pair_sum_",
                                                         "A_avg_wind","A_max_wind","A_precip",
                                                         "B_avg_wind","B_max_wind","B_precip",
                                                         "A_weather_dis","B_weather_dis"))]
    print(f"Added {len(new_cols)} new feature columns to sequence_features")
    return sf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(token: str):
    # Load airports
    af = pd.read_parquet(os.path.join(PROC_DIR, "airport_features.parquet"))
    airports = af["airport"].unique().tolist()

    known   = {a: s for a, s in AIRPORT_STATIONS.items() if a in airports}
    missing = [a for a in airports if a not in known]
    print(f"Airports in dataset: {len(airports)}")
    print(f"Station mapping found: {len(known)}  |  missing: {len(missing)}")
    if missing:
        print(f"  No station for: {sorted(missing)}")

    station_ids    = list(known.values())
    station_to_iata = {v.replace("GHCND:", ""): k for k, v in known.items()}

    # Fetch or load cached
    cache = os.path.join(PROC_DIR, "monthly_weather_gsom.parquet")
    if os.path.exists(cache):
        print(f"\nLoading cached GSOM data from {cache}")
        raw = pd.read_parquet(cache)
    else:
        print(f"\nFetching GSOM for {len(station_ids)} stations (2015–2024)...")
        raw = fetch_all_gsom(station_ids, token)
        if raw.empty:
            print("ERROR: No data returned — check token."); return
        raw.to_parquet(cache, index=False)
        print(f"Cached → {cache}")

    print(f"Raw GSOM records: {len(raw):,}")

    weather = build_weather_features(raw, station_to_iata)
    weather_cols = [c for c in weather.columns if c not in ("airport", "Month")]

    # Save enriched airport features
    af_enriched = af.merge(weather, on=["airport", "Month"], how="left")
    for col in weather_cols:
        af_enriched[col] = af_enriched[col].fillna(af_enriched[col].median())
    af_enriched.to_parquet(os.path.join(PROC_DIR, "airport_features.parquet"), index=False)
    print(f"Enriched airport_features: {af_enriched.shape}")

    # Save enriched sequence features
    sf = pd.read_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"))
    # Drop any existing weather columns to avoid duplicates on re-run
    drop = [c for c in sf.columns if any(c.startswith(p) for p in
            ("A_thunder","A_fog","A_high_wind","A_snow","A_precip","A_avg_wind",
             "A_max_wind","A_total","A_extreme","A_weather_dis",
             "B_thunder","B_fog","B_high_wind","B_snow","B_precip","B_avg_wind",
             "B_max_wind","B_total","B_extreme","B_weather_dis",
             "pair_max_","pair_sum_"))]
    sf = sf.drop(columns=drop, errors="ignore")
    sf = enrich_sequences(sf, weather)
    sf.to_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"), index=False)
    print(f"Enriched sequence_features: {sf.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("NCDC_TOKEN", ""))
    args = parser.parse_args()
    if not args.token:
        raise ValueError("Provide --token or set NCDC_TOKEN env var")
    main(args.token)
