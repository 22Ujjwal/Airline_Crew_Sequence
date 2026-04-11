"""
Out-of-Distribution (OOD) Validation for the crew sequence risk model.

Three OOD tests:
  1. Temporal OOD  — 2017 data (before training window, never seen)
  2. Extreme event — February 2021 (Texas winter storm Uri, massive disruption)
  3. Carrier OOD   — United Airlines (UA) at IAH hub, same feature logic
                     Tests whether airport-level risk is carrier-agnostic

All three use the model trained only on AA/DFW 2018–2023.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OOD_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "ood")
PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OOD_DIR, exist_ok=True)

BTS_URL = (
    "https://www.transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

KEEP_COLS = [
    "FlightDate", "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
    "Cancelled", "CancellationCode", "Diverted",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "DayOfWeek", "Month", "Year",
]

WEATHER_DELAY_THRESHOLD_MIN = 15
TURNAROUND_MAX_HRS = 4
TURNAROUND_MIN_HRS = 0.5

FEATURE_COLS = [
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    "Month", "is_spring_summer", "turnaround_min",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_months(year: int, months: list[int], airline: str, hub: str) -> pd.DataFrame:
    frames = []
    for month in months:
        url = BTS_URL.format(year=year, month=month)
        print(f"  Downloading {year}-{month:02d}...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, encoding="latin1", low_memory=False)
            available = [c for c in KEEP_COLS if c in df.columns]
            df = df[available]
            mask = (df["Reporting_Airline"] == airline) & (
                (df["Origin"] == hub) | (df["Dest"] == hub)
            )
            chunk = df[mask].copy()
            print(f"{len(chunk):,} rows")
            frames.append(chunk)
        except Exception as e:
            print(f"WARN: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Month"] = df["FlightDate"].dt.month
    df["Year"] = df["FlightDate"].dt.year
    df["Season"] = df["Month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    })
    return df


def _parse_hhmm(series: pd.Series) -> pd.Series:
    s = series.fillna(0).astype(int).astype(str).str.zfill(4)
    return s.str[:2].astype(int) * 60 + s.str[2:].astype(int)


def build_sequences(df: pd.DataFrame, hub: str) -> pd.DataFrame:
    inbound  = df[(df["Dest"] == hub)   & (df["Cancelled"] != 1)].copy()
    outbound = df[(df["Origin"] == hub) & (df["Cancelled"] != 1)].copy()
    inbound["arr_min"]  = _parse_hhmm(inbound["ArrTime"])
    outbound["dep_min"] = _parse_hhmm(outbound["DepTime"])

    ib = inbound[["FlightDate","Origin","arr_min","WeatherDelay","ArrDelay",
                   "ArrDelayMinutes","LateAircraftDelay","NASDelay","Month","Season","Year"]]\
         .rename(columns={"Origin":"airport_A","WeatherDelay":"weather_delay_A",
                           "ArrDelay":"arr_delay_A","ArrDelayMinutes":"arr_delay_min_A",
                           "LateAircraftDelay":"late_aircraft_A","NASDelay":"nas_delay_A"})
    ob = outbound[["FlightDate","Dest","dep_min","WeatherDelay","DepDelay",
                    "DepDelayMinutes","LateAircraftDelay","NASDelay","Distance"]]\
         .rename(columns={"Dest":"airport_B","WeatherDelay":"weather_delay_B",
                           "DepDelay":"dep_delay_B","DepDelayMinutes":"dep_delay_min_B",
                           "LateAircraftDelay":"late_aircraft_B","NASDelay":"nas_delay_B"})

    pairs = ib.merge(ob, on="FlightDate", how="inner")
    turnaround = pairs["dep_min"] - pairs["arr_min"]
    mask = (
        (turnaround >= TURNAROUND_MIN_HRS * 60) &
        (turnaround <= TURNAROUND_MAX_HRS * 60) &
        (pairs["airport_A"] != pairs["airport_B"])
    )
    pairs = pairs[mask].copy()
    pairs["turnaround_min"] = turnaround[mask]
    return pairs


def attach_airport_features(pairs: pd.DataFrame, airport_features: pd.DataFrame) -> pd.DataFrame:
    feat_a = airport_features.rename(columns=lambda c: f"A_{c}" if c not in ("airport","Month") else c)\
                             .rename(columns={"airport": "airport_A"})
    feat_b = airport_features.rename(columns=lambda c: f"B_{c}" if c not in ("airport","Month") else c)\
                             .rename(columns={"airport": "airport_B"})
    pairs = pairs.merge(feat_a, on=["airport_A","Month"], how="left")
    pairs = pairs.merge(feat_b, on=["airport_B","Month"], how="left")

    pairs["pair_combined_weather_rate"]  = pairs["A_weather_delay_rate"]  * pairs["B_weather_delay_rate"]
    pairs["pair_max_weather_rate"]       = pairs[["A_weather_delay_rate","B_weather_delay_rate"]].max(axis=1)
    pairs["pair_min_weather_rate"]       = pairs[["A_weather_delay_rate","B_weather_delay_rate"]].min(axis=1)
    pairs["pair_weather_rate_sum"]       = pairs["A_weather_delay_rate"]  + pairs["B_weather_delay_rate"]
    pairs["pair_avg_weather_delay_min"]  = (pairs["A_avg_weather_delay_min"] + pairs["B_avg_weather_delay_min"]) / 2
    pairs["both_high_risk"] = (
        (pairs["A_weather_delay_rate"] > pairs["A_weather_delay_rate"].quantile(0.75)) &
        (pairs["B_weather_delay_rate"] > pairs["B_weather_delay_rate"].quantile(0.75))
    ).astype(int)
    pairs = pd.get_dummies(pairs, columns=["Season"], prefix="season", drop_first=False)
    pairs["is_spring_summer"] = pairs.get("season_spring", 0) | pairs.get("season_summer", 0)
    return pairs


def label_sequences(pairs: pd.DataFrame) -> pd.DataFrame:
    weather_A = pairs["weather_delay_A"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    weather_B = pairs["weather_delay_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    cascade   = (
        (pairs["arr_delay_A"].fillna(0)    >= WEATHER_DELAY_THRESHOLD_MIN) &
        (pairs["late_aircraft_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN)
    )
    pairs["target"] = (weather_A | weather_B | cascade).astype(int)
    return pairs


def evaluate(name: str, model: xgb.XGBClassifier, X: pd.DataFrame,
             y: pd.Series, feature_cols: list[str]) -> dict:
    X_clean = X[feature_cols].astype(float)
    mask = X_clean.notna().all(axis=1)
    X_clean, y_clean = X_clean[mask], y[mask]

    if len(y_clean) == 0 or y_clean.nunique() < 2:
        print(f"[{name}] Not enough data or no positive labels — skipping.")
        return {}

    proba = model.predict_proba(X_clean)[:, 1]
    pred  = model.predict(X_clean)

    auc  = roc_auc_score(y_clean, proba)
    ap   = average_precision_score(y_clean, proba)
    rate = y_clean.mean()

    print(f"\n{'='*55}")
    print(f"OOD Test: {name}")
    print(f"  Sequences:      {len(y_clean):,}")
    print(f"  Positive rate:  {rate:.1%}")
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"  Avg Precision:  {ap:.4f}")
    print(classification_report(y_clean, pred, target_names=["low_risk","high_risk"], zero_division=0))

    return {"name": name, "n": len(y_clean), "positive_rate": rate, "roc_auc": auc, "avg_precision": ap}


# ---------------------------------------------------------------------------
# OOD Test 1: Temporal — AA/DFW 2017
# ---------------------------------------------------------------------------

def test_temporal_ood(model, airport_features, feature_cols):
    print("\n[OOD 1] Temporal — AA/DFW 2017 (before training window)")
    path = os.path.join(OOD_DIR, "ood_aa_dfw_2017.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = download_months(2017, list(range(1, 13)), airline="AA", hub="DFW")
        df.to_parquet(path, index=False)

    df = add_season(df)
    pairs = build_sequences(df, hub="DFW")
    pairs = attach_airport_features(pairs, airport_features)
    pairs = label_sequences(pairs)
    return evaluate("Temporal OOD — AA/DFW 2017", model, pairs, pairs["target"], feature_cols)


# ---------------------------------------------------------------------------
# OOD Test 2: Extreme event — Feb 2021 Texas Winter Storm Uri
# ---------------------------------------------------------------------------

def test_extreme_event_ood(model, airport_features, feature_cols):
    print("\n[OOD 2] Extreme event — Feb 2021 Texas Winter Storm Uri")
    path = os.path.join(OOD_DIR, "ood_aa_dfw_2021_02.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = download_months(2021, [2], airline="AA", hub="DFW")
        df.to_parquet(path, index=False)

    df = add_season(df)
    pairs = build_sequences(df, hub="DFW")
    pairs = attach_airport_features(pairs, airport_features)
    pairs = label_sequences(pairs)
    result = evaluate("Extreme Event OOD — Feb 2021 (Storm Uri)", model, pairs, pairs["target"], feature_cols)

    # Extra: show which pairs were worst during the storm
    if len(pairs) > 0:
        X = pairs[feature_cols].astype(float)
        mask = X.notna().all(axis=1)
        pairs_clean = pairs[mask].copy()
        pairs_clean["risk_score"] = model.predict_proba(X[mask])[:, 1]
        worst = (
            pairs_clean.groupby(["airport_A","airport_B"])
            .agg(avg_risk=("risk_score","mean"), n=("risk_score","count"),
                 actual_bad_rate=("target","mean"))
            .sort_values("avg_risk", ascending=False)
            .head(10)
        )
        print("\nTop 10 riskiest pairs during Storm Uri:")
        print(worst.to_string())
    return result


# ---------------------------------------------------------------------------
# OOD Test 3: Carrier OOD — United Airlines at IAH
# ---------------------------------------------------------------------------

def test_carrier_ood(model, airport_features, feature_cols):
    """
    Train features were built on AA/DFW. Here we apply the same airport-level
    feature lookup (from the AA/DFW training data) to UA flights at IAH.
    If airport risk is truly carrier-agnostic, we should still see signal.
    """
    print("\n[OOD 3] Carrier OOD — United Airlines / IAH hub (2022–2023)")
    path = os.path.join(OOD_DIR, "ood_ua_iah_2022_2023.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        frames = []
        for year in [2022, 2023]:
            chunk = download_months(year, list(range(1, 13)), airline="UA", hub="IAH")
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)

    df = add_season(df)
    pairs = build_sequences(df, hub="IAH")
    pairs = attach_airport_features(pairs, airport_features)
    pairs = label_sequences(pairs)
    return evaluate("Carrier OOD — UA/IAH 2022–2023", model, pairs, pairs["target"], feature_cols)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_ood_summary(results: list[dict], in_dist_metrics: dict):
    valid = [r for r in results if r]
    if not valid:
        return

    names    = [in_dist_metrics["name"]] + [r["name"] for r in valid]
    auc_vals = [in_dist_metrics["roc_auc"]] + [r["roc_auc"] for r in valid]
    ap_vals  = [in_dist_metrics["avg_precision"]] + [r["avg_precision"] for r in valid]
    colors   = ["steelblue"] + ["coral"] * len(valid)

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, auc_vals, width, label="ROC-AUC",        color=colors, alpha=0.85)
    ax.bar(x + width/2, ap_vals,  width, label="Avg Precision",   color=colors, alpha=0.5, hatch="//")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" — ", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("In-Distribution vs OOD Model Performance")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PROC_DIR, "ood_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"\nOOD comparison plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load trained model
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(PROC_DIR, "xgb_model.json"))

    # Load airport features (built from training data)
    airport_features = pd.read_parquet(os.path.join(PROC_DIR, "airport_features.parquet"))

    # Load in-distribution val metrics for comparison
    train_data = pd.read_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"))
    train_data["Year"] = pd.to_datetime(train_data["FlightDate"]).dt.year
    val_data = train_data[train_data["Year"] == train_data["Year"].max()]
    season_cols = [c for c in val_data.columns if c.startswith("season_")]
    feature_cols = FEATURE_COLS + season_cols
    feature_cols = [c for c in feature_cols if c in val_data.columns]

    in_dist = evaluate("In-Distribution (2024 val)", model, val_data, val_data["target"], feature_cols)

    # Run OOD tests
    results = [
        test_temporal_ood(model, airport_features, feature_cols),
        test_extreme_event_ood(model, airport_features, feature_cols),
        test_carrier_ood(model, airport_features, feature_cols),
    ]

    plot_ood_summary(results, in_dist)

    # Save summary table
    summary = pd.DataFrame([in_dist] + [r for r in results if r])
    summary.to_csv(os.path.join(PROC_DIR, "ood_summary.csv"), index=False)
    print(f"\nSummary saved → {os.path.join(PROC_DIR, 'ood_summary.csv')}")


if __name__ == "__main__":
    main()
