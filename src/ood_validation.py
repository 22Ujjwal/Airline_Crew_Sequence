"""
Out-of-Distribution (OOD) Validation for the crew sequence risk model.

Three OOD tests:
  1. Temporal OOD  — 2015 data (before training window, never seen)
  2. Extreme event — Jan/Feb 2015 (record NE blizzards, high disruption)
  3. Carrier OOD   — Non-AA carriers (DL, UA, WN) at DFW, 2022–2023
                     Tests whether airport-level risk is carrier-agnostic

KEY: OOD data is aggregated to pair×month level (same as training) before
     evaluation. Flight-level→aggregate mismatch was the prior bug.

All tests use the model trained on AA/DFW 2018–2023 (2015–2017 never seen).
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

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OOD_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "ood")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
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

# Must match training feature cols exactly
FEATURE_COLS = [
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    "Month", "is_spring_summer", "median_turnaround_min",
]

# Global bad-rate threshold from training (from feature_engineering output)
GLOBAL_BAD_RATE_THRESHOLD = 0.167


# ---------------------------------------------------------------------------
# Core pipeline helpers (mirrors feature_engineering.py logic)
# ---------------------------------------------------------------------------

def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Month"] = df["FlightDate"].dt.month
    df["Year"]  = df["FlightDate"].dt.year
    df["Season"] = df["Month"].map({
        12:"winter",1:"winter",2:"winter",
        3:"spring",4:"spring",5:"spring",
        6:"summer",7:"summer",8:"summer",
        9:"fall",10:"fall",11:"fall",
    })
    return df


def _parse_hhmm(series: pd.Series) -> pd.Series:
    s = series.fillna(0).astype(int).astype(str).str.zfill(4)
    return s.str[:2].astype(int) * 60 + s.str[2:].astype(int)


def build_sequences_agg(df: pd.DataFrame, hub: str) -> pd.DataFrame:
    """
    Mirrors updated feature_engineering.py: aggregate inbound/outbound to
    (date, airport) level first, then cross-join → pair×date rows.
    Returns flight-level-labeled pairs ready for aggregation.
    """
    inbound  = df[(df["Dest"]   == hub) & (df["Cancelled"] != 1)].copy()
    outbound = df[(df["Origin"] == hub) & (df["Cancelled"] != 1)].copy()

    inbound["arr_min"]  = _parse_hhmm(inbound["ArrTime"])
    outbound["dep_min"] = _parse_hhmm(outbound["DepTime"])

    ib_agg = (
        inbound.groupby(["FlightDate","Origin","Month","Season","Year"])
        .agg(
            arr_min_earliest = ("arr_min",          "min"),
            arr_min_latest   = ("arr_min",          "max"),
            weather_delay_A  = ("WeatherDelay",     lambda x: x.fillna(0).max()),
            arr_delay_A      = ("ArrDelay",         lambda x: x.fillna(0).max()),
            nas_delay_A      = ("NASDelay",         lambda x: x.fillna(0).max()),
            late_aircraft_A  = ("LateAircraftDelay",lambda x: x.fillna(0).max()),
        )
        .reset_index()
        .rename(columns={"Origin":"airport_A"})
    )

    ob_agg = (
        outbound.groupby(["FlightDate","Dest"])
        .agg(
            dep_min_earliest = ("dep_min",          "min"),
            dep_min_latest   = ("dep_min",          "max"),
            weather_delay_B  = ("WeatherDelay",     lambda x: x.fillna(0).max()),
            dep_delay_B      = ("DepDelay",         lambda x: x.fillna(0).max()),
            late_aircraft_B  = ("LateAircraftDelay",lambda x: x.fillna(0).max()),
        )
        .reset_index()
        .rename(columns={"Dest":"airport_B"})
    )

    pairs = ib_agg.merge(ob_agg, on="FlightDate", how="inner")
    pairs = pairs[pairs["airport_A"] != pairs["airport_B"]].copy()

    turnaround_mid = pairs["dep_min_earliest"] - pairs["arr_min_latest"]
    feasible = (
        (pairs["dep_min_latest"]   >= pairs["arr_min_earliest"] + TURNAROUND_MIN_HRS * 60) &
        (pairs["dep_min_earliest"] <= pairs["arr_min_latest"]   + TURNAROUND_MAX_HRS * 60)
    )
    pairs = pairs[feasible].copy()
    pairs["turnaround_min"] = turnaround_mid[feasible].clip(lower=0)
    return pairs


def label_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    weather_A = pairs["weather_delay_A"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    weather_B = pairs["weather_delay_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN
    cascade   = (
        (pairs["arr_delay_A"].fillna(0)    >= WEATHER_DELAY_THRESHOLD_MIN) &
        (pairs["late_aircraft_B"].fillna(0) >= WEATHER_DELAY_THRESHOLD_MIN)
    )
    pairs["raw_label"] = (weather_A | weather_B | cascade).astype(int)
    return pairs


def aggregate_to_pair_month(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse pair×date rows → pair×month rows, binarize using global threshold.
    Mirrors save_features() in feature_engineering.py.
    """
    agg = (
        pairs.groupby(["airport_A","airport_B","Month","Year"])
        .agg(
            observed_bad_rate    = ("raw_label",      "mean"),
            median_turnaround_min= ("turnaround_min", "median"),
            n_sequences          = ("raw_label",      "count"),
        )
        .reset_index()
    )
    agg["target"] = (agg["observed_bad_rate"] >= GLOBAL_BAD_RATE_THRESHOLD).astype(int)
    return agg


def attach_airport_features(pairs: pd.DataFrame, airport_features: pd.DataFrame) -> pd.DataFrame:
    feat_a = airport_features.rename(columns=lambda c: f"A_{c}" if c not in ("airport","Month") else c)\
                             .rename(columns={"airport":"airport_A"})
    feat_b = airport_features.rename(columns=lambda c: f"B_{c}" if c not in ("airport","Month") else c)\
                             .rename(columns={"airport":"airport_B"})
    pairs = pairs.merge(feat_a, on=["airport_A","Month"], how="left")
    pairs = pairs.merge(feat_b, on=["airport_B","Month"], how="left")

    pairs["pair_combined_weather_rate"] = pairs["A_weather_delay_rate"] * pairs["B_weather_delay_rate"]
    pairs["pair_max_weather_rate"]      = pairs[["A_weather_delay_rate","B_weather_delay_rate"]].max(axis=1)
    pairs["pair_min_weather_rate"]      = pairs[["A_weather_delay_rate","B_weather_delay_rate"]].min(axis=1)
    pairs["pair_weather_rate_sum"]      = pairs["A_weather_delay_rate"] + pairs["B_weather_delay_rate"]
    pairs["pair_avg_weather_delay_min"] = (pairs["A_avg_weather_delay_min"] + pairs["B_avg_weather_delay_min"]) / 2
    pairs["both_high_risk"] = (
        (pairs["A_weather_delay_rate"] > pairs["A_weather_delay_rate"].quantile(0.75)) &
        (pairs["B_weather_delay_rate"] > pairs["B_weather_delay_rate"].quantile(0.75))
    ).astype(int)

    # Season dummies from Month (no Season col at pair×month level)
    pairs["season_spring"]  = pairs["Month"].isin([3,4,5]).astype(int)
    pairs["season_summer"]  = pairs["Month"].isin([6,7,8]).astype(int)
    pairs["season_fall"]    = pairs["Month"].isin([9,10,11]).astype(int)
    pairs["season_winter"]  = pairs["Month"].isin([12,1,2]).astype(int)
    pairs["is_spring_summer"] = (pairs["season_spring"] | pairs["season_summer"])
    return pairs


def run_ood_pipeline(df: pd.DataFrame, hub: str, airport_features: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw flights → labeled pair×month rows with features."""
    df = add_season(df)
    pairs = build_sequences_agg(df, hub=hub)
    pairs = label_pairs(pairs)
    agg   = aggregate_to_pair_month(pairs)
    agg   = attach_airport_features(agg, airport_features)
    return agg


def evaluate(name: str, model: xgb.XGBClassifier,
             df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    if len(y) == 0 or y.nunique() < 2:
        print(f"[{name}] Skipped — insufficient data or single class.")
        return {}

    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    auc   = roc_auc_score(y, proba)
    ap    = average_precision_score(y, proba)
    rate  = y.mean()

    print(f"\n{'='*55}")
    print(f"OOD Test: {name}")
    print(f"  Rows (pair×month):  {len(y):,}")
    print(f"  Positive rate:      {rate:.1%}")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Avg Precision:      {ap:.4f}")
    print(classification_report(y, pred, target_names=["low_risk","high_risk"], zero_division=0))
    return {"name": name, "n": len(y), "positive_rate": rate, "roc_auc": auc, "avg_precision": ap}


# ---------------------------------------------------------------------------
# OOD Test 1: Temporal — all-carrier DFW 2015 (never in training)
# ---------------------------------------------------------------------------

def test_temporal_ood(model, airport_features, feature_cols):
    print("\n[OOD 1] Temporal — all-carrier DFW 2015 (before training window)")
    raw = os.path.join(RAW_DIR, "bts_all_dfw_2015.parquet")
    if not os.path.exists(raw):
        print("  bts_all_dfw_2015.parquet not found — skipping"); return {}
    df  = pd.read_parquet(raw)
    agg = run_ood_pipeline(df, hub="DFW", airport_features=airport_features)
    agg.to_parquet(os.path.join(OOD_DIR, "ood_temporal_2015.parquet"), index=False)
    return evaluate("Temporal OOD — all-carrier DFW 2015", model, agg, feature_cols)


# ---------------------------------------------------------------------------
# OOD Test 2: Extreme event — Jan/Feb 2015 record NE blizzards
# ---------------------------------------------------------------------------

def test_extreme_event_ood(model, airport_features, feature_cols):
    print("\n[OOD 2] Extreme event — Jan/Feb 2015 NE blizzards")
    raw = os.path.join(RAW_DIR, "bts_all_dfw_2015.parquet")
    if not os.path.exists(raw):
        print("  bts_all_dfw_2015.parquet not found — skipping"); return {}
    df  = pd.read_parquet(raw)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df  = df[df["FlightDate"].dt.month.isin([1, 2])].copy()
    agg = run_ood_pipeline(df, hub="DFW", airport_features=airport_features)
    agg.to_parquet(os.path.join(OOD_DIR, "ood_extreme_2015_janfeb.parquet"), index=False)

    result = evaluate("Extreme Event OOD — Jan/Feb 2015 blizzards", model, agg, feature_cols)

    # Show which pairs the model scored highest during blizzard months
    if len(agg) > 0:
        X = agg[feature_cols].astype(float)
        mask = X.notna().all(axis=1)
        agg_clean = agg[mask].copy()
        agg_clean["risk_score"] = model.predict_proba(X[mask])[:, 1]
        worst = (
            agg_clean.groupby(["airport_A","airport_B"])
            .agg(avg_risk=("risk_score","mean"), n=("risk_score","count"),
                 actual_bad_rate=("target","mean"))
            .sort_values("avg_risk", ascending=False)
            .head(10)
        )
        print("\nTop 10 riskiest pairs during Jan/Feb 2015 blizzards:")
        print(worst.to_string())
    return result


# ---------------------------------------------------------------------------
# OOD Test 3: Carrier OOD — non-AA carriers at DFW (DL, UA, WN, B6)
# ---------------------------------------------------------------------------

def test_carrier_ood(model, airport_features, feature_cols):
    """
    Raw files contain all carriers at DFW. Filter out AA → test whether
    airport-level weather risk generalises across carriers at same hub.
    """
    print("\n[OOD 3] Carrier OOD — non-AA carriers at DFW (2022–2023)")
    cached = os.path.join(OOD_DIR, "ood_non_aa_dfw_2022_2023.parquet")
    if os.path.exists(cached):
        agg = pd.read_parquet(cached)
    else:
        frames = []
        for year in [2022, 2023]:
            raw = os.path.join(RAW_DIR, f"bts_all_dfw_{year}.parquet")
            if not os.path.exists(raw):
                print(f"  {raw} not found — skipping {year}"); continue
            chunk = pd.read_parquet(raw)
            chunk = chunk[chunk["Reporting_Airline"] != "AA"].copy()
            print(f"  {year}: {len(chunk):,} non-AA rows ({chunk['Reporting_Airline'].nunique()} carriers)")
            frames.append(chunk)
        if not frames:
            print("  No data — skipping carrier OOD"); return {}
        df  = pd.concat(frames, ignore_index=True)
        agg = run_ood_pipeline(df, hub="DFW", airport_features=airport_features)
        agg.to_parquet(cached, index=False)

    return evaluate("Carrier OOD — non-AA/DFW 2022–2023", model, agg, feature_cols)


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_ood_summary(results: list[dict], in_dist: dict):
    valid = [r for r in results if r]
    if not valid:
        print("No OOD results to plot."); return

    all_res  = [in_dist] + valid
    names    = [r["name"] for r in all_res]
    auc_vals = [r["roc_auc"]       for r in all_res]
    ap_vals  = [r["avg_precision"] for r in all_res]
    colors   = ["steelblue"] + ["coral"] * len(valid)

    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, auc_vals, w, label="ROC-AUC",       color=colors, alpha=0.85)
    ax.bar(x + w/2, ap_vals,  w, label="Avg Precision",  color=colors, alpha=0.5, hatch="//")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" — ","\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("In-Distribution vs OOD Model Performance\n(blue = in-dist, coral = OOD)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PROC_DIR, "ood_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"\nOOD comparison plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(PROC_DIR, "xgb_model.json"))

    airport_features = pd.read_parquet(os.path.join(PROC_DIR, "airport_features.parquet"))

    # In-distribution: 2024 holdout (already aggregated)
    train_data   = pd.read_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"))
    val_data     = train_data[train_data["Year"] == train_data["Year"].max()].copy()
    season_cols  = [c for c in val_data.columns if c.startswith("season_")]
    feature_cols = [c for c in FEATURE_COLS + season_cols if c in val_data.columns]

    in_dist = evaluate("In-Distribution (2024 val)", model, val_data, feature_cols)

    results = [
        test_temporal_ood(model, airport_features, feature_cols),
        test_extreme_event_ood(model, airport_features, feature_cols),
        test_carrier_ood(model, airport_features, feature_cols),
    ]

    plot_ood_summary(results, in_dist)

    summary = pd.DataFrame([in_dist] + [r for r in results if r])
    out_csv = os.path.join(PROC_DIR, "ood_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Summary → {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
