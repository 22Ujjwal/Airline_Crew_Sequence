"""
Stress test & real-world verification for the crew sequence risk model.

Test 1 — 2024 holdout backtest
  Reload model, score 2024 rows (never seen during training), check:
  - ROC-AUC, Average Precision
  - Decile calibration: do higher-score deciles have proportionally higher actual bad rates?
  - Monotonicity: is each decile worse than the previous?

Test 2 — Historic storm event replay
  For known catastrophic weather events, download IEM METAR for affected airports,
  compute per-day weather severity, and check whether the model + weather penalty
  would have correctly flagged those sequences as high-risk.

  Events:
    A. Winter Storm Elliott  — Dec 23-26 2022 (ORD, BUF, CLT, DEN, MDW)
    B. DFW summer convective — Jul 10 2023 (DFW-centric, peak convective season)
    C. Southeast ice storm   — Jan 16 2024 (ATL, CLT, RDU, MIA)
"""

import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from weather import download_historical_weather, _compute_severity

PROC  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PLOTS = os.path.join(PROC, "plots")
os.makedirs(PLOTS, exist_ok=True)

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

STORM_EVENTS = {
    "Winter Storm Elliott (Dec 23-26 2022)": {
        "airports": ["ORD", "BUF", "CLT", "DEN", "MDW", "LGA", "DFW"],
        "year": 2022,
        "dates": pd.date_range("2022-12-23", "2022-12-26"),
        "expected": "CRITICAL — massive cancellation event across US",
    },
    "DFW Convective Outbreak (Jul 10 2023)": {
        "airports": ["DFW", "DAL", "AUS", "SAT", "HOU", "MSY", "ORD"],
        "year": 2023,
        "dates": pd.date_range("2023-07-08", "2023-07-12"),
        "expected": "HIGH — peak summer convective at DFW hub",
    },
    "Southeast Ice Storm (Jan 16-17 2024)": {
        "airports": ["ATL", "CLT", "RDU", "GSO", "BNA", "MIA", "DFW"],
        "year": 2024,
        "dates": pd.date_range("2024-01-15", "2024-01-18"),
        "expected": "HIGH — ice/freezing rain across Southeast",
    },
}


# ---------------------------------------------------------------------------
# Test 1: 2024 holdout backtest
# ---------------------------------------------------------------------------

def test1_holdout_backtest():
    print("\n" + "="*65)
    print("TEST 1 — 2024 Holdout Backtest")
    print("="*65)

    df = pd.read_parquet(os.path.join(PROC, "sequence_features.parquet"))
    season_cols = [c for c in df.columns if c.startswith("season_")]
    feat_cols = [c for c in FEATURE_COLS + season_cols if c in df.columns]

    val = df[df["Year"] == 2024].copy()
    X_val = val[feat_cols].astype(float)
    y_val = val["target"].astype(int)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(PROC, "xgb_model.json"))
    proba = model.predict_proba(X_val)[:, 1]

    roc  = roc_auc_score(y_val, proba)
    ap   = average_precision_score(y_val, proba)
    print(f"\nROC-AUC (2024):          {roc:.4f}")
    print(f"Average Precision (2024):{ap:.4f}")
    print(f"Baseline bad rate:       {y_val.mean():.3f} ({y_val.mean():.1%})")

    # Decile calibration
    val = val.copy()
    val["proba"] = proba
    val["decile"] = pd.qcut(val["proba"], q=10, labels=False, duplicates="drop")

    decile_tbl = (
        val.groupby("decile")
        .agg(
            n=("target", "count"),
            actual_bad_rate=("target", "mean"),
            mean_predicted=("proba", "mean"),
            mean_observed=("observed_bad_rate", "mean"),
        )
        .reset_index()
    )
    print("\nDecile Calibration (0=lowest predicted risk, 9=highest):")
    print(decile_tbl.to_string(index=False, float_format="{:.3f}".format))

    # Monotonicity check
    rates = decile_tbl["actual_bad_rate"].values
    monotone = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    print(f"\nMonotonicity (each decile worse than prev): {'PASS' if monotone else 'FAIL'}")

    # Lift: top decile vs bottom decile
    top_rate    = decile_tbl.iloc[-1]["actual_bad_rate"]
    bottom_rate = decile_tbl.iloc[0]["actual_bad_rate"]
    print(f"Lift (top/bottom decile): {top_rate/bottom_rate:.1f}x")

    # Plot calibration
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(decile_tbl["mean_predicted"], decile_tbl["actual_bad_rate"],
            "o-", color="steelblue", label="Actual bad rate")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Calibration — 2024 holdout\n(by predicted probability decile)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    bars = ax.bar(decile_tbl["decile"], decile_tbl["actual_bad_rate"],
                  color=plt.cm.RdYlGn_r(decile_tbl["actual_bad_rate"]))
    ax.axhline(y_val.mean(), color="navy", linestyle="--", label=f"Baseline {y_val.mean():.1%}")
    ax.set_xlabel("Risk score decile (0=lowest, 9=highest)")
    ax.set_ylabel("Actual bad rate")
    ax.set_title(f"Actual bad rate by decile — 2024 holdout\nROC-AUC={roc:.3f}  AP={ap:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS, "stress_test1_calibration.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nCalibration plot → {out}")

    return decile_tbl


# ---------------------------------------------------------------------------
# Test 2: Historic storm event replay
# ---------------------------------------------------------------------------

def test2_storm_replay():
    print("\n" + "="*65)
    print("TEST 2 — Historic Storm Event Replay")
    print("="*65)

    all_results = []

    for event_name, ev in STORM_EVENTS.items():
        print(f"\n--- {event_name} ---")
        print(f"Expected: {ev['expected']}")
        airports  = ev["airports"]
        year      = ev["year"]
        dates     = ev["dates"]

        # Download/cache IEM METAR for these airports × year
        print(f"Fetching IEM METAR for {airports} in {year}...")
        wx = download_historical_weather(airports, [year])

        if wx.empty:
            print("  WARNING: No weather data returned")
            continue

        # Filter to event dates
        wx["date"] = pd.to_datetime(wx["date"])
        ev_wx = wx[wx["date"].isin(dates)].copy()

        if ev_wx.empty:
            print("  WARNING: No data for event dates")
            continue

        print(f"\n  Airport weather during event:")
        cols = ["iata", "date", "weather_severity", "has_thunderstorm",
                "has_snow_ice", "has_fog", "min_visibility_mi",
                "max_wind_kt", "ceiling_ft"]
        cols_present = [c for c in cols if c in ev_wx.columns]
        print(ev_wx[cols_present].sort_values(["date","iata"]).to_string(index=False, float_format="{:.2f}".format))

        # Summary stats per airport
        summary = ev_wx.groupby("iata")["weather_severity"].agg(["mean","max"])
        summary.columns = ["avg_severity", "max_severity"]
        print(f"\n  Severity summary:")
        print(summary.sort_values("max_severity", ascending=False).to_string(float_format="{:.3f}".format))

        # For each airport pair involving DFW, compute what live_score would have been
        # using peak weather day severity
        print(f"\n  Pair-level impact (A→DFW→B, worst day):")
        dfwrow = ev_wx[ev_wx["iata"] == "DFW"]
        dfw_max_sev = dfwrow["weather_severity"].max() if not dfwrow.empty else 0.0

        for iata in airports:
            if iata == "DFW":
                continue
            rows = ev_wx[ev_wx["iata"] == iata]
            if rows.empty:
                continue
            ap_max_sev = rows["weather_severity"].max()
            # Simulate weather penalty formula from predict.py
            penalty = max(dfw_max_sev * 0.6, ap_max_sev * 0.4)
            all_results.append({
                "event":       event_name,
                "airport":     iata,
                "dfw_severity":dfw_max_sev,
                "ap_severity": ap_max_sev,
                "wx_penalty":  penalty,
                "flag":        "HIGH" if penalty > 0.3 else ("MOD" if penalty > 0.15 else "LOW"),
            })
            print(f"    {iata}: airport_sev={ap_max_sev:.2f}  dfw_sev={dfw_max_sev:.2f}  "
                  f"penalty={penalty:.2f}  → {all_results[-1]['flag']}")

    # Summary table
    if all_results:
        res_df = pd.DataFrame(all_results)
        print("\n\n" + "="*65)
        print("STORM REPLAY SUMMARY")
        print("="*65)
        print(res_df.groupby(["event","flag"]).size().unstack(fill_value=0).to_string())

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        events_short = {e: e.split("(")[0].strip() for e in res_df["event"].unique()}
        res_df["event_short"] = res_df["event"].map(events_short)
        colors = {"HIGH": "#d62728", "MOD": "#ff7f0e", "LOW": "#2ca02c"}
        for i, (ev_short, grp) in enumerate(res_df.groupby("event_short")):
            x = range(len(grp))
            bars = ax.bar(
                [j + i*0.28 for j in range(len(grp))],
                grp["wx_penalty"].values,
                width=0.25,
                label=ev_short,
                color=[colors[f] for f in grp["flag"]],
                alpha=0.8,
            )

        ax.axhline(0.30, color="red",    linestyle="--", linewidth=1, label="HIGH threshold (0.30)")
        ax.axhline(0.15, color="orange", linestyle="--", linewidth=1, label="MOD threshold (0.15)")
        ax.set_ylabel("Weather penalty")
        ax.set_title("Storm Event Replay — Weather Penalty per Airport Pair\n(higher = model would have flagged pair as high-risk)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOTS, "stress_test2_storm_replay.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\nStorm replay plot → {out}")

    return all_results


if __name__ == "__main__":
    t1 = test1_holdout_backtest()
    t2 = test2_storm_replay()

    print("\n" + "="*65)
    print("STRESS TEST COMPLETE")
    print("="*65)
    print("Plots saved to data/processed/plots/")
    print("  stress_test1_calibration.png")
    print("  stress_test2_storm_replay.png")
