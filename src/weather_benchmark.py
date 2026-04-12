"""
Weather Benchmark: correlate model risk scores with actual observed weather severity.

Pipeline:
  1. Load pair risk scores + unique airports from 2024 holdout
  2. Download IEM ASOS historical weather (free, no key) for those airports for 2024
  3. Aggregate to monthly airport-level severity
  4. Join onto pair scores: each pair gets severity_A, severity_B, pair_severity
  5. Compute and plot correlation: model risk score vs real weather severity
  6. Show top model-predicted risky pairs vs their actual weather

This answers: does our model flag pairs that are ACTUALLY bad per real weather data?
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from weather import download_historical_weather

PROC_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weather")
PLOTS_DIR   = os.path.join(PROC_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load pair scores and extract airports
# ---------------------------------------------------------------------------

def load_pairs() -> pd.DataFrame:
    pairs = pd.read_parquet(os.path.join(PROC_DIR, "pair_risk_scores.parquet"))
    # Filter to pairs with enough sequences to be reliable
    pairs = pairs[pairs["n_sequences"] >= 20].copy()
    print(f"Loaded {len(pairs):,} reliable pair×month combos")
    return pairs


def get_top_airports(pairs: pd.DataFrame, n: int = 60) -> list[str]:
    """Take the airports that appear most frequently in high-risk pairs."""
    all_airports = pd.concat([
        pairs.nlargest(500, "avg_risk_score")["airport_A"],
        pairs.nlargest(500, "avg_risk_score")["airport_B"],
    ]).value_counts().head(n).index.tolist()
    print(f"Benchmark airports: {sorted(all_airports)}")
    return all_airports


# ---------------------------------------------------------------------------
# Step 2: Download & aggregate weather to monthly airport level
# ---------------------------------------------------------------------------

def get_monthly_weather(airports: list[str], year: int = 2024, max_workers: int = 1) -> pd.DataFrame:
    print(f"\nDownloading IEM ASOS weather for {len(airports)} airports, {year}...")
    daily = download_historical_weather(airports, years=[year], max_workers=max_workers)

    if daily.empty:
        print("  No weather data returned.")
        return pd.DataFrame()

    daily["month"] = pd.to_datetime(daily["date"]).dt.month

    monthly = (
        daily.groupby(["iata", "month"])
        .agg(
            ts_days         = ("has_thunderstorm", "sum"),
            fog_days        = ("has_fog",          "sum"),
            snow_days       = ("has_snow_ice",     "sum"),
            low_ceil_days   = ("has_low_ceiling",  "sum"),
            avg_severity    = ("weather_severity", "mean"),
            max_severity    = ("weather_severity", "max"),
            avg_visibility  = ("min_visibility_mi","mean"),
            avg_wind_kt     = ("max_wind_kt",      "mean"),
            n_days          = ("weather_severity", "count"),
        )
        .reset_index()
        .rename(columns={"iata": "airport", "month": "Month"})
    )

    # Normalize day counts to rates
    for col in ["ts_days", "fog_days", "snow_days", "low_ceil_days"]:
        monthly[col.replace("_days", "_rate")] = monthly[col] / monthly["n_days"]

    print(f"Monthly weather: {len(monthly):,} airport×month rows "
          f"({monthly['airport'].nunique()} airports)")
    return monthly


# ---------------------------------------------------------------------------
# Step 3: Join weather onto pair scores
# ---------------------------------------------------------------------------

def join_weather_to_pairs(pairs: pd.DataFrame, monthly_wx: pd.DataFrame) -> pd.DataFrame:
    wx_a = monthly_wx.add_prefix("wxA_").rename(
        columns={"wxA_airport": "airport_A", "wxA_Month": "Month"}
    )
    wx_b = monthly_wx.add_prefix("wxB_").rename(
        columns={"wxB_airport": "airport_B", "wxB_Month": "Month"}
    )

    df = pairs.merge(wx_a, on=["airport_A", "Month"], how="inner")
    df = df.merge(wx_b, on=["airport_B", "Month"], how="inner")

    # Pair-level severity: worst of the two legs (a risky pair needs both bad)
    df["pair_avg_severity"]  = (df["wxA_avg_severity"]  + df["wxB_avg_severity"])  / 2
    df["pair_max_severity"]  = df[["wxA_avg_severity",   "wxB_avg_severity"]].max(axis=1)
    df["pair_min_severity"]  = df[["wxA_avg_severity",   "wxB_avg_severity"]].min(axis=1)
    df["pair_ts_rate"]       = (df["wxA_ts_rate"]        + df["wxB_ts_rate"])       / 2
    df["both_airports_bad"]  = (
        (df["wxA_avg_severity"] > df["wxA_avg_severity"].median()) &
        (df["wxB_avg_severity"] > df["wxB_avg_severity"].median())
    ).astype(int)

    print(f"Joined: {len(df):,} pairs have weather data for both airports")
    return df


# ---------------------------------------------------------------------------
# Step 4: Benchmark statistics
# ---------------------------------------------------------------------------

def compute_benchmark(df: pd.DataFrame) -> dict:
    r_avg, p_avg = stats.pearsonr(df["avg_risk_score"], df["pair_avg_severity"])
    r_ts,  p_ts  = stats.pearsonr(df["avg_risk_score"], df["pair_ts_rate"])
    r_min, p_min = stats.pearsonr(df["avg_risk_score"], df["pair_min_severity"])

    # Spearman (rank-based, more robust to outliers)
    rho_avg, _ = stats.spearmanr(df["avg_risk_score"], df["pair_avg_severity"])
    rho_ts,  _ = stats.spearmanr(df["avg_risk_score"], df["pair_ts_rate"])

    print("\n" + "="*55)
    print("WEATHER BENCHMARK — Model Risk Score vs Actual Severity")
    print("="*55)
    print(f"  Pairs in benchmark:        {len(df):,}")
    print(f"  Pearson r  (avg severity): {r_avg:.3f}  (p={p_avg:.2e})")
    print(f"  Pearson r  (TS rate):      {r_ts:.3f}  (p={p_ts:.2e})")
    print(f"  Pearson r  (min severity): {r_min:.3f}  (p={p_min:.2e})")
    print(f"  Spearman ρ (avg severity): {rho_avg:.3f}")
    print(f"  Spearman ρ (TS rate):      {rho_ts:.3f}")

    # Precision-at-K: are the model's top-K pairs actually the worst by weather?
    for k in [10, 25, 50]:
        top_k_model  = set(df.nlargest(k, "avg_risk_score").index)
        top_k_actual = set(df.nlargest(k, "pair_avg_severity").index)
        overlap = len(top_k_model & top_k_actual)
        print(f"  Precision@{k:2d}:               {overlap}/{k} top pairs overlap with actual worst")

    return {
        "n_pairs": len(df),
        "pearson_avg_severity": r_avg, "p_avg": p_avg,
        "pearson_ts_rate": r_ts,
        "spearman_avg_severity": rho_avg,
        "spearman_ts_rate": rho_ts,
    }


# ---------------------------------------------------------------------------
# Step 5: Plots
# ---------------------------------------------------------------------------

def plot_benchmark(df: pd.DataFrame, stats_dict: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Model Risk Score vs Actual Weather Severity (2024)\n"
                 "AA Crew Sequences — DFW Hub", fontsize=13, fontweight="bold")

    # 1. Scatter: risk score vs avg severity (colour = thunderstorm rate)
    ax = axes[0, 0]
    sc = ax.scatter(
        df["pair_avg_severity"], df["avg_risk_score"],
        c=df["pair_ts_rate"], cmap="YlOrRd",
        alpha=0.5, s=15, rasterized=True
    )
    plt.colorbar(sc, ax=ax, label="Thunderstorm rate (both airports avg)")
    r = stats_dict["pearson_avg_severity"]
    ax.set_xlabel("Actual Avg Weather Severity (IEM ASOS)")
    ax.set_ylabel("Model Risk Score")
    ax.set_title(f"Risk Score vs Actual Severity  r={r:.3f}")
    ax.grid(alpha=0.3)

    # 2. Monthly pattern: model avg risk vs actual avg severity across months
    ax = axes[0, 1]
    monthly = df.groupby("Month").agg(
        model_risk=("avg_risk_score", "mean"),
        actual_sev=("pair_avg_severity", "mean"),
    ).reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = monthly["Month"].values
    ax2 = ax.twinx()
    ax.bar(x, monthly["actual_sev"], color="steelblue", alpha=0.6, label="Actual severity")
    ax2.plot(x, monthly["model_risk"], color="coral", marker="o", linewidth=2, label="Model risk")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45, ha="right")
    ax.set_ylabel("Avg actual severity", color="steelblue")
    ax2.set_ylabel("Avg model risk score", color="coral")
    ax.set_title("Monthly: Model Risk vs Actual Severity")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    # 3. Top-25 model pairs: model risk vs actual severity bar comparison
    ax = axes[1, 0]
    top25 = df.nlargest(25, "avg_risk_score").copy()
    top25["pair"] = top25["airport_A"] + "→" + top25["airport_B"] + " M" + top25["Month"].astype(str)
    top25 = top25.sort_values("avg_risk_score")
    y = range(len(top25))
    ax.barh(y, top25["avg_risk_score"], color="coral",   alpha=0.8, label="Model risk score")
    ax.barh(y, top25["pair_avg_severity"], color="steelblue", alpha=0.6, label="Actual severity")
    ax.set_yticks(y)
    ax.set_yticklabels(top25["pair"], fontsize=7)
    ax.set_xlabel("Score / Severity")
    ax.set_title("Top 25 Model-Predicted Risky Pairs\nvs Actual Weather Severity")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # 4. Decile analysis: model risk deciles vs avg actual severity
    ax = axes[1, 1]
    df["risk_decile"] = pd.qcut(df["avg_risk_score"], q=10, labels=False, duplicates="drop")
    decile_stats = df.groupby("risk_decile").agg(
        actual_severity=("pair_avg_severity", "mean"),
        ts_rate=("pair_ts_rate", "mean"),
        n=("avg_risk_score", "count"),
    ).reset_index()
    ax.bar(decile_stats["risk_decile"] + 1, decile_stats["actual_severity"],
           color="steelblue", alpha=0.8, label="Avg actual severity")
    ax2b = ax.twinx()
    ax2b.plot(decile_stats["risk_decile"] + 1, decile_stats["ts_rate"],
              color="coral", marker="o", linewidth=2, label="TS rate")
    ax.set_xlabel("Model Risk Score Decile (1=lowest, 10=highest)")
    ax.set_ylabel("Avg actual weather severity", color="steelblue")
    ax2b.set_ylabel("Thunderstorm rate", color="coral")
    ax.set_title("Actual Severity by Model Risk Decile\n(good model → monotone increasing)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "weather_benchmark.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBenchmark plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pairs = load_pairs()
    airports = get_top_airports(pairs, n=60)

    monthly_wx = get_monthly_weather(airports, year=2024, max_workers=1)
    if monthly_wx.empty:
        print("No weather data — check IEM connectivity.")
        return

    monthly_wx.to_parquet(os.path.join(PROC_DIR, "monthly_weather_2024.parquet"), index=False)

    df = join_weather_to_pairs(pairs, monthly_wx)
    if len(df) < 20:
        print("Too few joined pairs — check airport coverage.")
        return

    benchmark_stats = compute_benchmark(df)
    plot_benchmark(df, benchmark_stats)

    df.to_parquet(os.path.join(PROC_DIR, "weather_benchmarked_pairs.parquet"), index=False)
    print(f"\nBenchmarked pairs saved → {os.path.join(PROC_DIR, 'weather_benchmarked_pairs.parquet')}")


if __name__ == "__main__":
    main()
