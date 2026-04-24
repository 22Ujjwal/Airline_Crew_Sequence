"""
FastAPI backend for AA Crew Risk web dashboard.

Run with:
    conda run -n aadata uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Or from project root:
    conda run -n aadata python -m uvicorn src.api:app --port 8000 --reload
"""
from __future__ import annotations
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RAW       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

app = FastAPI(title="AA Crew Risk API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (loaded at startup) ───────────────────────────────────────────────
_scores_df: pd.DataFrame = pd.DataFrame()
_predictor: Any = None
_feature_importance_df: pd.DataFrame = pd.DataFrame()


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


@app.on_event("startup")
async def startup() -> None:
    global _scores_df, _predictor, _feature_importance_df
    # Load pair risk scores
    scores_path = os.path.join(PROCESSED, "pair_risk_scores.parquet")
    if os.path.exists(scores_path):
        _scores_df = pd.read_parquet(scores_path)
        print(f"Loaded {len(_scores_df):,} pair-month scores")
    else:
        print("WARNING: pair_risk_scores.parquet not found")

    # Load predictor (model + features)
    try:
        from app.predictor import RiskPredictor, build_features_df
        df = build_features_df()
        _predictor = RiskPredictor(df)
        print("Predictor loaded")
    except Exception as e:
        print(f"WARNING: predictor load failed: {e}")

    # Load feature importance
    try:
        import xgboost as xgb
        from app.predictor import FEATURE_LABELS
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(PROCESSED, "xgb_model.json"))
        fnames = m.get_booster().feature_names
        fi     = m.feature_importances_

        def _group(f: str) -> str:
            if f.startswith(("A_weather", "A_overall", "A_nas_")):    return "Origin BTS"
            if f.startswith(("B_weather", "B_overall", "B_nas_")):    return "Dest BTS"
            if f.startswith("pair_") and "cascade" not in f and "wind" not in f and "precip" not in f: return "Pair BTS"
            if f in ("Month", "is_spring_summer", "median_turnaround_min") or f.startswith("season_"): return "Temporal"
            if f.startswith(("A_avg_wind", "A_precip", "A_extreme", "A_total_precip", "A_max_wind")): return "Origin GSOM"
            if f.startswith(("B_avg_wind", "B_precip", "B_extreme", "B_total_precip", "B_max_wind")): return "Dest GSOM"
            if f.startswith(("pair_max_avg_wind", "pair_max_precip", "pair_max_extreme",
                              "pair_max_total", "pair_max_max_wind")):                                  return "Pair GSOM"
            if f.startswith("DFW_"):   return "DFW Hub"
            if f.startswith("tc_"):    return "Tail-Chain / Duty"
            if f.startswith(("A_ap_", "B_ap_", "pair_cascade")): return "Airport Cascade"
            if f.startswith("mhc_"):   return "Multi-Hop Cascade"
            return "Other"

        _feature_importance_df = pd.DataFrame({
            "feature":    fnames,
            "importance": fi,
            "label":      [FEATURE_LABELS.get(f, f) for f in fnames],
            "group":      [_group(f) for f in fnames],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        _feature_importance_df["rank"] = _feature_importance_df.index + 1
        print("Feature importance loaded")
    except Exception as e:
        print(f"WARNING: feature importance load failed: {e}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "scores_loaded": len(_scores_df),
        "predictor_loaded": _predictor is not None,
    }


# ── Airports ──────────────────────────────────────────────────────────────────
@app.get("/api/airports")
def airports():
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")
    all_a = sorted(_scores_df["airport_A"].unique().tolist())
    all_b = sorted(_scores_df["airport_B"].unique().tolist())
    return sorted(set(all_a + all_b))


# ── Scores ────────────────────────────────────────────────────────────────────
@app.get("/api/scores")
def all_scores(month: int | None = Query(None)):
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")
    df = _scores_df if month is None else _scores_df[_scores_df["Month"] == month]
    return df[["airport_A", "airport_B", "Month", "avg_risk_score",
               "observed_bad_rate", "n_sequences"]].to_dict("records")


@app.get("/api/scores/top")
def top_pairs(month: int = 6, n: int = 20):
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")
    df = _scores_df[_scores_df["Month"] == month].nlargest(n, "avg_risk_score")
    return df[["airport_A", "airport_B", "Month", "avg_risk_score",
               "observed_bad_rate", "n_sequences"]].to_dict("records")


@app.get("/api/scores/heatmap/{month}")
def heatmap(month: int):
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")
    ms = _scores_df[_scores_df["Month"] == month]
    grp = (ms.groupby("airport_A")["avg_risk_score"].mean()
           .reset_index().rename(columns={"avg_risk_score": "avg_risk"}))
    return grp.to_dict("records")


# ── Prediction ────────────────────────────────────────────────────────────────
@app.get("/api/predict/{airport_a}/{airport_b}/{month}")
def predict(airport_a: str, airport_b: str, month: int):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    res = _predictor.predict_pair(airport_a.upper(), airport_b.upper(), month)
    if res is None:
        raise HTTPException(404, f"Pair {airport_a}→DFW→{airport_b} month={month} not in dataset")

    shap_df = _predictor.explain_pair(res["X"], top_n=15, gsom_imputed=res["gsom_imputed"])
    shap_records = [
        {
            "feature":       r.feature,
            "label":         r.label,
            "shap_value":    round(float(r.shap_value), 6),
            "feature_value": round(float(r.feature_value), 6),
            "imputed":       bool(r.imputed),
            "abs_shap":      round(float(r.abs_shap), 6),
        }
        for r in shap_df.itertuples()
    ]
    return {
        "airport_A":         airport_a.upper(),
        "airport_B":         airport_b.upper(),
        "month":             month,
        "risk_score":        round(float(res["risk_score"]), 5),
        "label":             res["label"],
        "observed_bad_rate": _safe_float(res["observed_bad_rate"]),
        "n_sequences":       _safe_int(res["n_sequences"]),
        "shap":              shap_records,
    }


@app.get("/api/timeline/{airport_a}/{airport_b}")
def pair_timeline(airport_a: str, airport_b: str):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    tl = _predictor.predict_all_months(airport_a.upper(), airport_b.upper())
    return [
        {"month": int(r.Month), "risk_score": round(float(r.risk_score), 5), "label": r.label}
        for r in tl.itertuples()
        if not math.isnan(r.risk_score)
    ]


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.get("/api/eval")
def eval_metrics():
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
        s = _scores_df.dropna(subset=["avg_risk_score", "observed_bad_rate"])
        y = (s["observed_bad_rate"] > 0.25).astype(int)
        p = s["avg_risk_score"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)

        # Decimate curves for payload size
        step = max(1, len(fpr) // 300)
        fpr_d = fpr[::step].tolist()
        tpr_d = tpr[::step].tolist()
        step2 = max(1, len(prec) // 300)
        prec_d = prec[::step2].tolist()
        rec_d  = rec[::step2].tolist()

        s2 = s.copy()
        s2["decile"] = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = (s2.groupby("decile")
               .agg(mean_score=("avg_risk_score", "mean"),
                    mean_obs=("observed_bad_rate", "mean"),
                    n=("avg_risk_score", "count"))
               .reset_index()
               .to_dict("records"))
        for row in cal:
            row["decile"] = int(row["decile"])

        pair_auc = float(roc_auc_score(y, p))
        pair_ap  = float(average_precision_score(y, p))

        dist = [
            {"band": "LOW (<20%)",      "fraction": float((p < 0.20).mean()), "count": int((p < 0.20).sum())},
            {"band": "MODERATE (20–30%)", "fraction": float(((p >= 0.20) & (p < 0.30)).mean()), "count": int(((p >= 0.20) & (p < 0.30)).sum())},
            {"band": "HIGH (≥30%)",     "fraction": float((p >= 0.30).mean()), "count": int((p >= 0.30).sum())},
        ]

        return {
            "auc":       round(pair_auc, 4),
            "ap":        round(pair_ap, 4),
            "pair_auc":  round(pair_auc, 4),
            "pair_ap":   round(pair_ap, 4),
            "fpr":       fpr_d,
            "tpr":       tpr_d,
            "precision": prec_d,
            "recall":    rec_d,
            "calibration":       cal,
            "score_distribution": dist,
        }
    except Exception as e:
        raise HTTPException(500, f"Eval failed: {e}")


# ── Feature importance ─────────────────────────────────────────────────────────
@app.get("/api/features")
def features():
    if _feature_importance_df.empty:
        raise HTTPException(503, "Feature importance not loaded")
    return [
        {
            "rank":       int(r.rank),
            "feature":    r.feature,
            "label":      r.label,
            "group":      r.group,
            "importance": round(float(r.importance), 7),
        }
        for r in _feature_importance_df.itertuples()
    ]


# ── Optimizer ─────────────────────────────────────────────────────────────────
class FlightEntry(BaseModel):
    airport: str
    time_str: str
    flight: str = ""


class OptimizeRequest(BaseModel):
    arrivals: list[FlightEntry]
    departures: list[FlightEntry]
    month: int


def _parse_time(t: str) -> int:
    """HH:MM → minutes from midnight."""
    parts = t.strip().split(":")
    return int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    if _scores_df.empty:
        raise HTTPException(503, "Data not loaded")

    try:
        from app.optimizer import optimize_sequences
        arr_df = pd.DataFrame([
            {
                "airport":  f.airport.upper(),
                "time_min": _parse_time(f.time_str),
                "time_str": f.time_str,
                "flight":   f.flight or "",
            }
            for f in req.arrivals
        ])
        dep_df = pd.DataFrame([
            {
                "airport":  f.airport.upper(),
                "time_min": _parse_time(f.time_str),
                "time_str": f.time_str,
                "flight":   f.flight or "",
            }
            for f in req.departures
        ])
        scores_idx = _scores_df.set_index(["airport_A", "airport_B", "Month"])
        seqs_df, stats = optimize_sequences(arr_df, dep_df, scores_idx, req.month)

        seq_records = [
            {
                "sequence":       r.Sequence,
                "airport_A":      r.airport_A,
                "airport_B":      r.airport_B,
                "flight_in":      str(r.flight_in),
                "arr_time":       str(r.arr_time),
                "flight_out":     str(r.flight_out),
                "dep_time":       str(r.dep_time),
                "turnaround_min": int(r.turnaround_min),
                "risk_score":     round(float(r.risk_score), 5),
                "risk_label":     r.risk_label,
            }
            for r in seqs_df.itertuples()
        ]

        return {
            "sequences": seq_records,
            "stats": {
                "n_arrivals":    stats["n_arrivals"],
                "n_departures":  stats["n_departures"],
                "n_matched":     stats["n_matched"],
                "feasible_pairs": stats["feasible_pairs"],
                "optimal_total": round(stats["optimal_total"], 4),
                "optimal_avg":   round(stats["optimal_avg"], 4),
                "worst_total":   round(stats["worst_total"], 4),
                "risk_saved":    round(stats["risk_saved"], 4),
                "pct_high":      round(stats["pct_high"], 4),
            },
        }
    except Exception as e:
        raise HTTPException(500, f"Optimization error: {e}")
