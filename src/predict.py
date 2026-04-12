"""
Real-time crew sequence risk predictor.

Usage:
  python predict.py ORD LAX          # predict risk for ORD→DFW→LAX right now
  python predict.py ORD LAX --month 7  # predict for July (historical patterns only)

How it works:
  1. Base risk  — XGBoost model trained on 2018–2024 historical delay patterns
  2. Live weather adjustment — fetches current METAR for airport_A, DFW, airport_B
                               via AWC API and adjusts the base score upward if
                               current conditions are severe.
  3. Returns a combined risk score [0, 1] with full explanation.
"""

import argparse
import datetime
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from weather import fetch_live_metar, _empty_weather

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

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

RISK_LABELS = {
    (0.0, 0.3): ("LOW",      "✓ Safe to sequence"),
    (0.3, 0.55): ("MODERATE", "⚠ Use caution — review turnaround buffer"),
    (0.55, 0.75): ("HIGH",    "✗ Avoid if possible — weather-prone pair"),
    (0.75, 1.01): ("CRITICAL","✗✗ Do not sequence — high cascade risk"),
}


def risk_label(score: float) -> tuple[str, str]:
    for (lo, hi), (label, advice) in RISK_LABELS.items():
        if lo <= score < hi:
            return label, advice
    return "CRITICAL", "Do not sequence"


class PairRiskPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(device="cuda", tree_method="hist")
        self.model.load_model(os.path.join(PROC, "xgb_model.json"))
        self.airport_features = pd.read_parquet(os.path.join(PROC, "airport_features.parquet"))
        self.pair_scores      = pd.read_parquet(os.path.join(PROC, "pair_risk_scores.parquet"))

    def _get_airport_features(self, airport: str, month: int) -> dict:
        row = self.airport_features[
            (self.airport_features["airport"] == airport) &
            (self.airport_features["Month"] == month)
        ]
        if row.empty:
            # Fall back to annual average for that airport
            row = self.airport_features[self.airport_features["airport"] == airport]
        if row.empty:
            return {}
        return row.mean(numeric_only=True).to_dict()

    def _build_feature_vector(self, airport_a: str, airport_b: str, month: int) -> pd.DataFrame:
        fa = self._get_airport_features(airport_a, month)
        fb = self._get_airport_features(airport_b, month)

        feat = {
            "A_weather_delay_rate":          fa.get("weather_delay_rate",       np.nan),
            "A_weather_cancel_rate":          fa.get("weather_cancel_rate",      np.nan),
            "A_avg_weather_delay_min":        fa.get("avg_weather_delay_min",    np.nan),
            "A_p75_weather_delay_min":        fa.get("p75_weather_delay_min",    np.nan),
            "A_p95_weather_delay_min":        fa.get("p95_weather_delay_min",    np.nan),
            "A_nas_delay_rate":              fa.get("nas_delay_rate",           np.nan),
            "A_overall_weather_delay_rate":   fa.get("overall_weather_delay_rate",np.nan),
            "A_overall_avg_weather_delay_min":fa.get("overall_avg_weather_delay_min",np.nan),
            "B_weather_delay_rate":          fb.get("weather_delay_rate",       np.nan),
            "B_weather_cancel_rate":          fb.get("weather_cancel_rate",      np.nan),
            "B_avg_weather_delay_min":        fb.get("avg_weather_delay_min",    np.nan),
            "B_p75_weather_delay_min":        fb.get("p75_weather_delay_min",    np.nan),
            "B_p95_weather_delay_min":        fb.get("p95_weather_delay_min",    np.nan),
            "B_nas_delay_rate":              fb.get("nas_delay_rate",           np.nan),
            "B_overall_weather_delay_rate":   fb.get("overall_weather_delay_rate",np.nan),
            "B_overall_avg_weather_delay_min":fb.get("overall_avg_weather_delay_min",np.nan),
            "Month":          month,
            "is_spring_summer": int(month in (3,4,5,6,7,8)),
            "median_turnaround_min": 90.0,  # default 90-min turnaround at DFW
        }

        a_rate = feat["A_weather_delay_rate"] or 0
        b_rate = feat["B_weather_delay_rate"] or 0
        feat["pair_combined_weather_rate"]  = a_rate * b_rate
        feat["pair_max_weather_rate"]       = max(a_rate, b_rate)
        feat["pair_min_weather_rate"]       = min(a_rate, b_rate)
        feat["pair_weather_rate_sum"]       = a_rate + b_rate
        feat["pair_avg_weather_delay_min"]  = (
            (feat["A_avg_weather_delay_min"] or 0) +
            (feat["B_avg_weather_delay_min"] or 0)
        ) / 2

        # both_high_risk: both airports above 75th percentile of delay rate
        all_rates = self.airport_features["weather_delay_rate"].dropna()
        p75 = all_rates.quantile(0.75)
        feat["both_high_risk"] = int(a_rate > p75 and b_rate > p75)

        # Season dummies
        season = {3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",
                  9:"fall",10:"fall",11:"fall",12:"winter",1:"winter",2:"winter"}[month]
        for s in ("fall","spring","summer","winter"):
            feat[f"season_{s}"] = int(season == s)

        return pd.DataFrame([feat])

    def predict_historical(self, airport_a: str, airport_b: str, month: int = None) -> dict:
        """Predict using historical patterns only (no live weather)."""
        if month is None:
            month = datetime.date.today().month

        fv = self._build_feature_vector(airport_a, airport_b, month)
        model_cols = self.model.get_booster().feature_names
        for col in model_cols:
            if col not in fv.columns:
                fv[col] = 0.0
        fv = fv[model_cols].astype(float)

        base_score = float(self.model.predict_proba(fv)[0, 1])

        # Look up observed rate from historical data if available
        hist = self.pair_scores[
            (self.pair_scores["airport_A"] == airport_a) &
            (self.pair_scores["airport_B"] == airport_b) &
            (self.pair_scores["Month"] == month)
        ]
        observed = float(hist["observed_bad_rate"].iloc[0]) if not hist.empty else None

        label, advice = risk_label(base_score)
        return {
            "airport_a":      airport_a,
            "airport_b":      airport_b,
            "hub":            "DFW",
            "month":          month,
            "base_risk_score": base_score,
            "risk_label":     label,
            "advice":         advice,
            "observed_bad_rate": observed,
        }

    def predict_live(self, airport_a: str, airport_b: str) -> dict:
        """
        Full real-time prediction: historical model + live weather adjustment.
        Fetches current METAR for airport_A, DFW, and airport_B.
        """
        month = datetime.date.today().month
        result = self.predict_historical(airport_a, airport_b, month)

        print(f"  Fetching live METAR for {airport_a}, DFW, {airport_b}...")
        weather = fetch_live_metar([airport_a, "DFW", airport_b])

        wa  = weather.get(airport_a, _empty_weather())
        wdfw= weather.get("DFW",     _empty_weather())
        wb  = weather.get(airport_b, _empty_weather())

        # DFW weather affects both legs — weight it higher
        weather_penalty = max(
            wdfw["weather_severity"] * 0.6,   # DFW: hub, both legs affected
            wa["weather_severity"]   * 0.4,   # leg 1: A→DFW
            wb["weather_severity"]   * 0.4,   # leg 2: DFW→B
        )

        base = result["base_risk_score"]
        # Blend: bad weather pushes score toward 1; clear weather doesn't reduce below base
        live_score = base + (1.0 - base) * weather_penalty
        live_score = float(np.clip(live_score, 0, 1))

        label, advice = risk_label(live_score)
        result.update({
            "live_risk_score":   live_score,
            "weather_penalty":   weather_penalty,
            "risk_label":        label,
            "advice":            advice,
            "weather_airport_a": wa,
            "weather_dfw":       wdfw,
            "weather_airport_b": wb,
        })
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_weather(label: str, w: dict) -> str:
    raw = w.get("raw", "")
    sev = w.get("weather_severity", 0)
    flags = []
    if w.get("has_thunderstorm"): flags.append("THUNDERSTORM")
    if w.get("has_fog"):          flags.append("FOG/MIST")
    if w.get("has_snow_ice"):     flags.append("SNOW/ICE")
    if w.get("has_low_ceiling"):  flags.append(f"LOW CEILING ({w.get('ceiling_ft',0):.0f}ft)")
    vis = w.get("min_visibility_mi", 10)
    if vis < 3:                   flags.append(f"LOW VIS ({vis:.1f}mi)")
    wind = w.get("max_wind_kt", 0)
    if wind > 20:                 flags.append(f"HIGH WIND ({wind:.0f}kt)")
    flag_str = ", ".join(flags) if flags else "Clear"
    return (
        f"  {label:12s}  severity={sev:.2f}  [{flag_str}]\n"
        f"             METAR: {raw[:80] if raw else 'unavailable'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Predict crew sequence risk for airport_A → DFW → airport_B")
    parser.add_argument("airport_a", help="Inbound airport IATA code (e.g. ORD)")
    parser.add_argument("airport_b", help="Outbound airport IATA code (e.g. LAX)")
    parser.add_argument("--month", type=int, default=None, help="Month 1-12 (default: current month)")
    parser.add_argument("--no-live", action="store_true", help="Skip live weather fetch")
    args = parser.parse_args()

    predictor = PairRiskPredictor()

    print(f"\n{'='*60}")
    print(f"  Crew Sequence Risk:  {args.airport_a.upper()} → DFW → {args.airport_b.upper()}")
    print(f"{'='*60}")

    if args.no_live or args.month:
        month = args.month or datetime.date.today().month
        result = predictor.predict_historical(args.airport_a.upper(), args.airport_b.upper(), month)
        score  = result["base_risk_score"]
        label  = result["risk_label"]
        print(f"\n  Historical risk score : {score:.3f}")
        print(f"  Risk level            : {label}")
        print(f"  Advice                : {result['advice']}")
        if result["observed_bad_rate"] is not None:
            print(f"  Observed bad rate     : {result['observed_bad_rate']:.1%}  (historical month {month})")
    else:
        result = predictor.predict_live(args.airport_a.upper(), args.airport_b.upper())
        base   = result["base_risk_score"]
        live   = result["live_risk_score"]
        print(f"\n  Historical base score : {base:.3f}")
        print(f"  Live weather penalty  : +{result['weather_penalty']:.3f}")
        print(f"  FINAL risk score      : {live:.3f}")
        print(f"  Risk level            : {result['risk_label']}")
        print(f"  Advice                : {result['advice']}")
        if result.get("observed_bad_rate"):
            print(f"  Historical bad rate   : {result['observed_bad_rate']:.1%}")
        print(f"\n  Current conditions:")
        print(_fmt_weather(args.airport_a.upper(), result["weather_airport_a"]))
        print(_fmt_weather("DFW (hub)",             result["weather_dfw"]))
        print(_fmt_weather(args.airport_b.upper(), result["weather_airport_b"]))

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
