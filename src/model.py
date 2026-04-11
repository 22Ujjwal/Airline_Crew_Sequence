"""
XGBoost model to classify high-risk crew sequences (A→DFW→B pairs).

Outputs:
  - Trained model saved to data/processed/xgb_model.json
  - Per-airport-pair risk scores saved to data/processed/pair_risk_scores.parquet
  - Feature importance plot
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

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


def load_features():
    path = os.path.join(PROCESSED_DIR, "sequence_features.parquet")
    df = pd.read_parquet(path)
    season_cols = [c for c in df.columns if c.startswith("season_")]
    return df, season_cols


def train(df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    # Class imbalance: weight the minority class
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance — negative: {neg:,}, positive: {pos:,}, scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",          # average precision — better for imbalanced
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    # Time-based split: train on earlier years, validate on most recent year
    train_mask = df["Year"] < df["Year"].max()
    val_mask = df["Year"] == df["Year"].max()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluation
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    print("\n--- Validation Results ---")
    print(f"ROC-AUC:           {roc_auc_score(y_val, y_pred_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["low_risk", "high_risk"]))

    return model


def plot_feature_importance(model: xgb.XGBClassifier, feature_cols: list[str]):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 20 XGBoost Feature Importances\nCrew Sequence Weather Risk")
    plt.tight_layout()
    out = os.path.join(PROCESSED_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    print(f"Feature importance plot saved → {out}")


def score_all_pairs(model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: list[str]):
    """
    Aggregate sequence-level predictions into per-pair (airport_A, airport_B, Month)
    risk scores. This is the final deliverable: which pairs are risky, in which months.
    """
    X = df[feature_cols].astype(float)
    df = df.copy()
    df["risk_score"] = model.predict_proba(X)[:, 1]

    pair_scores = (
        df.groupby(["airport_A", "airport_B", "Month"])
        .agg(
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            n_sequences=("n_sequences", "sum") if "n_sequences" in df.columns else ("risk_score", "count"),
            observed_bad_rate=("observed_bad_rate", "mean") if "observed_bad_rate" in df.columns else ("target", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )

    out = os.path.join(PROCESSED_DIR, "pair_risk_scores.parquet")
    pair_scores.to_parquet(out, index=False)
    print(f"\nPair risk scores saved → {out}")
    print("\nTop 20 riskiest airport pairs:")
    print(pair_scores.head(20).to_string(index=False))
    return pair_scores


def main():
    df, season_cols = load_features()
    feature_cols = FEATURE_COLS + season_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    model = train(df, feature_cols)

    # Save model
    model_path = os.path.join(PROCESSED_DIR, "xgb_model.json")
    model.save_model(model_path)
    print(f"\nModel saved → {model_path}")

    plot_feature_importance(model, feature_cols)
    score_all_pairs(model, df, feature_cols)


if __name__ == "__main__":
    main()
