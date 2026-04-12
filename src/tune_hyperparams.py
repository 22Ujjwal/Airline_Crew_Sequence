"""
Optuna hyperparameter tuning for LightGBM (GPU) on the duty-aware features.

Runs N trials of Bayesian optimization, each trial trains on 2018-2023
and evaluates on 2024 holdout. Saves best params + retrained final model.

Usage:
  python tune_hyperparams.py              # 50 trials, lgbm
  python tune_hyperparams.py --trials 100
  python tune_hyperparams.py --model xgb  # tune XGBoost instead
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PLOTS    = os.path.join(PROC_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

BASE_FEATURES = [
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
DUTY_FEATURES = [
    "A_dep_hour_median", "A_late_dep_rate", "A_early_dep_rate",
    "A_avg_block_min", "A_late_aircraft_delay_rate", "A_avg_late_aircraft_min",
    "B_dep_hour_median", "B_late_dep_rate", "B_early_dep_rate",
    "B_avg_block_min", "B_late_aircraft_delay_rate", "B_avg_late_aircraft_min",
    "tight_connection_rate", "very_tight_rate",
    "cascade_risk", "total_duty_block_min", "duty_overrun_risk", "late_dep_sequence",
]


def load_data(use_duty: bool = True):
    path = os.path.join(PROC_DIR,
                        "sequence_features_duty.parquet" if use_duty
                        else "sequence_features.parquet")
    df = pd.read_parquet(path)
    season_cols = [c for c in df.columns if c.startswith("season_")]
    all_feats   = BASE_FEATURES + (DUTY_FEATURES if use_duty else []) + season_cols
    feat_cols   = [c for c in all_feats if c in df.columns]
    return df, feat_cols


def lgbm_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "device": "cuda",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])


def xgb_objective(trial, X_train, y_train, X_val, y_val):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": neg / pos,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 40,
        "random_state": 42,
        "n_jobs": -1,
    }
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])


def run_tuning(model_type: str, n_trials: int, use_duty: bool):
    print(f"\nLoading {'duty-enriched' if use_duty else 'base'} features...")
    df, feat_cols = load_data(use_duty)
    print(f"Dataset: {df.shape}  |  Features: {len(feat_cols)}")

    train_mask = df["Year"] < 2024
    val_mask   = df["Year"] == 2024
    X_train = df[train_mask][feat_cols].astype(float)
    y_train = df[train_mask]["target"].astype(int)
    X_val   = df[val_mask][feat_cols].astype(float)
    y_val   = df[val_mask]["target"].astype(int)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}")
    print(f"Model: {model_type.upper()}  |  Trials: {n_trials}\n")

    objective = (lgbm_objective if model_type == "lgbm" else xgb_objective)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Save best params
    params_path = os.path.join(PROC_DIR, f"best_params_{model_type}.json")
    with open(params_path, "w") as f:
        json.dump({"best_auc": study.best_value, "params": study.best_params}, f, indent=2)
    print(f"Params saved → {params_path}")

    # Retrain final model with best params on full train set
    print("\nRetraining final model with best params...")
    best = study.best_params

    if model_type == "lgbm":
        final = lgb.LGBMClassifier(
            **best, class_weight="balanced",
            device="cuda", random_state=42, n_jobs=-1, verbose=-1,
        )
        final.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        model_path = os.path.join(PROC_DIR, f"lgbm_tuned_model.txt")
        final.booster_.save_model(model_path)
    else:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        final = xgb.XGBClassifier(
            **best, scale_pos_weight=neg/pos,
            tree_method="hist", device="cuda",
            eval_metric="aucpr", early_stopping_rounds=50,
            random_state=42, n_jobs=-1,
        )
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_path = os.path.join(PROC_DIR, f"xgb_tuned_model.json")
        final.save_model(model_path)

    proba = final.predict_proba(X_val)[:, 1]
    final_auc = roc_auc_score(y_val, proba)
    final_ap  = average_precision_score(y_val, proba)
    print(f"\nFinal tuned model — ROC-AUC: {final_auc:.4f}  AP: {final_ap:.4f}")
    print(f"Model saved → {model_path}")

    # Plot: optimization history + param importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trial history
    ax = axes[0]
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]
    ax.plot(values, alpha=0.4, color="steelblue", label="Trial AUC")
    ax.plot(best_so_far, color="red", linewidth=2, label="Best so far")
    ax.axhline(final_auc, color="green", linestyle="--", label=f"Final tuned={final_auc:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("ROC-AUC (2024 holdout)")
    ax.set_title(f"Optuna optimization history\n{model_type.upper()} — {n_trials} trials")
    ax.legend()
    ax.grid(alpha=0.3)

    # Param importance
    ax = axes[1]
    try:
        importances = optuna.importance.get_param_importances(study)
        params_sorted = list(importances.keys())[:10]
        vals_sorted   = [importances[p] for p in params_sorted]
        ax.barh(params_sorted[::-1], vals_sorted[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter importance\n(Fanova method)")
        ax.grid(axis="x", alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "Param importance\nnot available", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS, f"optuna_{model_type}_tuning.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved → {plot_path}")

    return study, final_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--model",  choices=["lgbm", "xgb", "both"], default="lgbm")
    parser.add_argument("--no-duty", action="store_true",
                        help="Use base features only (no duty enrichment)")
    args = parser.parse_args()

    use_duty = not args.no_duty
    models   = ["lgbm", "xgb"] if args.model == "both" else [args.model]

    results = {}
    for m in models:
        _, auc = run_tuning(m, args.trials, use_duty)
        results[m] = auc

    print("\n" + "="*50)
    print("TUNING SUMMARY")
    print("="*50)
    for m, auc in results.items():
        print(f"  {m.upper()}: {auc:.4f}")


if __name__ == "__main__":
    main()
