# Model Registry — AA Crew Sequence Risk

Tracks all trained model versions, their evaluation conditions, and artifacts.

**Evaluation notes:**
- `thresh=median` — old label: `observed_bad_rate > median` → 50% positive (meaningless split)
- `thresh=0.25` — new label: `observed_bad_rate > 0.25` → 33.4% positive (genuinely bad pairs)
- AUC numbers are NOT directly comparable across threshold schemes
- Val set = 2024 holdout (train on all prior years)

---

## Current Best Models

| Model | File | Val AUC | Val AP | Thresh | Features | Date |
|-------|------|---------|--------|--------|----------|------|
| **XGBoost v7** | `xgb_model.json` | **0.825** | 0.445 | 0.25 | BTS+GSOM+DFW+tail-chain+cascade+mhc (tail-matched seqs) | 2026-04-13 |
| **LGBM duty v5** | `lgbm_duty_model.txt` | **0.815** | 0.844 | 0.25 | BTS+GSOM+DFW+duty+tail-chain+cascade+mhc | 2026-04-12 |

> XGBoost v7 is trained on tail-matched sequences (398k obs, 11.4% positive rate, calibrated 1.2–58.3%).
> LGBM duty v5 is the prior version (corridor-day model, 42% positive, not yet rebuilt on tail-matched data).

---

## Full Version History

### v7 (XGBoost) — 2026-04-13 (current XGB)

**Key change:** Rebuilt sequence_features.parquet using tail-matched aircraft rotations
instead of corridor-day cross-join. Each row now represents a real A→DFW→B rotation
on the same tail number, same date, 30–240 min turn.

| Metric | Value |
|--------|-------|
| Val AUC | 0.825 |
| Val AP | 0.445 |
| Positive rate | 11.4% (vs 42.1% before) |
| scale_pos_weight | 7.73 |
| Training rows | 398,098 |
| Calibration | Isotonic, 309 breakpoints, fitted on full dataset |
| Calibrated range | 1.2% – 58.3% (≈ observed bad rate) |
| MCO→LAX June | calibrated 9.7%, observed 11.6% |

---

### v6 (XGBoost) — 2026-04-12

**Changes from XGB v2:**
- Added all features previously LGBM-only: tail-chain (16), airport cascade (8), multi-hop DFW cascade (9)
- `load_features()` now joins `tail_chain_features.parquet`, `airport_cascade_features.parquet`, `multihop_cascade_features.parquet`
- AUC flat vs v2 — confirms cascade features add narrative, not discriminative signal for XGB either

| Model | Val AUC | Val AP | Snapshot file |
|-------|---------|--------|---------------|
| XGBoost v3 | 0.833 | 0.830 | `model_snapshots/xgb_v3_fullfeatures_thresh025.json` |

---

### v5 — 2026-04-12 (current LGBM)

**Changes from v4:**
- Added multi-hop DFW cascade features (A→DFW→B→DFW→C→DFW→D...) from existing data
  - `mhc_n_hops_mean/max`, `mhc_total_late_min_mean/p75`, `mhc_cascade_hop_rate`, `mhc_cascade_depth_mean`, `mhc_unique_airports_mean`, `mhc_recovery_rate`
- AUC plateau reached — cascade features add report narrative value, not discriminative signal
- BTS full download (for non-DFW cascade B→C): blocked by BTS 503 outage; wired into model_lgbm.py — will auto-activate when `cascade_chain_features.parquet` is present

| Model | Val AUC | Val AP | Snapshot file |
|-------|---------|--------|---------------|
| LGBM duty v5 | 0.815 | 0.844 | `model_snapshots/lgbm_duty_v5_multihop_thresh025.txt` |

---

### v4 — 2026-04-12

**Changes from v3:**
- Added downstream cascade features: A→DFW→B→DFW three-hop chain via tail matching
  - `tc_downstream_rate`, `tc_cascade_late_rate`, `tc_cascade_late_min`, `tc_cascade_amplif_mean`
- Added airport-level cascade propagation index (all airports, no tail matching)
  - `A/B_ap_cascade_rate`, `A/B_ap_cascade_given_late`, `pair_cascade_product`, `pair_max_cascade_rate`
- 32-38% of sequences have identifiable B→DFW cascade leg; `ap_cascade_given_late` = 0.41 mean

| Model | Val AUC | Val AP | Snapshot file |
|-------|---------|--------|---------------|
| LGBM duty v4 | 0.815 | 0.844 | `model_snapshots/lgbm_duty_v4_cascade_thresh025.txt` |

---

### v3 — 2026-04-12

**Changes from v2:**
- Added tail-chain features: aircraft rotation as crew duty proxy (11 features)
- Same-FlightDate tail matching: `legs_before`, `block_min_before`, `duty_start_hour`, `total_duty_min`, `total_duty_p75`, `fdp_util_mean`, `fdp_util_p75`, `fdp_overrun_rate`, `wocl_rate`, `legs_after_mean`, `legs_in_day_mean`
- FAA Part 117 FDP limit table (simplified, by report hour + segment count)
- 24.2% pair×year coverage (high-volume routes); sparse pairs get NaN → LGBM handles natively

| Model | Val AUC | Val AP | Snapshot file |
|-------|---------|--------|---------------|
| LGBM duty v3 | 0.815 | 0.843 | `model_snapshots/lgbm_duty_v3_tailchain_thresh025.txt` |

---

### v2 — 2026-04-12

**Changes from v1:**
- Label threshold changed: median → 0.25 (absolute bad-rate)
- Added GSOM wind/precip features: `A/B_avg_wind_speed`, `precip_days`, `extreme_precip`, `total_precip`, `max_wind_gust` + `pair_max_*` variants
- Added DFW hub weather: `DFW_weather_delay_rate`, `DFW_weather_cancel_rate`, `DFW_avg_weather_delay_min`, `DFW_p95_weather_delay_min` (cached in `dfw_weather_monthly.parquet`)
- LGBM: GSOM cols backfilled from `sequence_features.parquet` at load time

| Model | Val AUC | Val AP | Snapshot file |
|-------|---------|--------|---------------|
| XGBoost v2 | 0.833 | 0.829 | `model_snapshots/xgb_v2_gsom_dfw_thresh025.json` |
| LGBM duty v2 | 0.813 | 0.841 | `model_snapshots/lgbm_duty_v2_gsom_dfw_thresh025.txt` |

**Feature set (XGB v2, 36 features):**
- 16 BTS airport-level (A and B weather/cancel/NAS rates)
- 6 pair-level BTS (combined, max, min, sum rates, avg delay, both_high_risk)
- 3 temporal (Month, is_spring_summer, median_turnaround_min)
- 10 GSOM wind/precip (A×5, B×5; ~55% airport coverage; NaN handled natively)
- 5 pair GSOM max aggregates
- 4 DFW hub weather (by month, full coverage)
- Season dummies (dynamic)

**LGBM duty v2 adds 18 duty/fatigue features on top of above:**
`A/B_dep_hour_median`, `late_dep_rate`, `early_dep_rate`, `avg_block_min`, `late_aircraft_delay_rate`, `avg_late_aircraft_min`, `tight_connection_rate`, `very_tight_rate`, `cascade_risk`, `total_duty_block_min`, `duty_overrun_risk`, `late_dep_sequence`

---

### v1 — 2026-04-11 (pre-GSOM, median threshold)

| Model | Val AUC | Val AP | Notes | Snapshot file |
|-------|---------|--------|-------|---------------|
| XGBoost base | 0.804 | 0.882 | 25 features, median thresh | *(not snapshotted — overwritten by v2)* |
| XGBoost tuned (Optuna) | 0.806 | — | Optuna 50 trials, same feature set | `model_snapshots/xgb_tuned_v1_optuna_medthresh.json` |
| LGBM tuned (Optuna) | 0.804 | — | Optuna, duty features, median thresh | `model_snapshots/lgbm_tuned_v1_optuna_medthresh.txt` |
| LGBM duty v1 | 0.804 | 0.882 | Duty features, median thresh | *(not snapshotted — overwritten by v2)* |

**OOD results (v1 models, median threshold):**

| Test | ROC-AUC | Avg Precision |
|------|---------|---------------|
| In-dist 2024 holdout | 0.804 | 0.882 |
| Temporal OOD — all-carrier DFW 2015 | 0.768 | 0.767 |
| Extreme Event OOD — Jan/Feb 2015 blizzards | 0.750 | 0.844 |
| Carrier OOD — non-AA/DFW 2022–2023 | 0.723 | 0.473 |

> OOD not yet rerun on v2 models.

---

## How to Restore a Snapshot

```bash
# Restore XGBoost v2
cp data/processed/model_snapshots/xgb_v2_gsom_dfw_thresh025.json data/processed/xgb_model.json

# Restore LGBM duty v2
cp data/processed/model_snapshots/lgbm_duty_v2_gsom_dfw_thresh025.txt data/processed/lgbm_duty_model.txt

# Restore Optuna-tuned XGBoost v1
cp data/processed/model_snapshots/xgb_tuned_v1_optuna_medthresh.json data/processed/xgb_tuned_model.json
```

---

## Snapshot Convention

When overwriting a model file, first copy it here with the pattern:

```
{model_name}_v{N}_{key_change}_{label_scheme}.{ext}
```

Examples:
- `xgb_v3_geographic_thresh025.json`
- `lgbm_duty_v3_calibrated_thresh025.txt`

Then add a row to this table before training the new version.
