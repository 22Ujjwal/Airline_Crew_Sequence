# AA EPPS Data Challenge — Project Guide

Complete reference for teammates joining the project. Covers environment setup,
data pipeline, model architecture, app deployment, and current status.

---

## 1. Project Overview

**Goal:** Build a risk-scoring system for American Airlines crew sequences at DFW hub.
A *sequence* is a tail-matched aircraft rotation: crew flies A→DFW (inbound), then
DFW→B (outbound) on the same aircraft with a 30–240 min turnaround. The model predicts
how likely that rotation is to experience weather disruption.

**Deliverables:**
- XGBoost classifier with isotonic calibration (scores ≈ observed disruption rate)
- Streamlit dashboard deployed on HuggingFace Spaces
- Sequence optimizer (Hungarian algorithm, minimizes total weather risk for a day's schedule)

**HuggingFace Space:** `itaykadosh/AA-EPPS-Data-Challenge`
**GitHub (primary):** `https://github.com/Optimax14/AA-EPPS-Data-Challenge`
**GitHub (team fork):** `https://github.com/22Ujjwal/Airline_Crew_Sequence`

---

## 2. Environment Setup

### Conda (recommended)

```bash
conda create -n aadata python=3.11
conda activate aadata
pip install -r requirements.txt
# Extra packages used in training/analysis (not in requirements.txt):
pip install lightgbm optuna huggingface_hub
```

### requirements.txt (for app deployment)

```
pandas>=2.0
numpy>=1.24
xgboost>=2.0
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
seaborn>=0.13
requests>=2.31
pyarrow>=14.0
tqdm>=4.66
streamlit>=1.35
plotly>=5.20
shap>=0.44
airportsdata>=20240101
pytz>=2024.1
```

### GPU (optional)

The XGBoost model uses `device="cuda"` if available. Training works fine on CPU —
just remove `device="cuda"` from `src/model.py:train()` or it falls back automatically.

---

## 3. Repository Structure

```
AAData/
├── app/                        # Streamlit dashboard
│   ├── app.py                  # Main UI (all tabs)
│   ├── predictor.py            # Model wrapper + GSOM imputation + SHAP
│   ├── optimizer.py            # Hungarian algorithm sequence optimizer
│   ├── live_flights.py         # Live schedule sources (AviationStack, BTS analog, OpenSky)
│   └── airports.py             # Airport metadata helpers
│
├── src/                        # Pipeline scripts (run in order, see §5)
│   ├── download_bts.py         # Download BTS On-Time data (DFW-filtered)
│   ├── download_bts_full.py    # Download full BTS data (all routes, for cascade chain)
│   ├── feature_engineering.py  # Build tail-matched sequences → sequence_features.parquet
│   ├── feature_engineering_duty.py  # Adds FAA Part 117 duty features (LGBM variant)
│   ├── enrich_features.py      # Add GSOM weather features (needs NOAA token)
│   ├── tail_chain_features.py  # Crew duty proxy via aircraft rotation chains
│   ├── cascade_chain_features.py    # Airport cascade propagation index
│   ├── multihop_dfw_cascade.py # Multi-hop A→DFW→B→DFW→C cascade features
│   ├── model.py                # Train XGBoost (primary model)
│   ├── model_lgbm.py           # Train LightGBM (alternate model with duty features)
│   ├── model_enhanced.py       # XGBoost + enhanced features variant
│   ├── tune_hyperparams.py     # Optuna hyperparameter search
│   ├── predict.py              # Batch prediction on new data
│   ├── ood_validation.py       # Out-of-distribution validation
│   ├── stress_test.py          # Calibration + storm replay stress tests
│   ├── weather_benchmark.py    # Benchmark GSOM vs BTS weather signal
│   ├── visualize.py            # Standalone plots
│   └── weather.py              # Weather utilities
│
├── data/
│   ├── raw/                    # BTS parquet files (not committed — large)
│   │   ├── bts_all_dfw_{year}.parquet   # All carriers, DFW-filtered (2015–2024)
│   │   └── bts_aa_dfw_{year}.parquet    # AA-only, DFW-filtered (subset of years)
│   └── processed/              # Model artifacts + feature tables (committed)
│       ├── sequence_features.parquet    # 398k tail-matched pair×year×month obs
│       ├── airport_features.parquet     # Airport-level BTS + GSOM stats
│       ├── pair_risk_scores.parquet     # 156k calibrated pair-month risk scores
│       ├── xgb_model.json               # Trained XGBoost v7
│       ├── calibration_isotonic.json    # 309-breakpoint isotonic calibration
│       ├── gsom_month_medians.json      # GSOM median imputation values
│       ├── tail_chain_features.parquet  # Tail-chain crew duty features
│       ├── airport_cascade_features.parquet
│       ├── multihop_cascade_features.parquet
│       ├── dfw_weather_monthly.parquet  # DFW hub weather by month
│       ├── monthly_weather_gsom.parquet # Raw NOAA GSOM download (cached)
│       └── model_snapshots/             # Versioned model backups
│
├── Dockerfile                  # HuggingFace Spaces deployment
├── requirements.txt
├── MODEL_REGISTRY.md           # All model versions with AUC/AP + notes
└── CLAUDE.md                   # This file
```

---

## 4. APIs & External Data

### 4.1 BTS On-Time Performance (free, no key needed)

Source: `https://www.transtats.bts.gov`

Downloaded automatically by `src/download_bts.py`. No API key required — it scrapes
the BTS bulk download portal.

```bash
conda run -n aadata python src/download_bts.py --years 2015 2016 2017 2018 2019 2022 2023 2024 --all-carriers
```

Years 2020–2021 are COVID years (available with `--include-covid` flag but excluded from
training by default due to abnormal operations).

**Already downloaded:** `data/raw/bts_all_dfw_{2015–2019,2022–2024}.parquet`
You do NOT need to re-download unless adding new years.

### 4.2 NOAA NCDC / CDO API (free, token required)

Used by `src/enrich_features.py` to fetch GSOM (Global Summary of Month) weather data.

**Get a token:** Register at `https://www.ncdc.noaa.gov/cdo-web/token`
(free, instant email delivery, rate limit: 1000 req/day)

**Token already used:** `SBRhkMknyZHEHobLChLzPRVSgtXUlmwL`

The GSOM data is already cached at `data/processed/monthly_weather_gsom.parquet` —
you don't need to re-fetch unless adding new airports or extending to post-2024 years.

If you do need to re-fetch:
```bash
conda run -n aadata python src/enrich_features.py --token YOUR_TOKEN
# Or: export NCDC_TOKEN=YOUR_TOKEN && python src/enrich_features.py
```

### 4.3 AviationStack (optional, live flight data)

Used in the Streamlit app's Optimizer tab for live AA schedule at DFW.

- Free tier: 100 requests/month
- Sign up at `https://aviationstack.com`
- Paste key in the sidebar when running the app
- **Not required** — the app falls back to BTS 2024 analog (same month + day-of-week)

### 4.4 OpenSky Network (optional, fallback)

Free, no key. Used as a last-resort fallback for live flights. Coverage is sparse (~20–40%
of AA flights). The app tries AviationStack → BTS analog → OpenSky in priority order.

### 4.5 HuggingFace (deployment)

```bash
pip install huggingface_hub
# Upload files to the Space:
python - << 'EOF'
from huggingface_hub import HfApi
api = HfApi(token="YOUR_HF_TOKEN")
api.upload_file(path_or_fileobj="app/app.py", path_in_repo="app/app.py",
                repo_id="itaykadosh/AA-EPPS-Data-Challenge", repo_type="space")
EOF
```

**Team HF token:** ask @itaykadosh for the token (HuggingFace → Settings → Access Tokens)

---

## 5. Data Pipeline (Run Order)

Run these once in sequence to rebuild everything from scratch. Most steps are already
done — only run what you actually need to update.

```
Step 1  download_bts.py          → data/raw/bts_all_dfw_*.parquet
Step 2  feature_engineering.py   → data/processed/sequence_features.parquet  ← DONE
Step 3  enrich_features.py       → adds GSOM cols to sequence_features         ← DONE
Step 4  tail_chain_features.py   → data/processed/tail_chain_features.parquet  ← DONE
Step 5  cascade_chain_features.py → airport_cascade_features.parquet           ← DONE
Step 6  multihop_dfw_cascade.py  → multihop_cascade_features.parquet           ← DONE
Step 7  model.py                 → xgb_model.json + pair_risk_scores.parquet   ← DONE
Step 8  fit_calibration          → calibration_isotonic.json                   ← DONE (see below)
```

### Commands

```bash
# Step 2: rebuild tail-matched sequences (takes ~5 min)
conda run -n aadata python src/feature_engineering.py

# Step 3: re-add GSOM weather (uses cached monthly_weather_gsom.parquet, no API call)
# Note: enrich_features.py requires --token but only fetches if cache is absent.
# To skip fetch and use cache, run the enrichment inline:
conda run -n aadata python - << 'EOF'
# (see /tmp/enrich_run.py for the standalone enrichment script that uses the cache)
EOF

# Step 7: retrain XGBoost (~3 min on GPU, ~15 min on CPU)
conda run -n aadata python src/model.py

# Step 8: refit isotonic calibration (run /tmp/fit_calibration.py)
conda run -n aadata python /tmp/fit_calibration.py
```

> **Important:** After any pipeline change, delete `data/processed/app_features_cache.parquet`
> (if it exists) so the app rebuilds its feature cache on the next launch.

### Standalone calibration script

The calibration fitting is not in `model.py` — it's a separate script at `/tmp/fit_calibration.py`.
This should be moved into the repo. It:
1. Loads the trained model + all features (same joins as `model.py:load_features()`)
2. Scores all 398k rows in 50k batches (avoids OOM)
3. Fits `IsotonicRegression(y=observed_bad_rate, sample_weight=sqrt(n_sequences))`
4. Saves `calibration_isotonic.json` (x=raw_score, y=calibrated_score breakpoints)
5. Saves `gsom_month_medians.json` (for predictor.py NaN imputation)
6. Overwrites `pair_risk_scores.parquet` with calibrated `avg_risk_score`

---

## 6. Model Architecture

### Primary: XGBoost v7 (current)

**File:** `data/processed/xgb_model.json`
**Trained by:** `src/model.py`

| Metric | Value |
|--------|-------|
| Val AUC | 0.825 |
| Val AP | 0.445 |
| Pair AUC | 0.803 |
| Positive rate | 11.4% |
| Training rows | 398,098 |
| Val split | Time-based (train < 2024, val = 2024) |

**70 features across 8 groups:**

| Group | Features | Count |
|-------|----------|-------|
| Origin BTS | weather_delay_rate, cancel_rate, avg/p75/p95 delay, NAS rate, overall stats | 8 |
| Dest BTS | same as origin but for airport B | 8 |
| Pair BTS | combined/max/min/sum weather rate, avg delay, both_high_risk | 6 |
| Temporal | Month, is_spring_summer, median_turnaround_min, season_* dummies | 6 |
| Origin GSOM | avg_wind_speed, precip_days, extreme_precip, total_precip, max_wind_gust | 5 |
| Dest GSOM | same as origin GSOM but for airport B | 5 |
| Pair GSOM | pair_max_* for each GSOM feature | 5 |
| DFW Hub | DFW_weather_delay_rate, cancel_rate, avg/p95 delay_min | 4 |
| Tail-Chain / Duty | legs_before, block_before, duty_start_hour, total_duty, fdp_util, fdp_overrun, wocl_rate, legs_after, legs_in_day, downstream_rate, cascade_late_rate/min/amplif | 13 |
| Airport Cascade | A/B ap_cascade_rate, ap_cascade_given_late, pair_cascade_product/max | 6 |
| Multi-Hop Cascade | n_hops, total_late_min, cascade_hop_rate, cascade_depth, unique_airports, recovery_rate | 8 |

**Labeling:**
- `target = 1` if `observed_bad_rate > 0.25` (>25% of tail-matched rotations disrupted)
- `scale_pos_weight = 7.73` (class imbalance compensation)
- `observed_bad_rate` = fraction of same-tail rotations with: weather delay ≥15 min on either leg, OR cascade (inbound late → late-aircraft departure on outbound)

**Calibration:**
Raw XGBoost log-odds → isotonic regression → observed bad rate scale.
Scores directly interpretable as "fraction of rotations historically disrupted."
MCO→LAX June: calibrated 9.7%, observed 11.6%.

### Alternate: LightGBM duty v5

**File:** `data/processed/lgbm_duty_model.txt`
**Trained by:** `src/model_lgbm.py`

Built on the old corridor-day dataset — **not yet rebuilt on tail-matched data**.
AUC 0.815, AP 0.844 (inflated — old 42% positive rate). Use XGBoost v7 for anything
production-facing. LGBM v5 is retained for comparison only.

---

## 7. Key Design Decisions (read this before changing anything)

### Tail-matched sequences, not corridor-days

Early versions used a "corridor-day" approach: cross-join ALL inbound flights from A with
ALL outbound flights to B on the same date. This produced ~30 "sequences" per pair-month
(≈ calendar days) but was fundamentally wrong:
- `n_sequences` was just calendar days, not real crew rotations
- Major hubs (MCO, ORD, ATL) were labeled positive ~90% of summer days because
  ANY one of dozens of flights triggering the threshold marked the whole corridor
- MCO→LAX June showed 87% risk (fabricated) vs 11.6% actual observed

Current approach links by `Tail_Number + FlightDate` with 30–240 min turnaround.
Result: mean 3.6 real rotations per pair-year-month, max 102, calibrated correctly.

### GSOM NaN imputation (not XGBoost native NaN routing)

~55% of airports have no NOAA GSOM station. Early versions let XGBoost handle NaN
natively — it learned "no GSOM station" ≈ lower risk (because small airports have fewer
weather events). This gave airports like ANC or ALB artificially low risk scores.

Fix: at inference time (`app/predictor.py:_apply_gsom_imputation()`), NaN GSOM
features are filled with month-level population medians from `gsom_month_medians.json`.
The app shows ★ and amber borders on SHAP bars for imputed features.

### Calibration scope

Isotonic regression was fitted on the **full** 398k-row dataset (no holdout for calibration).
This means calibration error on genuinely new route-months (routes that never appeared in
2015–2024) will be slightly higher. Acceptable for the competition scope.

### pair_risk_scores aggregation

`pair_risk_scores.parquet` aggregates across all years (2015–2024) per (airport_A, airport_B, Month).
`avg_risk_score` = mean calibrated score across year-months. This is what the app displays.
`avg_risk_score_raw` = pre-calibration XGBoost score (kept for diagnostics).

---

## 8. Running the App Locally

```bash
conda activate aadata
streamlit run app/app.py
# Opens at http://localhost:8501
```

The app caches heavy data loads with `@st.cache_resource` / `@st.cache_data`.
On first run it builds `data/processed/app_features_cache.parquet` (~24MB) — subsequent
runs are fast. Delete this file if you update any processed parquet files.

---

## 9. Deploying to HuggingFace

The Space rebuilds automatically when files are updated via the HF API or git push.

```bash
# Push all key app + model files:
python - << 'EOF'
from huggingface_hub import HfApi
import os

TOKEN = "YOUR_HF_TOKEN"  # ask @itaykadosh or generate at huggingface.co/settings/tokens
REPO  = "itaykadosh/AA-EPPS-Data-Challenge"
BASE  = os.path.abspath(".")  # run from project root

api = HfApi(token=TOKEN)
files = [
    "app/app.py",
    "app/predictor.py",
    "app/optimizer.py",
    "app/live_flights.py",
    "data/processed/xgb_model.json",
    "data/processed/pair_risk_scores.parquet",
    "data/processed/calibration_isotonic.json",
    "data/processed/gsom_month_medians.json",
    "data/processed/airport_features.parquet",
    "data/processed/sequence_features.parquet",
    "data/processed/tail_chain_features.parquet",
    "data/processed/airport_cascade_features.parquet",
    "data/processed/multihop_cascade_features.parquet",
    "data/processed/dfw_weather_monthly.parquet",
    "requirements.txt",
    "Dockerfile",
]
for f in files:
    path = os.path.join(BASE, f)
    if os.path.exists(path):
        print(f"Uploading {f}...")
        api.upload_file(path_or_fileobj=path, path_in_repo=f,
                        repo_id=REPO, repo_type="space", token=TOKEN)
print("Done.")
EOF
```

The Space runs the Dockerfile: `python:3.11-slim` → `pip install requirements.txt` →
`streamlit run app/app.py --server.port=7860`.

---

## 10. Current Status (as of 2026-04-14)

### Done

- [x] BTS 2015–2024 downloaded (all carriers, DFW-filtered)
- [x] Tail-matched sequence pipeline (`feature_engineering.py` rewritten)
- [x] GSOM enrichment (5 wind/precip features, cached in `monthly_weather_gsom.parquet`)
- [x] Tail-chain features (crew duty proxy via aircraft rotation)
- [x] Airport cascade propagation index
- [x] Multi-hop DFW cascade features
- [x] XGBoost v7 trained on tail-matched data (AUC 0.825, AP 0.445)
- [x] Isotonic calibration refit (MCO→LAX June: 9.7% calibrated vs 11.6% observed)
- [x] GSOM NaN imputation fairness fix
- [x] Streamlit dashboard with 6 tabs: Heatmap, Map, Query, Timeline, Optimizer, Methodology
- [x] Hungarian algorithm sequence optimizer
- [x] Live schedule integration (AviationStack / BTS analog / OpenSky)
- [x] SHAP explanations per pair-month query
- [x] HuggingFace Space deployed and current

### Pending / Nice-to-have

- [ ] **Rebuild LightGBM duty model on tail-matched data** — current LGBM was trained on
  the old corridor-day dataset (42% positive rate). Retrain with `src/model_lgbm.py`
  using the new `sequence_features.parquet`.
- [ ] **Re-run OOD validation** — `src/ood_validation.py` was last run on v1 models.
  Re-run on XGBoost v7 to get current temporal/carrier/extreme-event OOD numbers.
- [ ] **Move calibration fitting into `model.py`** — currently lives in `/tmp/fit_calibration.py`.
  Add a `fit_and_save_calibration()` function at the end of `model.py:main()`.
- [ ] **Extend BTS data to 2025** — 2025 data should be available from BTS by now.
  Run `download_bts.py --years 2025` and re-run the full pipeline.
- [ ] **Stress tests on new model** — `src/stress_test.py` (calibration decile check +
  storm replay) was last run on old models.

---

## 11. Common Gotchas

**Pandas suffix collision in merge:**
When merging two DataFrames and only one has a column (e.g., `ArrDelay` only exists in
inbound), pandas will NOT add `_x`/`_y` suffix — the column name stays unchanged.
Always pre-rename flight-specific columns before merging:
```python
ib = ib.rename(columns={"WeatherDelay": "WeatherDelay_A", "ArrDelay": "ArrDelay_A"})
ob = ob.rename(columns={"WeatherDelay": "WeatherDelay_B", "DepDelay": "DepDelay_B"})
pairs = ib.merge(ob, on=["FlightDate", "Tail_Number"])
```

**XGBoost feature name mismatch:**
If you add/remove features and try to score with an old model, XGBoost will raise
`ValueError: feature_names mismatch`. Always retrain after changing `FEATURE_COLS`.

**OOM on predict_proba:**
Full 398k-row scoring in one call killed the process. Score in batches of 50k:
```python
for start in range(0, len(df), 50_000):
    batch = df.iloc[start:start+50_000][feature_cols].astype(float)
    scores[start:start+50_000] = model.predict_proba(batch)[:, 1]
```

**Season dummy capitalization:**
`pd.get_dummies(df["Season"])` where Season has values `"Fall"/"Spring"/"Summer"/"Winter"`
produces `season_Fall` etc. (capital). The model was trained with lowercase `season_fall`.
The current `feature_engineering.py` generates the dummies correctly — don't change
the Season column value casing without retraining.

**app_features_cache.parquet:**
The app caches the joined feature table. If you update any processed parquet file and
don't delete this cache, the app will silently use stale data.
```bash
rm -f data/processed/app_features_cache.parquet
```
