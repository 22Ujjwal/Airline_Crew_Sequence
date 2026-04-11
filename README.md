# Airline Crew Sequence Risk Analysis

A data-driven framework for identifying airport pair sequences that are disproportionately likely to cause cascading weather-related delays for connecting crews. Built on BTS on-time performance data and real-time METAR weather observations.

## Problem

Airline crews often fly multi-leg sequences: **Airport A → DFW (hub) → Airport B**. When weather affects both legs simultaneously — or when a delay on the first leg propagates through the hub — the crew sequence becomes operationally risky. This project quantifies that risk at the pair level so schedulers can make informed sequencing decisions.

## Approach

1. **Historical risk profiling** — Five years of BTS on-time data (2018–2024) is used to compute per-airport weather delay rates, cancellation rates, and delay magnitude distributions for each month of the year.

2. **Pair-level feature engineering** — Airport risk profiles are combined into sequence-level features that capture multiplicative compounding risk (e.g., the product of both airports' weather delay rates), not just additive effects.

3. **Statistical significance testing** — Observed bad rates per pair×month are tested against a fleet-wide baseline using chi-square tests with Benjamini-Hochberg FDR correction and Wilson confidence intervals. Only pairs with sufficient sample size and statistically significant excess risk are flagged.

4. **Predictive modeling** — A gradient-boosted classifier trained on historical features predicts whether an unseen pair×month will be high-risk, enabling generalization beyond directly observed pairs.

5. **Real-time weather integration** — At inference time, live METAR observations from the Aviation Weather Center are fetched for all three airports (A, DFW, B) and used to adjust the historical base score upward if current conditions are severe.

## Repository Structure

```
src/
  download_bts.py          # Download BTS On-Time Performance data
  feature_engineering.py   # Build airport and pair-level features
  model.py                 # Train XGBoost classifier, generate pair risk scores
  significance.py          # Wilson CI, chi-square tests, FDR correction
  visualize.py             # All plots (pair heatmaps, volcano, CI bars, etc.)
  predict.py               # Real-time risk prediction CLI
  weather.py               # IEM historical + AWC live METAR integration

notebooks/
  analysis.ipynb           # End-to-end walkthrough with embedded visualizations

data/
  processed/               # Model artifacts, risk scores, significance results
    xgb_model.json
    airport_features.parquet
    pair_risk_scores.parquet
    pair_scores_with_significance.parquet
    sequence_features.parquet
    plots/

Crew_Sequences_Bad_Weather_Analytics_Challenge.pdf   # Original challenge brief
```

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Reproduce the full pipeline

```bash
# 1. Download 5 years of BTS data (~30 MB)
python src/download_bts.py

# 2. Build features and pair-level risk scores
python src/feature_engineering.py

# 3. Train the model and generate significance-tested pair scores
python src/model.py

# 4. Generate all visualizations
python src/visualize.py
```

### Real-time prediction

```bash
# Predict risk for ORD → DFW → MIA right now (fetches live weather)
python src/predict.py ORD MIA

# Historical-only prediction for a specific month (no API call)
python src/predict.py ORD MIA --month 7
```

**Output example:**
```
============================================================
  Crew Sequence Risk:  ORD → DFW → MIA
============================================================

  Historical base score : 0.412
  Live weather penalty  : +0.183
  FINAL risk score      : 0.595
  Risk level            : HIGH
  Advice                : Avoid if possible — weather-prone pair

  Current conditions:
  ORD           severity=0.20  [HIGH WIND (24kt)]
  DFW (hub)     severity=0.45  [THUNDERSTORM]
  MIA           severity=0.15  [FOG/MIST]
============================================================
```

### Explore interactively

```bash
jupyter lab notebooks/analysis.ipynb
```

## Risk Score Interpretation

| Score | Label    | Meaning |
|-------|----------|---------|
| 0.0–0.30 | LOW | Safe to sequence |
| 0.30–0.55 | MODERATE | Review turnaround buffer |
| 0.55–0.75 | HIGH | Avoid if possible |
| 0.75–1.0 | CRITICAL | Do not sequence |

The score combines a historical base probability (from the trained model) with a real-time weather penalty derived from current METAR severity at all three airports. The hub airport (DFW) is weighted more heavily since its weather affects both legs.

## Data Sources

| Source | What | Access |
|--------|------|--------|
| [BTS On-Time Performance](https://transtats.bts.gov) | Flight-level delay attribution by cause | Free, no key |
| [Iowa Environmental Mesonet (IEM)](https://mesonet.agron.iastate.edu) | Hourly METAR observations back to the 1970s | Free, no key |
| [Aviation Weather Center (AWC)](https://aviationweather.gov/api/data/metar) | Real-time METAR via REST API | Free, no key |

Raw BTS files are excluded from this repository (large, freely re-downloadable). Processed artifacts and the trained model are included in `data/processed/`.

## Statistical Methodology

- **Baseline**: Fleet-wide bad rate across all pair×month combinations
- **Bad sequence**: Any sequence with weather delay > 0 min or weather cancellation
- **Minimum sample**: n ≥ 30 sequences per pair×month before a pair is reported
- **Significance**: Chi-square goodness-of-fit vs baseline, corrected via Benjamini-Hochberg FDR (α = 0.05)
- **Confidence intervals**: Wilson score interval (exact binomial, 95%)
- **Effect size**: Risk ratio = observed bad rate / baseline bad rate
