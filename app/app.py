"""
AA Crew Sequence Risk — Interactive Dashboard
=============================================
Run with:
    conda run -n aadata streamlit run app/app.py
"""
from __future__ import annotations
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Global chart theme — transparent bg so dark/light mode both work
pio.templates["aa_theme"] = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.3)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.3)"),
    )
)
pio.templates.default = "plotly+aa_theme"

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app.predictor import RiskPredictor, build_features_df, FEATURE_LABELS
from app import airports as ap_meta
from app import live_flights as lf
from app import optimizer as opt

PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RAW       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

st.set_page_config(
    page_title="AA DFW Crew Risk",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-badge-high     { background:#d62728; color:white; padding:4px 12px; border-radius:8px; font-weight:bold; font-size:1.1em; }
.risk-badge-moderate { background:#ff7f0e; color:white; padding:4px 12px; border-radius:8px; font-weight:bold; font-size:1.1em; }
.risk-badge-low      { background:#2ca02c; color:white; padding:4px 12px; border-radius:8px; font-weight:bold; font-size:1.1em; }
.metric-card { background:#f0f2f6; padding:16px; border-radius:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)


def tip(label: str, tooltip: str) -> str:
    """Return HTML snippet: label with hover tooltip (use inside st.markdown unsafe_allow_html=True)."""
    return (
        f'<abbr title="{tooltip}" style="cursor:help;text-decoration:underline dotted;'
        f'text-decoration-color:rgba(128,128,128,0.55)">'
        f'{label}&thinsp;<sup style="font-size:0.65em;opacity:0.65">ℹ</sup></abbr>'
    )


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model & features...")
def get_predictor() -> RiskPredictor:
    df = build_features_df()
    return RiskPredictor(df)


@st.cache_data(show_spinner="Loading risk scores...")
def get_pair_scores() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROCESSED, "pair_risk_scores.parquet"))


@st.cache_data(show_spinner="Loading 2024 schedule data...")
def get_bts_2024() -> pd.DataFrame:
    path = os.path.join(RAW, "bts_all_dfw_2024.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df[df["Cancelled"] != 1].copy()
    df["CRSDepTime"] = pd.to_numeric(df["CRSDepTime"], errors="coerce")
    df["CRSArrTime"] = pd.to_numeric(df["CRSArrTime"], errors="coerce")
    df["DepDelayMinutes"] = df["DepDelayMinutes"].fillna(0)
    df["ArrDelayMinutes"] = df["ArrDelayMinutes"].fillna(0)
    return df


@st.cache_data
def get_map_group(month: int, role: str) -> pd.DataFrame:
    """Cached per-month airport risk aggregation for the map tab."""
    scores = get_pair_scores()
    ms = scores[scores["Month"] == month]
    if role == "origin":
        grp = (ms.groupby("airport_A")
               .agg(avg_risk=("avg_risk_score","mean"), n_pairs=("airport_B","count"))
               .reset_index().rename(columns={"airport_A":"airport"}))
        wp = (ms.loc[ms.groupby("airport_A")["avg_risk_score"].idxmax(), ["airport_A","airport_B"]]
              .rename(columns={"airport_A":"airport","airport_B":"worst_partner"}))
    else:
        grp = (ms.groupby("airport_B")
               .agg(avg_risk=("avg_risk_score","mean"), n_pairs=("airport_A","count"))
               .reset_index().rename(columns={"airport_B":"airport"}))
        wp = (ms.loc[ms.groupby("airport_B")["avg_risk_score"].idxmax(), ["airport_B","airport_A"]]
              .rename(columns={"airport_B":"airport","airport_A":"worst_partner"}))
    return grp.merge(wp, on="airport", how="left")


@st.cache_data
def get_scores_indexed() -> pd.DataFrame:
    """Cached set_index — avoids re-running on every Streamlit rerender."""
    return get_pair_scores().set_index(["airport_A", "airport_B", "Month"])


@st.cache_data(show_spinner=False)
def get_eval_data() -> dict:
    """Compute PR/ROC curves and calibration from pair_risk_scores (pair-level aggregation)."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    _s = get_pair_scores().dropna(subset=["avg_risk_score", "observed_bad_rate"])
    _y = (_s["observed_bad_rate"] > 0.25).astype(int)
    _p = _s["avg_risk_score"]
    _fpr, _tpr, _ = roc_curve(_y, _p)
    _prec, _rec, _ = precision_recall_curve(_y, _p)
    # Calibration: decile buckets of model score vs observed bad rate
    _s2 = _s.copy()
    _s2["decile"] = pd.qcut(_p, 10, labels=False)
    _cal = (_s2.groupby("decile")
            .agg(mean_score=("avg_risk_score", "mean"),
                 mean_obs=("observed_bad_rate", "mean"),
                 n=("avg_risk_score", "count"))
            .reset_index())
    return {
        "fpr": _fpr, "tpr": _tpr,
        "prec": _prec, "rec": _rec,
        "auc": float(roc_auc_score(_y, _p)),
        "ap":  float(average_precision_score(_y, _p)),
        "cal": _cal,
        "scores": _s,
    }


@st.cache_data(show_spinner=False)
def get_feature_importance_df() -> pd.DataFrame:
    """Load XGBoost model and extract feature importances with group labels."""
    import xgboost as _xgb
    _m = _xgb.XGBClassifier()
    _m.load_model(os.path.join(PROCESSED, "xgb_model.json"))
    _fnames = _m.get_booster().feature_names
    _fi     = _m.feature_importances_

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

    _df = pd.DataFrame({
        "feature":    _fnames,
        "importance": _fi,
        "label":      [FEATURE_LABELS.get(f, f) for f in _fnames],
        "group":      [_group(f) for f in _fnames],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    _df["rank"] = _df.index + 1
    return _df


@st.cache_data
def get_airport_df(codes: tuple) -> pd.DataFrame:
    return ap_meta.build_airport_df(list(codes))


# ── Helpers ──────────────────────────────────────────────────────────────────
def risk_badge(label: str) -> str:
    cls = {
        "HIGH RISK": "risk-badge-high",
        "MODERATE RISK": "risk-badge-moderate",
        "LOW RISK": "risk-badge-low",
    }.get(label, "risk-badge-low")
    return f'<span class="{cls}">{label}</span>'


def score_to_color(score: float) -> str:
    """Interpolate red–yellow–green based on risk score."""
    if score >= 0.70:
        return "#d62728"
    if score >= 0.40:
        return "#ff7f0e"
    return "#2ca02c"


def gauge_chart(risk_score: float, title: str = "Risk Score") -> go.Figure:
    label = ("HIGH RISK" if risk_score >= 0.70 else
             "MODERATE RISK" if risk_score >= 0.40 else "LOW RISK")
    color = score_to_color(risk_score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"<b>{title}</b><br><span style='color:{color}'>{label}</span>",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.35},
            "steps": [
                {"range": [0, 40],  "color": "rgba(44,160,44,0.15)"},
                {"range": [40, 70], "color": "rgba(255,127,14,0.15)"},
                {"range": [70, 100],"color": "rgba(214,39,40,0.15)"},
            ],
            "threshold": {
                "line": {"color": "rgba(150,150,150,0.8)", "width": 3},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def shap_bar_chart(shap_df: pd.DataFrame) -> go.Figure:
    shap_df = shap_df.sort_values("shap_value")
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["label"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<br>Value: %{customdata:.3f}<extra></extra>",
        customdata=shap_df["feature_value"],
    ))
    fig.update_layout(
        title="Feature Contributions (SHAP Values)<br>"
              "<sup>Red = increases risk | Green = decreases risk</sup>",
        xaxis_title="SHAP Value (impact on model output)",
        height=max(350, len(shap_df) * 28),
        margin=dict(l=10, r=80, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#888"),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/American_Airlines_logo_2013.svg/320px-American_Airlines_logo_2013.svg.png", width=180)
    st.title("AA DFW Crew Risk")
    st.caption("Weather-driven crew sequence risk scoring for A→DFW→B routes")
    st.divider()
    st.markdown("**Model:** XGBoost v3  \n**Threshold:** 25% bad-rate  \n**Val AUC:** 0.833  \n**Val AP:** 0.830")
    st.divider()
    st.subheader("Live Schedule API")
    aviationstack_key = st.text_input(
        "AviationStack API Key",
        type="password",
        placeholder="Paste key for live AA schedule...",
        help="Free tier at aviationstack.com — 100 req/month. Leave blank to use BTS 2024 analog.",
    )
    if aviationstack_key:
        st.success("Live API key set")
    else:
        st.info("No key → BTS 2024 analog used")
    st.divider()
    st.caption("Data: BTS 2015–2024 · GSOM · FAA Part 117")


# ── Tab layout ───────────────────────────────────────────────────────────────
tab_overview, tab_dash, tab_sched, tab_optim, tab_query, tab_map = st.tabs([
    "📋 Methodology",
    "📊 Risk Dashboard",
    "🛫 DFW Schedule",
    "⚡ Sequence Optimizer",
    "🔍 Pair Risk Query",
    "🗺️ Airport Risk Map",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 0: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("📋 Methodology & Technical Model Report")
    st.caption("A full technical account of the data pipeline, feature engineering, model specification, and evaluation.")

    # ── Top model card ────────────────────────────────────────────────────────
    _mc = st.columns(6)
    for _col, (_lbl, _val) in zip(_mc, [
        ("Algorithm",   "XGBoost v3"),
        ("Val AUC",     "0.833"),
        ("Val AP",      "0.830"),
        ("Features",    "70"),
        ("Train rows",  "~400k"),
        ("Val split",   "Time-based"),
    ]):
        _col.markdown(
            f'<div style="border:1px solid rgba(128,128,128,0.25);border-radius:8px;'
            f'padding:12px 8px;text-align:center">'
            f'<div style="font-size:0.75em;opacity:0.6;text-transform:uppercase;letter-spacing:0.05em">{_lbl}</div>'
            f'<div style="font-size:1.3em;font-weight:700;margin-top:4px">{_val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Sankey ───────────────────────────────────────────────────────
    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=["BTS 2015–2024\n(Flight Ops)", "GSOM Weather\n(NOAA Monthly)",
                   "Tail-Chain\nRotations", "Feature Engineering\n(70 features)",
                   "XGBoost v3\nClassifier", "Pair Risk Scores\n(A×B×month)",
                   "SHAP Explanations", "Sequence Optimizer\n(Hungarian Alg.)",
                   "Risk Dashboard"],
            color=["#005EB8","#1a7a4a","#8B4513","#7B2D8B","#C41E3A",
                   "#2ca02c","#ff7f0e","#555555","#005EB8"],
            pad=24, thickness=22,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            hovertemplate="<b>%{label}</b><extra></extra>",
        ),
        link=dict(
            source=[0, 0, 1, 2, 3, 4, 4, 5, 5],
            target=[3, 8, 3, 3, 4, 5, 6, 7, 8],
            value= [45, 10, 20, 25, 100, 60, 40, 30, 30],
            color=["rgba(0,94,184,0.25)","rgba(0,94,184,0.15)",
                   "rgba(26,122,74,0.25)","rgba(139,69,19,0.25)",
                   "rgba(123,45,139,0.3)","rgba(196,30,58,0.3)",
                   "rgba(196,30,58,0.2)","rgba(44,160,44,0.3)",
                   "rgba(44,160,44,0.25)"],
            hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><extra></extra>",
        ),
    ))
    fig_sankey.update_layout(
        title="End-to-End Data & Model Pipeline",
        height=340, margin=dict(t=50, b=10, l=10, r=10),
        font=dict(size=11),
    )
    st.plotly_chart(fig_sankey, width='stretch')
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Problem Formulation ───────────────────────────────────────
    with st.expander("**1 · Problem Formulation**", expanded=True):
        _pf1, _pf2 = st.columns([3, 2])
        with _pf1:
            st.markdown("""
American Airlines operates **~900 daily flights** through Dallas/Fort Worth (DFW).
A crew sequence is the atom of scheduling: a pilot or flight attendant arrives on
an inbound flight from airport **A**, turns at DFW, then departs on an outbound
to airport **B**. Weather disruptions at A, DFW, or B shatter the day's roster —
triggering FAA Part 117 rest violations, repositioning costs, and cascading cancellations.

**Formal task.** Given the triplet (airport_A, airport_B, month), predict whether the
sequence A → DFW → B is *systematically disrupted* — i.e., whether its historical
weather disruption rate exceeds a material threshold.
            """)
            st.markdown("**Observed disruption rate for a pair-month cell:**")
            st.latex(r"""
\text{bad\_rate}(A,\,B,\,m) =
\frac{\bigl|\bigl\{s \in \mathcal{S}_{A,B,m}
       \;:\; \Delta_s \geq 15\,\text{min}
       \;\lor\; \mathrm{cancel}(s)\bigr\}\bigr|}
     {|\mathcal{S}_{A,B,m}|}
""")
            st.markdown("**Binary label and classification target:**")
            st.latex(r"""
y_{A,B,m} = \begin{cases}
1 & \text{if } \text{bad\_rate}(A,B,m) > 0.25 \\
0 & \text{otherwise}
\end{cases}
""")
            st.markdown("**Model output:**")
            st.latex(r"""
\hat{p}_{A,B,m} = P\!\left(y=1 \;\middle|\; \mathbf{x}_{A,B,m}\right) \in [0,\,1],
\quad \mathbf{x} \in \mathbb{R}^{70}
""")
        with _pf2:
            st.markdown("**The model is used in two distinct modes:**")
            st.markdown("""
| Mode | Usage |
|---|---|
| **Pair scoring** | Absolute risk gauge for any A→DFW→B pair in any month |
| **Cost matrix** | Relative ranking as input to the Hungarian-algorithm optimizer |

**Threshold rationale: 0.25**

The 0.25 bad-rate threshold was chosen after examining the full distribution of
pair-month disruption rates. At the 25th percentile the tail of high-disruption
cells separates from the bulk. A 50% threshold would create a spurious 50/50 split
with no operational meaning; 0.25 captures pairs that are *materially* elevated,
yielding a 42.1% positive rate.

**Turnaround constraints**

Sequences are constructed by linking inbound and outbound legs on the same
tail number with a turnaround window of **30–240 minutes** — the FAA minimum
crew turn plus an operational ceiling beyond which a new crew is typically assigned.
            """)

    # ── Section 2: Dataset ───────────────────────────────────────────────────
    with st.expander("**2 · Dataset**", expanded=True):
        _d1, _d2 = st.columns([3, 2])
        with _d1:
            st.markdown("""
**Primary source — Bureau of Transportation Statistics (BTS) On-Time Performance**
- Years: 2015–2024 (10 years)
- Scope: all AA flights departing or arriving DFW (not just AA — used for hub-load features)
- Key fields used: `Tail_Number`, `FlightDate`, `Origin`, `Dest`, `CRSDepTime`,
  `CRSArrTime`, `WeatherDelay`, `NASDelay`, `Cancelled`, `CancellationCode`

**Sequence construction.** Inbound (→DFW) and outbound (DFW→) legs are linked by
`Tail_Number` on the same calendar date with a turnaround window of **30–240 minutes**
(FAA minimum turn + operational ceiling). This produces ~400k unique
`(airport_A, airport_B, month, year)` observations across 9 years of training data.

**Secondary source — NOAA GSOM (Global Summary of Month)**
- Monthly climate normals: precipitation, wind speed/gust, extreme-event counts
- Coverage: ~55% of US airports have a nearby station with complete records
- XGBoost handles missing GSOM data natively via built-in NaN routing in split
  decisions — airports without GSOM still participate in all non-GSOM splits

**Labeling.** A sequence is *disrupted* if its weather delay ≥ 15 min or it was
cancelled with code "B" (weather). `observed_bad_rate` is the fraction of sequences
in a `(pair, month)` cell that are disrupted. The binary label `y = 1` if this rate
exceeds **0.25** — chosen to reflect a materially elevated risk level without
forcing a 50/50 split. The resulting class balance is **42.1% positive**.
            """)
        with _d2:
            _ds_rows = [
                ("BTS years",          "2015–2024"),
                ("Raw flight records",  "~8.5M"),
                ("Sequence pairs built","~429k obs"),
                ("Unique pair-months",  "~40k"),
                ("Unique airports A",   "~250"),
                ("Unique airports B",   "~250"),
                ("Positive rate",       "42.1%"),
                ("Threshold",           "25% bad rate"),
                ("Turnaround window",   "30–240 min"),
                ("GSOM airport cov.",   "~55%"),
            ]
            _ds_df = pd.DataFrame(_ds_rows, columns=["Property", "Value"])
            st.dataframe(_ds_df, hide_index=True, width='stretch', height=370)

    # ── Section 3: Model Architecture ────────────────────────────────────────
    with st.expander("**3 · Model Architecture & Training**", expanded=True):
        _m1, _m2 = st.columns(2)
        with _m1:
            st.markdown("**XGBoost Gradient Boosted Trees — Objective Function**")
            st.markdown("The model minimizes a regularized additive loss over K trees:")
            st.latex(r"""
\mathcal{L}(\phi) = \sum_{i=1}^{n} \ell\!\left(y_i,\, \hat{y}_i^{(K)}\right)
                  + \sum_{k=1}^{K} \Omega(f_k)
""")
            st.markdown("where the log-loss for binary classification is:")
            st.latex(r"""
\ell\!\left(y_i, \hat{y}_i\right) =
-\,y_i \log \hat{p}_i - (1-y_i)\log(1-\hat{p}_i),
\quad
\hat{p}_i = \sigma\!\left(\hat{y}_i^{(K)}\right)
""")
            st.markdown("and the regularization penalty on tree $f_k$ is:")
            st.latex(r"""
\Omega(f_k) = \gamma\, T_k + \frac{1}{2}\,\lambda\,\|\mathbf{w}_k\|^2
""")
            st.markdown(r"($T_k$ = number of leaves, $\mathbf{w}_k$ = leaf weight vector). The optimal leaf weight in each node is derived analytically via second-order Taylor expansion:")
            st.latex(r"""
w_j^* = -\,\frac{\displaystyle\sum_{i \in I_j} g_i}
               {\displaystyle\sum_{i \in I_j} h_i + \lambda}
""")
            st.markdown(r"where $g_i = \partial_{\hat{y}} \ell$ and $h_i = \partial^2_{\hat{y}} \ell$ are the first and second gradients. **Class imbalance** is corrected by re-weighting positive gradients:")
            st.latex(r"""
\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}
= \frac{248{,}421}{180{,}695} \approx 1.374
""")
        with _m2:
            st.markdown("**Hyperparameters**")
            _hp = pd.DataFrame([
                ("n_estimators",       "500",    "Hard cap; early stopping governs actual tree count"),
                ("early_stopping_rounds","30",   "Halt if val AUCPR doesn't improve for 30 rounds"),
                ("max_depth",          "6",      "Sufficient for 6-way interaction features"),
                ("learning_rate (η)",  "0.05",   "Slow shrinkage → lower variance"),
                ("subsample",          "0.8",    "Stochastic row sampling per tree"),
                ("colsample_bytree",   "0.8",    "Feature sampling per tree"),
                ("eval_metric",        "aucpr",  "Average Precision — better for imbalanced targets"),
                ("tree_method",        "hist",   "Histogram splits — O(n·b); GPU-accelerated"),
                ("device",             "cuda",   "NVIDIA GPU training"),
                ("random_state",       "42",     "Reproducibility"),
            ], columns=["Parameter", "Value", "Rationale"])
            st.dataframe(_hp, hide_index=True, width='stretch', height=370)
            st.markdown("""
**Validation: strict temporal split**

```
Train:  Year ∈ {2015, …, 2023}  (~85%)
Val:    Year = 2024              (~15%)
```
Standard k-fold would leak future information (2023 data training on 2024 labels in some folds).
Time-based holdout tests true out-of-sample generalization.

**NaN passthrough.** XGBoost learns a default branching direction for each split
when a feature value is missing — GSOM features (absent for ~45% of airports) are
handled natively without imputation.
            """)

    # ── Section 4: Feature Engineering ───────────────────────────────────────
    with st.expander("**4 · Feature Engineering — All 70 Features**", expanded=True):
        _fi_df = get_feature_importance_df()
        _group_colors = {
            "Origin BTS":        "#005EB8",
            "Dest BTS":          "#0088CC",
            "Pair BTS":          "#00AADD",
            "Temporal":          "#7B2D8B",
            "Origin GSOM":       "#1a7a4a",
            "Dest GSOM":         "#2ca02c",
            "Pair GSOM":         "#5cb85c",
            "DFW Hub":           "#C41E3A",
            "Tail-Chain / Duty": "#8B4513",
            "Airport Cascade":   "#ff7f0e",
            "Multi-Hop Cascade": "#B8860B",
            "Other":             "#888888",
        }

        # ── Sunburst: group → feature ─────────────────────────────────────
        _sun_df = _fi_df[_fi_df["importance"] > 0].copy()
        _sun_df["pct"] = (_sun_df["importance"] / _sun_df["importance"].sum() * 100).round(2)
        fig_sun = px.sunburst(
            _sun_df,
            path=["group", "label"],
            values="importance",
            color="group",
            color_discrete_map=_group_colors,
            custom_data=["feature", "pct"],
            title="Feature Importance Hierarchy — Group → Individual Feature (XGBoost Gain)",
        )
        fig_sun.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Group: %{parent}<br>"
                "Importance: %{value:.4f}<br>"
                "Share: %{customdata[1]:.2f}%<extra></extra>"
            ),
            textfont_size=11,
            insidetextorientation="radial",
        )
        fig_sun.update_layout(height=560, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_sun, width='stretch')

        # ── Bar (top 25) + group bar side by side ─────────────────────────
        _bc1, _bc2 = st.columns([3, 2])
        with _bc1:
            _top25 = _fi_df.head(25).sort_values("importance")
            fig_fi = go.Figure(go.Bar(
                x=_top25["importance"],
                y=_top25["label"],
                orientation="h",
                marker=dict(
                    color=[_group_colors.get(g, "#888") for g in _top25["group"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.3f}" for v in _top25["importance"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<br>Feature: %{customdata}<extra></extra>",
                customdata=_top25["feature"],
            ))
            fig_fi.update_layout(
                title="Top 25 Features by Gain",
                xaxis=dict(title="Normalized gain", range=[0, _top25["importance"].max() * 1.22]),
                height=640,
                margin=dict(l=10, r=90, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fi, width='stretch')
        with _bc2:
            _grp_sum = (_fi_df.groupby("group")["importance"].sum()
                        .reset_index().sort_values("importance", ascending=False))
            fig_grp = go.Figure(go.Bar(
                x=_grp_sum["importance"],
                y=_grp_sum["group"],
                orientation="h",
                marker=dict(color=[_group_colors.get(g, "#888") for g in _grp_sum["group"]]),
                text=[f"{v:.3f}" for v in _grp_sum["importance"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Total gain: %{x:.4f}<extra></extra>",
            ))
            fig_grp.update_layout(
                title="Total Importance by Group",
                xaxis=dict(title="Sum of gain", range=[0, _grp_sum["importance"].max() * 1.25]),
                height=640,
                margin=dict(l=10, r=90, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_grp, width='stretch')

        # ── Full feature table ────────────────────────────────────────────
        with st.expander("Show all 70 features"):
            _tbl_cols = _fi_df[["rank", "group", "label", "feature", "importance"]].copy()
            _tbl_cols.columns = ["Rank", "Group", "Description", "Raw Name", "Importance (gain)"]
            _tbl_cols["Importance (gain)"] = _tbl_cols["Importance (gain)"].map("{:.5f}".format)
            st.dataframe(_tbl_cols, hide_index=True, width='stretch', height=500)

    # ── Section 5: Evaluation ─────────────────────────────────────────────────
    with st.expander("**5 · Model Evaluation**", expanded=True):
        st.markdown("""
**Validation set:** BTS 2024 held out entirely from training (time-based split).
Pair-level metrics below are computed on all aggregated pair-month scores vs. observed bad rates.
        """)

        # Metric cards row
        _ev_cols = st.columns(5)
        for _col, (_lbl, _val, _note) in zip(_ev_cols, [
            ("Val AUC",         "0.833", "sequence-level (2024)"),
            ("Val AP",          "0.830", "sequence-level (2024)"),
            ("Pair AUC",        "0.938", "pair-level aggregation"),
            ("Pair AP",         "0.892", "pair-level aggregation"),
            ("F1 @ 0.50",       "0.688", "High Risk class"),
        ]):
            _col.markdown(
                f'<div style="border:1px solid rgba(128,128,128,0.25);border-radius:8px;'
                f'padding:14px 8px;text-align:center">'
                f'<div style="font-size:0.72em;opacity:0.6;text-transform:uppercase;letter-spacing:0.04em">{_lbl}</div>'
                f'<div style="font-size:1.8em;font-weight:700;margin:4px 0">{_val}</div>'
                f'<div style="font-size:0.72em;opacity:0.55">{_note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

        _eval = get_eval_data()

        # Row 1: ROC + PR
        _r1a, _r1b = st.columns(2)
        with _r1a:
            # ROC curve
            _fpr_s = _eval["fpr"][::max(1, len(_eval["fpr"])//500)]
            _tpr_s = _eval["tpr"][::max(1, len(_eval["tpr"])//500)]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="rgba(150,150,150,0.5)", width=1.5),
                name="Random (AUC=0.50)", showlegend=True,
            ))
            fig_roc.add_trace(go.Scatter(
                x=_fpr_s, y=_tpr_s, mode="lines",
                line=dict(color="#005EB8", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,94,184,0.08)",
                name=f"XGBoost (AUC = {_eval['auc']:.3f})",
            ))
            fig_roc.update_layout(
                title="ROC Curve (pair-level)",
                xaxis=dict(title="False Positive Rate", range=[0,1]),
                yaxis=dict(title="True Positive Rate", range=[0,1.02]),
                height=380, margin=dict(t=50, b=50, l=50, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.55, y=0.08),
            )
            fig_roc.add_annotation(
                x=0.65, y=0.35, text=f"AUC = {_eval['auc']:.3f}",
                font=dict(size=15, color="#005EB8"), showarrow=False,
            )
            st.plotly_chart(fig_roc, width='stretch')

        with _r1b:
            # Precision-Recall curve
            _step = max(1, len(_eval["prec"]) // 500)
            _pr_p = _eval["prec"][::_step]
            _pr_r = _eval["rec"][::_step]
            _baseline = float((_eval["scores"]["observed_bad_rate"] > 0.25).mean())
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=[0, 1], y=[_baseline, _baseline], mode="lines",
                line=dict(dash="dash", color="rgba(150,150,150,0.5)", width=1.5),
                name=f"Baseline (AP={_baseline:.2f})", showlegend=True,
            ))
            fig_pr.add_trace(go.Scatter(
                x=_pr_r, y=_pr_p, mode="lines",
                line=dict(color="#C41E3A", width=2.5),
                fill="tozeroy", fillcolor="rgba(196,30,58,0.08)",
                name=f"XGBoost (AP = {_eval['ap']:.3f})",
            ))
            fig_pr.update_layout(
                title="Precision-Recall Curve (pair-level)",
                xaxis=dict(title="Recall", range=[0,1]),
                yaxis=dict(title="Precision", range=[0,1.02]),
                height=380, margin=dict(t=50, b=50, l=50, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.02, y=0.08),
            )
            fig_pr.add_annotation(
                x=0.35, y=0.35, text=f"AP = {_eval['ap']:.3f}",
                font=dict(size=15, color="#C41E3A"), showarrow=False,
            )
            st.plotly_chart(fig_pr, width='stretch')

        # Row 2: Calibration + Confusion matrix
        _r2a, _r2b = st.columns(2)
        with _r2a:
            # Calibration scatter
            _cal = _eval["cal"]
            _cal_colors = [score_to_color(float(s)) for s in _cal["mean_score"]]
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dot", color="rgba(150,150,150,0.5)", width=1.5),
                name="Perfect calibration", showlegend=True,
            ))
            fig_cal.add_trace(go.Scatter(
                x=_cal["mean_score"],
                y=_cal["mean_obs"],
                mode="markers+lines",
                marker=dict(
                    size=_cal["n"] / _cal["n"].max() * 28 + 10,
                    color=_cal_colors,
                    line=dict(width=1.5, color="white"),
                ),
                line=dict(color="rgba(128,128,128,0.4)", width=1.5),
                text=[f"Decile {i}<br>Score: {s:.3f}<br>Obs bad rate: {o:.3f}<br>n={n:,}"
                      for i, (s, o, n) in enumerate(zip(_cal["mean_score"], _cal["mean_obs"], _cal["n"]))],
                hovertemplate="%{text}<extra></extra>",
                name="Model (decile means)",
            ))
            fig_cal.update_layout(
                title="Calibration Plot — Model Score vs. Observed Bad Rate<br>"
                      "<sup>Dot size ∝ number of pair-months in decile. Above diagonal = overestimates risk.</sup>",
                xaxis=dict(title="Mean Model Risk Score (decile)", range=[0, 1], tickformat=".0%"),
                yaxis=dict(title="Mean Observed Bad Rate (decile)", range=[0, 0.55], tickformat=".0%"),
                height=400, margin=dict(t=70, b=50, l=60, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.02, y=0.92),
            )
            st.plotly_chart(fig_cal, width='stretch')

        with _r2b:
            # Confusion matrix
            _cm = np.array([[220253, 28168], [71170, 109525]])
            _cm_pct = _cm / _cm.sum()
            _ann = [[f"<b>{_cm[i,j]:,}</b><br>{_cm_pct[i,j]:.1%}" for j in range(2)] for i in range(2)]
            _cm_colors = [["rgba(0,94,184,0.55)", "rgba(196,30,58,0.25)"],
                          ["rgba(196,30,58,0.25)", "rgba(0,94,184,0.55)"]]
            fig_cm = go.Figure()
            for _ri in range(2):
                for _ci in range(2):
                    fig_cm.add_shape(type="rect",
                        x0=_ci-0.5, y0=_ri-0.5, x1=_ci+0.5, y1=_ri+0.5,
                        fillcolor=_cm_colors[_ri][_ci], line=dict(color="rgba(128,128,128,0.3)", width=1))
                    fig_cm.add_annotation(
                        x=_ci, y=_ri, text=_ann[_ri][_ci],
                        font=dict(size=15), showarrow=False, align="center")
            fig_cm.update_layout(
                title="Confusion Matrix (threshold = 0.50, full dataset)",
                xaxis=dict(tickvals=[0,1], ticktext=["Pred Low", "Pred High"],
                           side="top", range=[-0.5, 1.5]),
                yaxis=dict(tickvals=[0,1], ticktext=["Actual Low", "Actual High"],
                           range=[-0.5, 1.5], autorange="reversed"),
                height=400, margin=dict(t=80, b=20, l=100, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cm, width='stretch')

        # Score distribution histogram
        _bands = ["LOW\n(<0.40)", "MODERATE\n(0.40–0.70)", "HIGH\n(≥0.70)"]
        _pcts  = [0.5869, 0.2466, 0.1665]
        _cnts  = [int(p * 429116) for p in _pcts]
        fig_dist = go.Figure(go.Bar(
            x=_bands, y=_pcts,
            marker=dict(color=["#2ca02c","#ff7f0e","#d62728"],
                        line=dict(width=0)),
            text=[f"{p:.1%}<br>({c:,} pairs)" for p, c in zip(_pcts, _cnts)],
            textposition="outside",
        ))
        fig_dist.update_layout(
            title="Model Score Distribution — All 429k Pair-Months",
            yaxis=dict(tickformat=".0%", range=[0, 0.72], title="Fraction of pair-months"),
            xaxis=dict(title="Risk Band"),
            height=310, margin=dict(t=50, b=50, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, width='stretch')

        st.markdown("""
**Interpreting the calibration plot**

Every dot lies **above** the diagonal — model scores consistently exceed observed bad rates.
This is expected for uncalibrated XGBoost: the model is trained to rank pairs
(maximizing AUCPR), not to output calibrated probabilities. The scores are
monotonically ordered with observed rates, which is what matters operationally.
A score of 0.65 means *"this pair ranks in the top ~17% riskiest"*, not
*"65% of sequences will be disrupted."*

**Known limitations**

1. **Uncalibrated probabilities.** Use scores for ranking and relative comparison, not as literal disruption rates.
2. **AA-only training.** Tail-chain and cascade features reflect AA operational patterns.
3. **Climate stationarity.** Features derived from 2015–2024 GSOM climatology; structural climate shifts would require retraining.
4. **No real-time weather.** Captures climatological risk only — overlay live NWS products for day-of decisions.
        """)

    # ── Section 6: Feature Group Deep Dive ───────────────────────────────────
    with st.expander("**6 · Feature Group Deep Dive**"):
        _group_details = [
            ("Origin & Dest BTS Weather", "#005EB8", """
**Source:** BTS On-Time Performance database (FAA Form 41).
**Computed per airport × month** over 2015–2024 AA flights at DFW.

Each airport has 8 features split into *AA-specific* (when that airport appears
in an AA DFW sequence) and *overall* (all carriers, all routes):

| Feature | Definition |
|---|---|
| `weather_delay_rate` | Fraction of flights with `WeatherDelay ≥ 15 min` |
| `weather_cancel_rate` | Fraction of flights cancelled with code "B" (weather) |
| `avg_weather_delay_min` | Mean `WeatherDelay` across delayed flights |
| `p75_weather_delay_min` | 75th percentile of weather delay distribution |
| `p95_weather_delay_min` | 95th percentile — captures tail risk |
| `nas_delay_rate` | Fraction with `NASDelay ≥ 15 min` (ATC/system, correlated with weather) |
| `overall_weather_delay_rate` | All-carrier version of `weather_delay_rate` |
| `overall_avg_weather_delay_min` | All-carrier average weather delay |

Pair-level features (`pair_*`) are computed as max, min, sum, or product across A and B,
capturing compounding effects (both airports simultaneously bad → highest risk).
            """),
            ("GSOM Weather (NOAA)", "#1a7a4a", """
**Source:** NOAA Global Summary of Month, downloaded via IEM API.
**Coverage:** ~55% of US commercial airports have a nearby GSOM station with complete data.
Missing values are left as NaN — XGBoost learns a default split direction for each feature,
effectively learning "if GSOM is unavailable, use the BTS-based priors."

| Feature | Definition |
|---|---|
| `avg_wind_speed` | Monthly mean surface wind speed (knots) |
| `max_wind_gust` | Maximum recorded wind gust in month (knots) |
| `precip_days` | Number of days with measurable precipitation |
| `total_precip` | Total monthly precipitation (inches) |
| `extreme_precip` | Days with precipitation ≥ 1 inch |

These five features exist for both A and B, plus pair-level max-aggregations (`pair_max_*`).
GSOM captures the *climatological* pattern — e.g., Boston in January has high `precip_days`
and elevated `max_wind_gust` regardless of BTS delay attribution.
            """),
            ("DFW Hub Weather", "#C41E3A", """
Every A→DFW→B sequence transits DFW — so DFW weather is a **universal covariate**
shared across all pairs. We compute it separately from airport-level features because
it is not specific to A or B.

DFW weather is computed from all flights in the BTS files (both departing and arriving DFW),
aggregated by month. Four features:

| Feature | Definition |
|---|---|
| `DFW_weather_delay_rate` | Fraction of DFW flights delayed by weather ≥ 15 min |
| `DFW_weather_cancel_rate` | Fraction of DFW flights cancelled (weather) |
| `DFW_avg_weather_delay_min` | Mean weather delay at DFW |
| `DFW_p95_weather_delay_min` | 95th-percentile weather delay — captures severe weather events |

DFW hub weather ranks ~15th in feature importance, suggesting that pair-specific factors
dominate over hub-wide conditions — which makes sense, since DFW weather is a constant
backdrop, not a differentiator between pairs.
            """),
            ("Tail-Chain & FAA Part 117 Duty", "#8B4513", """
**Motivation.** A crew sequence A→DFW→B is not isolated: the aircraft (tail number)
arrives at DFW having already flown earlier that day (e.g., LGA→DFW). Each prior leg
adds fatigue and reduces buffer. FAA Part 117 limits Flight Duty Period (FDP) to
typically 9–13 hours depending on report time and number of legs.

**Construction.** For each tail number we reconstruct the full day's rotation from
BTS data. The DFW sequence is the focal leg; we look at preceding and succeeding
legs on the same tail.

| Feature | Definition |
|---|---|
| `tc_legs_before_mean` | Avg number of legs the aircraft flew before the DFW arrival leg |
| `tc_block_before_mean` | Avg total block time (min) before DFW arrival |
| `tc_duty_start_hour` | Avg local hour of the crew's first departure of the day |
| `tc_total_duty_mean/p75` | Total duty period (first departure → last arrival + ground time) |
| `tc_fdp_util_mean/p75` | FDP utilization: duty period / FAA Part 117 legal FDP limit |
| `tc_fdp_overrun_rate` | Fraction of sequences where FDP utilization > 0.95 (near-limit) |
| `tc_wocl_rate` | Fraction of sequences where duty period overlaps 02:00–05:59 local (Window of Circadian Low — highest fatigue risk) |
| `tc_legs_after_mean` | Avg legs the aircraft flies after the DFW departure leg |
| `tc_legs_in_day_mean` | Total legs in the full rotation day |
| `tc_downstream_rate` | Fraction of sequences where the leg after B is late (propagation) |
| `tc_cascade_late_rate` | Fraction of sequences where B→DFW arrival is late due to A→DFW delay |
| `tc_cascade_late_min` | Avg minutes the cascade adds to B→DFW arrival |
| `tc_cascade_amplif_mean` | Delay amplification factor: late minutes out / late minutes in |

`tc_cascade_amplif_mean` is the **4th most important feature overall** — sequences where
a small inbound delay reliably amplifies into a large outbound delay are systematically risky.
            """),
            ("Airport Cascade Propagation", "#ff7f0e", """
**Motivation.** Some airports are network hubs where delays propagate outward more aggressively
than others. A delay at ORD ripples through dozens of downstream AA sequences; a delay at SBA
(Santa Barbara) is largely contained.

**Construction.** For each airport and month we compute the probability that a late inbound
at that airport causes a late outbound on the next leg.

| Feature | Definition |
|---|---|
| `A_ap_cascade_rate` | P(outbound late \| airport A appears in the sequence) |
| `A_ap_cascade_given_late` | P(outbound late \| airport A's inbound is late) |
| `B_ap_cascade_rate` | Same for airport B |
| `B_ap_cascade_given_late` | Same for airport B |
| `pair_cascade_product` | A_rate × B_rate — joint cascade exposure |
| `pair_max_cascade_rate` | max(A_rate, B_rate) — worst single endpoint |
            """),
            ("Multi-Hop DFW Cascade", "#B8860B", """
**Motivation.** The A→DFW→B sequence is embedded in a longer chain. If the crew
then operates B→DFW→C→DFW→D, a delay on the focal leg propagates downstream.
These features capture how deeply a delay on A→DFW→B reverberates.

**Construction.** We trace downstream rotations from BTS data: after B departs DFW,
where does the next leg go, and does it too connect through DFW? We follow up to
3 downstream hops.

| Feature | Definition |
|---|---|
| `mhc_n_hops_mean/max` | Number of downstream DFW hops after the focal B departure |
| `mhc_total_late_min_mean/p75` | Total accumulated late minutes across all downstream hops |
| `mhc_cascade_hop_rate` | Fraction of downstream hops that are late |
| `mhc_cascade_depth_mean` | Avg depth at which disruption first appears downstream |
| `mhc_unique_airports_mean` | Number of distinct airports affected by a cascading delay |
| `mhc_recovery_rate` | Fraction of downstream chains that recover (no more late hops after 1st) |

`mhc_n_hops_mean` is the **6th most important feature** in the model — pairs with more
downstream rotations passing through DFW are inherently riskier because a single delay
has higher blast radius.
            """),
        ]
        for _gname, _gcolor, _gdesc in _group_details:
            st.markdown(
                f'<div style="border-left:4px solid {_gcolor};padding-left:16px;margin-bottom:8px">'
                f'<b style="font-size:1.05em">{_gname}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown(_gdesc)
            st.markdown("---")

    # ── Section 7: Key Findings ───────────────────────────────────────────────
    with st.expander("**7 · Key Findings & Operational Implications**"):
        st.markdown("""
**Finding 1: Seasonality dominates all other signals (26% of total importance)**

The top 3 features are all temporal: `is_spring_summer` (15.0%), `season_summer` (9.1%),
`season_spring` (1.9%). This reflects a non-obvious result: spring/summer, not winter,
is the riskiest season for DFW crew sequences. DFW is a convective storm hub — afternoon
thunderstorm activity peaks June–August, generating rapid-onset ground stops that freeze
both inbound and outbound operations simultaneously. Winter snow/ice events at DFW are
relatively rare; the real risk is summer convection.

**Finding 2: Destination-side weather drives more risk than origin-side**

`B_avg_wind_speed` (11.1%) outranks `A_avg_wind_speed` (2.5%). BTS features also
show B-side dominance. Hypothesis: the outbound (DFW→B) leg is more operationally
constrained — the crew has already absorbed the inbound leg's delays, has a shorter
buffer, and faces regulatory FDP limits. A weather event at B that closes the airport
or causes long ground delays has no recovery valve.

**Finding 3: Cascade amplification is the highest-signal non-seasonal feature**

`tc_cascade_amplif_mean` — the ratio of outbound delay minutes to inbound delay minutes —
is the 4th most important feature (4.2%). Sequences where a 20-minute inbound delay
routinely becomes a 45-minute outbound delay are structurally risky regardless of season.
This identifies aircraft rotations with tight turns and no slack.

**Finding 4: Multi-hop depth matters more than multi-hop rate**

`mhc_n_hops_mean` (3.5%) ranks above `mhc_cascade_hop_rate` (0.7%). The number of
downstream DFW connections is more predictive than whether those connections are late.
High-degree nodes in the DFW rotation network carry systemic risk even in good weather —
any disruption propagates to many flights.

**Finding 5: FDP overrun is a leading indicator, not a lagging one**

`tc_fdp_overrun_rate` (1.2%) predicts disruption *before* it happens. Sequences where
crews are routinely flying near their legal FDP limits have elevated bad rates — consistent
with fatigue-induced error under weather pressure. This validates the regulatory basis of
Part 117 limits as a risk proxy.

**Optimization uplift:** Running the Hungarian algorithm on a representative daily schedule
(n=120 arrivals, n=140 departures) reduces total risk score by 15–25% vs. random assignment,
and by 8–12% vs. greedy (highest-priority-first) assignment. The gains concentrate in the
moderate-risk band: the optimizer systematically avoids creating HIGH-risk sequences and
distributes unavoidable risk across pairs more evenly.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.header("Airport Pair Risk Dashboard")
    scores = get_pair_scores()

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        month_sel = st.slider("Filter by Month", 1, 12, 6, key="dash_month",
                              format="%d")
        month_name = ap_meta.MONTH_NAMES[month_sel]
    with col_ctrl2:
        top_n = st.selectbox("Show top N pairs", [10, 20, 50, 100, 200], index=1)
    with col_ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_all_months = st.checkbox("All months", value=False)

    df_view = scores if show_all_months else scores[scores["Month"] == month_sel]
    df_top = df_view.nlargest(top_n, "avg_risk_score")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    pct_high = (df_view["avg_risk_score"] >= 0.70).mean() * 100
    m1.metric("Pairs Analyzed", f"{len(df_view):,}")
    m2.metric("High Risk (>70%)", f"{pct_high:.1f}%")
    m3.metric("Avg Risk Score", f"{df_view['avg_risk_score'].mean():.1%}")
    top_a = df_view.groupby("airport_A")["avg_risk_score"].mean().idxmax()
    m4.metric("Riskiest Origin", top_a)

    st.divider()

    # Top pairs bar chart
    col_bar, col_tbl = st.columns([3, 2])
    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=df_top["avg_risk_score"],
            y=[f"{r.airport_A}→DFW→{r.airport_B}" for r in df_top.itertuples()],
            orientation="h",
            marker_color=[score_to_color(s) for s in df_top["avg_risk_score"]],
            text=[f"{s:.1%}" for s in df_top["avg_risk_score"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Risk: %{x:.1%}<extra></extra>",
        ))
        fig_bar.update_layout(
            title=f"Top {top_n} Riskiest Sequences — {month_name if not show_all_months else 'All Months'}",
            xaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=max(400, top_n * 22),
            margin=dict(l=10, r=80, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, width='stretch')

    with col_tbl:
        st.subheader("Risk Table")
        display = df_top[["airport_A", "airport_B", "Month", "avg_risk_score",
                           "observed_bad_rate", "n_sequences"]].copy()
        display.columns = ["Origin", "Dest", "Month", "Model Risk", "Observed Bad %", "N Sequences"]
        display["Model Risk"] = display["Model Risk"].map("{:.1%}".format)
        display["Observed Bad %"] = display["Observed Bad %"].map("{:.1%}".format)
        st.dataframe(display, width='stretch', height=420)
        st.markdown(
            tip("Model Risk", "XGBoost predicted probability that this pair-month systematically "
                "exceeds the 25% disruption threshold — trained on 60+ features") +
            " vs " +
            tip("Observed Bad %", "Raw historical fraction of A→DFW→B sequences in this "
                "pair-month that exceeded the weather disruption threshold (2015–2024)") +
            " — model risk is typically higher than observed bad rate due to XGBoost calibration "
            "and the definitional difference (probability vs. rate).",
            unsafe_allow_html=True,
        )

    st.divider()

    # Monthly heatmap: Top 15 origin airports × month
    st.subheader("Monthly Risk Heatmap — Top Origins")
    top_origins = scores.groupby("airport_A")["avg_risk_score"].mean().nlargest(15).index.tolist()
    heat_df = (
        scores[scores["airport_A"].isin(top_origins)]
        .groupby(["airport_A", "Month"])["avg_risk_score"]
        .mean()
        .unstack("Month")
        .reindex(columns=range(1, 13))
    )
    heat_df.columns = [ap_meta.MONTH_NAMES[m][:3] for m in heat_df.columns]
    fig_heat = px.imshow(
        heat_df.values,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=1,
        aspect="auto",
        labels=dict(color="Avg Risk"),
        title="Average Risk Score by Origin Airport × Month",
    )
    fig_heat.update_layout(height=420, margin=dict(t=40, b=40))
    st.plotly_chart(fig_heat, width='stretch')


# ── Shared helper ────────────────────────────────────────────────────────────
def _render_sequences(seqs: pd.DataFrame, date_label: str, dep_col: str | None = None,
                      key_suffix: str = ""):
    """Render scored sequences: risk filter, colored table, timeline chart."""
    if seqs.empty:
        st.info(f"No feasible A→DFW→B sequences found for {date_label}.")
        return

    risk_filter = st.multiselect(
        "Filter by risk level", ["HIGH", "MODERATE", "LOW", "N/A"],
        default=["HIGH", "MODERATE", "LOW", "N/A"], key=f"rf_{key_suffix}"
    )
    seqs_view = seqs[seqs["risk_label"].isin(risk_filter)]

    s1, s2, s3 = st.columns(3)
    s1.metric("Sequences Found", len(seqs))
    high_count = (seqs["risk_label"] == "HIGH").sum()
    s2.metric("High Risk", high_count,
              delta=f"{high_count/max(len(seqs),1):.0%} of total")
    s3.metric("Period", date_label[:30])

    show_cols = [c for c in ["Sequence", "flight_in", "arr_time", "flight_out",
                               "dep_time", "turnaround_min", "risk_score", "risk_label"]
                 if c in seqs_view.columns]
    disp = seqs_view[show_cols].copy().rename(columns={
        "flight_in": "Inbound", "arr_time": "Arrived",
        "flight_out": "Outbound", "dep_time": "Departed",
        "turnaround_min": "Turnaround (min)",
        "risk_score": "Risk Score", "risk_label": "Risk Level",
    })
    if "Risk Score" in disp.columns:
        disp["Risk Score"] = disp["Risk Score"].map(
            lambda x: f"{x:.1%}" if isinstance(x, float) and not np.isnan(x) else "N/A"
        )

    def _color(row):
        c = {"HIGH": "rgba(214,39,40,0.25)", "MODERATE": "rgba(255,127,14,0.25)", "LOW": "rgba(44,160,44,0.25)"}.get(
            str(row.get("Risk Level", "")), "")
        return [f"background-color:{c}" for _ in row]

    st.dataframe(disp.style.apply(_color, axis=1), width='stretch', height=400)
    st.download_button("Download CSV", disp.to_csv(index=False),
                       file_name=f"dfw_risk_{date_label[:10]}.csv", mime="text/csv",
                       key=f"dl_{key_suffix}")

    # Timeline
    st.subheader("Risk Timeline")
    plot_seqs = seqs.dropna(subset=["risk_score"])
    if dep_col and dep_col in plot_seqs.columns:
        x_vals  = plot_seqs[dep_col] / 60
        x_axis  = dict(title="Scheduled Departure Hour",
                       tickvals=list(range(0, 25)),
                       ticktext=[f"{h:02d}:00" for h in range(25)])
    else:
        x_vals  = np.arange(len(plot_seqs), dtype=float)
        x_axis  = dict(title="Sequence (sorted by departure)")

    fig_tl = go.Figure()
    for x_v, (_, row) in zip(x_vals, plot_seqs.iterrows()):
        fig_tl.add_trace(go.Scatter(
            x=[x_v], y=[row["risk_score"]], mode="markers+text",
            marker=dict(size=11, color=score_to_color(row["risk_score"]),
                        line=dict(width=1, color="black")),
            text=[str(row.get("airport_B", ""))], textposition="top center",
            hovertemplate=(f"<b>{row.get('Sequence','')}</b><br>Risk: {row['risk_score']:.1%}"
                           f"<br>Turnaround: {row.get('turnaround_min','?')} min<extra></extra>"),
            showlegend=False,
        ))
    fig_tl.add_hline(y=0.70, line_dash="dash", line_color="red",
                     annotation_text="High Risk", annotation_position="right")
    fig_tl.add_hline(y=0.40, line_dash="dash", line_color="orange",
                     annotation_text="Moderate", annotation_position="right")
    fig_tl.update_layout(xaxis=x_axis,
                          yaxis=dict(title="Risk Score", range=[-0.05,1.05], tickformat=".0%"),
                          height=360, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_tl, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: DFW SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════
with tab_sched:
    st.header("DFW Schedule — Sequence Risk Overlay")
    st.caption(
        "AA flights at DFW scored for weather disruption risk. "
        "Live: AviationStack API (key in sidebar). Current schedule: BTS 2024 analog "
        "(same month + day-of-week). Historical: pick any 2024 date."
    )

    bts = get_bts_2024()
    if bts.empty:
        st.warning("BTS 2024 data not found at `data/raw/bts_all_dfw_2024.parquet`.")
    else:
        scores_sched = get_pair_scores()
        scores_idx   = get_scores_indexed()

        data_mode = st.radio(
            "Data Source", ["🔴 Live (AviationStack)", "📅 Current Schedule (BTS Analog)", "📂 Historical (BTS 2024)"],
            horizontal=True, key="sched_mode",
        )
        st.divider()

        def _bts_day_to_seqs(day_df: pd.DataFrame, month_val: int,
                              arr_h0: int = 0, arr_h1: int = 24,
                              dep_h0: int = 0, dep_h1: int = 24) -> pd.DataFrame:
            arrivals   = day_df[day_df["Dest"]   == "DFW"].copy()
            departures = day_df[day_df["Origin"] == "DFW"].copy()
            arrivals["arr_min"]   = arrivals["CRSArrTime"]  // 100 * 60 + arrivals["CRSArrTime"]  % 100
            departures["dep_min"] = departures["CRSDepTime"] // 100 * 60 + departures["CRSDepTime"] % 100
            arrivals   = arrivals[(arrivals["arr_min"]   >= arr_h0 * 60) & (arrivals["arr_min"]   < arr_h1 * 60)]
            departures = departures[(departures["dep_min"] >= dep_h0 * 60) & (departures["dep_min"] < dep_h1 * 60)]
            arr_s = arrivals[["Origin","arr_min","Tail_Number","Flight_Number_Reporting_Airline"]].copy()
            arr_s.columns = ["airport_A","arr_min","tail","flight_in"]
            dep_s = departures[["Dest","dep_min","Tail_Number","Flight_Number_Reporting_Airline"]].copy()
            dep_s.columns = ["airport_B","dep_min","tail","flight_out"]
            seqs = arr_s.merge(dep_s, on="tail", how="inner")
            seqs["turnaround_min"] = seqs["dep_min"] - seqs["arr_min"]
            seqs = seqs[(seqs["turnaround_min"] >= 30) & (seqs["turnaround_min"] <= 240)
                        & (seqs["airport_A"] != seqs["airport_B"])].copy()
            seqs["Sequence"] = seqs["airport_A"] + " → DFW → " + seqs["airport_B"]
            seqs["Month"]    = month_val
            seqs["arr_time"] = (seqs["arr_min"]//60).astype(int).astype(str).str.zfill(2)+":"+\
                               (seqs["arr_min"]%60).astype(int).astype(str).str.zfill(2)
            seqs["dep_time"] = (seqs["dep_min"]//60).astype(int).astype(str).str.zfill(2)+":"+\
                               (seqs["dep_min"]%60).astype(int).astype(str).str.zfill(2)
            return seqs

        # ── LIVE MODE (AviationStack) ──────────────────────────────────────
        if data_mode.startswith("🔴"):
            if not aviationstack_key:
                st.warning("Add your **AviationStack API key** in the sidebar to enable live flights. "
                           "Free tier at aviationstack.com (100 req/month).")
            else:
                col_l1, col_l2 = st.columns([4, 1])
                cache_key = "as_live_seqs"
                with col_l2:
                    fetch_btn = st.button("🔄 Fetch Live", key="sched_refresh_live")
                if fetch_btn:
                    with st.spinner("Fetching live AA schedule from AviationStack..."):
                        arr_raw, dep_raw, status = lf.fetch_aviationstack(aviationstack_key)
                        st.session_state[cache_key] = (arr_raw, dep_raw, status)
                if cache_key not in st.session_state:
                    st.info("Press **🔄 Fetch Live** to load the current AA schedule from AviationStack.")
                    st.stop()
                arr_raw, dep_raw, status = st.session_state[cache_key]
                arr_df = opt.aviationstack_to_arrivals(arr_raw, 0, 24)
                dep_df = opt.aviationstack_to_departures(dep_raw, 0, 24)
                with col_l1:
                    if "error" in status.lower():
                        st.error(status)
                    elif len(arr_df) == 0 and len(dep_df) == 0:
                        st.warning(f"{status}  \n⚠️ No AA flights parsed — API may be rate-limited (100 req/month free) or key invalid.")
                    else:
                        st.caption(status)

                with st.expander(f"Raw: {len(arr_df)} AA arrivals / {len(dep_df)} AA departures"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**→ DFW arrivals**")
                        if not arr_df.empty:
                            st.dataframe(arr_df[["flight","airport","time_str"]].rename(
                                columns={"flight":"Flight","airport":"From","time_str":"Time"}),
                                width='stretch', height=260)
                    with c2:
                        st.markdown("**DFW → departures**")
                        if not dep_df.empty:
                            st.dataframe(dep_df[["flight","airport","time_str"]].rename(
                                columns={"flight":"Flight","airport":"To","time_str":"Time"}),
                                width='stretch', height=260)

                # Build seqs from live data — vectorized cross-join
                from datetime import datetime as _dt
                month_val = _dt.now().month
                seqs = pd.DataFrame()
                if not arr_df.empty and not dep_df.empty:
                    _a = arr_df[["airport","time_min","time_str","flight"]].copy()
                    _d = dep_df[["airport","time_min","time_str","flight"]].copy()
                    cross = _a.merge(_d, how="cross", suffixes=("_a","_b"))
                    _ta = cross["time_min_b"] - cross["time_min_a"]
                    cross = cross[(_ta >= 30) & (_ta <= 240) & (cross["airport_a"] != cross["airport_b"])].copy()
                    if not cross.empty:
                        cross["turnaround_min"] = _ta[cross.index].astype(int)
                        cross["Month"]    = month_val
                        cross["Sequence"] = cross["airport_a"] + " → DFW → " + cross["airport_b"]
                        cross = cross.rename(columns={
                            "airport_a": "airport_A", "airport_b": "airport_B",
                            "flight_a": "flight_in", "time_str_a": "arr_time",
                            "flight_b": "flight_out", "time_str_b": "dep_time",
                        })
                        seqs = cross[["airport_A","airport_B","flight_in","arr_time",
                                      "flight_out","dep_time","turnaround_min","Month","Sequence"]]

                if not seqs.empty:
                    seqs = lf.score_sequences(seqs, scores_sched)
                    _render_sequences(seqs, f"Live {_dt.now().strftime('%Y-%m-%d %H:%M UTC')}",
                                      key_suffix="live")

        # ── CURRENT SCHEDULE ANALOG ────────────────────────────────────────
        elif data_mode.startswith("📅"):
            col_r1, col_r2 = st.columns([4,1])
            with col_r2:
                refresh = st.button("🔄 Refresh", key="sched_refresh_analog")
            cache_key = "bts_analog"
            if refresh or cache_key not in st.session_state:
                day_df, status = lf.get_bts_analog(bts[bts["Reporting_Airline"]=="AA"])
                st.session_state[cache_key] = (day_df, status)
            else:
                day_df, status = st.session_state[cache_key]
            with col_r1:
                st.caption(status)

            month_val = int(pd.to_datetime(day_df["FlightDate"].iloc[0]).month)
            seqs = _bts_day_to_seqs(day_df, month_val)
            if not seqs.empty:
                seqs = lf.score_sequences(seqs, scores_sched)
                seqs = seqs.sort_values("risk_score", ascending=False)
                _render_sequences(seqs, f"Current schedule analog ({day_df['FlightDate'].iloc[0]})",
                                   dep_col="dep_min", key_suffix="analog")
            else:
                st.info("No sequences found in analog day.")

        # ── HISTORICAL MODE ────────────────────────────────────────────────
        else:
            avail_dates = sorted(bts["FlightDate"].unique())
            col_d1, col_d2, col_d3 = st.columns([2, 2, 2])
            with col_d1:
                sel_date = st.selectbox("Date", avail_dates,
                                        index=min(180, len(avail_dates)-1))
            with col_d2:
                carrier_filter = st.radio("Carrier", ["AA only", "All carriers"],
                                           horizontal=True, key="sched_carrier")
            if carrier_filter == "All carriers":
                st.caption(
                    "⚠️ Risk scores were trained exclusively on AA sequences. "
                    "Non-AA tail numbers use the same pair-month risk lookup, "
                    "which may not reflect other carriers' operational patterns."
                )
            month_val = int(pd.to_datetime(sel_date).month)
            day_df = bts[bts["FlightDate"] == sel_date].copy()
            if carrier_filter == "AA only":
                day_df = day_df[day_df["Reporting_Airline"] == "AA"]

            seqs = _bts_day_to_seqs(day_df, month_val)
            if not seqs.empty:
                seqs = lf.score_sequences(seqs, scores_sched)
                seqs = seqs.sort_values("risk_score", ascending=False)
                _render_sequences(seqs, sel_date, dep_col="dep_min", key_suffix="hist")
            else:
                st.info(f"No sequences on {sel_date}.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: SEQUENCE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
with tab_optim:
    st.header("⚡ Sequence Optimizer")
    st.caption(
        "Given a pool of DFW arrivals and departures, find the minimum-risk one-to-one "
        "assignment of inbound → outbound sequences using the Hungarian algorithm. "
        "Constrains turnaround time to 30–240 min per FAA guidelines."
    )

    bts_o = get_bts_2024()
    scores_o_idx = get_scores_indexed()

    if bts_o.empty:
        st.warning("BTS 2024 data not found.")
    else:
        # ── Data source ────────────────────────────────────────────────────
        opt_source = st.radio(
            "Schedule Source",
            ["🔴 Live (AviationStack)", "📅 Current Schedule (BTS Analog)", "📂 Historical (BTS 2024)"],
            horizontal=True, key="opt_source",
        )
        st.divider()

        # ── Time window controls ───────────────────────────────────────────
        st.subheader("Time Window")
        tw1, tw2, tw3 = st.columns(3)
        with tw1:
            opt_carrier = st.radio("Carrier", ["AA only", "All carriers"],
                                    key="opt_carrier", horizontal=True)
        with tw2:
            st.markdown("**Arrival Window (→ DFW)**")
            arr_h0 = st.slider("Arrival from hour", 0, 23, 6,  key="arr_h0")
            arr_h1 = st.slider("Arrival to hour",   1, 24, 20, key="arr_h1")
        with tw3:
            st.markdown("**Departure Window (DFW →)**")
            dep_h0 = st.slider("Departure from hour", 0, 23, 7,  key="dep_h0")
            dep_h1 = st.slider("Departure to hour",   1, 24, 22, key="dep_h1")

        # Extra control: historical date picker (only shown in historical mode)
        if opt_source.startswith("📂"):
            avail_dates_o = sorted(bts_o["FlightDate"].unique())
            opt_date = st.selectbox("Schedule Date", avail_dates_o,
                                     index=min(180, len(avail_dates_o)-1), key="opt_date")
        else:
            opt_date = None

        st.divider()

        # ── Load arrivals + departures based on source ─────────────────────
        arrivals_o   = pd.DataFrame()
        departures_o = pd.DataFrame()
        opt_month    = datetime.now().month
        opt_source_label = ""

        if opt_source.startswith("🔴"):
            if not aviationstack_key:
                st.warning("Add AviationStack API key in sidebar for live data.")
            else:
                cache_key_as = "as_live_seqs"
                col_oa, col_ob = st.columns([4,1])
                with col_ob:
                    fetch_live = st.button("🔄 Fetch Live", key="opt_refresh_live")
                if fetch_live:
                    with st.spinner("Fetching live AA schedule from AviationStack..."):
                        arr_raw, dep_raw, status = lf.fetch_aviationstack(aviationstack_key)
                        st.session_state[cache_key_as] = (arr_raw, dep_raw, status)
                if cache_key_as not in st.session_state:
                    st.info("Press **🔄 Fetch Live** to load the current AA DFW schedule.")
                else:
                    arr_raw, dep_raw, status = st.session_state[cache_key_as]
                    arr_df_as = opt.aviationstack_to_arrivals(arr_raw, arr_h0, arr_h1)
                    dep_df_as = opt.aviationstack_to_departures(dep_raw, dep_h0, dep_h1)
                with col_oa:
                    st.caption(status)
                    if len(arr_df_as) == 0 and len(dep_df_as) == 0:
                        st.warning("No flights returned — check API key or try BTS Analog.")
                    elif len(arr_df_as) < 50:
                        st.info(
                            f"Only {len(arr_df_as)} arrivals found. "
                            "All times shown in **DFW local (CDT)**. "
                            "Widen the arrival/departure hour sliders if flights are missing."
                        )
                arrivals_o   = arr_df_as
                departures_o = dep_df_as
                opt_month    = datetime.now().month
                opt_source_label = f"Live {datetime.now().strftime('%Y-%m-%d')}"

        elif opt_source.startswith("📅"):
            col_oa, col_ob = st.columns([4,1])
            with col_ob:
                if st.button("🔄 Refresh", key="opt_refresh_analog"):
                    if "bts_analog" in st.session_state:
                        del st.session_state["bts_analog"]
            cache_key_an = "bts_analog"
            if cache_key_an not in st.session_state:
                aa_bts = bts_o[bts_o["Reporting_Airline"] == "AA"] if opt_carrier == "AA only" else bts_o
                day_df_an, status_an = lf.get_bts_analog(aa_bts)
                st.session_state[cache_key_an] = (day_df_an, status_an)
            day_df_an, status_an = st.session_state[cache_key_an]
            with col_oa:
                st.caption(status_an)
            opt_month = int(pd.to_datetime(day_df_an["FlightDate"].iloc[0]).month)
            arrivals_o   = opt.bts_to_arrivals(day_df_an, arr_h0, arr_h1)
            departures_o = opt.bts_to_departures(day_df_an, dep_h0, dep_h1)
            opt_source_label = f"Analog {day_df_an['FlightDate'].iloc[0]}"

        else:  # historical
            opt_day = bts_o[bts_o["FlightDate"] == opt_date].copy()
            if opt_carrier == "AA only":
                opt_day = opt_day[opt_day["Reporting_Airline"] == "AA"]
            opt_month    = int(pd.to_datetime(opt_date).month)
            arrivals_o   = opt.bts_to_arrivals(opt_day, arr_h0, arr_h1)
            departures_o = opt.bts_to_departures(opt_day, dep_h0, dep_h1)
            opt_source_label = opt_date

        if arr_h0 >= arr_h1 or dep_h0 >= dep_h1:
            st.error("End hour must be > start hour.")
        elif not arrivals_o.empty or not departures_o.empty:

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Arrivals in window",   len(arrivals_o))
            sc2.metric("Departures in window", len(departures_o))
            feasible_count = 0
            if not arrivals_o.empty and not departures_o.empty:
                # Vectorized cross-join feasibility check
                _a = arrivals_o["time_min"].to_numpy(dtype=float)[:, None]   # (n,1)
                _d = departures_o["time_min"].to_numpy(dtype=float)[None, :] # (1,m)
                _ta = _d - _a
                _ap_a = arrivals_o["airport"].to_numpy(dtype=str)[:, None]
                _ap_d = departures_o["airport"].to_numpy(dtype=str)[None, :]
                _same = _ap_a == _ap_d
                feasible_count = int(((_ta >= 30) & (_ta <= 240) & ~_same).sum())
            sc3.metric("Feasible pairs", feasible_count)

            st.divider()

            if st.button("⚡ Run Optimizer", type="primary", key="run_optim"):
                if arrivals_o.empty or departures_o.empty:
                    st.error("Need at least 1 arrival and 1 departure in the time windows.")
                elif feasible_count == 0:
                    st.error("No feasible A→DFW→B pairs in the selected windows. "
                             "Widen arrival/departure windows or reduce turnaround constraints.")
                else:
                    with st.spinner("Running Hungarian algorithm..."):
                        result_df, stats = opt.optimize_sequences(
                            arrivals_o, departures_o, scores_o_idx, opt_month
                        )
                    st.session_state["opt_result"] = (result_df, stats, arrivals_o, departures_o)

            if "opt_result" in st.session_state:
                result_df, stats, arrivals_o_r, departures_o_r = st.session_state["opt_result"]

                # ── Summary metrics ───────────────────────────────────────
                st.subheader("Optimization Results")
                rm1, rm2, rm3, rm4, rm5 = st.columns(5)
                rm1.metric("Sequences Assigned",   stats["n_matched"])
                rm2.metric("Avg Risk (Optimal)",   f"{stats['optimal_avg']:.1%}")
                worst_avg = stats["worst_total"] / max(stats["n_matched"], 1)
                rm3.metric("Avg Risk (Worst-case)", f"{worst_avg:.1%}")
                risk_saved_pct = (stats["risk_saved"] / max(stats["worst_total"], 0.001)) * 100
                rm4.metric("Risk Reduction",       f"{risk_saved_pct:.1f}%",
                           delta=f"-{stats['risk_saved']:.2f} total score")
                rm5.metric("High Risk Sequences",  f"{stats['pct_high']:.0%}")

                st.divider()

                col_res1, col_res2 = st.columns([3, 2])

                with col_res1:
                    st.subheader("Optimal Assignment")
                    if result_df.empty:
                        st.info("No feasible assignments found.")
                    else:
                        disp_r = result_df[["Sequence", "flight_in", "arr_time",
                                             "flight_out", "dep_time",
                                             "turnaround_min", "risk_score", "risk_label"]].copy()
                        disp_r.columns = ["Sequence", "Inbound", "Arrived", "Outbound",
                                           "Departs", "Turnaround (min)", "Risk Score", "Risk Level"]
                        disp_r["Risk Score"] = disp_r["Risk Score"].map("{:.1%}".format)

                        def _cr(row):
                            c = {"HIGH":"rgba(214,39,40,0.25)","MODERATE":"rgba(255,127,14,0.25)","LOW":"rgba(44,160,44,0.25)"}.get(
                                str(row.get("Risk Level","")), "")
                            return [f"background-color:{c}" for _ in row]

                        st.dataframe(disp_r.style.apply(_cr, axis=1),
                                     width='stretch', height=420)
                        st.download_button("Download Optimal Schedule",
                                           disp_r.to_csv(index=False),
                                           file_name=f"optimal_sequences_{opt_source_label}.csv",
                                           mime="text/csv", key="dl_opt")

                with col_res2:
                    st.subheader("Risk Distribution")
                    if not result_df.empty:
                        counts = result_df["risk_label"].value_counts().reindex(
                            ["HIGH","MODERATE","LOW"], fill_value=0)
                        fig_pie = go.Figure(go.Pie(
                            labels=counts.index, values=counts.values,
                            marker_colors=["#d62728","#ff7f0e","#2ca02c"],
                            hole=0.45,
                            textinfo="label+percent+value",
                        ))
                        fig_pie.update_layout(title="Assigned Sequences by Risk Level",
                                               height=280, margin=dict(t=40,b=0,l=0,r=0))
                        st.plotly_chart(fig_pie, width='stretch')

                    # Optimal vs worst bar
                    fig_cmp = go.Figure([
                        go.Bar(name="Worst-case", x=["Total Risk Score"],
                               y=[stats["worst_total"]], marker_color="#d62728"),
                        go.Bar(name="Optimal",    x=["Total Risk Score"],
                               y=[stats["optimal_total"]], marker_color="#2ca02c"),
                    ])
                    fig_cmp.update_layout(
                        barmode="group", title="Optimal vs Worst-case Total Risk",
                        height=260, plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=40,b=40,l=40,r=20),
                        legend=dict(orientation="h", y=-0.2),
                    )
                    st.plotly_chart(fig_cmp, width='stretch')

                # ── Gantt-style timeline ──────────────────────────────────
                if not result_df.empty:
                    st.subheader("Sequence Timeline (Gantt)")
                    fig_g = go.Figure()
                    for i, row in result_df.iterrows():
                        arr_m = arrivals_o_r[arrivals_o_r["airport"] == row["airport_A"]]["time_min"]
                        dep_m = departures_o_r[departures_o_r["airport"] == row["airport_B"]]["time_min"]
                        a_t = float(arr_m.iloc[0]) if not arr_m.empty else 0
                        d_t = float(dep_m.iloc[0]) if not dep_m.empty else a_t + 90
                        color = score_to_color(row["risk_score"])
                        fig_g.add_trace(go.Scatter(
                            x=[a_t/60, d_t/60], y=[i, i], mode="lines+markers",
                            line=dict(color=color, width=8),
                            marker=dict(size=8, color=["#555", color]),
                            name=row["Sequence"],
                            hovertemplate=(
                                f"<b>{row['Sequence']}</b><br>"
                                f"Arr: {a_t/60:.2f}h | Dep: {d_t/60:.2f}h<br>"
                                f"Turnaround: {row['turnaround_min']} min<br>"
                                f"Risk: {row['risk_score']:.1%}<extra></extra>"
                            ),
                            showlegend=False,
                        ))
                        fig_g.add_annotation(
                            x=a_t/60, y=i, text=row["airport_A"],
                            showarrow=False, xanchor="right", font=dict(size=9)
                        )
                        fig_g.add_annotation(
                            x=d_t/60, y=i, text=row["airport_B"],
                            showarrow=False, xanchor="left", font=dict(size=9)
                        )
                    fig_g.update_layout(
                        xaxis=dict(title="Time (hour)", tickvals=list(range(0,25)),
                                   ticktext=[f"{h:02d}:00" for h in range(25)]),
                        yaxis=dict(visible=False),
                        height=max(300, len(result_df) * 22 + 60),
                        plot_bgcolor="rgba(0,0,0,0)",
                        title="Each bar = one A→DFW→B sequence (color = risk level)",
                        margin=dict(l=80, r=80, t=50, b=50),
                    )
                    st.plotly_chart(fig_g, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: PAIR RISK QUERY
# ═══════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("Pair Risk Query")
    st.caption("Select an inbound origin (A) and outbound destination (B) to score the A→DFW→B sequence.")

    pred = get_predictor()

    col_qa, col_qb, col_qm = st.columns(3)
    with col_qa:
        a_opts = pred.airports_a
        default_a = a_opts.index("MCO") if "MCO" in a_opts else 0
        airport_a = st.selectbox(
            "Inbound Origin (A)",
            a_opts,
            index=default_a,
            format_func=ap_meta.label,
            key="q_a",
        )
    with col_qb:
        b_opts = pred.airports_b
        default_b = b_opts.index("LAX") if "LAX" in b_opts else 0
        airport_b = st.selectbox(
            "Outbound Destination (B)",
            b_opts,
            index=default_b,
            format_func=ap_meta.label,
            key="q_b",
        )
    with col_qm:
        q_month = st.slider("Month", 1, 12, 6, key="q_month",
                             format="%d — " + "%s")
        st.caption(ap_meta.MONTH_NAMES[q_month])

    st.markdown("---")

    if airport_a == "DFW" or airport_b == "DFW":
        st.warning("DFW is the hub — select non-DFW airports for A and B.")
    else:
        result = pred.predict_pair(airport_a, airport_b, q_month)

        if result is None:
            st.warning(
                f"No historical data for **{airport_a} → DFW → {airport_b}** in month {q_month}. "
                "This pair-month combination wasn't observed in BTS 2015–2024."
            )
        else:
            # Layout: gauge + explanation side by side
            col_g, col_e = st.columns([1, 2])

            with col_g:
                st.plotly_chart(
                    gauge_chart(result["risk_score"],
                                f"{airport_a} → DFW → {airport_b}"),
                    width='stretch',
                )

                # Key metrics
                st.markdown(
                    f"| | |\n|--|--|\n"
                    f"| **Sequence** | {airport_a} → DFW → {airport_b} |\n"
                    f"| **Month** | {ap_meta.MONTH_NAMES[q_month]} |\n"
                    f"| {tip('Model Risk Score', 'XGBoost predicted probability that this pair-month systematically exceeds the 25% disruption threshold')} | {result['risk_score']:.1%} |\n"
                    f"| {tip('Observed Bad Rate', 'Fraction of historical A→DFW→B sequences in this pair-month that exceeded the weather disruption threshold (2015–2024 BTS data)')} | {result['observed_bad_rate']:.1%} |\n"
                    f"| **Historical Sequences** | {result['n_sequences']:,} |\n",
                    unsafe_allow_html=True,
                )

                # Recommendation box
                score = result["risk_score"]
                if score >= 0.70:
                    st.error(
                        f"**Recommendation: Do Not Assign**\n\n"
                        f"This sequence has a {score:.0%} weather disruption risk. "
                        f"Weather patterns at **{airport_a}** and **{airport_b}** create "
                        f"compounding disruption potential at DFW hub. In {ap_meta.MONTH_NAMES[q_month]}, "
                        f"{result['observed_bad_rate']:.0%} of historical sequences experienced "
                        f"significant weather delays."
                    )
                elif score >= 0.40:
                    st.warning(
                        f"**Recommendation: Caution**\n\n"
                        f"This sequence has moderate weather risk ({score:.0%}). "
                        f"Consider adding buffer time or monitoring weather. "
                        f"Historical disruption rate: {result['observed_bad_rate']:.0%}."
                    )
                else:
                    st.success(
                        f"**Recommendation: Acceptable**\n\n"
                        f"This sequence has low weather risk ({score:.0%}). "
                        f"Historical disruption rate: {result['observed_bad_rate']:.0%}."
                    )

            with col_e:
                # SHAP explanation — cached in session_state to avoid re-init every render
                shap_key = f"shap_{airport_a}_{airport_b}_{q_month}"
                if shap_key not in st.session_state:
                    with st.spinner("Computing feature contributions (first time only)..."):
                        try:
                            st.session_state[shap_key] = pred.explain_pair(result["X"], top_n=15)
                        except Exception as ex:
                            st.session_state[shap_key] = ex
                shap_result = st.session_state[shap_key]
                if isinstance(shap_result, Exception):
                    st.info(f"SHAP explanation unavailable: {shap_result}")
                    feat_vals = result["X"].T.rename(columns={0: "Value"})
                    feat_vals.index = [FEATURE_LABELS.get(f, f) for f in feat_vals.index]
                    feat_vals = feat_vals.dropna()
                    st.dataframe(feat_vals.style.format("{:.4f}"), height=350)
                else:
                    st.plotly_chart(shap_bar_chart(shap_result), width='stretch')

        st.divider()

        # Month-by-month risk for selected pair
        st.subheader(f"Month-by-Month Risk: {airport_a} → DFW → {airport_b}")
        monthly = pred.predict_all_months(airport_a, airport_b)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly["Month"],
            y=monthly["risk_score"],
            mode="lines+markers",
            marker=dict(
                size=12,
                color=[score_to_color(s) if not np.isnan(s) else "#aaa"
                       for s in monthly["risk_score"]],
                line=dict(width=1.5, color="black"),
            ),
            line=dict(color="#555", width=2),
            text=monthly["risk_score"].map(
                lambda s: f"{s:.1%}" if not np.isnan(s) else "N/A"
            ),
            hovertemplate="<b>%{x}</b><br>Risk: %{text}<extra></extra>",
        ))
        fig_monthly.add_hrect(y0=0.70, y1=1.05, fillcolor="red", opacity=0.07, line_width=0)
        fig_monthly.add_hrect(y0=0.40, y1=0.70, fillcolor="orange", opacity=0.07, line_width=0)
        fig_monthly.add_hline(y=0.70, line_dash="dash", line_color="red", opacity=0.4)
        fig_monthly.add_hline(y=0.40, line_dash="dash", line_color="orange", opacity=0.4)
        fig_monthly.update_layout(
            xaxis=dict(tickvals=list(range(1, 13)),
                       ticktext=[ap_meta.MONTH_NAMES[m][:3] for m in range(1, 13)]),
            yaxis=dict(title="Risk Score", range=[0, 1.05], tickformat=".0%"),
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            title="Seasonal Risk Profile",
            showlegend=False,
        )
        if q_month:
            fig_monthly.add_vline(x=q_month, line_dash="dot", line_color="#333",
                                  annotation_text=ap_meta.MONTH_NAMES[q_month][:3],
                                  annotation_position="top")
        st.plotly_chart(fig_monthly, width='stretch')

        # Compare with reversed sequence
        with st.expander("Compare: reversed sequence B → DFW → A"):
            result_rev = pred.predict_pair(airport_b, airport_a, q_month)
            if result_rev:
                col_rev1, col_rev2 = st.columns(2)
                with col_rev1:
                    st.plotly_chart(
                        gauge_chart(result["risk_score"] if result else 0,
                                    f"{airport_a}→DFW→{airport_b}"),
                        width='stretch'
                    )
                with col_rev2:
                    st.plotly_chart(
                        gauge_chart(result_rev["risk_score"],
                                    f"{airport_b}→DFW→{airport_a}"),
                        width='stretch'
                    )
            else:
                st.info(f"No data for {airport_b} → DFW → {airport_a} in month {q_month}.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: AIRPORT RISK MAP
# ═══════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header("Airport Risk Map")
    st.caption("Airports sized and colored by average model risk score. DFW is the hub for all sequences.")

    scores_map = get_pair_scores()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        map_month = st.slider("Month", 1, 12, 6, key="map_month")
        st.caption(ap_meta.MONTH_NAMES[map_month])
    with col_m2:
        map_role = st.radio("Airport Role", ["As Origin (A)", "As Destination (B)"], horizontal=True)
    with col_m3:
        map_top_n = st.slider("Show top N airports", 10, 200, 30)
    with col_m4:
        map_dark = st.toggle("Dark map", value=True, key="map_dark")

    # Map geo colors based on user toggle
    if map_dark:
        _land  = "rgba(40,40,40,0.7)"
        _lake  = "rgba(30,80,120,0.5)"
        _coast = "rgba(160,160,160,0.5)"
        _sub   = "rgba(160,160,160,0.3)"
        _bg    = "rgba(15,17,22,0.0)"
    else:
        _land  = "#e8ecf0"
        _lake  = "#c6dff0"
        _coast = "#aaaaaa"
        _sub   = "#cccccc"
        _bg    = "rgba(0,0,0,0)"

    role_key   = "origin" if map_role == "As Origin (A)" else "dest"
    role_label = "Origin"  if map_role == "As Origin (A)" else "Destination"
    grp = get_map_group(map_month, role_key)

    grp = grp.nlargest(map_top_n, "avg_risk")
    ap_df = get_airport_df(tuple(grp["airport"].tolist() + ["DFW"]))
    grp = grp.merge(ap_df, left_on="airport", right_on="iata", how="left")
    grp = grp.dropna(subset=["lat", "lon"])

    fig_map = go.Figure()

    # DFW hub marker
    dfw_info = ap_meta.get("DFW")
    if dfw_info.get("lat"):
        fig_map.add_trace(go.Scattergeo(
            lon=[dfw_info["lon"]], lat=[dfw_info["lat"]],
            mode="markers+text",
            marker=dict(size=18, color="#1f77b4", symbol="star",
                        line=dict(width=2, color="white")),
            text=["DFW"], textposition="top right",
            name="DFW Hub",
            hovertemplate="<b>DFW — Dallas/Fort Worth</b><br>Hub airport (all sequences pass through)<extra></extra>",
        ))
        # Draw spoke lines to top-10 riskiest
        for _, row in grp.head(10).iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                fig_map.add_trace(go.Scattergeo(
                    lon=[dfw_info["lon"], row["lon"]],
                    lat=[dfw_info["lat"], row["lat"]],
                    mode="lines",
                    line=dict(width=1, color=score_to_color(row["avg_risk"])),
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Airport markers
    fig_map.add_trace(go.Scattergeo(
        lon=grp["lon"],
        lat=grp["lat"],
        mode="markers+text",
        marker=dict(
            size=grp["avg_risk"] * 30 + 6,
            color=grp["avg_risk"],
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Avg Risk", tickformat=".0%", x=1.0),
            line=dict(width=0.8, color="black"),
        ),
        text=grp["airport"],
        textposition="top center",
        textfont=dict(size=9),
        name=f"Airports as {role_label}",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Avg Risk: %{marker.color:.1%}<br>"
            "N Pairs: %{customdata[0]}<br>"
            "Worst Partner: %{customdata[1]}<extra></extra>"
        ),
        customdata=grp[["n_pairs", "worst_partner"]].values,
    ))

    fig_map.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            bgcolor=_bg,
            showland=True,       landcolor=_land,
            showlakes=True,      lakecolor=_lake,
            showcoastlines=True, coastlinecolor=_coast,
            showsubunits=True,   subunitcolor=_sub,
            showframe=False,
        ),
        title=f"Airport Risk Map — {role_label} — {ap_meta.MONTH_NAMES[map_month]}",
        height=560,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_map, width='stretch')

    # Table below map
    st.subheader(f"Top {map_top_n} Airports by Risk ({ap_meta.MONTH_NAMES[map_month]})")
    tbl = grp[["airport", "city", "state", "avg_risk", "n_pairs", "worst_partner"]].copy()
    tbl.columns = ["Airport", "City", "State", "Avg Risk", "N Pairs", "Worst Partner"]
    tbl["Avg Risk"] = tbl["Avg Risk"].map("{:.1%}".format)
    st.dataframe(tbl, width='stretch', height=320)
