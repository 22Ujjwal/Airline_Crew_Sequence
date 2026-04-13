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
tab_dash, tab_sched, tab_optim, tab_query, tab_map = st.tabs([
    "📊 Risk Dashboard",
    "🛫 DFW Schedule",
    "⚡ Sequence Optimizer",
    "🔍 Pair Risk Query",
    "🗺️ Airport Risk Map",
])


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
        top_n = st.selectbox("Show top N pairs", [10, 20, 50], index=1)
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
                st.markdown(f"""
                | | |
                |--|--|
                | **Sequence** | {airport_a} → DFW → {airport_b} |
                | **Month** | {ap_meta.MONTH_NAMES[q_month]} |
                | **Model Score** | {result['risk_score']:.1%} |
                | **Observed Bad Rate** | {result['observed_bad_rate']:.1%} |
                | **Historical Sequences** | {result['n_sequences']:,} |
                """)

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

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        map_month = st.slider("Month", 1, 12, 6, key="map_month")
        st.caption(ap_meta.MONTH_NAMES[map_month])
    with col_m2:
        map_role = st.radio("Airport Role", ["As Origin (A)", "As Destination (B)"], horizontal=True)
    with col_m3:
        map_top_n = st.slider("Show top N airports", 10, 200, 30)

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
            bgcolor="rgba(0,0,0,0)",
            showland=True,    landcolor="rgba(40,40,40,0.6)",
            showlakes=True,   lakecolor="rgba(30,80,120,0.4)",
            showcoastlines=True, coastlinecolor="rgba(150,150,150,0.5)",
            showsubunits=True,   subunitcolor="rgba(150,150,150,0.3)",
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
