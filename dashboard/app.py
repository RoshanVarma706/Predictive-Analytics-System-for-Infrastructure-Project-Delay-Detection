"""
Executive Dashboard — Predictive Analytics for Infrastructure Delay Detection
Streamlit + Plotly multi-page app with dark theme.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.risk_scoring import compute_risk_scores, assign_risk_tier, build_risk_dataframe
from src.feature_engineering import engineer_features

# ─────────────────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraRisk AI | Delay Prediction Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0d1117; color: #e6edf3; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s;
  }
  [data-testid="metric-container"]:hover { border-color: #58a6ff; }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.9rem; font-weight: 700; }
  [data-testid="stMetricDelta"] { font-size: 0.82rem; }

  /* Cards */
  .risk-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 12px;
  }
  .tier-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
  }
  .tier-Low      { background: #0d4a1e; color: #3fb950; border: 1px solid #238636; }
  .tier-Medium   { background: #4a3500; color: #e3b341; border: 1px solid #9e6a03; }
  .tier-High     { background: #4a2000; color: #f0883e; border: 1px solid #bd561d; }
  .tier-Critical { background: #4a0000; color: #ff7b72; border: 1px solid #da3633; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #58a6ff;
    border-bottom: 2px solid #21262d; padding-bottom: 8px; margin-bottom: 20px;
  }
  .page-title {
    font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px;
  }
  .page-subtitle { font-size: 0.9rem; color: #8b949e; margin-bottom: 24px; }

  /* Plotly plots background */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Divider */
  hr { border-color: #21262d; }

  /* Inputs */
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #8b949e !important; }
  .stSelectbox > div > div { background: #161b22 !important; border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Low": "#3fb950", "Medium": "#e3b341",
    "High": "#f0883e", "Critical": "#ff7b72",
}
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="Inter"),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, path in [("XGBoost",  "outputs/models/xgboost_model.pkl"),
                       ("LightGBM", "outputs/models/lightgbm_model.pkl")]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_preprocessor():
    path = "outputs/preprocessor.pkl"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_risk_scores():
    path = "outputs/risk_scores.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_metrics():
    path = "outputs/metrics_report.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

@st.cache_data
def load_feature_importance():
    path = "outputs/feature_importance.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_shap_importance():
    xgb_path = "outputs/shap/xgboost_feature_importance.csv"
    lgb_path  = "outputs/shap/lightgbm_feature_importance.csv"
    dfs = {}
    if os.path.exists(xgb_path):  dfs["XGBoost"]  = pd.read_csv(xgb_path)
    if os.path.exists(lgb_path):  dfs["LightGBM"] = pd.read_csv(lgb_path)
    return dfs

@st.cache_data
def load_raw_data():
    path = "data/raw/project_data.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_x_test():
    path = "outputs/X_test.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def check_pipeline_run():
    return (
        os.path.exists("outputs/models/xgboost_model.pkl") and
        os.path.exists("outputs/risk_scores.csv")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-size:2.5rem;'>🏗️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#58a6ff;'>InfraRisk AI</div>
        <div style='font-size:0.75rem; color:#8b949e;'>Delay Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview", "⚠️ Risk Analysis", "🔍 Model Insights",
         "🔮 Predict", "🌿 Root Cause"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; padding: 8px 0;'>
        <b style='color:#58a6ff;'>Models</b><br>
        XGBoost · LightGBM<br><br>
        <b style='color:#58a6ff;'>Explainability</b><br>
        SHAP · LIME<br><br>
        <b style='color:#58a6ff;'>Imbalance</b><br>
        SMOTE · Class Weights
    </div>
    """, unsafe_allow_html=True)


if not check_pipeline_run():
    st.warning("⚠️ Pipeline outputs not found. Run `python train.py` first.")
    st.code("python generate_data.py\npython train.py", language="bash")
    st.stop()

models     = load_models()
metrics    = load_metrics()
risk_df    = load_risk_scores()
fi_df      = load_feature_importance()
shap_fi    = load_shap_importance()
raw_df     = load_raw_data()
X_test_df  = load_x_test()
preprocessor = load_preprocessor()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tier badge HTML
# ─────────────────────────────────────────────────────────────────────────────
def tier_badge(tier: str) -> str:
    return f'<span class="tier-badge tier-{tier}">{tier}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown('<div class="page-title">📊 Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Portfolio-level risk summary and model performance</div>', unsafe_allow_html=True)

    xgb_m  = next((m for m in metrics if m["model"] == "XGBoost"),  {})
    lgbm_m = next((m for m in metrics if m["model"] == "LightGBM"), {})

    total     = len(risk_df)
    at_risk   = len(risk_df[risk_df["risk_tier"].isin(["High", "Critical"])])
    critical  = len(risk_df[risk_df["risk_tier"] == "Critical"])
    avg_score = risk_df["risk_score"].mean() if not risk_df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Projects",    f"{total:,}")
    col2.metric("At-Risk Projects",  f"{at_risk:,}",  f"{at_risk/max(total,1):.0%} of portfolio")
    col3.metric("Critical Projects", f"{critical:,}", delta_color="inverse")
    col4.metric("Avg Risk Score",    f"{avg_score:.1f}/100")
    col5.metric("XGBoost AUC-ROC",   f"{xgb_m.get('AUC-ROC', 'N/A')}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            risk_df, x="risk_score", nbins=40,
            color_discrete_sequence=["#58a6ff"],
        )
        fig.add_vrect(x0=0,  x1=30,  fillcolor="#3fb950", opacity=0.07, line_width=0)
        fig.add_vrect(x0=30, x1=60,  fillcolor="#e3b341", opacity=0.07, line_width=0)
        fig.add_vrect(x0=60, x1=80,  fillcolor="#f0883e", opacity=0.07, line_width=0)
        fig.add_vrect(x0=80, x1=100, fillcolor="#ff7b72", opacity=0.07, line_width=0)
        fig.update_layout(**PLOT_THEME, height=320,
                          xaxis_title="Risk Score", yaxis_title="Count",
                          showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Risk Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = risk_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig2 = px.pie(
            tier_counts, names="Tier", values="Count",
            color="Tier",
            color_discrete_map=TIER_COLORS,
            hole=0.55,
        )
        fig2.update_traces(textfont_size=12)
        fig2.update_layout(**PLOT_THEME, height=320,
                           margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="v", x=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    if metrics:
        metric_keys = ["AUC-ROC", "Avg-Precision", "F1 (weighted)", "Precision", "Recall", "MCC"]
        fig3 = go.Figure()
        for m in metrics:
            fig3.add_trace(go.Bar(
                name=m["model"],
                x=metric_keys,
                y=[m.get(k, 0) for k in metric_keys],
                text=[f"{m.get(k,0):.3f}" for k in metric_keys],
                textposition="outside",
            ))
        fig3.update_layout(
            **PLOT_THEME, barmode="group", height=360,
            yaxis=dict(range=[0, 1.15], gridcolor="#21262d"),
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Risk Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ Risk Analysis":
    st.markdown('<div class="page-title">⚠️ Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Project-level risk scores, tier distribution, and portfolio drill-down</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2 = st.columns(2)
    selected_tiers = col_f1.multiselect(
        "Filter by Risk Tier",
        options=["Low", "Medium", "High", "Critical"],
        default=["High", "Critical"],
    )
    score_range = col_f2.slider("Risk Score Range", 0, 100, (0, 100))

    filtered = risk_df.copy()
    if selected_tiers:
        filtered = filtered[filtered["risk_tier"].isin(selected_tiers)]
    filtered = filtered[
        (filtered["risk_score"] >= score_range[0]) &
        (filtered["risk_score"] <= score_range[1])
    ]

    # Merge with raw data for project type / region
    if not raw_df.empty and not filtered.empty:
        meta = raw_df[["project_id", "project_type", "region",
                        "project_budget_musd", "contractor_tier"]].copy()
        filtered = filtered.merge(meta, on="project_id", how="left")

    col_c1, col_c2 = st.columns([1.3, 1])

    with col_c1:
        st.markdown('<div class="section-header">Risk Score vs. Delay Probability</div>', unsafe_allow_html=True)
        if "project_type" in filtered.columns:
            fig = px.scatter(
                filtered, x="delay_prob", y="risk_score",
                color="risk_tier", color_discrete_map=TIER_COLORS,
                symbol="project_type", size="risk_score",
                size_max=16, opacity=0.75,
                hover_data=["project_id", "project_type", "risk_tier"],
            )
            fig.update_layout(**PLOT_THEME, height=380,
                              xaxis_title="Delay Probability", yaxis_title="Risk Score",
                              legend=dict(orientation="v", x=1.01),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        st.markdown('<div class="section-header">Region × Tier Heatmap</div>', unsafe_allow_html=True)
        if "region" in filtered.columns:
            heat = filtered.groupby(["region", "risk_tier"]).size().reset_index(name="count")
            heat_pivot = heat.pivot(index="region", columns="risk_tier", values="count").fillna(0)
            for t in ["Low", "Medium", "High", "Critical"]:
                if t not in heat_pivot.columns:
                    heat_pivot[t] = 0
            heat_pivot = heat_pivot[["Low", "Medium", "High", "Critical"]]

            fig2 = px.imshow(
                heat_pivot, color_continuous_scale=["#0d1117", "#1f4068", "#58a6ff"],
                text_auto=True, aspect="auto",
            )
            fig2.update_layout(**PLOT_THEME, height=380,
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown(f'<div class="section-header">Project Risk Table ({len(filtered):,} records)</div>', unsafe_allow_html=True)

    display_cols = ["project_id", "delay_prob", "risk_score", "risk_tier", "actual_delayed"]
    if "project_type" in filtered.columns:
        display_cols += ["project_type", "region", "contractor_tier"]

    show_df = filtered[display_cols].head(200).copy()
    show_df["delay_prob"] = show_df["delay_prob"].apply(lambda x: f"{x:.2%}")
    show_df["risk_score"] = show_df["risk_score"].apply(lambda x: f"{x:.1f}")
    st.dataframe(show_df, use_container_width=True, height=340)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Model Insights":
    st.markdown('<div class="page-title">🔍 Model Insights & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Feature importance, SHAP global analysis, and model comparison</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Feature Importance", "🎯 SHAP Analysis"])

    with tab1:
        if not fi_df.empty:
            top_n = st.slider("Top N Features", 5, min(25, len(fi_df)), 15)
            fi_plot = fi_df.head(top_n)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="XGBoost", x=fi_plot["feature"][::-1],
                y=fi_plot["xgb_importance"][::-1],
                marker_color="#58a6ff", orientation="v",
            ))
            fig.add_trace(go.Bar(
                name="LightGBM", x=fi_plot["feature"][::-1],
                y=fi_plot["lgbm_importance"][::-1],
                marker_color="#f0883e", orientation="v",
            ))
            fig.update_layout(
                **PLOT_THEME, barmode="group", height=440,
                xaxis_tickangle=-35,
                yaxis_title="Importance Score",
                legend=dict(orientation="h", y=1.05),
                margin=dict(l=10, r=10, t=30, b=90),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not found. Run train.py first.")

    with tab2:
        col_sh1, col_sh2 = st.columns(2)

        for col, model_name, key in [
            (col_sh1, "XGBoost",  "XGBoost"),
            (col_sh2, "LightGBM", "LightGBM"),
        ]:
            with col:
                st.markdown(f'<div class="section-header">{model_name} — SHAP Importance</div>', unsafe_allow_html=True)
                if key in shap_fi and not shap_fi[key].empty:
                    sf = shap_fi[key].head(12)
                    fig = go.Figure(go.Bar(
                        x=sf["mean_abs_shap"],
                        y=sf["feature"],
                        orientation="h",
                        marker_color="#58a6ff" if key == "XGBoost" else "#f0883e",
                    ))
                    fig.update_layout(
                        **PLOT_THEME, height=400,
                        xaxis_title="Mean |SHAP Value|",
                        yaxis=dict(autorange="reversed", gridcolor="#21262d"),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: display saved plot image
                    img_path = f"outputs/shap/{key.lower()}_shap_bar.png"
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.info(f"Run train.py to generate {model_name} SHAP outputs.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Predict
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown('<div class="page-title">🔮 Real-Time Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter project parameters to get an instant delay risk score</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown('<div class="section-header">Project Parameters</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        project_type   = c1.selectbox("Project Type",    ["road", "bridge", "pipeline", "building", "rail"])
        region         = c2.selectbox("Region",          ["North", "South", "East", "West", "Central"])
        contractor_tier = c1.selectbox("Contractor Tier", ["Tier1", "Tier2", "Tier3"])
        planned_days   = c2.number_input("Planned Duration (days)", 90, 2000, 365)

        c3, c4 = st.columns(2)
        start_delay    = c3.number_input("Start Delay (days)", 0, 180, 10)
        permit_days    = c4.number_input("Permit Processing (days)", 1, 240, 30)

        c5, c6 = st.columns(2)
        budget_util    = c5.slider("Budget Utilization (%)",     30, 130, 80)
        workforce_avail = c6.slider("Workforce Availability (%)", 20, 100, 72)

        c7, c8 = st.columns(2)
        equipment_avail = c7.slider("Equipment Availability (%)", 20, 100, 75)
        budget_musd    = c8.number_input("Budget (M USD)", 0.5, 500.0, 25.0, step=0.5)

        c9, c10 = st.columns(2)
        land_acq       = c9.selectbox("Land Acquisition Issues", [0, 1], format_func=lambda x: "Yes" if x else "No")
        reg_hurdles    = c10.number_input("Regulatory Hurdles", 0, 10, 1)

        c11, c12 = st.columns(2)
        weather        = c11.number_input("Weather Events", 0, 10, 1)
        stakeholder    = c12.selectbox("Stakeholder Conflicts", [0, 1], format_func=lambda x: "Yes" if x else "No")

        c13, c14 = st.columns(2)
        past_delays    = c13.number_input("Contractor Past Delays", 0, 15, 1)
        sim_avg_delay  = c14.number_input("Similar Project Avg Delay (days)", 0, 300, 20)

        predict_btn = st.button("🔮 Compute Risk Score", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Risk Assessment Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build input row
            input_data = pd.DataFrame([{
                "project_id":                   "PRED-001",
                "project_type":                 project_type,
                "region":                       region,
                "contractor_tier":              contractor_tier,
                "planned_duration_days":        planned_days,
                "actual_start_delay_days":      start_delay,
                "permit_processing_days":       permit_days,
                "budget_utilization_pct":       budget_util,
                "workforce_availability_pct":   workforce_avail,
                "equipment_availability_pct":   equipment_avail,
                "land_acquisition_issues":      land_acq,
                "regulatory_hurdles":           reg_hurdles,
                "weather_events":               weather,
                "stakeholder_conflicts":        stakeholder,
                "contractor_past_delays":       past_delays,
                "similar_project_avg_delay_days": sim_avg_delay,
                "project_budget_musd":          budget_musd,
                "delayed":                      0,  # placeholder
            }])

            try:
                engineered = engineer_features(input_data)
                X_pred, _, _, feat_names = __import__(
                    "src.data_preprocessing", fromlist=["preprocess"]
                ).preprocess(engineered, preprocessor=preprocessor, fit=False)

                xgb_model  = models.get("XGBoost")
                lgbm_model = models.get("LightGBM")

                if xgb_model:
                    proba = xgb_model.predict_proba(X_pred)[0, 1]
                    risk  = compute_risk_scores(
                        np.array([proba]),
                        project_types=np.array([project_type]),
                        project_budgets=np.array([budget_musd]),
                    )[0]
                    tier  = assign_risk_tier(risk)

                    # Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk,
                        delta={"reference": 50, "increasing": {"color": "#ff7b72"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                            "bar":  {"color": TIER_COLORS[tier], "thickness": 0.3},
                            "bgcolor": "#161b22",
                            "borderwidth": 1, "bordercolor": "#21262d",
                            "steps": [
                                {"range": [0,  30], "color": "#0d4a1e"},
                                {"range": [30, 60], "color": "#4a3500"},
                                {"range": [60, 80], "color": "#4a2000"},
                                {"range": [80, 100],"color": "#4a0000"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3},
                                          "thickness": 0.85, "value": risk},
                        },
                        number={"suffix": "/100", "font": {"size": 36, "color": TIER_COLORS[tier]}},
                        title={"text": f"Risk Score", "font": {"size": 14, "color": "#8b949e"}},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", height=300,
                        margin=dict(l=30, r=30, t=30, b=10),
                        font=dict(color="#e6edf3"),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown(f"""
                    <div class='risk-card' style='text-align:center;'>
                        <div style='font-size:1.8rem; font-weight:700; color:{TIER_COLORS[tier]};'>{tier.upper()}</div>
                        <div style='color:#8b949e; font-size:0.85rem; margin-top:6px;'>
                            Delay Probability: <b style='color:#e6edf3;'>{proba:.1%}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if lgbm_model:
                        lgbm_proba = lgbm_model.predict_proba(X_pred)[0, 1]
                        lgbm_risk  = compute_risk_scores(
                            np.array([lgbm_proba]),
                            project_types=np.array([project_type]),
                            project_budgets=np.array([budget_musd]),
                        )[0]
                        st.markdown(f"""
                        <div class='risk-card'>
                            <b>LightGBM</b> — Delay Prob: {lgbm_proba:.1%} | Risk Score: {lgbm_risk:.1f}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Models not loaded. Please run train.py first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#8b949e;'>
                <div style='font-size:3rem;'>🔮</div>
                <div style='margin-top:12px;'>Fill in the project parameters<br>and click <b>Compute Risk Score</b></div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Root Cause
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌿 Root Cause":
    st.markdown('<div class="page-title">🌿 Root Cause Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">SHAP-based drill-down: identify top delay drivers per project</div>', unsafe_allow_html=True)

    # SHAP assets
    shap_vals_path  = "outputs/shap_values_xgb.npy"
    shap_sample_path = "outputs/shap_sample_X.csv"
    explainer_path  = "outputs/shap_explainer_xgb.pkl"

    if os.path.exists(shap_vals_path) and os.path.exists(shap_sample_path):
        shap_vals_arr = np.load(shap_vals_path)
        shap_X        = pd.read_csv(shap_sample_path)
        explainer_obj = joblib.load(explainer_path) if os.path.exists(explainer_path) else None

        n_samples = len(shap_X)
        sample_idx = st.slider(f"Select Project Sample (0 – {n_samples-1})", 0, n_samples-1, 0)

        sv = shap_vals_arr[sample_idx]
        top_n = st.radio("Show top N features", [5, 8, 12], horizontal=True)
        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_n]

        col_wf, col_tbl = st.columns([1.3, 1])

        with col_wf:
            st.markdown('<div class="section-header">SHAP Contribution Waterfall</div>', unsafe_allow_html=True)
            features_sel = shap_X.columns[top_idx].tolist()
            shap_sel     = sv[top_idx]
            expected     = explainer_obj.expected_value if explainer_obj else 0
            if isinstance(expected, list): expected = expected[1]

            colors = [TIER_COLORS["Critical"] if s > 0 else TIER_COLORS["Low"] for s in shap_sel]
            fig = go.Figure(go.Waterfall(
                orientation="h",
                measure=["relative"] * len(features_sel) + ["total"],
                y=features_sel + ["Prediction"],
                x=list(shap_sel) + [float(expected) + shap_sel.sum()],
                connector={"line": {"color": "#21262d"}},
                decreasing={"marker": {"color": TIER_COLORS["Low"]}},
                increasing={"marker": {"color": TIER_COLORS["Critical"]}},
                totals={"marker": {"color": "#58a6ff"}},
                text=[f"{v:+.3f}" for v in list(shap_sel) + [expected + shap_sel.sum()]],
            ))
            fig.update_layout(
                **PLOT_THEME, height=420,
                xaxis_title="SHAP Value (impact on delay prediction)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_tbl:
            st.markdown('<div class="section-header">Top Risk Drivers</div>', unsafe_allow_html=True)
            driver_data = []
            for i, idx in enumerate(top_idx):
                feat = shap_X.columns[idx]
                val  = shap_X.iloc[sample_idx, idx]
                shap_v = sv[idx]
                direction = "↑ Increases" if shap_v > 0 else "↓ Reduces"
                driver_data.append({
                    "Rank":      i + 1,
                    "Feature":   feat,
                    "Value":     round(float(val), 3),
                    "SHAP":      round(float(shap_v), 4),
                    "Effect":    direction,
                })
            driver_df = pd.DataFrame(driver_data)
            st.dataframe(driver_df, use_container_width=True, height=380, hide_index=True)

        # Saved waterfall image
        wf_img = f"outputs/shap/xgboost_waterfall_sample0.png"
        if sample_idx == 0 and os.path.exists(wf_img):
            st.markdown("---")
            st.markdown('<div class="section-header">Pre-generated Waterfall Plot</div>', unsafe_allow_html=True)
            st.image(wf_img, use_container_width=True)
    else:
        st.info("SHAP artifacts not found. Run `python train.py` to generate them.")
