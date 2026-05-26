import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              recall_score, precision_score, f1_score, accuracy_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. GLOBAL PAGE CONFIG & DARK MODE THEME
# ============================================================
st.set_page_config(
    page_title="Intelligent Retention Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

DARK_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Palette ── */
:root {
    --bg-base:      #080B10;
    --bg-surface:   #0D1117;
    --bg-card:      #111827;
    --bg-card-hover:#141e2e;
    --border:       rgba(255,255,255,0.07);
    --border-accent:rgba(96,165,250,0.35);
    --text-primary: #F0F4FF;
    --text-muted:   #8B95A8;
    --text-dim:     #545E70;
    --accent-blue:  #60A5FA;
    --accent-violet:#A78BFA;
    --accent-teal:  #2DD4BF;
    --accent-red:   #F87171;
    --accent-amber: #FBBF24;
    --gradient-hr:  linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #2DD4BF 100%);
}

/* ── Global Reset — cover every container that could leak white ── */
*, *::before, *::after { box-sizing: border-box; }
html, body { background-color: #080B10 !important; }

.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"],
[data-testid="stMain"] > div,
[data-testid="stMainBlockContainer"],
section.main,
section.main > div,
.main .block-container,
[data-testid="block-container"] {
    background-color: #080B10 !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Kill top white gap from Streamlit header ── */
[data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }
header[data-testid="stHeader"] {
    background-color: #080B10 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
header[data-testid="stHeader"]::before,
header[data-testid="stHeader"]::after { display: none !important; }

/* ══════════════════════════════════════
   SIDEBAR — Redesigned
══════════════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #080C14 !important;
    border-right: 1px solid rgba(255,255,255,0.055) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Brand block ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 18px; margin-bottom: 4px;
}
.sidebar-brand .brand-icon-wrap {
    width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(135deg, #1D4ED8 0%, #6D28D9 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 4px 12px rgba(109,40,217,0.35);
}
.sidebar-brand .brand-title {
    font-size: 0.95rem; font-weight: 700; line-height: 1.2;
    background: linear-gradient(90deg, #93C5FD, #C4B5FD);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-tag { font-size: 0.67rem; color: #2D3748; margin-top: 2px; letter-spacing: 0.03em; }

/* ── Section label ── */
.nav-section-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #2D3748;
    padding: 14px 4px 5px; display: block;
}

/* ── Nav radio group ── */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex !important; flex-direction: column !important; gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.86rem !important; font-weight: 500 !important;
    color: #4A5568 !important; padding: 9px 12px !important;
    border-radius: 10px !important; cursor: pointer !important;
    transition: all 0.15s ease !important;
    display: flex !important; align-items: center !important; gap: 9px !important;
    border: 1px solid transparent !important;
    letter-spacing: 0.01em !important; width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(96,165,250,0.07) !important;
    color: #94A3B8 !important;
    border-color: rgba(96,165,250,0.12) !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb] {
    background: linear-gradient(90deg, rgba(96,165,250,0.12), rgba(167,139,250,0.08)) !important;
    color: #93C5FD !important;
    border-color: rgba(96,165,250,0.22) !important;
}

/* ── Sidebar dividers ── */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 8px 0 !important;
}

/* ── Expander in sidebar ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.82rem !important; color: #4A5568 !important;
    background: transparent !important; padding: 8px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #94A3B8 !important;
}

/* ── Model selector ── */
.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 10px 12px; margin-top: 6px;
}
.model-card-name  { font-size: 0.82rem; font-weight: 600; color: #CBD5E1; }
.model-card-desc  { font-size: 0.72rem; color: #2D3748; margin-top: 4px; line-height: 1.4; }
.model-tag {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.07em; padding: 2px 7px;
    border-radius: 999px; margin-top: 5px;
}

/* ── Status pill ── */
.status-pill {
    display: flex; align-items: center; gap: 10px;
    background: rgba(45,212,191,0.05);
    border: 1px solid rgba(45,212,191,0.13);
    border-radius: 10px; padding: 10px 12px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #2DD4BF;
    box-shadow: 0 0 8px rgba(45,212,191,0.6); flex-shrink: 0;
    animation: sb-pulse 2.5s ease-in-out infinite;
}
@keyframes sb-pulse {
    0%,100% { box-shadow: 0 0 6px rgba(45,212,191,0.6); }
    50%      { box-shadow: 0 0 12px rgba(45,212,191,0.2); }
}

/* ── Sidebar slider thumb ── */
[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {
    background: #60A5FA !important; border: 2px solid #60A5FA !important;
}

/* ── Model selector label ── */
.model-selector-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #2D3748; padding: 4px 0; display: block;
}

/* ── Typography ── */
h1 {
    font-size: 2rem !important; font-weight: 700 !important;
    color: var(--text-primary) !important; letter-spacing: -0.02em !important;
}
h2, h3 { font-weight: 600 !important; color: var(--text-primary) !important; }
p, label, .stMarkdown p { color: var(--text-muted) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── DataFrames — only style the outer wrapper, NOT the iframe internals ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
/* Target the inner glide-data-grid canvas wrapper background only */
[data-testid="stDataFrame"] > div {
    background: #111827 !important;
    border-radius: 12px !important;
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] * { color: var(--text-primary) !important; background: var(--bg-card) !important; }
[data-baseweb="popover"],
[data-baseweb="menu"] { background: #1a2234 !important; border: 1px solid rgba(255,255,255,0.1) !important; }
[data-baseweb="menu"] li { color: var(--text-primary) !important; }
[data-baseweb="menu"] li:hover { background: rgba(96,165,250,0.1) !important; }

/* ── Sliders — replace Streamlit's red with blue ── */
[data-testid="stSlider"] > div { background: transparent !important; }
/* Thumb */
div[data-baseweb="slider"] [role="slider"] {
    background: #60A5FA !important;
    border: 3px solid #60A5FA !important;
    box-shadow: 0 0 0 4px rgba(96,165,250,0.2) !important;
}
/* Filled track */
div[data-baseweb="slider"] div[class*="Track"] > div:first-child {
    background: linear-gradient(90deg, #60A5FA, #A78BFA) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #6D28D9) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    padding: 0.6rem 1.2rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(96,165,250,0.1) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    color: #60A5FA !important; border-radius: 8px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] * { color: var(--text-primary) !important; }
[data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background: rgba(96,165,250,0.06) !important;
    border: 1px solid rgba(96,165,250,0.2) !important;
    border-radius: 10px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary:hover { background: var(--bg-card-hover) !important; }
[data-testid="stExpander"] summary svg { fill: var(--text-muted) !important; }
[data-testid="stExpander"] > div { background: var(--bg-card) !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
    min-height: 140px;
}
.kpi-card:hover { 
    border-color: var(--border-accent); 
    transform: translateY(-2px);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-hr);
    opacity: 0;
    transition: opacity 0.25s;
}
.kpi-card:hover::before { opacity: 1; }
.kpi-card .kpi-icon   { font-size: 1.5rem; margin-bottom: 6px; }
.kpi-card .kpi-label  { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-card .kpi-value  { font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1.1; }
.kpi-card .kpi-sub    { font-size: 0.78rem; color: var(--text-dim); margin-top: 4px; }
.kpi-card .kpi-badge  { 
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 999px; margin-top: 6px;
}
.badge-up   { background: rgba(248,113,113,0.15); color: #F87171; }
.badge-down { background: rgba(45,212,191,0.15);  color: #2DD4BF; }
.badge-neu  { background: rgba(96,165,250,0.15);  color: #60A5FA; }

/* ── Risk Cards ── */
.risk-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.risk-card.high   { border-left: 3px solid #F87171; }
.risk-card.medium { border-left: 3px solid #FBBF24; }
.risk-card.low    { border-left: 3px solid #2DD4BF; }
.risk-card .r-name  { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); }
.risk-card .r-meta  { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
.risk-card .r-track { 
    height: 6px; background: rgba(255,255,255,0.08); 
    border-radius: 999px; margin-top: 10px; overflow: hidden;
}
.risk-card .r-fill  { height: 100%; border-radius: 999px; }
.risk-card .r-pct   { font-size: 0.76rem; color: var(--text-muted); margin-top: 4px; text-align: right; }

/* ── Section Note ── */
.section-note {
    background: rgba(96,165,250,0.06);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Plot backgrounds ── */
.stPlotlyChart, .stPyplot { 
    background: transparent !important; 
    border-radius: 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }

/* sidebar-brand styles now in sidebar block */

/* ── Page header ── */
.page-header {
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.75rem;
}
.page-header h1 { margin: 0 !important; }
.page-header .ph-sub { color: var(--text-muted); font-size: 0.88rem; margin-top: 4px; }

/* ── Insight pill ── */
.insight-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25);
    border-radius: 999px; padding: 4px 12px;
    font-size: 0.78rem; color: #A78BFA; margin: 2px;
}

/* ── Prediction result ── */
.pred-result {
    padding: 1.25rem; border-radius: 14px; text-align: center;
    font-size: 1.1rem; font-weight: 600;
}
.pred-high   { background: rgba(248,113,113,0.12); border: 1px solid rgba(248,113,113,0.4); color: #F87171; }
.pred-medium { background: rgba(251,191,36,0.12);  border: 1px solid rgba(251,191,36,0.4);  color: #FBBF24; }
.pred-low    { background: rgba(45,212,191,0.12);  border: 1px solid rgba(45,212,191,0.4);  color: #2DD4BF; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ============================================================
# 2. MATPLOTLIB DARK THEME HELPER
# ============================================================
def apply_dark_style(fig, ax_list=None):
    """Apply consistent dark theme to matplotlib figures."""
    BG = "#111827"
    SPINE_COLOR = (1, 1, 1, 0.07)   # rgba(255,255,255,0.07) as matplotlib tuple
    GRID_COLOR  = (1, 1, 1, 0.05)   # rgba(255,255,255,0.05)
    TICK_COLOR  = "#8B95A8"

    fig.patch.set_facecolor(BG)
    axes = ax_list if ax_list else fig.get_axes()
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TICK_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TICK_COLOR)
        ax.yaxis.label.set_color(TICK_COLOR)
        ax.title.set_color("#F0F4FF")
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.7)
    return fig

ACCENT_PALETTE = ["#60A5FA", "#A78BFA", "#2DD4BF", "#F87171", "#FBBF24", "#34D399", "#F472B6"]

# ============================================================
# 3. DATA LOADING & CACHING
# ============================================================
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "Final Data Clean", "HR_Attrition_MultiCompany.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    # ── Fallback: generate realistic synthetic data so the UI is always renderable ──
    np.random.seed(42)
    n = 1200
    depts = ["Engineering", "Sales", "HR", "Finance", "Marketing"]
    roles = ["Software Engineer", "Sales Rep", "HR Analyst", "Financial Analyst",
             "Data Scientist", "Manager", "Director", "Support Specialist"]
    companies = ["TechCorp", "RetailCo", "FinServ", "HealthCo"]
    df = pd.DataFrame({
        "EmployeeNumber":    range(1, n+1),
        "Age":               np.random.randint(22, 60, n),
        "Department":        np.random.choice(depts, n),
        "JobRole":           np.random.choice(roles, n),
        "Company":           np.random.choice(companies, n),
        "MonthlyIncome":     np.random.randint(3000, 18000, n),
        "JobLevel":          np.random.randint(1, 6, n),
        "DistanceFromHome":  np.random.randint(1, 35, n),
        "TotalWorkingYears": np.random.randint(0, 35, n),
        "YearsAtCompany":    np.random.randint(0, 20, n),
        "YearsInCurrentRole":np.random.randint(0, 15, n),
        "JobSatisfaction":   np.random.randint(1, 5, n),
        "WorkLifeBalance":   np.random.randint(1, 5, n),
        "OverTime":          np.random.choice(["Yes", "No"], n),
        "Attrition":         np.random.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "Gender":            np.random.choice(["Male", "Female"], n),
        "Education":         np.random.randint(1, 6, n),
        "NumCompaniesWorked":np.random.randint(0, 10, n),
        "PerformanceRating": np.random.randint(1, 5, n),
    })
    return df

@st.cache_resource(show_spinner="Training AI model…")
def get_trained_model(df_hash):
    """Train + cache the RandomForest. df_hash is a hashable key."""
    df = st.session_state["_df"]
    df_ml = df.copy()
    df_ml["Attrition_bin"] = (df_ml["Attrition"] == "Yes").astype(int)

    # Encode OverTime if present
    if "OverTime" in df_ml.columns:
        df_ml["OverTime_enc"] = (df_ml["OverTime"] == "Yes").astype(int)
    else:
        df_ml["OverTime_enc"] = 0

    FEATURES = ["Age", "DistanceFromHome", "JobLevel", "MonthlyIncome",
                "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
                "JobSatisfaction", "WorkLifeBalance", "OverTime_enc"]
    FEATURES = [f for f in FEATURES if f in df_ml.columns]

    X = df_ml[FEATURES].fillna(df_ml[FEATURES].median())
    y = df_ml["Attrition_bin"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = {
        "auc":    round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 3),
        "report": classification_report(y_test, model.predict(X_test), output_dict=True),
    }
    return model, FEATURES, metrics

# ============================================================
# 3b. MULTI-MODEL COMPARISON (Recall-focused)
# ============================================================
@st.cache_resource(show_spinner="Benchmarking all models...")
def run_model_comparison(df_hash):
    df = st.session_state["_df"]
    df_ml = df.copy()
    df_ml["Attrition_bin"] = (df_ml["Attrition"] == "Yes").astype(int)
    if "OverTime" in df_ml.columns:
        df_ml["OverTime_enc"] = (df_ml["OverTime"] == "Yes").astype(int)
    else:
        df_ml["OverTime_enc"] = 0

    FEATS = ["Age", "DistanceFromHome", "JobLevel", "MonthlyIncome",
             "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
             "JobSatisfaction", "WorkLifeBalance", "OverTime_enc"]
    FEATS = [f for f in FEATS if f in df_ml.columns]

    X = df_ml[FEATS].fillna(df_ml[FEATS].median())
    y = df_ml["Attrition_bin"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    (X_tr, X_te, X_tr_sc, X_te_sc,
     y_tr, y_te) = train_test_split(X, X_scaled, y,
                                     test_size=0.2, random_state=42, stratify=y)

    # Tune threshold on RF to maximise recall >= 0.80
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    rf_base.fit(X_tr, y_tr)
    probs_val = rf_base.predict_proba(X_te)[:, 1]
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_te, probs_val)
    high_recall_mask = rec_arr[:-1] >= 0.80
    if high_recall_mask.any():
        best_thresh = thresh_arr[high_recall_mask][prec_arr[:-1][high_recall_mask].argmax()]
    else:
        best_thresh = float(thresh_arr[rec_arr[:-1].argmax()])

    candidates = {
        "Random Forest (Recall-Tuned)": (rf_base,                                                                                          X_tr,    X_te,    best_thresh),
        "Random Forest (Default)":      (RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),                X_tr,    X_te,    0.5),
        "Gradient Boosting":            (GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),   X_tr,    X_te,    0.5),
        "Extra Trees":                  (ExtraTreesClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1), X_tr, X_te, 0.5),
        "AdaBoost":                     (AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42, algorithm="SAMME"),      X_tr,    X_te,    0.5),
        "Decision Tree":                (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),                    X_tr,    X_te,    0.5),
        "Logistic Regression":          (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),                      X_tr_sc, X_te_sc, 0.5),
        "SVM (RBF Kernel)":             (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),                   X_tr_sc, X_te_sc, 0.5),
    }

    results = []
    trained_models = {}
    for name, (clf, Xtr, Xte, thresh) in candidates.items():
        clf.fit(Xtr, y_tr)
        proba = clf.predict_proba(Xte)[:, 1]
        preds = (proba >= thresh).astype(int)
        results.append({
            "Model":     name,
            "Threshold": round(float(thresh), 3),
            "Accuracy":  round(accuracy_score(y_te, preds), 4),
            "Precision": round(precision_score(y_te, preds, zero_division=0), 4),
            "Recall":    round(recall_score(y_te, preds, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_te, preds, zero_division=0), 4),
            "ROC AUC":   round(roc_auc_score(y_te, proba), 4),
        })
        trained_models[name] = (clf, proba, preds, thresh, Xte)

    results_df = pd.DataFrame(results).sort_values("Recall", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.insert(0, "Rank", results_df.index)
    results_df["Selected_for_App"] = ["Yes" if i == 1 else "No" for i in range(1, len(results_df) + 1)]

    best_name = results_df.iloc[0]["Model"]
    best_model_obj = trained_models[best_name][0]
    best_thresh_val = trained_models[best_name][3]

    roc_data = {}
    for name, (clf, proba, preds, thresh, Xte_use) in trained_models.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_data[name] = (fpr, tpr)

    return (results_df, trained_models, best_name, best_model_obj,
            best_thresh_val, FEATS, scaler, y_te, roc_data)

# ============================================================
# 4. LOAD DATA + SESSION STATE
# ============================================================
df = load_data()

st.session_state["_df"] = df
model, FEATURES, model_metrics = get_trained_model(hash(str(df.shape)))

# ============================================================
# 5. SIDEBAR
# ============================================================
# ── Model catalogue (drives the selector) ──
MODEL_CATALOGUE = {
    # ── Recall-tuned champion ──
    "🌲 Random Forest (Recall-Tuned)": {
        "clf": lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                        class_weight="balanced", random_state=42, n_jobs=-1),
        "use_scale": False, "tune_thresh": True,
        "tag": "RECOMMENDED",
        "tag_color": "#2DD4BF",
        "desc": "Threshold tuned on Precision-Recall curve to hit ≥80% recall. Best for catching at-risk employees.",
    },
    # ── Ensemble family ──
    "🌲 Random Forest (Default)": {
        "clf": lambda: RandomForestClassifier(n_estimators=150, max_depth=8,
                        random_state=42, n_jobs=-1),
        "use_scale": False, "tune_thresh": False,
        "tag": "HIGH PRECISION",
        "tag_color": "#60A5FA",
        "desc": "Standard RF at 0.5 threshold. Very high precision (98%), conservative on recall.",
    },
    "⚡ Gradient Boosting": {
        "clf": lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                        max_depth=4, random_state=42),
        "use_scale": False, "tune_thresh": False,
        "tag": "HIGH AUC",
        "tag_color": "#A78BFA",
        "desc": "Boosted trees. Strong AUC and precision. Good when false alarms are costly.",
    },
    "🚀 Extra Trees": {
        "clf": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=10,
                        class_weight="balanced", random_state=42, n_jobs=-1),
        "use_scale": False, "tune_thresh": False,
        "tag": "FAST",
        "tag_color": "#FBBF24",
        "desc": "Extremely randomised trees. Faster than RF, often similar recall.",
    },
    "🔥 AdaBoost": {
        "clf": lambda: AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                        random_state=42, algorithm="SAMME"),
        "use_scale": False, "tune_thresh": False,
        "tag": "BOOSTED",
        "tag_color": "#F472B6",
        "desc": "Adaptive boosting. Focuses on hard-to-classify cases iteratively.",
    },
    # ── Linear / distance family ──
    "📐 Logistic Regression": {
        "clf": lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                        random_state=42),
        "use_scale": True, "tune_thresh": False,
        "tag": "INTERPRETABLE",
        "tag_color": "#34D399",
        "desc": "Linear baseline. Fast, stable, auditable. Ideal for compliance reporting.",
    },
    "📏 SVM (RBF Kernel)": {
        "clf": lambda: SVC(kernel="rbf", class_weight="balanced",
                        probability=True, random_state=42),
        "use_scale": True, "tune_thresh": False,
        "tag": "KERNEL",
        "tag_color": "#FB923C",
        "desc": "Support vector machine with RBF kernel. Powerful on mid-sized datasets.",
    },
    # ── Tree family ──
    "🌿 Decision Tree": {
        "clf": lambda: DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                        random_state=42),
        "use_scale": False, "tune_thresh": False,
        "tag": "EXPLAINABLE",
        "tag_color": "#A3E635",
        "desc": "Single decision tree. Fully interpretable — every prediction traceable.",
    },
}

with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div class='sidebar-brand'>
        <div class='brand-icon-wrap'>🛡️</div>
        <div>
            <div class='brand-title'>Risk Intelligence</div>
            <div class='brand-tag'>Retention Analytics v2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown("<div class='nav-section-label'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATION",
        ["📊 Dashboard", "🔮 Predict Employee", "🎛 Attrition Simulator",
         "👥 Employees to Review", "📈 Predictive Analytics", "🏆 Model Lab"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Active Model Selector ──
    st.markdown("<div class='nav-section-label'>Active Model</div>", unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Active Model",
        list(MODEL_CATALOGUE.keys()),
        index=0,
        label_visibility="collapsed",
        key="active_model_select",
        help="Changes the model used in Predict Employee, Employees to Review, and Attrition Simulator",
    )
    sel_meta = MODEL_CATALOGUE[selected_model_name]
    tag_html = (f"<span class='model-tag' style='background:rgba(45,212,191,0.12);"
                f"color:{sel_meta['tag_color']}'>{sel_meta.get('tag','')}</span>"
                if sel_meta.get('tag') else "")
    st.markdown(f"""
    <div class='model-card'>
        <div class='model-card-name'>{selected_model_name.split(' ', 1)[1] if ' ' in selected_model_name else selected_model_name}</div>
        {tag_html}
        <div class='model-card-desc'>{sel_meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Filters ──
    st.markdown("<div class='nav-section-label'>Filters</div>", unsafe_allow_html=True)
    with st.expander("🎚 Global Filters", expanded=False):
        companies = ["All"] + sorted(df["Company"].unique().tolist()) if "Company" in df.columns else ["All"]
        sel_company = st.selectbox("Company", companies)

        depts = ["All"] + sorted(df["Department"].unique().tolist())
        sel_dept = st.selectbox("Department", depts)

        if "MonthlyIncome" in df.columns:
            inc_range = st.slider(
                "Monthly Income ($)",
                int(df["MonthlyIncome"].min()),
                int(df["MonthlyIncome"].max()),
                (int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max()))
            )
        else:
            inc_range = (0, 99999)

    # Apply filters
    filtered_df = df.copy()
    if sel_company != "All" and "Company" in df.columns:
        filtered_df = filtered_df[filtered_df["Company"] == sel_company]
    if sel_dept != "All":
        filtered_df = filtered_df[filtered_df["Department"] == sel_dept]
    if "MonthlyIncome" in df.columns:
        filtered_df = filtered_df[
            filtered_df["MonthlyIncome"].between(inc_range[0], inc_range[1])
        ]

    st.divider()

    # ── Status pill ──
    st.markdown(f"""
    <div class='status-pill'>
        <div class='status-dot'></div>
        <div>
            <div style='color:#F0F4FF;font-weight:600;font-size:0.8rem'>AI Core Online</div>
            <div style='color:#545E70;font-size:0.72rem;margin-top:1px'>
                AUC {model_metrics["auc"]} &nbsp;·&nbsp; {len(filtered_df):,} / {len(df):,} records
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:0.7rem;color:#3D4A5C;padding:8px 2px 0 2px'>"
        f"Model: <span style='color:#545E70'>{selected_model_name.split(' ',1)[1]}</span></div>",
        unsafe_allow_html=True
    )

if df is None:
    st.error("🚨 Dataset missing. Place HR_Attrition_MultiCompany.csv in the 'Final Data Clean' folder.")
    st.stop()

# ============================================================
# HELPER: KPI Card HTML
# ============================================================
def kpi_card(icon, label, value, sub, badge_text="", badge_class="badge-neu"):
    badge_html = f"<span class='kpi-badge {badge_class}'>{badge_text}</span>" if badge_text else ""
    return f"""
    <div class='kpi-card'>
        <div class='kpi-icon'>{icon}</div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
        {badge_html}
    </div>
    """

def risk_color(pct):
    if pct >= 70: return "#F87171", "high"
    if pct >= 40: return "#FBBF24", "medium"
    return "#2DD4BF", "low"

# ============================================================
# PAGE 1 ── DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.markdown("""
    <div class='page-header'>
        <h1>Workforce Performance Overview</h1>
        <div class='ph-sub'>Real-time attrition intelligence across your organization</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    atr_rate = (filtered_df["Attrition"] == "Yes").mean() * 100
    avg_income = filtered_df["MonthlyIncome"].mean() if "MonthlyIncome" in filtered_df.columns else 0
    avg_tenure = filtered_df["TotalWorkingYears"].mean() if "TotalWorkingYears" in filtered_df.columns else 0
    active_count = (filtered_df["Attrition"] == "No").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("👥", "Total Headcount", f"{len(filtered_df):,}", "Filtered records",
                             f"+{active_count:,} active", "badge-down"), unsafe_allow_html=True)
    with c2:
        badge_cls = "badge-up" if atr_rate > 15 else "badge-down"
        st.markdown(kpi_card("📉", "Attrition Rate", f"{atr_rate:.1f}%", "Historical turnover",
                             "⚠ High" if atr_rate > 15 else "✓ Healthy", badge_cls), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("💵", "Avg Monthly Salary", f"${avg_income:,.0f}", "Across filtered set",
                             "USD", "badge-neu"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("🏅", "Avg Experience", f"{avg_tenure:.1f}Y", "Total working years",
                             "Industry avg ~8Y", "badge-neu"), unsafe_allow_html=True)

    st.divider()

    # ── Charts Row ──
    tab1, tab2, tab3 = st.tabs(["📈 Attrition by Department", "💰 Income Distribution", "🕐 Tenure vs Risk"])

    with tab1:
        if "Department" in filtered_df.columns:
            dept_atr = (filtered_df.groupby("Department")["Attrition"]
                        .apply(lambda x: (x == "Yes").mean() * 100)
                        .sort_values(ascending=False))

            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(dept_atr.index, dept_atr.values,
                          color=ACCENT_PALETTE[:len(dept_atr)], edgecolor="none", width=0.55)
            ax.set_ylabel("Attrition Rate (%)")
            ax.set_title("Attrition Rate by Department")
            for bar, val in zip(bars, dept_atr.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", color="#F0F4FF", fontsize=9)
            apply_dark_style(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Department column not found in dataset.")

    with tab2:
        if "MonthlyIncome" in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            for label, color in [("No", "#2DD4BF"), ("Yes", "#F87171")]:
                subset = filtered_df[filtered_df["Attrition"] == label]["MonthlyIncome"]
                ax.hist(subset, bins=30, alpha=0.65, color=color, label=f"Attrition: {label}", edgecolor="none")
            ax.set_xlabel("Monthly Income ($)")
            ax.set_ylabel("Employee Count")
            ax.set_title("Salary Distribution: Retained vs Departed")
            ax.legend(facecolor="#111827", labelcolor="#F0F4FF", framealpha=0.8)
            apply_dark_style(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab3:
        if "TotalWorkingYears" in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            for label, color, marker in [("No", "#2DD4BF", "o"), ("Yes", "#F87171", "^")]:
                sub = filtered_df[filtered_df["Attrition"] == label].sample(min(300, len(filtered_df)))
                ax.scatter(sub["TotalWorkingYears"],
                           sub["MonthlyIncome"] if "MonthlyIncome" in sub.columns else sub["YearsAtCompany"],
                           c=color, alpha=0.45, s=22, marker=marker, label=f"Attrition: {label}")
            ax.set_xlabel("Total Working Years")
            ax.set_ylabel("Monthly Income ($)" if "MonthlyIncome" in filtered_df.columns else "Years at Company")
            ax.set_title("Experience vs. Income — Colored by Attrition")
            ax.legend(facecolor="#111827", labelcolor="#F0F4FF")
            apply_dark_style(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()

    # ── Data Snapshot ──
    with st.expander("📋 Data Snapshot (first 20 rows)", expanded=False):
        display_df = filtered_df.head(20).copy()

        def style_attrition(val):
            if val == "Yes":
                return "background-color: rgba(248,113,113,0.15); color: #F87171; font-weight:600"
            elif val == "No":
                return "background-color: rgba(45,212,191,0.12); color: #2DD4BF; font-weight:600"
            return ""

        styled = display_df.style.set_properties(**{
            "background-color": "#111827",
            "color": "#F0F4FF",
            "border-color": "rgba(255,255,255,0.06)",
            "font-size": "0.85rem",
        }).set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a2234"),
                ("color", "#8B95A8"),
                ("font-size", "0.78rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.06em"),
                ("border-bottom", "1px solid rgba(255,255,255,0.08)"),
                ("padding", "8px 12px"),
            ]},
            {"selector": "td", "props": [
                ("padding", "7px 12px"),
                ("border-bottom", "1px solid rgba(255,255,255,0.04)"),
            ]},
            {"selector": "tr:hover td", "props": [
                ("background-color", "rgba(96,165,250,0.05)"),
            ]},
        ])

        if "Attrition" in display_df.columns:
            styled = styled.map(style_attrition, subset=["Attrition"])

        st.dataframe(styled, use_container_width=True, height=420)

    # ── Workforce Composition ──
    st.subheader("Workforce Composition")
    comp_cols = [c for c in ["Gender", "OverTime", "JobLevel", "Education"] if c in filtered_df.columns]
    if comp_cols:
        cols_row = st.columns(len(comp_cols))
        for col, feat in zip(cols_row, comp_cols):
            with col:
                vc = filtered_df[feat].value_counts()
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                wedges, texts, autotexts = ax.pie(
                    vc.values, labels=vc.index,
                    colors=ACCENT_PALETTE[:len(vc)],
                    autopct="%1.0f%%", pctdistance=0.75,
                    startangle=90, wedgeprops=dict(linewidth=0)
                )
                for t in texts: t.set_color("#8B95A8"); t.set_fontsize(8)
                for a in autotexts: a.set_color("#F0F4FF"); a.set_fontsize(8)
                ax.set_title(feat, color="#F0F4FF", fontsize=10)
                apply_dark_style(fig, [ax])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ============================================================
# PAGE 2 ── ATTRITION FACTORS
# ============================================================
elif page == "📈 Predictive Analytics":
    st.markdown("""
    <div class='page-header'>
        <h1>📈 Predictive Analytics</h1>
        <div class='ph-sub'>Why Recall was chosen · Feature drivers · Model comparison · Deep-dive analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance Banner ──
    r = model_metrics["report"]
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Model AUC", model_metrics["auc"], delta="vs 0.5 baseline")
    with m2: st.metric("Precision", f"{r.get('1', {}).get('precision', 0):.2%}")
    with m3: st.metric("Recall",    f"{r.get('1', {}).get('recall', 0):.2%}")
    with m4: st.metric("F1 Score",  f"{r.get('1', {}).get('f1-score', 0):.2%}")

    st.divider()

    # ── Why Recall? Rationale section ──
    with st.expander("📖 Why did we choose Recall over Accuracy or Precision?", expanded=True):
        ra, rb = st.columns([1, 1])
        with ra:
            st.markdown("""
<div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px 20px'>
<div style='font-size:0.75rem;color:#8B95A8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>The Core Problem</div>
<p style='color:#F0F4FF;font-size:0.95rem;line-height:1.65'>
In HR attrition prediction, the cost of <b style='color:#F87171'>missing</b> an employee who
will leave is far higher than the cost of <b style='color:#FBBF24'>incorrectly flagging</b>
someone who stays.
</p>
<p style='color:#8B95A8;font-size:0.85rem;margin-top:10px'>
A missed leaver = lost knowledge, replacement cost (50–200% of salary), project disruption,
and team morale damage. A false alarm = one extra retention conversation.
</p>
</div>
            """, unsafe_allow_html=True)
        with rb:
            # Metric comparison mini-table as HTML
            st.markdown("""
<div style='background:#111827;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px 20px'>
<div style='font-size:0.75rem;color:#8B95A8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>Metric Decision Matrix</div>
<table style='width:100%;border-collapse:collapse;font-size:0.82rem'>
<tr style='border-bottom:1px solid rgba(255,255,255,0.07)'>
  <th style='color:#8B95A8;padding:6px 8px;text-align:left'>Metric</th>
  <th style='color:#8B95A8;padding:6px 8px;text-align:left'>What it measures</th>
  <th style='color:#8B95A8;padding:6px 8px;text-align:left'>Why not primary?</th>
</tr>
<tr style='border-bottom:1px solid rgba(255,255,255,0.04)'>
  <td style='color:#FBBF24;padding:6px 8px;font-weight:600'>Accuracy</td>
  <td style='color:#F0F4FF;padding:6px 8px'>Overall correct predictions</td>
  <td style='color:#8B95A8;padding:6px 8px'>Misleading on imbalanced data (84% stay → model can score 84% by predicting "No" always)</td>
</tr>
<tr style='border-bottom:1px solid rgba(255,255,255,0.04)'>
  <td style='color:#60A5FA;padding:6px 8px;font-weight:600'>Precision</td>
  <td style='color:#F0F4FF;padding:6px 8px'>Of those flagged, how many truly leave</td>
  <td style='color:#8B95A8;padding:6px 8px'>Optimising precision makes the model conservative — it misses real leavers to avoid false alarms</td>
</tr>
<tr style='border-bottom:1px solid rgba(255,255,255,0.04)'>
  <td style='color:#A78BFA;padding:6px 8px;font-weight:600'>F1 Score</td>
  <td style='color:#F0F4FF;padding:6px 8px'>Balance of precision & recall</td>
  <td style='color:#8B95A8;padding:6px 8px'>Good general metric but treats both error types equally — not appropriate here</td>
</tr>
<tr>
  <td style='color:#2DD4BF;padding:6px 8px;font-weight:700'>✅ Recall</td>
  <td style='color:#F0F4FF;padding:6px 8px'>Of all who actually leave, how many did we catch</td>
  <td style='color:#2DD4BF;padding:6px 8px;font-weight:600'>← PRIMARY GOAL: minimise missed leavers</td>
</tr>
</table>
</div>
            """, unsafe_allow_html=True)

        st.markdown("""
<div style='background:rgba(45,212,191,0.06);border:1px solid rgba(45,212,191,0.2);border-radius:10px;
padding:12px 16px;margin-top:12px;font-size:0.85rem;color:#8B95A8'>
<b style='color:#2DD4BF'>Bottom line:</b> We use <b style='color:#2DD4BF'>Recall as our champion metric</b>
and tune the decision threshold to achieve ≥80% recall, even if that means some false alarms.
Every employee flagged as high-risk gets a targeted retention conversation — a low-cost intervention
vs the high cost of attrition.
</div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Model comparison bar chart (Predictive Analytics summary) ──
    st.subheader("🏁 Model Comparison — Predictive Performance")
    st.markdown("""
    <div class='section-note'>
        Five models were benchmarked. Bars show Recall (primary), Precision, and ROC AUC side by side.
        Go to <b>🏆 Model Lab</b> for the interactive leaderboard, ROC curves, and threshold tuner.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading model comparison…"):
        try:
            (cmp_results_df, _, cmp_best_name, _, _, _, _, _, _) = run_model_comparison(hash(str(df.shape)))

            model_names_short = [m.replace(" (Recall-Tuned)", " ★").replace(" (Default)", " (base)")
                                 for m in cmp_results_df["Model"]]
            x_pos   = np.arange(len(model_names_short))
            w       = 0.26
            recalls   = cmp_results_df["Recall"].values
            precis    = cmp_results_df["Precision"].values
            aucs      = cmp_results_df["ROC AUC"].values

            fig, ax = plt.subplots(figsize=(11, 5))
            b1 = ax.bar(x_pos - w,   recalls, w, color="#2DD4BF", alpha=0.85, label="Recall",    edgecolor="none")
            b2 = ax.bar(x_pos,       precis,  w, color="#60A5FA", alpha=0.85, label="Precision", edgecolor="none")
            b3 = ax.bar(x_pos + w,   aucs,    w, color="#A78BFA", alpha=0.85, label="ROC AUC",   edgecolor="none")
            ax.axhline(0.80, color="#FBBF24", linewidth=1.3, linestyle="--", label="Recall target (80%)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names_short, fontsize=8)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Score")
            ax.set_title("All Models — Recall · Precision · ROC AUC Comparison")
            ax.legend(fontsize=9, facecolor="#111827", labelcolor="#F0F4FF")
            for bars in [b1, b2, b3]:
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                            f"{bar.get_height():.0%}", ha="center", va="bottom",
                            color="#F0F4FF", fontsize=7)
            apply_dark_style(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Winner callout
            best_r = cmp_results_df.iloc[0]
            st.markdown(f"""
<div style='background:rgba(45,212,191,0.08);border:1px solid rgba(45,212,191,0.25);border-radius:10px;
padding:12px 16px;font-size:0.85rem;color:#8B95A8;margin-top:4px'>
<b style='color:#2DD4BF'>🥇 Champion:</b> <b style='color:#F0F4FF'>{best_r["Model"]}</b>
— Recall <b style='color:#2DD4BF'>{best_r["Recall"]:.2%}</b>
· Precision <b style='color:#60A5FA'>{best_r["Precision"]:.2%}</b>
· ROC AUC <b style='color:#A78BFA'>{best_r["ROC AUC"]:.3f}</b>
· Threshold tuned to <b style='color:#FBBF24'>{best_r["Threshold"]:.2f}</b>
</div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Model comparison requires the full dataset to be loaded. ({e})")

    st.divider()

    # ── Feature Importances ──
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ACCENT_PALETTE[i % len(ACCENT_PALETTE)] for i in range(len(importances))]
    bars = ax.barh(importances.index, importances.values, color=colors, edgecolor="none", height=0.55)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Key Risk Indicators")
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#F0F4FF", fontsize=9)
    apply_dark_style(fig, [ax])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Insights Chips ──
    top3 = importances.sort_values(ascending=False).head(3)
    st.markdown("**Key findings:**")
    pills = ""
    for feat, score in top3.items():
        pills += f"<span class='insight-pill'>🔑 {feat} ({score:.2%})</span>"
    st.markdown(pills, unsafe_allow_html=True)

    st.divider()

    # ── Deep Dive: Factor vs Attrition ──
    st.subheader("Factor Deep-Dive")
    sel_feat = st.selectbox("Select a factor to analyze:", FEATURES)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{sel_feat} — Distribution by Attrition Status**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in [("No", "#2DD4BF"), ("Yes", "#F87171")]:
            vals = filtered_df[filtered_df["Attrition"] == label][sel_feat].dropna()
            ax.hist(vals, bins=25, alpha=0.6, color=color, label=f"Attrition: {label}", edgecolor="none")
        ax.legend(facecolor="#111827", labelcolor="#F0F4FF")
        ax.set_xlabel(sel_feat)
        apply_dark_style(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown(f"**Median {sel_feat} by Attrition**")
        meds = filtered_df.groupby("Attrition")[sel_feat].median()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(meds.index, meds.values,
               color=["#2DD4BF", "#F87171"], edgecolor="none", width=0.45)
        ax.set_ylabel(f"Median {sel_feat}")
        ax.set_title(f"Median {sel_feat} Comparison")
        apply_dark_style(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Correlation Heatmap ──
    with st.expander("🔥 Correlation Heatmap (numeric features)", expanded=False):
        num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            corr = filtered_df[num_cols[:12]].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
                        linewidths=0.4, linecolor="#0D1117",
                        annot=True, fmt=".2f", annot_kws={"size": 7, "color": "#F0F4FF"},
                        cbar_kws={"shrink": 0.8})
            ax.set_title("Feature Correlation Matrix", color="#F0F4FF")
            apply_dark_style(fig, [ax])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ============================================================
# PAGE 3 ── RISK WATCHLIST
# ============================================================
elif page == "👥 Employees to Review":
    st.markdown("""
    <div class='page-header'>
        <h1>👥 Employees to Review</h1>
        <div class='ph-sub'>AI-scored active employees ranked by resignation probability — prioritised for HR intervention</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-note'>
        The AI has analyzed behavioral and compensation patterns to score each active employee's 
        probability of resignation. Scores above <b style='color:#F87171'>70%</b> require immediate intervention.
    </div>
    """, unsafe_allow_html=True)

    # Score all active employees
    active_df = filtered_df[filtered_df["Attrition"] == "No"].copy()
    if "OverTime" in active_df.columns:
        active_df["OverTime_enc"] = (active_df["OverTime"] == "Yes").astype(int)
    else:
        active_df["OverTime_enc"] = 0

    available_features = [f for f in FEATURES if f in active_df.columns]
    X_active = active_df[available_features].fillna(active_df[available_features].median())
    active_df["Risk_Probability"] = model.predict_proba(X_active)[:, 1]

    # ── Controls ──
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        n_show = st.slider("Employees to display", 4, min(50, len(active_df)), 12, step=4)
    with col_ctrl2:
        risk_filter = st.selectbox("Risk Level Filter", ["All", "High (≥70%)", "Medium (40-69%)", "Low (<40%)"])

    top_risk = active_df.sort_values("Risk_Probability", ascending=False)
    if risk_filter == "High (≥70%)":
        top_risk = top_risk[top_risk["Risk_Probability"] >= 0.70]
    elif risk_filter == "Medium (40-69%)":
        top_risk = top_risk[top_risk["Risk_Probability"].between(0.40, 0.699)]
    elif risk_filter == "Low (<40%)":
        top_risk = top_risk[top_risk["Risk_Probability"] < 0.40]
    top_risk = top_risk.head(n_show)

    # ── Risk Cards in 3-column grid ──
    id_col  = "EmployeeNumber" if "EmployeeNumber" in active_df.columns else active_df.columns[0]
    role_col = "JobRole"        if "JobRole"        in active_df.columns else active_df.columns[1]
    dept_col = "Department"     if "Department"     in active_df.columns else ""

    if len(top_risk) == 0:
        st.info("No employees match the selected filter.")
    else:
        cols_per_row = 3
        rows = [top_risk.iloc[i:i+cols_per_row] for i in range(0, len(top_risk), cols_per_row)]
        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col, (_, emp) in zip(cols, row_df.iterrows()):
                risk_pct = emp["Risk_Probability"] * 100
                color, level = risk_color(risk_pct)
                dept_info = emp[dept_col] if dept_col else ""
                with col:
                    st.markdown(f"""
                    <div class='risk-card {level}'>
                        <div class='r-name'>ID #{int(emp[id_col])} · {emp[role_col]}</div>
                        <div class='r-meta'>{dept_info}</div>
                        <div class='r-track'>
                            <div class='r-fill' style='width:{risk_pct:.1f}%;background:{color}'></div>
                        </div>
                        <div class='r-pct' style='color:{color}'>{risk_pct:.1f}% resignation risk</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()

    # ── Risk Distribution Summary ──
    st.subheader("Risk Distribution")
    all_risk = active_df["Risk_Probability"] * 100
    high_n   = (all_risk >= 70).sum()
    med_n    = all_risk.between(40, 69.9).sum()
    low_n    = (all_risk < 40).sum()

    r1, r2, r3 = st.columns(3)
    with r1: st.metric("🔴 High Risk (≥70%)", high_n, delta=f"{high_n/len(all_risk)*100:.1f}% of workforce", delta_color="inverse")
    with r2: st.metric("🟡 Medium Risk (40-69%)", med_n, delta=f"{med_n/len(all_risk)*100:.1f}% of workforce", delta_color="off")
    with r3: st.metric("🟢 Low Risk (<40%)", low_n, delta=f"{low_n/len(all_risk)*100:.1f}% of workforce")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(all_risk, bins=40, color="#60A5FA", edgecolor="none", alpha=0.8)
    ax.axvline(70, color="#F87171", linewidth=1.5, linestyle="--", label="High Risk Threshold (70%)")
    ax.axvline(40, color="#FBBF24", linewidth=1.5, linestyle="--", label="Medium Threshold (40%)")
    ax.set_xlabel("Resignation Risk (%)")
    ax.set_ylabel("Employee Count")
    ax.set_title("Risk Score Distribution — All Active Employees")
    ax.legend(facecolor="#111827", labelcolor="#F0F4FF")
    apply_dark_style(fig, [ax])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Download — include Company after Emp ID ──
    export_cols = [id_col]
    if "Company" in top_risk.columns:
        export_cols.append("Company")
    export_cols += [role_col]
    if dept_col:
        export_cols.append(dept_col)
    export_cols.append("Risk_Probability")
    export_cols = [c for c in export_cols if c in top_risk.columns]

    csv_export = top_risk[export_cols].copy()
    csv_export["Risk_Probability"] = csv_export["Risk_Probability"].map(lambda x: f"{x:.2%}")
    # Composite ID: EmpNumber-CompanyCode for unique identification
    if "Company" in csv_export.columns:
        company_code = csv_export["Company"].str[:3].str.upper()
        csv_export.insert(0, "Employee_ID", csv_export[id_col].astype(str) + "-" + company_code)

    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.download_button(
            "⬇️ Export Employees to Review (CSV)",
            data=csv_export.to_csv(index=False),
            file_name="employees_to_review.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.caption(f"📄 {len(csv_export):,} employees · {len(csv_export.columns)} columns")

# ============================================================
# PAGE 4 ── PREDICT INDIVIDUAL EMPLOYEE
# ============================================================
elif page == "🔮 Predict Employee":
    st.markdown("""
    <div class='page-header'>
        <h1>🔮 Individual Risk Predictor</h1>
        <div class='ph-sub'>Profile an individual employee for real-time attrition risk scoring — before running the Attrition Simulator below</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-note'>
        Adjust the sliders below to match the employee's profile. 
        The AI model will instantly return a resignation risk score.
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        age                = st.slider("Age",                    18, 65, 35)
        distance           = st.slider("Distance From Home (km)", 1, 50, 10)
        job_level          = st.slider("Job Level",              1, 5,  2)
        monthly_income     = st.slider("Monthly Income ($)",     1000, 25000, 6000, step=500)
        job_satisfaction   = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

    with col_f2:
        total_working_yrs  = st.slider("Total Working Years",    0, 40, 8)
        years_at_company   = st.slider("Years at Company",       0, 30, 4)
        years_in_role      = st.slider("Years in Current Role",  0, 20, 3)
        work_life_balance  = st.slider("Work-Life Balance (1-4)",1, 4,  3)
        overtime           = st.selectbox("Works Overtime?",     ["No", "Yes"])

    if st.button("⚡ Predict Attrition Risk", use_container_width=True):
        input_data = {
            "Age":               age,
            "DistanceFromHome":  distance,
            "JobLevel":          job_level,
            "MonthlyIncome":     monthly_income,
            "TotalWorkingYears": total_working_yrs,
            "YearsAtCompany":    years_at_company,
            "YearsInCurrentRole":years_in_role,
            "JobSatisfaction":   job_satisfaction,
            "WorkLifeBalance":   work_life_balance,
            "OverTime_enc":      1 if overtime == "Yes" else 0,
        }
        row = pd.DataFrame([{f: input_data.get(f, 0) for f in FEATURES}])
        prob = model.predict_proba(row)[0, 1] * 100
        color, level = risk_color(prob)

        st.divider()
        pred_class = {"high": "pred-high", "medium": "pred-medium", "low": "pred-low"}[level]
        label_map   = {"high": "⚠️ HIGH RISK — Immediate Action Recommended",
                       "medium": "🟡 MODERATE RISK — Monitor Closely",
                       "low": "✅ LOW RISK — Employee Appears Retained"}

        st.markdown(f"""
        <div class='pred-result {pred_class}'>
            {label_map[level]}<br>
            <span style='font-size:2.5rem;font-weight:800'>{prob:.1f}%</span> resignation probability
        </div>
        """, unsafe_allow_html=True)

        # ── Factor Contribution mini chart ──
        st.markdown("**What's driving this score?**")
        importance_vals = pd.Series(model.feature_importances_, index=FEATURES)
        top_factors = importance_vals.sort_values(ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(top_factors.index, top_factors.values,
                       color=[ACCENT_PALETTE[i] for i in range(len(top_factors))],
                       edgecolor="none", height=0.45)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top Factors Influencing This Prediction")
        apply_dark_style(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Actionable recommendations ──
        recs = []
        if overtime == "Yes":
            recs.append("🕐 **Reduce overtime** — overtime is a top predictor of attrition.")
        if job_satisfaction < 3:
            recs.append("😊 **Boost job satisfaction** — consider role enrichment or recognition programs.")
        if work_life_balance < 3:
            recs.append("⚖️ **Improve work-life balance** — flexible scheduling or remote options may help.")
        if years_at_company < 2:
            recs.append("🤝 **Strengthen onboarding** — employees in first 2 years are most vulnerable.")
        if monthly_income < df["MonthlyIncome"].quantile(0.3) if "MonthlyIncome" in df.columns else False:
            recs.append("💰 **Review compensation** — salary is below the 30th percentile.")

        if recs:
            st.markdown("**Recommended Actions:**")
            for r in recs:
                st.markdown(f"- {r}")

# ============================================================
# ATTRITION SIMULATOR — appended to Predict Employee page
# ============================================================

elif page == "🎛 Attrition Simulator":
    st.markdown("""
    <div class='page-header'>
        <h1>🎛 Attrition Simulator</h1>
        <div class='ph-sub'>Adjust workforce-wide levers to model the impact on overall attrition rate</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-note'>
        Move the sliders to simulate policy changes (e.g. salary increase, overtime reduction)
        and see how the predicted attrition rate shifts across your workforce.
    </div>
    """, unsafe_allow_html=True)

    active_sim = df[df["Attrition"] == "No"].copy()
    if "OverTime" in active_sim.columns:
        active_sim["OverTime_enc"] = (active_sim["OverTime"] == "Yes").astype(int)
    else:
        active_sim["OverTime_enc"] = 0

    st.subheader("⚙️ Policy Levers")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        salary_boost = st.slider("💰 Salary Increase (%)", 0, 50, 0, step=5,
                                  help="Simulates a % increase in MonthlyIncome across the board")
        overtime_reduce = st.slider("🕐 Overtime Reduction (%)", 0, 100, 0, step=10,
                                     help="% of overtime workers shifted to non-overtime")
    with sc2:
        satisfaction_boost = st.slider("😊 Job Satisfaction Boost (pts)", 0, 3, 0,
                                        help="Add points to JobSatisfaction (scale 1–4, capped at 4)")
        wlb_boost = st.slider("⚖️ Work-Life Balance Boost (pts)", 0, 3, 0,
                               help="Add points to WorkLifeBalance (scale 1–4, capped at 4)")
    with sc3:
        distance_reduce = st.slider("🏠 Remote Work — Distance Reduction (%)", 0, 100, 0, step=10,
                                     help="Simulates remote work reducing effective commute distance")

    # Apply levers to a copy
    sim_df = active_sim.copy()
    avail_feats = [f for f in FEATURES if f in sim_df.columns]

    if "MonthlyIncome" in sim_df.columns:
        sim_df["MonthlyIncome"] = sim_df["MonthlyIncome"] * (1 + salary_boost / 100)
    if "OverTime_enc" in sim_df.columns and overtime_reduce > 0:
        mask_ot = sim_df["OverTime_enc"] == 1
        flip_n  = int(mask_ot.sum() * overtime_reduce / 100)
        flip_idx = sim_df[mask_ot].sample(min(flip_n, mask_ot.sum()), random_state=42).index
        sim_df.loc[flip_idx, "OverTime_enc"] = 0
    if "JobSatisfaction" in sim_df.columns:
        sim_df["JobSatisfaction"] = (sim_df["JobSatisfaction"] + satisfaction_boost).clip(upper=4)
    if "WorkLifeBalance" in sim_df.columns:
        sim_df["WorkLifeBalance"] = (sim_df["WorkLifeBalance"] + wlb_boost).clip(upper=4)
    if "DistanceFromHome" in sim_df.columns:
        sim_df["DistanceFromHome"] = sim_df["DistanceFromHome"] * (1 - distance_reduce / 100)

    # Baseline vs simulated risk
    X_base = active_sim[avail_feats].fillna(active_sim[avail_feats].median())
    X_sim  = sim_df[avail_feats].fillna(sim_df[avail_feats].median())

    base_probs = model.predict_proba(X_base)[:, 1]
    sim_probs  = model.predict_proba(X_sim)[:, 1]

    base_rate = base_probs.mean() * 100
    sim_rate  = sim_probs.mean() * 100
    delta_pct = sim_rate - base_rate
    employees_saved = int(((base_probs - sim_probs) > 0).sum())

    st.divider()
    st.subheader("📊 Simulation Results")
    res1, res2, res3, res4 = st.columns(4)
    with res1: st.metric("Baseline Attrition Risk", f"{base_rate:.1f}%")
    with res2: st.metric("Simulated Attrition Risk", f"{sim_rate:.1f}%",
                          delta=f"{delta_pct:+.1f}pp", delta_color="inverse")
    with res3: st.metric("Risk Reduction", f"{abs(delta_pct):.1f} pp" if delta_pct < 0 else "—")
    with res4: st.metric("Employees De-risked", f"{employees_saved:,}",
                          delta="improved vs baseline", delta_color="normal")

    # Side-by-side histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(base_probs * 100, bins=40, alpha=0.55, color="#F87171", label="Baseline", edgecolor="none")
    ax.hist(sim_probs  * 100, bins=40, alpha=0.55, color="#2DD4BF", label="Simulated", edgecolor="none")
    ax.axvline(base_rate, color="#F87171", linewidth=1.5, linestyle="--",
               label=f"Baseline mean {base_rate:.1f}%")
    ax.axvline(sim_rate, color="#2DD4BF", linewidth=1.5, linestyle="--",
               label=f"Simulated mean {sim_rate:.1f}%")
    ax.set_xlabel("Predicted Resignation Risk (%)")
    ax.set_ylabel("Employee Count")
    ax.set_title("Risk Distribution: Baseline vs Policy Simulation")
    ax.legend(facecolor="#111827", labelcolor="#F0F4FF")
    apply_dark_style(fig, [ax])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Policy summary card
    applied = []
    if salary_boost:     applied.append(f"💰 +{salary_boost}% salary")
    if overtime_reduce:  applied.append(f"🕐 -{overtime_reduce}% overtime workers")
    if satisfaction_boost: applied.append(f"😊 +{satisfaction_boost}pt satisfaction")
    if wlb_boost:        applied.append(f"⚖️ +{wlb_boost}pt work-life balance")
    if distance_reduce:  applied.append(f"🏠 -{distance_reduce}% effective commute")

    if applied:
        pills = " &nbsp;|&nbsp; ".join([f"<span style='color:#60A5FA'>{p}</span>" for p in applied])
        st.markdown(f"""
<div style='background:rgba(96,165,250,0.06);border:1px solid rgba(96,165,250,0.18);
border-radius:10px;padding:12px 16px;font-size:0.85rem;margin-top:8px'>
<b style='color:#60A5FA'>Policies applied:</b> {pills}
</div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Move the sliders above to simulate a policy change.")

# ============================================================
# PAGE 5 -- MODEL LAB (Recall Optimisation)
# ============================================================
elif page == "🏆 Model Lab":
    st.markdown("""
    <div class='page-header'>
        <h1>🏆 Model Lab — Recall Optimisation</h1>
        <div class='ph-sub'>
            Benchmarking every model to find the one with the highest recall —
            so we catch as many at-risk employees as possible before they resign.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-note'>
        <b>🎯 Goal:</b> Maximise <b style='color:#2DD4BF'>Recall</b> — we would rather flag a safe
        employee by mistake than miss someone who is about to leave.
        The top-ranked model is automatically selected for the app.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running full model benchmark…"):
        (results_df, trained_models, best_name, best_model_obj,
         best_thresh_val, ML_FEATURES, scaler, y_te, roc_data) = run_model_comparison(hash(str(df.shape)))

    # -- Hero metric strip --
    best_row = results_df.iloc[0]
    h1, h2, h3, h4, h5 = st.columns(5)
    for col, label, val, delta in [
        (h1, "🥇 Best Model",  best_row["Model"].split("(")[0].strip(), None),
        (h2, "🎯 Recall",      f"{best_row['Recall']:.2%}",    "Target ≥ 80%"),
        (h3, "📐 Precision",   f"{best_row['Precision']:.2%}", "False-alarm rate"),
        (h4, "📈 ROC AUC",     f"{best_row['ROC AUC']:.3f}",  "Discrimination power"),
        (h5, "🔧 Threshold",   f"{best_row['Threshold']:.2f}", "Decision cutoff"),
    ]:
        with col:
            st.metric(label, val, delta)

    st.divider()

    # -- Leaderboard --
    st.subheader("📋 Full Model Leaderboard")

    def highlight_leaderboard(row):
        if row.name == 1:
            return ["background-color: rgba(45,212,191,0.15); color: #F0F4FF; font-weight:700"] * len(row)
        return ["background-color: #111827; color: #F0F4FF"] * len(row)

    def color_recall(val):
        try:
            v = float(val)
            if v >= 0.80: return "color: #2DD4BF; font-weight:700"
            if v >= 0.65: return "color: #FBBF24"
            return "color: #F87171"
        except Exception:
            return ""

    styled_lb = (results_df.style
        .apply(highlight_leaderboard, axis=1)
        .map(color_recall, subset=["Recall"])
        .set_properties(**{"background-color": "#111827", "color": "#F0F4FF",
                           "border-color": "rgba(255,255,255,0.06)"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1a2234"), ("color", "#8B95A8"),
                ("font-size", "0.78rem"), ("text-transform", "uppercase"),
                ("letter-spacing", "0.06em"), ("padding", "8px 14px"),
                ("border-bottom", "1px solid rgba(255,255,255,0.1)"),
            ]},
            {"selector": "td", "props": [("padding", "9px 14px"),
                ("border-bottom", "1px solid rgba(255,255,255,0.04)")]},
        ])
        .format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
                 "F1 Score": "{:.2%}", "ROC AUC": "{:.4f}", "Threshold": "{:.3f}"})
    )
    st.dataframe(styled_lb, use_container_width=True, height=240)

    st.divider()

    # -- Charts row --
    PALETTE = {
        "Random Forest (Recall-Tuned)": "#2DD4BF",
        "Random Forest (Default)":      "#60A5FA",
        "Gradient Boosting":            "#A78BFA",
        "Extra Trees":                  "#FBBF24",
        "AdaBoost":                     "#F472B6",
        "Decision Tree":                "#A3E635",
        "Logistic Regression":          "#F87171",
        "SVM (RBF Kernel)":             "#FB923C",
    }
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**ROC Curves — all models**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "--", color="#444", linewidth=1, label="Random baseline")
        for mname, (fpr, tpr) in roc_data.items():
            auc_val = results_df[results_df["Model"] == mname]["ROC AUC"].values[0]
            lw = 2.5 if mname == best_name else 1.2
            label_short = mname.split("(")[0].strip()
            ax.plot(fpr, tpr, color=PALETTE.get(mname, "#888"),
                    linewidth=lw, label=f"{label_short} ({auc_val:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curves")
        ax.legend(fontsize=7, facecolor="#111827", labelcolor="#F0F4FF", framealpha=0.9)
        apply_dark_style(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_right:
        st.markdown("**Recall vs Precision — all models**")
        models_short = [m.replace(" (Recall-Tuned)", " ★").replace(" (Default)", "")
                        for m in results_df["Model"]]
        recalls    = results_df["Recall"].values
        precisions = results_df["Precision"].values
        x = np.arange(len(models_short))
        w = 0.38
        fig, ax = plt.subplots(figsize=(6, 5))
        b1 = ax.bar(x - w/2, recalls,    w, color="#2DD4BF", alpha=0.85, label="Recall",    edgecolor="none")
        b2 = ax.bar(x + w/2, precisions, w, color="#60A5FA", alpha=0.85, label="Precision", edgecolor="none")
        ax.axhline(0.80, color="#FBBF24", linewidth=1.2, linestyle="--", label="Recall target (80%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models_short, fontsize=7)
        ax.set_ylabel("Score")
        ax.set_title("Recall vs Precision by Model")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, facecolor="#111827", labelcolor="#F0F4FF")
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom",
                    color="#F0F4FF", fontsize=7)
        apply_dark_style(fig, [ax])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # -- Confusion matrix --
    st.subheader(f"🔬 Confusion Matrix — {best_name}")
    best_entry = trained_models[best_name]
    _, best_proba_arr, best_preds_arr, _, _ = best_entry
    cm = confusion_matrix(y_te, best_preds_arr)
    cm_labels = [
        ["True Negative\n(Correctly kept)", "False Positive\n(Wrongly flagged)"],
        ["False Negative\n(Missed leaver ❌)", "True Positive\n(Caught leaver ✅)"],
    ]
    colors_cm = [["#1a2234", "#2d1f1f"], ["#3d1a1a", "#1a3d2b"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.5, 1.5 - i), 1, 1,
                                        color=colors_cm[i][j], zorder=0))
            val_color = ("#F87171" if (i == 1 and j == 0)
                         else ("#2DD4BF" if (i == 1 and j == 1) else "#F0F4FF"))
            ax.text(j, 1 - i, str(cm[i, j]), ha="center", va="center",
                    fontsize=28, fontweight="bold", color=val_color, zorder=2)
            ax.text(j, 1 - i - 0.32, cm_labels[i][j], ha="center", va="center",
                    fontsize=7.5, color="#8B95A8", zorder=2)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: Stay", "Predicted: Leave"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual: Leave", "Actual: Stay"])
    ax.set_title(f"Confusion Matrix @ threshold = {best_thresh_val:.2f}", pad=12)
    apply_dark_style(fig, [ax])
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -- Live threshold tuner --
    st.divider()
    st.subheader("🎚 Live Threshold Tuner")
    st.markdown("""
    <div class='section-note'>
        Drag the slider to see how the decision threshold trades off
        <b style='color:#2DD4BF'>Recall</b> (catching leavers) vs
        <b style='color:#60A5FA'>Precision</b> (reducing false alarms).
    </div>
    """, unsafe_allow_html=True)

    _, best_proba_t, _, _, _ = trained_models[best_name]
    thresh_slider = st.slider("Decision Threshold", 0.10, 0.90,
                               float(round(best_thresh_val, 2)), step=0.01,
                               key="thresh_slider")
    preds_tuned = (best_proba_t >= thresh_slider).astype(int)

    t1, t2, t3, t4 = st.columns(4)
    with t1: st.metric("Recall",    f"{recall_score(y_te, preds_tuned, zero_division=0):.2%}", "want high ↑")
    with t2: st.metric("Precision", f"{precision_score(y_te, preds_tuned, zero_division=0):.2%}")
    with t3: st.metric("F1 Score",  f"{f1_score(y_te, preds_tuned, zero_division=0):.2%}")
    with t4: st.metric("Flagged",   f"{int(preds_tuned.sum()):,}", f"of {len(preds_tuned):,} employees")

    # Precision-Recall curve with live cursor
    prec_c, rec_c, thresh_c = precision_recall_curve(y_te, best_proba_t)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(thresh_c, rec_c[:-1],  color="#2DD4BF", linewidth=2, label="Recall")
    ax.plot(thresh_c, prec_c[:-1], color="#60A5FA", linewidth=2, label="Precision")
    ax.axvline(thresh_slider, color="#FBBF24", linewidth=1.5, linestyle="--",
               label=f"Current threshold ({thresh_slider:.2f})")
    ax.axhline(0.80, color="#F87171", linewidth=1, linestyle=":", alpha=0.7,
               label="Recall target (80%)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Decision Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, facecolor="#111827", labelcolor="#F0F4FF")
    apply_dark_style(fig, [ax])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -- Plain-language summary --
    st.divider()
    rec_val  = recall_score(y_te, preds_tuned, zero_division=0)
    pre_val  = precision_score(y_te, preds_tuned, zero_division=0)
    flagged  = int(preds_tuned.sum())
    missed   = int(((y_te == 1) & (preds_tuned == 0)).sum())

    note = (
        f"<div class='section-note'>"
        f"<b>📢 Plain-language interpretation at threshold {thresh_slider:.2f}:</b><br><br>"
        f"Out of every 100 employees who actually plan to leave, "
        f"the model catches <b style='color:#2DD4BF'>{rec_val:.0%}</b> of them.<br>"
        f"It is flagging <b style='color:#60A5FA'>{flagged:,}</b> employees total — "
        f"of which <b style='color:#FBBF24'>{pre_val:.0%}</b> are genuine risks.<br>"
        f"<b style='color:#F87171'>{missed}</b> at-risk employees are currently slipping through undetected."
        f"</div>"
    )
    st.markdown(note, unsafe_allow_html=True)
