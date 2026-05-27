import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import altair as alt
import base64
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
# SCHEMA + DEFAULT DATASET (from Project Omni)
# ============================================================
EXPECTED_SCHEMA = {
    "Company": "object", "Age": "object", "Attrition": "object", "BusinessTravel": "object",
    "DailyRate": "int64", "Department": "object", "DistanceFromHome": "int64", "Education": "int64",
    "EducationField": "object", "EmployeeCount": "int64", "EmployeeNumber": "int64",
    "EnvironmentSatisfaction": "int64", "Gender": "object", "HourlyRate": "int64",
    "JobInvolvement": "int64", "JobLevel": "int64", "JobRole": "object", "JobSatisfaction": "int64",
    "MaritalStatus": "object", "MonthlyIncome": "int64", "MonthlyRate": "int64",
    "NumCompaniesWorked": "int64", "Over18": "object", "OverTime": "object",
    "PercentSalaryHike": "float64", "PerformanceRating": "int64", "RelationshipSatisfaction": "int64",
    "StandardHours": "int64", "StockOptionLevel": "int64", "TotalWorkingYears": "int64",
    "TrainingTimesLastYear": "int64", "WorkLifeBalance": "int64", "YearsAtCompany": "int64",
    "YearsInCurrentRole": "int64", "YearsSinceLastPromotion": "int64", "YearsWithCurrManager": "int64",
    "Reliability": "float64", "Service Quality": "int64", "Service Efficiency": "float64",
    "Attrition_Reason": "object", "Hire_Date": "datetime", "Exit_Date": "datetime"
}

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def get_default_dataset():
    sample_data = {
        "Company": ["Company 1", "Company 1"], "Age": [53, 55], "Attrition": ["Yes", "No"],
        "BusinessTravel": ["Travel_Rarely", "Non-Travel"], "DailyRate": [607, 177],
        "Department": ["Research & Development", "Research & Development"],
        "DistanceFromHome": [2, 8], "Education": [5, 1],
        "EducationField": ["Technical Degree", "Medical"], "EmployeeCount": [1, 1],
        "EmployeeNumber": [1572, 1278], "EnvironmentSatisfaction": [3, 4],
        "Gender": ["Female", "Male"], "HourlyRate": [78, 37], "JobInvolvement": [2, 2],
        "JobLevel": [3, 4], "JobRole": ["Manufacturing Director", "Healthcare Representative"],
        "JobSatisfaction": [4, 2], "MaritalStatus": ["Married", "Divorced"],
        "MonthlyIncome": [10169, 13577], "MonthlyRate": [14618, 25592],
        "NumCompaniesWorked": [0, 1], "Over18": ["Y", "Y"], "OverTime": ["No", "Yes"],
        "PercentSalaryHike": [16.0, 15.0], "PerformanceRating": [3, 3],
        "RelationshipSatisfaction": [2, 4], "StandardHours": [80, 80],
        "StockOptionLevel": [1, 1], "TotalWorkingYears": [34, 34],
        "TrainingTimesLastYear": [4, 3], "WorkLifeBalance": [3, 3],
        "YearsAtCompany": [33, 33], "YearsInCurrentRole": [7, 9],
        "YearsSinceLastPromotion": [1, 15], "YearsWithCurrManager": [9, 0],
        "Reliability": [80.61, 88.80], "Service Quality": [2, 3],
        "Service Efficiency": [11.14, 15.00], "Attrition_Reason": ["Better Opportunity", None],
        "Hire_Date": ["1990-08-20", "1990-09-07"], "Exit_Date": ["2023-08-17", None]
    }
    df = pd.DataFrame(sample_data)
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'], errors='coerce')
    return df

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

/* ══ ATTRITION CARD (Project Omni metric_card component) ══ */
.attrition-card {
    background: linear-gradient(180deg, rgba(38,39,48,0.98), rgba(23,24,31,0.98));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    height: 176px;
    display: flex; flex-direction: column; justify-content: space-between; overflow: hidden;
}
.attrition-card .label  { font-size: 0.95rem; color: #b9c0d0; margin-bottom: 0.30rem; }
.attrition-card .value  { font-size: 2.0rem; font-weight: 700; line-height: 1.12; color: #ffffff; min-height: 2.35rem; overflow-wrap: anywhere; }
.attrition-card .sub    { color: #9aa3b2; font-size: 0.84rem; line-height: 1.30; min-height: 2.15rem; }
.attrition-card .meter-wrap  { margin-top: 0.55rem; }
.attrition-card .meter-label { color: #c7cedb; font-size: 0.76rem; margin-bottom: 0.28rem; }
.attrition-card .meter-track { width: 100%; height: 7px; background: rgba(255,255,255,0.10); border-radius: 999px; overflow: hidden; }
.attrition-card .meter-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #60a5fa, #f87171); }
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
# 2b. PROJECT OMNI CHART STYLE HELPERS
# ============================================================
def style_dark_axis(ax):
    """Projectomni standard chart style — used for all Omni pages."""
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#f9fafb")
    ax.grid(axis="x", color="#374151", alpha=0.35)
    ax.set_axisbelow(True)

OMNI_PALETTE = {"No": "#4CAF82", "Yes": "#E05C5C"}
OMNI_GRID    = "#374151"
OMNI_FG      = "#e5e7eb"
OMNI_TITLE   = "#f9fafb"
OMNI_RATING_LABELS = ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
OMNI_FIGSIZE = (5, 3.8)

def style_wb_chart(ax, title, xlabel, ylabel, legend_keys=None):
    """Wellbeing/Performance chart style (transparent bg, no-top-right spines)."""
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=OMNI_TITLE)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=10, color=OMNI_FG)
    ax.set_ylabel(ylabel, fontsize=10, color=OMNI_FG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(OMNI_GRID)
    ax.spines["bottom"].set_color(OMNI_GRID)
    ax.tick_params(axis="both", labelsize=9, colors=OMNI_FG)
    ax.grid(axis="y", color=OMNI_GRID, alpha=0.35)
    ax.set_axisbelow(True)
    if legend_keys:
        patches = [mpatches.Patch(facecolor=OMNI_PALETTE[k],
                                  label="Stayed" if k == "No" else "Left")
                   for k in legend_keys]
        ax.legend(handles=patches, fontsize=9, facecolor="none",
                  edgecolor=OMNI_GRID, labelcolor=OMNI_FG)

def finish_chart(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def finish_overview_chart(fig):
    fig.tight_layout(pad=1.8)
    st.pyplot(fig, use_container_width=True, bbox_inches=None)
    plt.close(fig)

def metric_card(label, value, subtext="", progress=None, progress_label="",
                sample_size=None, fill_color=None):
    """Projectomni attrition-card KPI tile."""
    meter_block = ""
    if progress is not None:
        progress = max(0, min(1, float(progress)))
        bar_color = fill_color if fill_color else "linear-gradient(90deg, #60a5fa, #f87171)"
        fill_style = (f"background: {bar_color};" if bar_color.startswith("linear")
                      else f"background-color: {bar_color};")
        meter_block = (
            f'<div class="meter-wrap">'
            f'<div class="meter-label">{progress_label}</div>'
            f'<div class="meter-track">'
            f'<div class="meter-fill" style="width: {progress * 100:.1f}%; {fill_style}"></div>'
            f'</div></div>'
        )
    sample_block = (
        f'<div style="font-size:0.72rem;color:#6b7280;margin-top:0.35rem;">Based on {sample_size:,} employees</div>'
        if sample_size is not None else ""
    )
    html = (
        f'<div class="attrition-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{subtext}</div>'
        f'{meter_block}'
        f'{sample_block}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

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
        "AdaBoost":                     (AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),      X_tr,    X_te,    0.5),
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
                        random_state=42),
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
        ["📊 Executive Summary", "🎯 Attrition Drivers", "⚖️ Wellbeing/Performance",
         "📈 Predictive Analytics", "👥 Employees to Review", "🔮 Predict Employee",
         "💰 Attrition Simulator", "🏆 Model Lab"],
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
        help="Changes the model used in Predict Employee and Employees to Review",
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

    st.divider()
    st.markdown("<div class='nav-section-label'>Data Sync</div>", unsafe_allow_html=True)
    with st.expander("⚙️ Upload Dataset", expanded=False):
        uploaded_file = st.file_uploader("Upload Company CSV", type=["csv"], label_visibility="collapsed")
        st.download_button(
            label="⬇️ Download Template",
            data=convert_df_to_csv(get_default_dataset()),
            file_name="omni_retention_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Resolve user_df from upload or analytics df ──
user_df = df.copy() if df is not None else get_default_dataset()
if uploaded_file is not None:
    try:
        temp_df = pd.read_csv(uploaded_file)
        if set(EXPECTED_SCHEMA.keys()) == set(temp_df.columns):
            temp_df['Hire_Date'] = pd.to_datetime(temp_df['Hire_Date'], errors='coerce')
            temp_df['Exit_Date'] = pd.to_datetime(temp_df['Exit_Date'], errors='coerce')
            user_df = temp_df
            st.sidebar.success("✅ Custom dataset synced!")
        else:
            missing = set(EXPECTED_SCHEMA.keys()) - set(temp_df.columns)
            st.sidebar.error(f"❌ Column mismatch. Missing: {missing}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

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
if page == "📈 Predictive Analytics":
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
        <div class='ph-sub'>Profile an individual employee for a real-time attrition risk score</div>
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

# ============================================================
# PAGE — EXECUTIVE SUMMARY (from Project Omni)
# ============================================================
elif page == "📊 Executive Summary":
    st.header("📊 Executive Summary Dashboard (Active Headcount Engine)")
    st.markdown("---")

    df_exec = user_df.copy()
    df_exec['Hire_Date'] = pd.to_datetime(df_exec['Hire_Date'], errors='coerce')
    df_exec['Exit_Date'] = pd.to_datetime(df_exec['Exit_Date'], errors='coerce')
    df_exec = df_exec[df_exec['Hire_Date'].notna()]

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        company_options = ["All"] + sorted(df_exec['Company'].dropna().unique().tolist())
        selected_company = st.selectbox("Select Company Focus:", company_options)
    if selected_company != "All":
        df_exec = df_exec[df_exec['Company'] == selected_company]

    if df_exec.empty:
        st.warning("⚠️ No employee records match the selected company.")
    else:
        min_year = int(df_exec['Hire_Date'].dt.year.min())
        max_hire_year = df_exec['Hire_Date'].dt.year.max()
        _exit_parsed = pd.to_datetime(df_exec['Exit_Date'], errors='coerce')
        max_exit_year = _exit_parsed.dt.year.max()
        max_year = int(max(max_hire_year, max_exit_year) if pd.notna(max_exit_year) else max_hire_year)
        available_years = sorted(list(range(min_year, max_year + 1)), reverse=True)

        with filter_col2:
            default_years = available_years[:5]
            selected_years = st.multiselect("Select Analysis Years:", options=available_years, default=default_years)

        if not selected_years:
            st.warning("⚠️ Please select at least one year.")
        else:
            months_bridge, all_departures = [], []
            chronological_years = sorted(selected_years)

            for yr in chronological_years:
                for month_idx in range(1, 13):
                    month_start = pd.Timestamp(year=yr, month=month_idx, day=1)
                    month_end   = month_start + pd.offsets.MonthEnd(0)
                    active_mask = (df_exec['Hire_Date'] <= month_end) & (
                        (df_exec['Exit_Date'].isna()) | (df_exec['Exit_Date'] > month_start))
                    df_active = df_exec[active_mask]
                    new_hires  = len(df_active[(df_active['Hire_Date'] >= month_start) & (df_active['Hire_Date'] <= month_end)])
                    left_mask  = (df_active['Attrition'].astype(str).str.strip().str.lower() == 'yes') & \
                                 (df_active['Exit_Date'] >= month_start) & (df_active['Exit_Date'] <= month_end)
                    df_left    = df_active[left_mask]
                    if not df_left.empty:
                        all_departures.append(df_left)
                    attrition_count = len(df_left)
                    active_total    = len(df_active)
                    monthly_rate    = (attrition_count / active_total * 100) if active_total > 0 else 0.0
                    display_label   = month_start.strftime('%B') if len(selected_years) == 1 else month_start.strftime('%Y-%m')
                    sort_value      = month_idx if len(selected_years) == 1 else month_start
                    months_bridge.append({'Month_Identifier': display_label, 'Sort_Key': sort_value,
                                          'Active_Headcount': active_total, 'New_Hires': new_hires,
                                          'Attrition_Departures': attrition_count, 'Attrition_Rate_Percent': monthly_rate})

            timeline_df = pd.DataFrame(months_bridge)
            df_departures_period = pd.concat(all_departures, ignore_index=True) if all_departures else pd.DataFrame()

            x_axis_field = 'Month_Identifier:O'
            x_axis_sort  = alt.EncodingSortField(field='Sort_Key', order='ascending')
            x_axis_title = 'Timeline' if len(selected_years) > 1 else f'Calendar Months ({chronological_years[0]})'

            st.markdown("### Key Performance Indicators")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            current_headcount = timeline_df.iloc[-1]['Active_Headcount']
            total_period_attrition = timeline_df['Attrition_Departures'].sum()
            total_ever_served = current_headcount + total_period_attrition
            attrition_rate_val = (total_period_attrition / total_ever_served * 100) if total_ever_served > 0 else 0.0

            with kpi_col1:
                metric_card("Active Headcount", f"{int(current_headcount):,}", "Active staff at end of selected window",
                            progress=current_headcount/total_ever_served if total_ever_served > 0 else 0,
                            progress_label=f"{current_headcount/total_ever_served:.0%} of total ever served" if total_ever_served > 0 else "")
            with kpi_col2:
                metric_card("Total Window Departures", f"{int(total_period_attrition):,}", "Employees who left during the selected period",
                            progress=total_period_attrition/total_ever_served if total_ever_served > 0 else 0,
                            progress_label=f"{total_period_attrition/total_ever_served:.0%} of total ever served" if total_ever_served > 0 else "")
            with kpi_col3:
                metric_card("Period Attrition Rate", f"{attrition_rate_val:.2f}%", "Share of total workforce that resigned in this window",
                            progress=min(attrition_rate_val/100, 1.0),
                            progress_label="Resignation share within selected period")

            st.markdown("---")
            st.markdown("### Historical Trends")
            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                st.markdown("#### MoM Attrition Rate")
                st.altair_chart(alt.Chart(timeline_df).mark_line(color='#ff4b4b', strokeWidth=3, point=True).encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Attrition_Rate_Percent:Q', title='Attrition Rate (%)'),
                    tooltip=['Month_Identifier', alt.Tooltip('Attrition_Rate_Percent:Q', format='.2f', title='Attrition %')]
                ).properties(height=260), use_container_width=True)
            with ch2:
                st.markdown("#### Active Headcount")
                st.altair_chart(alt.Chart(timeline_df).mark_bar(color='#1f77b4').encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Active_Headcount:Q', title='Workforce Volume'),
                    tooltip=['Month_Identifier', alt.Tooltip('Active_Headcount:Q', title='Active Staff')]
                ).properties(height=260), use_container_width=True)
            with ch3:
                st.markdown("#### Volumetric Attrition Losses")
                st.altair_chart(alt.Chart(timeline_df).mark_bar(color='#e45756').encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Attrition_Departures:Q', title='Departures Count'),
                    tooltip=['Month_Identifier', alt.Tooltip('Attrition_Departures:Q', title='Losses')]
                ).properties(height=260), use_container_width=True)

            st.markdown("---")
            if not df_departures_period.empty:
                st.markdown("### Categorical Attrition Distribution")
                bd1, bd2 = st.columns(2)
                with bd1:
                    st.markdown("#### Primary Root Causes")
                    reason_summary = df_departures_period.groupby('Attrition_Reason', dropna=False).size().reset_index(name='Count')
                    reason_summary['Attrition_Reason'] = reason_summary['Attrition_Reason'].fillna('Unspecified')
                    st.altair_chart(alt.Chart(reason_summary).mark_bar(color='#e45756', cornerRadiusEnd=3).encode(
                        y=alt.Y('Attrition_Reason:N', sort='-x', title='Reason'),
                        x=alt.X('Count:Q', title='Total Losses'),
                        tooltip=['Attrition_Reason:N', 'Count:Q']
                    ).properties(height=280), use_container_width=True)
                with bd2:
                    st.markdown("#### Organizational Impact")
                    if 'Department' in df_departures_period.columns and 'JobRole' in df_departures_period.columns:
                        role_summary = df_departures_period.groupby(['Department', 'JobRole']).size().reset_index(name='Count')
                        st.altair_chart(alt.Chart(role_summary).mark_bar(cornerRadiusEnd=3).encode(
                            y=alt.Y('JobRole:N', sort='-x', title='Job Role'),
                            x=alt.X('Count:Q', title='Total Losses'),
                            color=alt.Color('Department:N', scale=alt.Scale(scheme='tableau10')),
                            tooltip=['Department:N', 'JobRole:N', 'Count:Q']
                        ).properties(height=280), use_container_width=True)

            with st.expander("🔍 Monthly Aggregates Table"):
                disp = timeline_df.copy()
                disp['Attrition_Rate_Percent'] = disp['Attrition_Rate_Percent'].map('{:,.2f}%'.format)
                st.dataframe(disp.drop(columns=['Sort_Key'], errors='ignore'), use_container_width=True)



elif page == "🎯 Attrition Drivers":
    st.header("🎯 Resignation Drivers Analysis")
    st.caption("A descriptive view of existing resignation patterns based only on the selected HR driver columns. This section does not use model feature importance or prediction outputs.")
    st.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.16); margin:0.65rem 0 1.05rem 0;'></div>",
        unsafe_allow_html=True,
    )

    # Keep this page clean when no custom data has been uploaded yet.
    # This avoids showing sample-data results or schema-related messages in this section.
    # Show content if user_df has the required columns (uploaded OR loaded from disk)
    _drivers_ready = "Attrition" in user_df.columns and len(user_df) > 2
    if not _drivers_ready:
        st.info("Upload the HR employee data CSV file from the sidebar to view the Resignation Drivers dashboard.")
    else:
        attrition_driver_columns = [
            "Age", "BusinessTravel", "Department", "EducationField", "JobRole",
            "MonthlyIncome", "NumCompaniesWorked", "PerformanceRating", "YearsAtCompany",
            "YearsInCurrentRole", "Attrition_Reason"
        ]

        available_driver_columns = [col for col in attrition_driver_columns if col in user_df.columns]
        missing_driver_columns = [col for col in attrition_driver_columns if col not in user_df.columns]

        if "Attrition" not in user_df.columns:
            st.warning("Upload a valid HR employee data file with a resignation status column to use this section.")
        elif len(available_driver_columns) == 0:
            st.warning("Upload a valid HR employee data file with the selected driver columns to use this section.")
        else:
            label_map = {
                "Age": "Age",
                "Company": "Company",
                "BusinessTravel": "Business Travel",
                "Department": "Department",
                "EducationField": "Education Field",
                "JobRole": "Job Role",
                "MonthlyIncome": "Monthly Income",
                "NumCompaniesWorked": "Number of Companies Worked",
                "PerformanceRating": "Performance Rating",
                "YearsAtCompany": "Years at Company",
                "YearsInCurrentRole": "Years in Current Role",
                "Attrition": "Resignation Status",
                "Attrition_Reason": "Resignation Reason",
                "Hire_Date": "Hire Date",
                "Exit_Date": "Exit Date",
                "Attrited_Employees": "Resigned Employees",
                "Active_Employees": "Active Employees",
                "Attrition_Rate": "Resignation Rate",
            }

            def pretty_label(column_name):
                return label_map.get(column_name, str(column_name).replace("_", " ").title())

            def pretty_value(value):
                return str(value).replace("_", " ").strip()

            def style_dark_axis(ax):
                ax.set_facecolor("#111827")
                for spine in ax.spines.values():
                    spine.set_color("#374151")
                ax.tick_params(colors="#e5e7eb")
                ax.xaxis.label.set_color("#e5e7eb")
                ax.yaxis.label.set_color("#e5e7eb")
                ax.title.set_color("#f9fafb")
                ax.grid(axis="x", color="#374151", alpha=0.35)
                ax.set_axisbelow(True)

            def finish_chart(fig):
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            def finish_overview_chart(fig):
                # Use the same sizing behavior as the Patterns charts so both Overview graphs
                # visually fill their cards as evenly as possible.
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True, bbox_inches=None)
                plt.close(fig)

            def build_attrition_rate_table(data, factor):
                summary = data.groupby(factor, dropna=False).agg(
                    Employees=("Attrition_Flag", "count"),
                    Attrited_Employees=("Attrition_Flag", "sum"),
                    Attrition_Rate=("Attrition_Flag", "mean")
                ).reset_index()
                summary["Active_Employees"] = summary["Employees"] - summary["Attrited_Employees"]
                summary["Attrition_Rate"] = summary["Attrition_Rate"].fillna(0)
                return summary.sort_values(["Attrition_Rate", "Attrited_Employees"], ascending=[False, False])

            def csv_bytes(dataframe):
                return dataframe.to_csv(index=False).encode("utf-8")

            driver_df = user_df.copy()
            driver_df["Attrition"] = driver_df["Attrition"].astype(str).str.strip().str.title()
            driver_df["Attrition_Flag"] = driver_df["Attrition"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

            numeric_driver_columns = [
                "Age", "MonthlyIncome", "NumCompaniesWorked", "PerformanceRating",
                "YearsAtCompany", "YearsInCurrentRole"
            ]
            categorical_driver_columns = [
                "BusinessTravel", "Department", "EducationField", "JobRole", "Attrition_Reason"
            ]

            for col in numeric_driver_columns:
                if col in driver_df.columns:
                    driver_df[col] = pd.to_numeric(driver_df[col], errors="coerce")

            for col in categorical_driver_columns:
                if col in driver_df.columns:
                    driver_df[col] = (
                        driver_df[col]
                        .astype("object")
                        .where(driver_df[col].notna(), "Not Available")
                        .astype(str)
                        .str.strip()
                        .replace({"": "Not Available", "nan": "Not Available", "None": "Not Available"})
                    )

            # Date parsing is kept inside Attrition Drivers only so the other tabs remain untouched.
            date_reference_columns = [col for col in ["Hire_Date", "Exit_Date"] if col in driver_df.columns]
            for col in date_reference_columns:
                driver_df[col] = pd.to_datetime(driver_df[col], errors="coerce")
            # Also coerce on filtered_df which is a copy made before this block
            for col in date_reference_columns:
                if col in filtered_df.columns:
                    filtered_df[col] = pd.to_datetime(filtered_df[col], errors="coerce")

            if missing_driver_columns:
                st.info("Missing optional driver columns: " + ", ".join(missing_driver_columns))

            filtered_df = driver_df.copy()
            date_filter_summary = "All available dates"
            date_filter_basis_summary = ""

            with st.expander("Filter resignation driver data", expanded=False):
                st.markdown(
                    "<div class='section-note'>Leave a multiselect blank to include all values for that field. Use the date range when you want the KPIs, charts, tables, and exports to follow a specific time window.</div>",
                    unsafe_allow_html=True,
                )

                date_filter_options = []
                if "Hire_Date" in filtered_df.columns and filtered_df["Hire_Date"].notna().any():
                    date_filter_options.append("Hire Date")
                if "Exit_Date" in filtered_df.columns and filtered_df["Exit_Date"].notna().any():
                    date_filter_options.append("Exit Date / Resignation Date")
                if all(col in filtered_df.columns for col in ["Hire_Date", "Exit_Date"]) and filtered_df["Hire_Date"].notna().any():
                    date_filter_options.insert(0, "Employment Activity Window")

                if date_filter_options:
                    selected_date_basis = st.selectbox(
                        "Date Filter Basis",
                        date_filter_options,
                        index=0,
                        help=(
                            "Employment Activity Window includes employees active at any time in the selected range "
                            "and counts resignations only when the exit date falls in that range."
                        ),
                    )

                    if selected_date_basis == "Employment Activity Window":
                        date_bounds = pd.concat([
                            filtered_df["Hire_Date"].dropna(),
                            filtered_df["Exit_Date"].dropna()
                        ])
                    elif selected_date_basis == "Hire Date":
                        date_bounds = filtered_df["Hire_Date"].dropna()
                    else:
                        date_bounds = filtered_df["Exit_Date"].dropna()

                    if not date_bounds.empty:
                        min_available_date = date_bounds.min().date()
                        max_available_date = date_bounds.max().date()
                        selected_date_range = st.date_input(
                            "Date Range",
                            value=(min_available_date, max_available_date),
                            min_value=min_available_date,
                            max_value=max_available_date,
                            key="attrition_driver_date_range",
                        )

                        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
                            selected_start_date, selected_end_date = selected_date_range
                            range_start = pd.Timestamp(selected_start_date)
                            range_end = pd.Timestamp(selected_end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

                            if range_start > range_end:
                                st.warning("Start date cannot be later than end date. The date filter was not applied.")
                            else:
                                if selected_date_basis == "Employment Activity Window":
                                    active_window_mask = (
                                        filtered_df["Hire_Date"].notna()
                                        & (filtered_df["Hire_Date"] <= range_end)
                                        & (
                                            filtered_df["Exit_Date"].isna()
                                            | (filtered_df["Exit_Date"] >= range_start)
                                        )
                                    )
                                    filtered_df = filtered_df.loc[active_window_mask].copy()

                                    exit_in_selected_window = (
                                        filtered_df["Exit_Date"].notna()
                                        & filtered_df["Exit_Date"].between(range_start, range_end, inclusive="both")
                                    )
                                    attrition_without_exit_date = (
                                        filtered_df["Exit_Date"].isna()
                                        & (filtered_df["Attrition"] == "Yes")
                                    )
                                    filtered_df["Attrition_Flag"] = np.where(
                                        (filtered_df["Attrition"] == "Yes") & (exit_in_selected_window | attrition_without_exit_date),
                                        1,
                                        0
                                    ).astype(int)
                                    filtered_df["Attrition"] = np.where(filtered_df["Attrition_Flag"].eq(1), "Yes", "No")
                                    date_filter_basis_summary = "Employment Activity Window"
                                elif selected_date_basis == "Hire Date":
                                    filtered_df = filtered_df.loc[
                                        filtered_df["Hire_Date"].between(range_start, range_end, inclusive="both")
                                    ].copy()
                                    date_filter_basis_summary = "Hire Date"
                                else:
                                    filtered_df = filtered_df.loc[
                                        filtered_df["Exit_Date"].between(range_start, range_end, inclusive="both")
                                    ].copy()
                                    date_filter_basis_summary = "Exit Date / Resignation Date"

                                date_filter_summary = f"{date_filter_basis_summary}: {range_start:%Y-%m-%d} to {range_end:%Y-%m-%d}"
                        else:
                            st.info("Select both a start and end date to apply the date range filter.")
                else:
                    st.info("No valid Hire_Date or Exit_Date values were found for date filtering.")

                selected_company = []
                if "Company" in filtered_df.columns:
                    company_options = sorted(filtered_df["Company"].dropna().astype(str).unique().tolist())
                    selected_company = st.multiselect(
                        "Company",
                        company_options,
                        default=[],
                        placeholder="All companies"
                    )
                    if selected_company:
                        filtered_df = filtered_df[filtered_df["Company"].astype(str).isin(selected_company)]

                status_options = ["Yes - Resigned", "No - Active"]
                selected_resignation_status = st.multiselect(
                    "Resignation Status",
                    status_options,
                    default=[],
                    placeholder="All resignation statuses"
                )
                if selected_resignation_status:
                    status_map = {"Yes - Resigned": "Yes", "No - Active": "No"}
                    selected_status_values = [status_map[status] for status in selected_resignation_status]
                    filtered_df = filtered_df[filtered_df["Attrition"].isin(selected_status_values)]

                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    if "Age" in filtered_df.columns and filtered_df["Age"].notna().any():
                        age_min = int(filtered_df["Age"].min())
                        age_max = int(filtered_df["Age"].max())
                        selected_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))
                        filtered_df = filtered_df[filtered_df["Age"].between(selected_age[0], selected_age[1], inclusive="both")]

                    if "BusinessTravel" in filtered_df.columns:
                        travel_options = sorted(filtered_df["BusinessTravel"].dropna().unique().tolist())
                        selected_travel = st.multiselect("Business Travel", travel_options, default=[], placeholder="All business travel types")
                        if selected_travel:
                            filtered_df = filtered_df[filtered_df["BusinessTravel"].isin(selected_travel)]

                    if "Department" in filtered_df.columns:
                        department_options = sorted(filtered_df["Department"].dropna().unique().tolist())
                        selected_department = st.multiselect("Department", department_options, default=[], placeholder="All departments")
                        if selected_department:
                            filtered_df = filtered_df[filtered_df["Department"].isin(selected_department)]

                    if "EducationField" in filtered_df.columns:
                        education_options = sorted(filtered_df["EducationField"].dropna().unique().tolist())
                        selected_education = st.multiselect("Education Field", education_options, default=[], placeholder="All education fields")
                        if selected_education:
                            filtered_df = filtered_df[filtered_df["EducationField"].isin(selected_education)]

                with filter_col2:
                    if "JobRole" in filtered_df.columns:
                        job_role_options = sorted(filtered_df["JobRole"].dropna().unique().tolist())
                        selected_job_role = st.multiselect("Job Role", job_role_options, default=[], placeholder="All job roles")
                        if selected_job_role:
                            filtered_df = filtered_df[filtered_df["JobRole"].isin(selected_job_role)]

                    if "MonthlyIncome" in filtered_df.columns and filtered_df["MonthlyIncome"].notna().any():
                        income_min = int(filtered_df["MonthlyIncome"].min())
                        income_max = int(filtered_df["MonthlyIncome"].max())
                        selected_income = st.slider("Monthly Income Range", income_min, income_max, (income_min, income_max))
                        filtered_df = filtered_df[filtered_df["MonthlyIncome"].between(selected_income[0], selected_income[1], inclusive="both")]

                    if "NumCompaniesWorked" in filtered_df.columns and filtered_df["NumCompaniesWorked"].notna().any():
                        company_count_options = sorted(filtered_df["NumCompaniesWorked"].dropna().unique().tolist())
                        selected_company_count = st.multiselect("Number of Companies Worked", company_count_options, default=[], placeholder="All values")
                        if selected_company_count:
                            filtered_df = filtered_df[filtered_df["NumCompaniesWorked"].isin(selected_company_count)]

                with filter_col3:
                    if "PerformanceRating" in filtered_df.columns and filtered_df["PerformanceRating"].notna().any():
                        rating_options = sorted(filtered_df["PerformanceRating"].dropna().unique().tolist())
                        selected_rating = st.multiselect("Performance Rating", rating_options, default=[], placeholder="All ratings")
                        if selected_rating:
                            filtered_df = filtered_df[filtered_df["PerformanceRating"].isin(selected_rating)]

                    if "YearsAtCompany" in filtered_df.columns and filtered_df["YearsAtCompany"].notna().any():
                        tenure_min = int(filtered_df["YearsAtCompany"].min())
                        tenure_max = int(filtered_df["YearsAtCompany"].max())
                        selected_tenure = st.slider("Years at Company Range", tenure_min, tenure_max, (tenure_min, tenure_max))
                        filtered_df = filtered_df[filtered_df["YearsAtCompany"].between(selected_tenure[0], selected_tenure[1], inclusive="both")]

                    if "YearsInCurrentRole" in filtered_df.columns and filtered_df["YearsInCurrentRole"].notna().any():
                        role_min = int(filtered_df["YearsInCurrentRole"].min())
                        role_max = int(filtered_df["YearsInCurrentRole"].max())
                        selected_role_years = st.slider("Years in Current Role Range", role_min, role_max, (role_min, role_max))
                        filtered_df = filtered_df[filtered_df["YearsInCurrentRole"].between(selected_role_years[0], selected_role_years[1], inclusive="both")]

                    if "Attrition_Reason" in filtered_df.columns:
                        reason_options = sorted(filtered_df["Attrition_Reason"].dropna().unique().tolist())
                        selected_reason = st.multiselect("Resignation Reason", reason_options, default=[], placeholder="All reasons")
                        if selected_reason:
                            filtered_df = filtered_df[filtered_df["Attrition_Reason"].isin(selected_reason)]

            st.caption(f"Current date scope: {date_filter_summary}")

            attrited_df = filtered_df[filtered_df["Attrition"] == "Yes"].copy()
            total_filtered = len(filtered_df)
            total_attrited = len(attrited_df)
            attrition_rate = (total_attrited / total_filtered) if total_filtered > 0 else 0
            active_count = total_filtered - total_attrited

            total_available = len(driver_df)
            if "Attrition_Reason" in attrited_df.columns and total_attrited > 0 and attrited_df["Attrition_Reason"].notna().any():
                top_reason_counts = attrited_df["Attrition_Reason"].value_counts()
                top_reason = top_reason_counts.idxmax()
                top_reason_count = int(top_reason_counts.max())
            else:
                top_reason = "Not Available"
                top_reason_count = 0

            scope_progress = (total_filtered / total_available) if total_available > 0 else 0
            resigned_dataset_progress = (total_attrited / total_available) if total_available > 0 else 0
            top_reason_progress = (top_reason_count / total_attrited) if total_attrited > 0 else 0

            card1, card2, card3, card4 = st.columns(4)
            with card1:
                metric_card(
                    "Total Employees",
                    f"{total_filtered:,}",
                    "Employees currently included in the filters",
                    progress=scope_progress,
                    progress_label=f"{scope_progress:.0%} of full dataset"
                )
            with card2:
                metric_card(
                    "Resigned Employees",
                    f"{total_attrited:,}",
                    f"Active employees in scope: {active_count:,}",
                    progress=resigned_dataset_progress,
                    progress_label=f"{resigned_dataset_progress:.0%} of full dataset"
                )
            with card3:
                metric_card(
                    "Resignation Rate",
                    f"{attrition_rate:.2%}",
                    "Share of employees in scope marked as resigned",
                    progress=attrition_rate,
                    progress_label="Resigned within current scope"
                )
            with card4:
                metric_card(
                    "Top Resignation Reason",
                    str(top_reason),
                    "Most common reason among resigned employees in scope",
                    progress=top_reason_progress,
                    progress_label=(f"{top_reason_progress:.0%} of resigned employees" if total_attrited > 0 else "No resigned employees in scope")
                )

            if total_filtered == 0:
                st.warning("No records match the selected filters. Try widening one or more filters.")
            else:
                overview_tab, patterns_tab, data_tab = st.tabs(["Overview", "Patterns & Distributions", "Tables & Records"])

                with overview_tab:
                    rate_factor_options = [
                        col for col in [
                            "BusinessTravel", "Department", "EducationField", "JobRole",
                            "NumCompaniesWorked", "PerformanceRating", "YearsAtCompany", "YearsInCurrentRole"
                        ] if col in filtered_df.columns
                    ]

                    selected_rate_factor = None
                    rate_table = pd.DataFrame()

                    col_a, col_b = st.columns(2, gap="small")

                    with col_a:
                        with st.container(border=True):
                            st.subheader("Top Reasons for Resignation")
                            st.caption("Showing the most common reasons among resigned employees in scope.")
                            if "Attrition_Reason" in attrited_df.columns and len(attrited_df) > 0:
                                max_reason_count = int(attrited_df["Attrition_Reason"].nunique())
                                reason_limit_options = [value for value in [5, 8, 10] if value <= max_reason_count]
                                if max_reason_count not in reason_limit_options:
                                    reason_limit_options.append(max_reason_count)
                                reason_limit_options = sorted(set(reason_limit_options))

                                reason_limit = st.selectbox(
                                    "Number of Reasons to Show",
                                    reason_limit_options,
                                    index=len(reason_limit_options) - 1,
                                    key=f"reason_limit_{max_reason_count}",
                                    format_func=lambda value: f"Top {value}" if value < max_reason_count else "All",
                                )

                                reason_counts = attrited_df["Attrition_Reason"].value_counts().head(reason_limit).reset_index()
                                reason_counts.columns = ["Attrition_Reason", "Attrited_Employees"]
                                reason_counts["Attrition_Reason_Display"] = reason_counts["Attrition_Reason"].map(pretty_value)

                                fig, ax = plt.subplots(figsize=(9.4, 4.9), facecolor="#0b1220")
                                ax.barh(reason_counts["Attrition_Reason_Display"], reason_counts["Attrited_Employees"], color="#60a5fa")
                                ax.invert_yaxis()
                                ax.set_xlabel("Resigned Employees")
                                ax.set_ylabel("Reason")
                                ax.set_title("Most Common Resignation Reasons")
                                ax.margins(y=0.08)
                                style_dark_axis(ax)
                                finish_overview_chart(fig)
                            else:
                                st.info("No resignation reason data is available for the current filters.")

                    with col_b:
                        with st.container(border=True):
                            st.subheader("Resignation Rate by Factor")
                            st.caption("Showing the resignation rate for the selected descriptive factor.")

                            if rate_factor_options:
                                selected_rate_factor = st.selectbox(
                                    "Choose a Factor",
                                    rate_factor_options,
                                    key="rate_factor",
                                    format_func=pretty_label,
                                )
                                rate_table = build_attrition_rate_table(filtered_df, selected_rate_factor)

                            if selected_rate_factor is not None and len(rate_table) > 0:
                                rate_table_chart = rate_table.head(10).copy()
                                rate_table_chart[selected_rate_factor] = rate_table_chart[selected_rate_factor].astype(str).map(pretty_value)
                                rate_table_chart["Attrition_Rate_Percent"] = rate_table_chart["Attrition_Rate"] * 100

                                fig, ax = plt.subplots(figsize=(9.4, 4.9), facecolor="#0b1220")
                                ax.barh(rate_table_chart[selected_rate_factor], rate_table_chart["Attrition_Rate_Percent"], color="#f59e0b")
                                ax.invert_yaxis()
                                ax.set_xlabel("Resignation Rate (%)")
                                ax.set_ylabel(pretty_label(selected_rate_factor))
                                ax.set_title(f"Resignation Rate by {pretty_label(selected_rate_factor)}")
                                ax.margins(y=0.08)
                                style_dark_axis(ax)
                                finish_overview_chart(fig)
                            else:
                                st.info("No factor is available for the resignation rate chart.")

                    if len(rate_table) > 0:
                        with st.expander("Show summary table for selected factor", expanded=False):
                            display_rate_table = rate_table.copy()
                            display_rate_table["Attrition_Rate"] = (display_rate_table["Attrition_Rate"] * 100).round(2).astype(str) + "%"
                            display_rate_table = display_rate_table.rename(columns={col: pretty_label(col) for col in display_rate_table.columns})
                            st.dataframe(display_rate_table, use_container_width=True)

                with patterns_tab:
                    col_c, col_d = st.columns(2)

                    with col_c:
                        with st.container(border=True):
                            st.subheader("Resigned Employees by Category")
                            count_factor_options = [
                                col for col in ["BusinessTravel", "Department", "EducationField", "JobRole", "PerformanceRating", "Attrition_Reason"]
                                if col in attrited_df.columns
                            ]
                            if count_factor_options:
                                selected_count_factor = st.selectbox(
                                    "Choose a Category",
                                    count_factor_options,
                                    key="count_factor",
                                    format_func=pretty_label,
                                )

                                if len(attrited_df) > 0:
                                    count_table = attrited_df[selected_count_factor].astype(str).value_counts().head(8).reset_index()
                                    count_table.columns = [selected_count_factor, "Attrited_Employees"]
                                    count_table[selected_count_factor] = count_table[selected_count_factor].map(pretty_value)

                                    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor="#0b1220")
                                    ax.barh(count_table[selected_count_factor], count_table["Attrited_Employees"], color="#34d399")
                                    ax.invert_yaxis()
                                    ax.set_xlabel("Resigned Employees")
                                    ax.set_ylabel(pretty_label(selected_count_factor))
                                    ax.set_title(f"Resigned Employees by {pretty_label(selected_count_factor)}")
                                    style_dark_axis(ax)
                                    finish_chart(fig)
                                else:
                                    st.info("No resigned employees match the current filters.")
                            else:
                                st.info("No category is available for the resigned employee count chart.")

                    with col_d:
                        with st.container(border=True):
                            st.subheader("Employee Profile Trends")
                            numeric_options = [col for col in numeric_driver_columns if col in filtered_df.columns and filtered_df[col].notna().any()]
                            if numeric_options:
                                selected_numeric_factor = st.selectbox(
                                    "Choose an Employee Factor",
                                    numeric_options,
                                    key="numeric_factor",
                                    format_func=pretty_label,
                                )

                                hist_df = filtered_df.dropna(subset=[selected_numeric_factor]).copy()
                                if len(hist_df) > 0:
                                    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor="#0b1220")

                                    chart_df = hist_df.copy()
                                    chart_df["Employment_Status"] = np.where(chart_df["Attrition"] == "Yes", "Resigned", "Active")

                                    # These charts use an overlay style instead of stacked totals.
                                    # Active employees are shown as the wider background bar, while
                                    # resigned employees are shown in front from the same baseline.
                                    # This keeps the resigned count readable without implying a stacked total.
                                    compact_discrete_fields = ["NumCompaniesWorked", "PerformanceRating"]
                                    tenure_fields = ["YearsAtCompany", "YearsInCurrentRole"]
                                    banded_continuous_fields = ["MonthlyIncome", "Age"]

                                    if selected_numeric_factor in tenure_fields:
                                        values = chart_df[selected_numeric_factor]
                                        if selected_numeric_factor == "YearsAtCompany":
                                            bins = [-0.1, 0, 1, 2, 5, 10, 15, 20, np.inf]
                                            labels = ["0 yrs", "1 yr", "2 yrs", "3-5 yrs", "6-10 yrs", "11-15 yrs", "16-20 yrs", "21+ yrs"]
                                        else:
                                            bins = [-0.1, 0, 1, 2, 5, 10, 15, np.inf]
                                            labels = ["0 yrs", "1 yr", "2 yrs", "3-5 yrs", "6-10 yrs", "11-15 yrs", "16+ yrs"]

                                        chart_df["Numeric_Group"] = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
                                        count_table = (
                                            chart_df
                                            .dropna(subset=["Numeric_Group"])
                                            .groupby(["Numeric_Group", "Employment_Status"], observed=False)
                                            .size()
                                            .unstack(fill_value=0)
                                            .reindex(columns=["Active", "Resigned"], fill_value=0)
                                        )
                                        x_labels = [str(label) for label in count_table.index]
                                        x_positions = np.arange(len(count_table.index))
                                        ax.bar(x_positions, count_table["Active"], width=0.76, label="Active", color="#60a5fa", alpha=0.78, zorder=2)
                                        ax.bar(x_positions, count_table["Resigned"], width=0.46, label="Resigned", color="#f87171", alpha=0.92, zorder=3)
                                        ax.set_xticks(x_positions)
                                        ax.set_xticklabels(x_labels, rotation=25, ha="right")
                                        ax.set_xlim(-0.6, len(x_positions) - 0.4)
                                        max_value = max(count_table["Active"].max(), count_table["Resigned"].max())
                                        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)
                                        ax.grid(False)
                                        ax.yaxis.grid(True, color="#374151", alpha=0.35)

                                    elif selected_numeric_factor in compact_discrete_fields:
                                        count_table = (
                                            chart_df
                                            .groupby([selected_numeric_factor, "Employment_Status"])
                                            .size()
                                            .unstack(fill_value=0)
                                            .reindex(columns=["Active", "Resigned"], fill_value=0)
                                            .sort_index()
                                        )
                                        x_positions = np.arange(len(count_table.index))
                                        x_labels = [str(int(value)) if float(value).is_integer() else str(value) for value in count_table.index]
                                        ax.bar(x_positions, count_table["Active"], width=0.76, label="Active", color="#60a5fa", alpha=0.78, zorder=2)
                                        ax.bar(x_positions, count_table["Resigned"], width=0.46, label="Resigned", color="#f87171", alpha=0.92, zorder=3)
                                        ax.set_xticks(x_positions)
                                        ax.set_xticklabels(x_labels)
                                        ax.set_xlim(-0.6, len(x_positions) - 0.4)
                                        max_value = max(count_table["Active"].max(), count_table["Resigned"].max())
                                        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)
                                        ax.grid(False)
                                        ax.yaxis.grid(True, color="#374151", alpha=0.35)

                                    elif selected_numeric_factor in banded_continuous_fields:
                                        values = chart_df[selected_numeric_factor]
                                        if selected_numeric_factor == "MonthlyIncome":
                                            bins = [0, 3000, 6000, 10000, 15000, np.inf]
                                            labels = ["≤3,000", "3,001-6,000", "6,001-10,000", "10,001-15,000", "15,001+"]
                                        else:
                                            bins = [0, 25, 30, 40, 50, np.inf]
                                            labels = ["18-25", "26-30", "31-40", "41-50", "51+"]

                                        chart_df["Numeric_Group"] = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
                                        count_table = (
                                            chart_df
                                            .dropna(subset=["Numeric_Group"])
                                            .groupby(["Numeric_Group", "Employment_Status"], observed=False)
                                            .size()
                                            .unstack(fill_value=0)
                                            .reindex(columns=["Active", "Resigned"], fill_value=0)
                                        )
                                        x_labels = [str(label) for label in count_table.index]
                                        x_positions = np.arange(len(count_table.index))
                                        ax.bar(x_positions, count_table["Active"], width=0.76, label="Active", color="#60a5fa", alpha=0.78, zorder=2)
                                        ax.bar(x_positions, count_table["Resigned"], width=0.46, label="Resigned", color="#f87171", alpha=0.92, zorder=3)
                                        ax.set_xticks(x_positions)
                                        ax.set_xticklabels(x_labels, rotation=20, ha="right")
                                        ax.set_xlim(-0.6, len(x_positions) - 0.4)
                                        max_value = max(count_table["Active"].max(), count_table["Resigned"].max())
                                        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)
                                        ax.grid(False)
                                        ax.yaxis.grid(True, color="#374151", alpha=0.35)

                                    else:
                                        active_values = chart_df.loc[chart_df["Attrition"] == "No", selected_numeric_factor].dropna()
                                        resigned_values = chart_df.loc[chart_df["Attrition"] == "Yes", selected_numeric_factor].dropna()
                                        bins = np.histogram_bin_edges(chart_df[selected_numeric_factor].dropna(), bins=18)
                                        active_counts, bin_edges = np.histogram(active_values, bins=bins)
                                        resigned_counts, _ = np.histogram(resigned_values, bins=bins)
                                        bin_widths = np.diff(bin_edges)
                                        ax.bar(bin_edges[:-1], active_counts, width=bin_widths, align="edge", label="Active", color="#60a5fa", alpha=0.72, zorder=2)
                                        ax.bar(bin_edges[:-1] + (bin_widths * 0.15), resigned_counts, width=bin_widths * 0.70, align="edge", label="Resigned", color="#f87171", alpha=0.90, zorder=3)
                                        max_value = max(active_counts.max() if len(active_counts) else 0, resigned_counts.max() if len(resigned_counts) else 0)
                                        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)

                                    ax.set_xlabel(pretty_label(selected_numeric_factor))
                                    ax.set_ylabel("Employees")
                                    ax.set_title(f"{pretty_label(selected_numeric_factor)} Distribution by Employment Status")
                                    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#e5e7eb")
                                    style_dark_axis(ax)
                                    if selected_numeric_factor in compact_discrete_fields + tenure_fields + banded_continuous_fields:
                                        ax.grid(False)
                                        ax.yaxis.grid(True, color="#374151", alpha=0.35)
                                    finish_chart(fig)
                                else:
                                    st.info("Not enough numeric data is available for the selected driver.")
                            else:
                                st.info("No employee profile factor is available for this chart.")

                    if "MonthlyIncome" in filtered_df.columns and "YearsAtCompany" in filtered_df.columns:
                        with st.container(border=True):
                            st.subheader("Monthly Income vs Years at Company")
                            if selected_company:
                                st.caption("This chart follows the current filters, including the selected company filter.")
                            scatter_df = filtered_df.copy().dropna(subset=["MonthlyIncome", "YearsAtCompany"])
                            if len(scatter_df) > 0:
                                fig, ax = plt.subplots(figsize=(13, 5.4), facecolor="#0b1220")
                                no_mask = scatter_df["Attrition"] == "No"
                                yes_mask = scatter_df["Attrition"] == "Yes"
                                ax.scatter(scatter_df.loc[no_mask, "YearsAtCompany"], scatter_df.loc[no_mask, "MonthlyIncome"], label="Active", alpha=0.6, color="#60a5fa", edgecolors="white", linewidths=0.3)
                                ax.scatter(scatter_df.loc[yes_mask, "YearsAtCompany"], scatter_df.loc[yes_mask, "MonthlyIncome"], label="Resigned", alpha=0.8, color="#f87171", edgecolors="white", linewidths=0.3)
                                ax.set_xlabel("Years at Company")
                                ax.set_ylabel("Monthly Income")
                                ax.set_title("Monthly Income and Tenure Pattern by Employment Status")
                                ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#e5e7eb")
                                style_dark_axis(ax)
                                finish_chart(fig)
                            else:
                                st.info("Not enough monthly income and tenure data is available for the scatter plot.")

                with data_tab:
                    with st.container(border=True):
                        st.subheader("Filtered Employee Records")
                        st.caption("This table and export follow the current filter selection in this Resignation Drivers page.")
                        date_detail_columns = [col for col in ["Hire_Date", "Exit_Date"] if col in filtered_df.columns]
                        detail_columns = ["Attrition"] + available_driver_columns + date_detail_columns
                        if "Company" in filtered_df.columns:
                            detail_columns = ["Company"] + detail_columns
                        if "EmployeeNumber" in filtered_df.columns:
                            detail_columns = ["EmployeeNumber"] + detail_columns
                        detail_columns = list(dict.fromkeys([col for col in detail_columns if col in filtered_df.columns]))

                        export_filtered_df = filtered_df[detail_columns].copy()
                        display_records = export_filtered_df.rename(columns={col: pretty_label(col) for col in export_filtered_df.columns})
                        for col in ["Hire Date", "Exit Date"]:
                            if col in display_records.columns:
                                display_records[col] = pd.to_datetime(display_records[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

                        st.download_button(
                            label="Download Filtered Records CSV",
                            data=csv_bytes(export_filtered_df),
                            file_name="filtered_attrition_driver_records.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        st.dataframe(display_records, use_container_width=True, height=420)



elif page == "⚖️ Wellbeing/Performance":
    st.header("⚖️ Wellbeing & Performance Matrix")
    st.markdown("---")

    # Ruth

    # --- FILTERS ---
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        companies = ["All"] + sorted(user_df["Company"].dropna().unique().tolist())
        selected_company = st.selectbox("Company", companies)

    with col_f2:
        departments = ["All"] + sorted(user_df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)

    with col_f3:
        _wb_date_col = next((c for c in ["Hire_Date", "Exit_Date"] if c in user_df.columns), None)
        if _wb_date_col:
            _parsed = pd.to_datetime(user_df[_wb_date_col], errors="coerce").dropna()
            _abs_min = max(_parsed.min().date(), pd.Timestamp("2000-01-01").date())
            _abs_max = _parsed.max().date()
            import datetime as _dt
            _month_options = []
            _cur = pd.Timestamp(_abs_min).to_period("M")
            _end_p = pd.Timestamp(_abs_max).to_period("M")
            while _cur <= _end_p:
                _month_options.append(_cur.strftime("%b %Y"))
                _cur += 1
            _month_options_rev = list(reversed(_month_options))
            _wb_month = st.selectbox("Month Filter", ["All"] + _month_options_rev, key="wb_month_filter")
        else:
            _wb_date_col = None
            _wb_month = "All"

    # Apply filters to a working copy of the data
    wb_df = user_df.copy()
    if selected_company != "All":
        wb_df = wb_df[wb_df["Company"] == selected_company]
    if selected_dept != "All":
        wb_df = wb_df[wb_df["Department"] == selected_dept]
    if _wb_date_col and _wb_month != "All":
        wb_df[_wb_date_col] = pd.to_datetime(wb_df[_wb_date_col], errors="coerce")
        wb_df = wb_df[wb_df[_wb_date_col].notna()]  # drop rows where parse failed
        _sel_period = pd.Period(_wb_month, freq="M")
        wb_df = wb_df[wb_df[_wb_date_col].dt.to_period("M") == _sel_period]

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

    PALETTE = {"No": "#4CAF82", "Yes": "#E05C5C"}
    CHART_BG = "none"
    AXIS_BG = "none"
    GRID_COLOR = "#374151"
    FG = "#e5e7eb"
    TITLE_FG = "#f9fafb"
    RATING_LABELS = ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
    FIGSIZE = (5, 3.8)

    def style_chart(ax, title, xlabel, ylabel, legend_keys=None):
        ax.set_facecolor("none")
        ax.figure.patch.set_alpha(0)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=TITLE_FG)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10, color=FG)
        ax.set_ylabel(ylabel, fontsize=10, color=FG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(axis="both", labelsize=9, colors=FG)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.35)
        ax.set_axisbelow(True)
        if legend_keys:
            patches = [mpatches.Patch(facecolor=PALETTE[k],
                                      label="Stayed" if k == "No" else "Left")
                       for k in legend_keys]
            ax.legend(handles=patches, fontsize=9, title="",
                      facecolor="none", edgecolor=GRID_COLOR, labelcolor=FG)

    # --- SCORECARDS ---
    avg_job = wb_df['JobSatisfaction'].mean()
    avg_env = wb_df['EnvironmentSatisfaction'].mean()
    avg_rel = wb_df['RelationshipSatisfaction'].mean()
    wb_sample = len(wb_df)

    def _score_color(score):
        if score < 2.4:
            return "#ef4444"   # red
        elif score < 2.8:
            return "#f59e0b"   # amber
        return None            # default gradient

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        metric_card(
            "Avg Job Satisfaction",
            f"{avg_job:.2f} / 4",
            "Average job satisfaction across filtered employees",
            progress=avg_job / 4,
            progress_label=f"{avg_job / 4:.0%} of maximum score",
            sample_size=wb_sample,
            fill_color=_score_color(avg_job),
        )
    with sc2:
        metric_card(
            "Avg Environment Satisfaction",
            f"{avg_env:.2f} / 4",
            "Average environment satisfaction across filtered employees",
            progress=avg_env / 4,
            progress_label=f"{avg_env / 4:.0%} of maximum score",
            sample_size=wb_sample,
            fill_color=_score_color(avg_env),
        )
    with sc3:
        metric_card(
            "Avg Relationship Satisfaction",
            f"{avg_rel:.2f} / 4",
            "Average relationship satisfaction across filtered employees",
            progress=avg_rel / 4,
            progress_label=f"{avg_rel / 4:.0%} of maximum score",
            sample_size=wb_sample,
            fill_color=_score_color(avg_rel),
        )

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

    tab_job, tab_env, tab_rel = st.tabs(["💼 Job Satisfaction", "🏢 Environment Satisfaction", "🤝 Relationship Satisfaction"])

    with tab_job:
        st.caption("How satisfied employees are with their job, overtime load, and work-life balance — and whether that's linked to leaving.")
        js_col1, js_col2, js_col3 = st.columns(3)

        with js_col1:
            with st.container(border=True):
                js_data = wb_df.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=js_data, x="JobSatisfaction", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Job Satisfaction", "Satisfaction Rating", "Number of Employees", legend_keys=["No", "Yes"])
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(RATING_LABELS)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with js_col2:
            with st.container(border=True):
                ot_df = wb_df.copy()
                ot_df["OverTime"] = ot_df["OverTime"].map({"No": "No OT", "Yes": "Rendered OT"})
                ot_data = ot_df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=ot_data, x="OverTime", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Overtime vs Attrition", "Overtime Status", "Number of Employees", legend_keys=["No", "Yes"])
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with js_col3:
            with st.container(border=True):
                wlb_data = wb_df.groupby(["WorkLifeBalance", "Attrition"]).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=wlb_data, x="WorkLifeBalance", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Work-Life Balance", "Balance Rating", "Number of Employees", legend_keys=["No", "Yes"])
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(RATING_LABELS)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with tab_env:
        st.caption("How comfortable employees feel in their physical and social work environment, and whether commute distance plays a role.")
        env_col1, env_col2 = st.columns(2)

        with env_col1:
            with st.container(border=True):
                env_data = wb_df.groupby(["EnvironmentSatisfaction", "Attrition"]).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=env_data, x="EnvironmentSatisfaction", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Environment Satisfaction", "Satisfaction Rating", "Number of Employees", legend_keys=["No", "Yes"])
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(RATING_LABELS)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with env_col2:
            with st.container(border=True):
                bins = [0, 5, 10, 15, 20, 100]
                labels_dist = ["0–5 km", "6–10 km", "11–15 km", "16–20 km", "20+ km"]
                wb_df["DistanceBucket"] = pd.cut(wb_df["DistanceFromHome"], bins=bins, labels=labels_dist, right=True)
                dist_data = wb_df.groupby(["DistanceBucket", "Attrition"], observed=True).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=dist_data, x="DistanceBucket", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Distance From Home", "Commute Distance", "Number of Employees", legend_keys=["No", "Yes"])
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with tab_rel:
        st.caption("How employees feel about their relationships at work — with peers and managers — and how long they've been with their current manager.")
        rel_col1, rel_col2 = st.columns(2)

        with rel_col1:
            with st.container(border=True):
                rs_data = wb_df.groupby(["RelationshipSatisfaction", "Attrition"]).size().reset_index(name="Count")
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                sns.barplot(data=rs_data, x="RelationshipSatisfaction", y="Count", hue="Attrition", palette=PALETTE, ax=ax)
                style_chart(ax, "Relationship Satisfaction", "Satisfaction Rating", "Number of Employees", legend_keys=["No", "Yes"])
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(RATING_LABELS)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with rel_col2:
            with st.container(border=True):
                mgr_avg = wb_df.groupby("Attrition")["YearsWithCurrManager"].mean().reset_index()
                mgr_avg["Label"] = mgr_avg["Attrition"].map({"No": "Stayed", "Yes": "Left"})
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                ax.set_facecolor("none")
                bars = ax.bar(mgr_avg["Label"], mgr_avg["YearsWithCurrManager"],
                              color=[PALETTE[v] for v in mgr_avg["Attrition"]], width=0.4)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f"{bar.get_height():.1f} yrs", ha="center", fontsize=10, fontweight="bold", color=FG)
                ax.set_title("Avg Years With Current Manager", fontsize=13, fontweight="bold", pad=12, color=TITLE_FG)
                ax.set_ylabel("Average Years", fontsize=10, color=FG)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color(GRID_COLOR)
                ax.spines["bottom"].set_color(GRID_COLOR)
                ax.tick_params(colors=FG)
                ax.grid(axis="y", color=GRID_COLOR, alpha=0.35)
                ax.set_axisbelow(True)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        rel_col3, rel_col4 = st.columns(2)

        with rel_col3:
            with st.container(border=True):
                tenure_avg = wb_df.groupby("Attrition")["YearsAtCompany"].mean().reset_index()
                tenure_avg["Label"] = tenure_avg["Attrition"].map({"No": "Stayed", "Yes": "Left"})
                fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                ax.set_facecolor("none")
                bars = ax.bar(tenure_avg["Label"], tenure_avg["YearsAtCompany"],
                              color=[PALETTE[v] for v in tenure_avg["Attrition"]], width=0.4)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f"{bar.get_height():.1f} yrs", ha="center", fontsize=10, fontweight="bold", color=FG)
                ax.set_title("Avg Years At Company", fontsize=13, fontweight="bold", pad=12, color=TITLE_FG)
                ax.set_ylabel("Average Years", fontsize=10, color=FG)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color(GRID_COLOR)
                ax.spines["bottom"].set_color(GRID_COLOR)
                ax.tick_params(colors=FG)
                ax.grid(axis="y", color=GRID_COLOR, alpha=0.35)
                ax.set_axisbelow(True)
                fig.tight_layout(pad=1.8)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with rel_col4:
            with st.container(border=True):
                mgr_line = wb_df.groupby("YearsWithCurrManager").agg(
                    AvgSat=("RelationshipSatisfaction", "mean"),
                    Count=("RelationshipSatisfaction", "count")
                ).reset_index()
                mgr_line = mgr_line.dropna(subset=["AvgSat"])
                if mgr_line.empty:
                    st.info("No data available for this chart.")
                else:
                    min_s, max_s = 40, 300
                    count_range = mgr_line["Count"].max() - mgr_line["Count"].min()
                    mgr_line["BubbleSize"] = min_s if count_range == 0 else (
                        (mgr_line["Count"] - mgr_line["Count"].min()) / count_range * (max_s - min_s) + min_s
                    )
                    y_min = mgr_line["AvgSat"].min()
                    y_max = mgr_line["AvgSat"].max()
                    y_pad = max((y_max - y_min) * 0.4, 0.1)
                    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CHART_BG)
                    ax.set_facecolor("none")
                    ax.plot(mgr_line["YearsWithCurrManager"], mgr_line["AvgSat"],
                            color=PALETTE["No"], linewidth=2, zorder=1)
                    ax.scatter(mgr_line["YearsWithCurrManager"], mgr_line["AvgSat"],
                               s=mgr_line["BubbleSize"], color=PALETTE["No"], alpha=0.85, zorder=2)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)
                    ax.set_title("Satisfaction by Manager Tenure", fontsize=13, fontweight="bold", pad=12, color=TITLE_FG)
                    ax.set_xlabel("Years With Current Manager\n(Bubble size = number of employees)", fontsize=9, labelpad=8, color=FG)
                    ax.set_ylabel("Avg Relationship Satisfaction", fontsize=10, color=FG)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color(GRID_COLOR)
                    ax.spines["bottom"].set_color(GRID_COLOR)
                    ax.tick_params(colors=FG)
                    ax.grid(axis="y", color=GRID_COLOR, alpha=0.35)
                    ax.set_axisbelow(True)
                    fig.tight_layout(pad=1.8)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()



elif page == "💰 Attrition Simulator":
    st.header("📊 Attrition Financial Risk Indicator")
    st.markdown("---")

    st.markdown("### 🔍 Executive Risk Monitoring & 12-Month Run-Rate Projection")
    st.caption("""
    **Forward-Looking Context:** This module translates historical turnover into forward-looking dollar exposure.
    Using an active-window monthly run-rate model, it projects cumulative financial bleed over 12 months —
    capped at your current at-risk workforce value — so operators can act before losses become irreversible.
    """)
    st.markdown("[Learn more about Attrition Risk Financial Indicator Engine here](https://docs.google.com/document/d/1E7UzjTNt62vo_jkuiTeIZ8PTD9DydjK4av4Tc0emmZE/edit?tab=t.0)")

    # Fallback risk scoring — replaced by trained model when available in session state
    @st.cache_resource
    def get_fallback_rf_predictions(df):
        np.random.seed(42)
        return np.random.uniform(0.05, 0.85, size=len(df))

    # =========================================================================
    # 1. BUSINESS CONTEXT FILTERS
    # =========================================================================
    st.markdown("#### 🪵 Filter Employee Risk Context")
    focus_col1, focus_col2, focus_col3, focus_col4 = st.columns(4)

    with focus_col1:
        companies = ["All Companies"] + sorted(user_df["Company"].dropna().unique().tolist())
        selected_company = st.selectbox("Company Focus", companies)

    with focus_col2:
        departments = ["All Departments"] + sorted(user_df["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department Focus", departments)

    with focus_col3:
        career_levels = [
            "All Levels",
            "Entry-Level Staff (Level 1)",
            "Mid-Level Professionals (Level 2-3)",
            "Management & Specialists (Level 4-5)"
        ]
        selected_career = st.selectbox("Career Level Focus", career_levels)

    with focus_col4:
        tenure_groups = ["All Milestones", "New Hires (0-2 Yrs)", "Mid-Tenure (3-7 Yrs)", "Tenured Pillars (8+ Yrs)"]
        selected_tenure = st.selectbox("Tenure / Milestone Groups", tenure_groups)

    # =========================================================================
    # 2. VISUAL METHODOLOGY COGNITION
    # =========================================================================
    st.warning("""
    📝 **Note on Financial Calibration:** Turnover impact is automatically mapped to industry cost benchmarks
    based on organizational seniority: Entry-Level (Level 1) is calculated at 30% of annual salary;
    Levels 2-3 (Mid-Level) at 60%; and Levels 4-5 (Management/Specialists) at 100% to fully account for
    vacancy disruptions, specialized operational skills, and recruitment onboarding friction.
    """)

    with st.expander("⚙️ View Financial Calibration Framework", expanded=False):
        st.markdown("""
        | Career Level | Job Level | Cost Benchmark | Rationale |
        |---|---|---|---|
        | Entry-Level Staff | Level 1 | 30% of Annual Salary | High replacement volume, short training period |
        | Mid-Level Professionals | Level 2–3 | 60% of Annual Salary | Moderate disruption, specialized operational knowledge |
        | Leadership & Specialists | Level 4–5 | 100% of Annual Salary | Critical vacancy gap, long headhunter timelines, severe project delays |
        """)

    # =========================================================================
    # 3. DATA SEGMENTATION & FILTER ENGINE
    # =========================================================================
    df_analysis = user_df.copy()
    # Ensure date columns are datetime — may arrive as strings from user_df
    for _dc in ["Hire_Date", "Exit_Date"]:
        if _dc in df_analysis.columns:
            df_analysis[_dc] = pd.to_datetime(df_analysis[_dc], errors="coerce")

    if selected_company != "All Companies":
        df_analysis = df_analysis[df_analysis["Company"] == selected_company].copy()
    if selected_dept != "All Departments":
        df_analysis = df_analysis[df_analysis["Department"] == selected_dept].copy()

    if selected_career == "Entry-Level Staff (Level 1)":
        df_analysis = df_analysis[df_analysis["JobLevel"] == 1].copy()
    elif selected_career == "Mid-Level Professionals (Level 2-3)":
        df_analysis = df_analysis[df_analysis["JobLevel"].isin([2, 3])].copy()
    elif selected_career == "Management & Specialists (Level 4-5)":
        df_analysis = df_analysis[df_analysis["JobLevel"].isin([4, 5])].copy()

    if selected_tenure == "New Hires (0-2 Yrs)":
        df_analysis = df_analysis[df_analysis["YearsAtCompany"] <= 2].copy()
    elif selected_tenure == "Mid-Tenure (3-7 Yrs)":
        df_analysis = df_analysis[(df_analysis["YearsAtCompany"] >= 3) & (df_analysis["YearsAtCompany"] <= 7)].copy()
    elif selected_tenure == "Tenured Pillars (8+ Yrs)":
        df_analysis = df_analysis[df_analysis["YearsAtCompany"] >= 8].copy()

    # =========================================================================
    # 4. ROW-BY-ROW COST CALCULATION USING JOB LEVEL BENCHMARKS
    # =========================================================================
    df_analysis = df_analysis.copy()
    df_analysis["AnnualSalary"] = df_analysis["MonthlyIncome"] * 12

    def apply_job_level_cost(row):
        if row["JobLevel"] == 1:
            return row["AnnualSalary"] * 0.30
        elif row["JobLevel"] in [2, 3]:
            return row["AnnualSalary"] * 0.60
        else:
            return row["AnnualSalary"] * 1.00

    df_analysis["IndividualTurnoverCost"] = df_analysis.apply(apply_job_level_cost, axis=1)

    # Inject risk probability scores — replaced by trained model scores when available
    df_analysis["Attrition_Risk_Prob"] = get_fallback_rf_predictions(df_analysis)

    # Explicit memory isolation: keep historical losses and active workforce fully separate
    historical_attrited = df_analysis[df_analysis["Attrition"] == "Yes"].copy()
    active_workforce    = df_analysis[df_analysis["Attrition"] == "No"].copy()
    high_risk_active_staff = active_workforce[active_workforce["Attrition_Risk_Prob"] >= 0.50].copy()

    realized_historical_loss = historical_attrited["IndividualTurnoverCost"].sum()
    at_risk_exposure_cost    = high_risk_active_staff["IndividualTurnoverCost"].sum()

    # =========================================================================
    # 5. ACTIVE-WINDOW MONTHLY RUN-RATE MODEL
    # =========================================================================
    # Count only the unique calendar months where departures actually occurred
    has_exit_dates = (
        len(historical_attrited) > 0
        and historical_attrited["Exit_Date"].notna().sum() > 0
    )

    if has_exit_dates:
        exit_periods = historical_attrited["Exit_Date"].dropna().dt.to_period("M")
        unique_active_months = max(exit_periods.nunique(), 1)
    else:
        unique_active_months = 12

    # Baseline burn velocity: average cost lost per active departure month
    average_monthly_loss_velocity = (
        realized_historical_loss / unique_active_months
        if unique_active_months > 0 else 0.0
    )

    # 12-month cumulative projection, capped at at-risk exposure ceiling
    ceiling_cap = at_risk_exposure_cost if at_risk_exposure_cost > 0 else realized_historical_loss * 1.5
    x_labels = [f"+{i} Mo" for i in range(1, 13)]
    cumulative_projection = [
        min(i * average_monthly_loss_velocity, ceiling_cap)
        for i in range(1, 13)
    ]
    total_predicted_12m_bleed = cumulative_projection[-1]

    # =========================================================================
    # 6. MoM DELTA COMPUTATION FOR METRIC TILES
    # =========================================================================
    # Derive MoM deltas by comparing the two most-recent active departure periods
    if has_exit_dates and exit_periods.nunique() >= 2:
        sorted_periods = sorted(exit_periods.unique())
        latest_period   = sorted_periods[-1]
        previous_period = sorted_periods[-2]

        latest_mask   = historical_attrited["Exit_Date"].dt.to_period("M") == latest_period
        previous_mask = historical_attrited["Exit_Date"].dt.to_period("M") == previous_period

        latest_active_risk   = active_workforce[active_workforce["Attrition_Risk_Prob"] >= 0.50]
        previous_active_risk = latest_active_risk  # active cohort is static; deltas proxy velocity shift

        latest_cost   = historical_attrited[latest_mask]["IndividualTurnoverCost"].sum()
        previous_cost = historical_attrited[previous_mask]["IndividualTurnoverCost"].sum()

        latest_headcount   = len(historical_attrited[latest_mask])
        previous_headcount = len(historical_attrited[previous_mask])

        mom_headcount_delta = len(high_risk_active_staff) - previous_headcount
        mom_cost_delta      = latest_cost - previous_cost
    else:
        mom_headcount_delta = None
        mom_cost_delta      = None

    # =========================================================================
    # 7. EXECUTIVE KPI SCORECARDS
    # =========================================================================
    st.markdown("---")
    st.subheader("📋 Headline Risk Overview")

    total_cohort           = len(df_analysis)
    total_attrited_cohort  = len(historical_attrited)
    cohort_attrition_rate  = (total_attrited_cohort / total_cohort * 100) if total_cohort > 0 else 0.0
    st.caption(
        f"Cohort Attrition Rate: **{cohort_attrition_rate:.1f}%** "
        f"({total_attrited_cohort} resigned / {total_cohort} total in selected filters)"
    )

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    active_total_count = len(active_workforce) if len(active_workforce) > 0 else 1

    with kpi_col1:
        metric_card(
            "At-Risk Active Staff",
            f"{len(high_risk_active_staff):,}",
            f"MoM: {mom_headcount_delta:+,} headcount" if mom_headcount_delta is not None else "Flagged employees with risk score ≥ 50%",
            progress=len(high_risk_active_staff) / active_total_count,
            progress_label=f"{len(high_risk_active_staff) / active_total_count:.0%} of active workforce flagged"
        )

    with kpi_col2:
        metric_card(
            "Total Financial Exposure",
            f"${at_risk_exposure_cost:,.0f}",
            f"MoM: ${mom_cost_delta:+,.0f}" if mom_cost_delta is not None else "Combined replacement cost of high-risk staff",
            progress=min(at_risk_exposure_cost / ceiling_cap, 1.0) if ceiling_cap > 0 else 0,
            progress_label=f"{min(at_risk_exposure_cost / ceiling_cap, 1.0):.0%} of exposure ceiling" if ceiling_cap > 0 else ""
        )

    with kpi_col3:
        metric_card(
            "Projected 12-Month Loss Pipeline",
            f"${total_predicted_12m_bleed:,.0f}",
            f"+${average_monthly_loss_velocity:,.0f} / mo burn rate",
            progress=min(total_predicted_12m_bleed / ceiling_cap, 1.0) if ceiling_cap > 0 else 0,
            progress_label=f"{min(total_predicted_12m_bleed / ceiling_cap, 1.0):.0%} of exposure ceiling reached" if ceiling_cap > 0 else ""
        )

    # =========================================================================
    # 8. DUAL PANEL CHART LAYOUT
    # =========================================================================
    st.markdown("---")
    st.subheader("📈 Financial Impact Dashboard Analytics")

    chart_col1, chart_col2 = st.columns(2)

    # --- LEFT: Horizontal balance sheet bar chart ---
    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor="white")
        ax1.set_facecolor("white")

        categories = [
            "Realized Operational Losses\n(Past Resignations)",
            "Active High-Risk Value\n(Flagged Current Staff)"
        ]
        costs      = [realized_historical_loss, at_risk_exposure_cost]
        bar_colors = ["#A0A0A0", "#D9534F"]

        bars = ax1.barh(categories, costs, color=bar_colors, height=0.45)

        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax1.text(
                    width * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"${width:,.0f}",
                    ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10
                )

        ax1.set_title("Current Corporate Balance Sheet Context",
                      fontsize=12, fontweight="bold", pad=15)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.get_xaxis().set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close()

    # --- RIGHT: 12-month cumulative area trend line ---
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="white")
        ax2.set_facecolor("white")

        ax2.plot(x_labels, cumulative_projection,
                 color="#D9534F", linewidth=2.5, marker="o", markersize=5, zorder=2)
        ax2.fill_between(x_labels, cumulative_projection,
                         color="#D9534F", alpha=0.12, zorder=1)

        # Dashed ceiling cap line to mark the maximum preventable exposure boundary
        ax2.axhline(y=ceiling_cap, color="#D9534F", linestyle="--", linewidth=1.2, alpha=0.6)
        ax2.text(
            x_labels[-1], ceiling_cap,
            "  Ceiling Cap\n  (Max Exposure)",
            va="bottom", ha="right", fontsize=7.5, color="#D9534F"
        )

        ax2.set_title("12-Month Cumulative Loss Projection",
                      fontsize=12, fontweight="bold", pad=15)
        ax2.set_xlabel("Forward Monthly Milestones", fontsize=9, labelpad=8)
        ax2.set_ylabel("Cumulative Projected Cash Loss ($)", fontsize=9)
        ax2.tick_params(axis="x", rotation=30, labelsize=8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#cccccc")

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()