import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #0D1117 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Radio — remove coloured dots, add pill highlight ── */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
    transition: background 0.15s, color 0.15s !important;
    display: flex !important; align-items: center !important; gap: 8px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(96,165,250,0.08) !important;
    color: var(--accent-blue) !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }

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

/* ── Sidebar brand ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 1rem 0; margin-bottom: 0.5rem;
}
.sidebar-brand .brand-icon { font-size: 1.6rem; }
.sidebar-brand .brand-title { 
    font-size: 1rem; font-weight: 700; 
    background: var(--gradient-hr); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-tag { font-size: 0.7rem; color: var(--text-dim); margin-top: -2px; }

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
# 4. LOAD DATA + SESSION STATE
# ============================================================
df = load_data()

st.session_state["_df"] = df
model, FEATURES, model_metrics = get_trained_model(hash(str(df.shape)))

# ============================================================
# 5. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
        <span class='brand-icon'>🛡️</span>
        <div>
            <div class='brand-title'>Risk Intelligence</div>
            <div class='brand-tag'>Retention Analytics v2</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["📊 Dashboard", "🔍 Attrition Factors", "⚠️ Risk Watchlist", "🔮 Predict Employee"],
        label_visibility="visible"
    )

    st.divider()

    # ── Filters ──
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
    st.markdown("""
    <div class='section-note'>
        <b>Analytics</b><br>
        Explore and dive in the dataset
    </div>
    """.format(auc=model_metrics["auc"]), unsafe_allow_html=True)
    st.caption(f"Dataset: {len(filtered_df):,} of {len(df):,} records")

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
elif page == "🔍 Attrition Factors":
    st.markdown("""
    <div class='page-header'>
        <h1>🤖 Underlying Drivers of Turnover</h1>
        <div class='ph-sub'>AI-ranked risk factors derived from Random Forest feature importances</div>
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
elif page == "⚠️ Risk Watchlist":
    st.markdown("""
    <div class='page-header'>
        <h1>⚠️ High-Risk Retention Watchlist</h1>
        <div class='ph-sub'>AI-scored employees ranked by resignation probability</div>
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

    # ── Download ──
    csv = top_risk[[id_col, role_col] + ([dept_col] if dept_col else []) + ["Risk_Probability"]].copy()
    csv["Risk_Probability"] = csv["Risk_Probability"].map(lambda x: f"{x:.2%}")
    st.download_button(
        "⬇️ Export Watchlist CSV",
        data=csv.to_csv(index=False),
        file_name="risk_watchlist.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 4 ── PREDICT INDIVIDUAL EMPLOYEE
# ============================================================
elif page == "🔮 Predict Employee":
    st.markdown("""
    <div class='page-header'>
        <h1>🔮 Individual Risk Predictor</h1>
        <div class='ph-sub'>Enter employee attributes to get a real-time attrition probability score</div>
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
