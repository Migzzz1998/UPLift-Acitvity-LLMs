import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from anthropic import Anthropic
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from streamlit_option_menu import option_menu

# --- APP CONFIGURATION ---
st.set_page_config(page_title='Intelligent Retention Analytics', layout='wide')

# --- STRICT SCHEMA DEFINITION ---
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
DATE_FORMAT = "%Y-%m-%d"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Define fallback sample dataset so the app can render metrics even if no file is uploaded yet
@st.cache_data
def get_default_dataset():
    sample_data = {
        "Company": ["Company 1", "Company 1"], "Age": [53, 55], "Attrition": ["Yes", "No"], "BusinessTravel": ["Travel_Rarely", "Non-Travel"],
        "DailyRate": [607, 177], "Department": ["Research & Development", "Research & Development"], "DistanceFromHome": [2, 8], "Education": [5, 1],
        "EducationField": ["Technical Degree", "Medical"], "EmployeeCount": [1, 1], "EmployeeNumber": [1572, 1278],
        "EnvironmentSatisfaction": [3, 4], "Gender": ["Female", "Male"], "HourlyRate": [78, 37], "JobInvolvement": [2, 2],
        "JobLevel": [3, 4], "JobRole": ["Manufacturing Director", "Healthcare Representative"], "JobSatisfaction": [4, 2], "MaritalStatus": ["Married", "Divorced"],
        "MonthlyIncome": [10169, 13577], "MonthlyRate": [14618, 25592], "NumCompaniesWorked": [0, 1], "Over18": ["Y", "Y"], "OverTime": ["No", "Yes"],
        "PercentSalaryHike": [16.0, 15.0], "PerformanceRating": [3, 3], "RelationshipSatisfaction": [2, 4], "StandardHours": [80, 80],
        "StockOptionLevel": [1, 1], "TotalWorkingYears": [34, 34], "TrainingTimesLastYear": [4, 3], "WorkLifeBalance": [3, 3],
        "YearsAtCompany": [33, 33], "YearsInCurrentRole": [7, 9], "YearsSinceLastPromotion": [1, 15], "YearsWithCurrManager": [9, 0],
        "Reliability": [80.61, 88.80], "Service Quality": [2, 3], "Service Efficiency": [11.14, 15.00], "Attrition_Reason": ["Better Opportunity", None],
        "Hire_Date": ["1990-08-20", "1990-09-07"], "Exit_Date": ["2023-08-17", None]
    }
    df = pd.DataFrame(sample_data)
    for col in [c for c, d in EXPECTED_SCHEMA.items() if d == "datetime"]:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors='coerce')
    return df


# =========================================================================
# --- SIDEBAR INTERFACE LAYOUT ---
# =========================================================================

with st.sidebar:
    st.markdown("**Omni-Retention Engine**")
    st.markdown("---")
    
    # 1. NAVIGATION HUB (Positioned Top)
    app_feature = option_menu(
        menu_title=None,
        options=[
            "Executive Summary", 
            "Attrition Drivers", 
            "Wellbeing/Performance", 
            "Predictive Analytics", 
            "Employees to Review", 
            "Attrition Simulator"
        ],
        icons=["speedometer2", "graph-down-arrow", "heart-pulse", "cpu", "person-lines-fill", "sliders"],
        default_index=0,
        styles={
            "container": {"padding": "0px!important", "background-color": "transparent"},
            "icon": {"color": "#ff4b4b", "font-size": "16px"},
            "nav-link": {"font-size": "13px", "text-align": "left", "margin": "6px 0px", "padding": "8px"},
            "nav-link-selected": {"background-color": "#262730", "color": "white"},
        }
    )
    
    st.markdown("---")
    
    # 2. FILE MANAGEMENT SYSTEM (Positioned Bottom)
    with st.expander("⚙️ Data Sync Options", expanded=False):
        uploaded_file = st.file_uploader("Upload Company CSV Dataset", type=["csv"])
        
        # Reference template downloader built inside expander box
        template_df = get_default_dataset()
        csv_template = convert_df_to_csv(template_df)
        st.download_button(
            label="Download Reference Template", 
            data=csv_template, 
            file_name="omni_retention_template.csv", 
            mime="text/csv",
            use_container_width=True
        )


# =========================================================================
# --- LIVE RUNTIME DATA LOADING ---
# =========================================================================

# Process uploaded file if available; otherwise drop back onto default sample dataset effortlessly
user_df = get_default_dataset()

if uploaded_file is not None:
    try:
        temp_df = pd.read_csv(uploaded_file)
        if list(EXPECTED_SCHEMA.keys()) == list(temp_df.columns):
            for col in [c for c, d in EXPECTED_SCHEMA.items() if d == "datetime"]:
                temp_df[col] = pd.to_datetime(temp_df[col], format=DATE_FORMAT, errors='coerce')
            user_df = temp_df
            st.sidebar.success("✅ Custom data compiled successfully!")
        else:
            st.sidebar.error("❌ File mismatch. Please use the reference template.")
    except Exception as e:
        st.sidebar.error(f"Error parsing uploaded file: {e}")


# =========================================================================
# --- MAIN DATA DISPLAY MANAGEMENT ---
# =========================================================================

st.caption("PROJECT OMNI-RETENTION")

if app_feature == "Executive Summary":
    st.header("📊 Executive Summary Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Headcount", f"{len(user_df):,} Employees")
    col2.metric("Total Attrited Records", len(user_df[user_df['Attrition'] == 'Yes']))
    col3.metric("Attrition Rate", f"{(len(user_df[user_df['Attrition'] == 'Yes']) /len(user_df))*100:.2f}%")
    
    st.subheader("Raw Data Operational Overview")
    st.dataframe(user_df.head(10), use_container_width=True)
    # IR

elif app_feature == "Attrition Drivers":
    st.header("🎯 Attrition Drivers Analysis")
    st.markdown("---")
    # Matt


elif app_feature == "Wellbeing/Performance":
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
        sat_levels = {"All": None, "Low (1)": 1, "Medium (2)": 2, "High (3)": 3, "Very High (4)": 4}
        selected_sat_label = st.selectbox("Satisfaction Level (Job)", list(sat_levels.keys()))

    # Apply filters to a working copy of the data
    wb_df = user_df.copy()
    if selected_company != "All":
        wb_df = wb_df[wb_df["Company"] == selected_company]
    if selected_dept != "All":
        wb_df = wb_df[wb_df["Department"] == selected_dept]
    if sat_levels[selected_sat_label] is not None:
        wb_df = wb_df[wb_df["JobSatisfaction"] == sat_levels[selected_sat_label]]

    st.markdown("---")

    # Soft, easy-on-the-eyes colors: teal for Stayed, coral for Left
    PALETTE = {"No": "#5BA4A4", "Yes": "#E07B6A"}
    RATING_LABELS = ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"]
    FIGSIZE = (5, 3.8)

    # Consistent card container styling for the 2x2 grid
    st.markdown("""
    <style>
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        background-color: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1rem 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    def style_chart(ax, title, xlabel, ylabel, legend_keys=None):
        # Consistent styling with solid square legend patches — no line indicators
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        if legend_keys:
            patches = [mpatches.Patch(facecolor=PALETTE[k],
                                      label="Stayed" if k == "No" else "Left")
                       for k in legend_keys]
            ax.legend(handles=patches, fontsize=9, title="")

    # --- SCORECARDS ---
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Avg Job Satisfaction", f"{wb_df['JobSatisfaction'].mean():.2f} / 4")
    sc2.metric("Avg Environment Satisfaction", f"{wb_df['EnvironmentSatisfaction'].mean():.2f} / 4")
    sc3.metric("Avg Relationship Satisfaction", f"{wb_df['RelationshipSatisfaction'].mean():.2f} / 4")

    st.markdown("---")

    # --- JOB SATISFACTION SECTION ---
    st.subheader("💼 Job Satisfaction")
    st.caption("How satisfied employees are with their job, overtime load, and work-life balance — and whether that's linked to leaving.")

    js_col1, js_col2, js_col3 = st.columns(3)

    with js_col1:
        js_data = wb_df.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=js_data, x="JobSatisfaction", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Job Satisfaction", "Satisfaction Rating", "Number of Employees",
                    legend_keys=["No", "Yes"])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(RATING_LABELS)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with js_col2:
        # Rename overtime values to plain business language before plotting
        ot_df = wb_df.copy()
        ot_df["OverTime"] = ot_df["OverTime"].map({"No": "No OT", "Yes": "Rendered OT"})
        ot_data = ot_df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=ot_data, x="OverTime", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Overtime vs Attrition", "Overtime Status", "Number of Employees",
                    legend_keys=["No", "Yes"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with js_col3:
        wlb_data = wb_df.groupby(["WorkLifeBalance", "Attrition"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=wlb_data, x="WorkLifeBalance", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Work-Life Balance", "Balance Rating", "Number of Employees",
                    legend_keys=["No", "Yes"])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(RATING_LABELS)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- ENVIRONMENT SATISFACTION SECTION ---
    st.subheader("🏢 Environment Satisfaction")
    st.caption("How comfortable employees feel in their physical and social work environment, and whether commute distance plays a role.")

    env_col1, env_col2 = st.columns(2)

    with env_col1:
        env_data = wb_df.groupby(["EnvironmentSatisfaction", "Attrition"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=env_data, x="EnvironmentSatisfaction", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Environment Satisfaction", "Satisfaction Rating", "Number of Employees",
                    legend_keys=["No", "Yes"])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(RATING_LABELS)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with env_col2:
        bins = [0, 5, 10, 15, 20, 100]
        labels_dist = ["0–5 km", "6–10 km", "11–15 km", "16–20 km", "20+ km"]
        wb_df["DistanceBucket"] = pd.cut(wb_df["DistanceFromHome"], bins=bins, labels=labels_dist, right=True)
        dist_data = wb_df.groupby(["DistanceBucket", "Attrition"], observed=True).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=dist_data, x="DistanceBucket", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Distance From Home", "Commute Distance", "Number of Employees",
                    legend_keys=["No", "Yes"])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- RELATIONSHIP SATISFACTION SECTION ---
    st.subheader("🤝 Relationship Satisfaction")
    st.caption("How employees feel about their relationships at work — with peers and managers — and how long they've been with their current manager.")

    rel_col1, rel_col2 = st.columns(2)

    with rel_col1:
        rs_data = wb_df.groupby(["RelationshipSatisfaction", "Attrition"]).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        sns.barplot(data=rs_data, x="RelationshipSatisfaction", y="Count", hue="Attrition",
                    palette=PALETTE, ax=ax)
        style_chart(ax, "Relationship Satisfaction", "Satisfaction Rating", "Number of Employees",
                    legend_keys=["No", "Yes"])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(RATING_LABELS)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with rel_col2:
        mgr_avg = wb_df.groupby("Attrition")["YearsWithCurrManager"].mean().reset_index()
        mgr_avg["Label"] = mgr_avg["Attrition"].map({"No": "Stayed", "Yes": "Left"})
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        bars = ax.bar(mgr_avg["Label"], mgr_avg["YearsWithCurrManager"],
                      color=[PALETTE[v] for v in mgr_avg["Attrition"]], width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f} yrs", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Avg Years With Current Manager", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Average Years", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    rel_col3, rel_col4 = st.columns(2)

    with rel_col3:
        tenure_avg = wb_df.groupby("Attrition")["YearsAtCompany"].mean().reset_index()
        tenure_avg["Label"] = tenure_avg["Attrition"].map({"No": "Stayed", "Yes": "Left"})
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        bars = ax.bar(tenure_avg["Label"], tenure_avg["YearsAtCompany"],
                      color=[PALETTE[v] for v in tenure_avg["Attrition"]], width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{bar.get_height():.1f} yrs", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Avg Years At Company", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Average Years", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with rel_col4:
        # Line chart with bubble sizes — larger dot means more employees at that tenure year
        mgr_line = wb_df.groupby("YearsWithCurrManager").agg(
            AvgSat=("RelationshipSatisfaction", "mean"),
            Count=("RelationshipSatisfaction", "count")
        ).reset_index()

        # Scale bubble size so the reader can see which data points are based on more employees
        min_s, max_s = 40, 300
        count_range = mgr_line["Count"].max() - mgr_line["Count"].min()
        if count_range == 0:
            mgr_line["BubbleSize"] = min_s
        else:
            mgr_line["BubbleSize"] = ((mgr_line["Count"] - mgr_line["Count"].min())
                                       / count_range * (max_s - min_s) + min_s)

        # Dynamic y-axis — don't start at 0 so small differences are visible
        y_min = mgr_line["AvgSat"].min()
        y_max = mgr_line["AvgSat"].max()
        y_pad = max((y_max - y_min) * 0.4, 0.1)

        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.set_facecolor("white")
        ax.plot(mgr_line["YearsWithCurrManager"], mgr_line["AvgSat"],
                color="#5BA4A4", linewidth=2, zorder=1)
        ax.scatter(mgr_line["YearsWithCurrManager"], mgr_line["AvgSat"],
                   s=mgr_line["BubbleSize"], color="#5BA4A4", alpha=0.85, zorder=2)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title("Satisfaction by Manager Tenure", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Years With Current Manager\n(Bubble size = number of employees)", fontsize=9, labelpad=8)
        ax.set_ylabel("Avg Relationship Satisfaction", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


elif app_feature == "Predictive Analytics":
    st.header("🔮 Machine Learning Predictive Analytics")
    st.markdown("---")
    # Your visualization logic goes here

elif app_feature == "Employees to Review":
    st.header("🔍 Employees Designated for Review")
    st.markdown("---")
    # Your visualization logic goes here

elif app_feature == "Attrition Simulator":
    st.header("🕹️ Scenario Attrition Simulator ('What-If' Analysis)")
    st.markdown("---")
    # Your visualization logic goes here