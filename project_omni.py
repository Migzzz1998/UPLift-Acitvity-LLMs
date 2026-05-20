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

elif app_feature == "Attrition Drivers":
    st.header("🎯 Attrition Drivers Analysis")
    st.markdown("---")
    # Your visualization logic goes here 

elif app_feature == "Wellbeing/Performance":
    st.header("⚖️ Wellbeing & Performance Matrix")
    st.markdown("---")
    # Your visualization logic goes here

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