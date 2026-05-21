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
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'], errors='coerce')
    return df


# =========================================================================
# --- SIDEBAR INTERFACE LAYOUT ---
# =========================================================================

with st.sidebar:
    st.markdown("### 🏢 **Omni-Retention Engine**")
    st.markdown("---")
    
    # 1. NAVIGATION HUB
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
    
    # 2. FILE MANAGEMENT SYSTEM
    with st.expander("⚙️ Data Sync Options", expanded=False):
        uploaded_file = st.file_uploader("Upload Company CSV Dataset", type=["csv"])
        
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
# --- LIVE RUNTIME DATA LOADING (FIXED GLOBAL PARSING) ---
# =========================================================================

user_df = get_default_dataset()

if uploaded_file is not None:
    try:
        temp_df = pd.read_csv(uploaded_file)
        
        # Check order-independent matching using set equality
        if set(EXPECTED_SCHEMA.keys()) == set(temp_df.columns):
            # GLOBAL FIXED PARSER: Removed hardcoded formatting loops completely. 
            # This cleanly handles both 'YYYY-MM-DD' and slash layouts like 'M/D/YYYY'
            temp_df['Hire_Date'] = pd.to_datetime(temp_df['Hire_Date'], errors='coerce')
            temp_df['Exit_Date'] = pd.to_datetime(temp_df['Exit_Date'], errors='coerce')
            
            user_df = temp_df
            st.sidebar.success("✅ Custom dataset synced successfully!")
        else:
            st.sidebar.error("❌ File mismatch. Column headers do not line up with the standard layout.")
            missing = set(EXPECTED_SCHEMA.keys()) - set(temp_df.columns)
            if missing:
                st.sidebar.write(f"Missing fields: {missing}")
    except Exception as e:
        st.sidebar.error(f"Error parsing uploaded file: {e}")


# =========================================================================
# --- MAIN DATA DISPLAY MANAGEMENT ---
# =========================================================================

st.caption("PROJECT OMNI-RETENTION")

if app_feature == "Executive Summary":
    st.header("📊 Executive Summary Dashboard (Active Headcount Engine)")
    st.markdown("---")

    df_exec = user_df.copy()

    # Standardize data types and dates across the workspace
    df_exec['Hire_Date'] = pd.to_datetime(df_exec['Hire_Date'], errors='coerce')
    df_exec['Exit_Date'] = pd.to_datetime(df_exec['Exit_Date'], errors='coerce')
    
    # Drop rows without a valid hire date to preserve pipeline security
    df_exec = df_exec[df_exec['Hire_Date'].notna()]

    # =========================================================================
    # --- FILTERS ROW ---
    # =========================================================================
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        company_options = ["All"] + sorted(df_exec['Company'].dropna().unique().tolist())
        selected_company = st.selectbox("Select Company Focus:", company_options)

    if selected_company != "All":
        df_exec = df_exec[df_exec['Company'] == selected_company]

    # Check if data exists after applying company filter
    if df_exec.empty:
        st.warning("⚠️ No employee records match the selected company.")
    else:
        # Generate the complete dynamic year range based on actual dataset bounds safely
        min_year = int(df_exec['Hire_Date'].dt.year.min())
        max_hire_year = df_exec['Hire_Date'].dt.year.max()
        max_exit_year = df_exec['Exit_Date'].dt.year.max()
        max_year = int(max(max_hire_year, max_exit_year) if pd.notna(max_exit_year) else max_hire_year)
        
        # Complete list of available years sorted descending (most recent first)
        available_years = sorted(list(range(min_year, max_year + 1)), reverse=True)
        
        with filter_col2:
            # Default selection automatically set to the past 5 years
            default_years = available_years[:5]
            selected_years = st.multiselect(
                "Select View Analysis Years (Multi-Select):", 
                options=available_years, 
                default=default_years
            )

        # Handle case where no years are selected
        if not selected_years:
            st.warning("⚠️ Please select at least one year to view the analytics summary.")
        else:
            # =========================================================================
            # --- ACTIVE HEADCOUNT SNAPSHOT ENGINE ---
            # =========================================================================
            months_bridge = []
            all_departures = []
            
            import altair as alt

            # Sort selected years chronologically (ascending) for the timeline view
            chronological_years = sorted(selected_years)
            
            for yr in chronological_years:
                for month_idx in range(1, 13):
                    month_start = pd.Timestamp(year=yr, month=month_idx, day=1)
                    month_end = month_start + pd.offsets.MonthEnd(0)
                    
                    active_mask = (df_exec['Hire_Date'] <= month_end) & (
                        (df_exec['Exit_Date'].isna()) | (df_exec['Exit_Date'] > month_start)
                    )
                    df_active_this_month = df_exec[active_mask]
                    
                    new_hires_count = len(df_active_this_month[
                        (df_active_this_month['Hire_Date'] >= month_start) & (df_active_this_month['Hire_Date'] <= month_end)
                    ])
                    
                    # Capture exact departures matching the filter scope timeline
                    left_mask = (df_active_this_month['Attrition'].astype(str).str.strip().str.lower() == 'yes') & \
                                (df_active_this_month['Exit_Date'] >= month_start) & \
                                (df_active_this_month['Exit_Date'] <= month_end)
                    df_left_this_month = df_active_this_month[left_mask]
                    
                    if not df_left_this_month.empty:
                        all_departures.append(df_left_this_month)
                    
                    attrition_count = len(df_left_this_month)
                    active_total = len(df_active_this_month)
                    monthly_attrition_rate = (attrition_count / active_total * 100) if active_total > 0 else 0.0
                    
                    # Format dynamic timeline tags
                    if len(selected_years) == 1:
                        display_label = month_start.strftime('%B')
                        sort_value = month_idx
                    else:
                        display_label = month_start.strftime('%Y-%m')
                        sort_value = month_start
                        
                    months_bridge.append({
                        'Month_Identifier': display_label,
                        'Sort_Key': sort_value,
                        'Active_Headcount': active_total,
                        'New_Hires': new_hires_count,
                        'Attrition_Departures': attrition_count,
                        'Attrition_Rate_Percent': monthly_attrition_rate
                    })
                    
            timeline_summary_df = pd.DataFrame(months_bridge)
            
            # Reconstruct consolidated departure dataframe for category breakdowns
            df_departures_period = pd.concat(all_departures, ignore_index=True) if all_departures else pd.DataFrame()
            
            x_axis_field = 'Month_Identifier:O'
            x_axis_sort = alt.EncodingSortField(field='Sort_Key', order='ascending')
            x_axis_title = 'Timeline' if len(selected_years) > 1 else f'Calendar Months ({chronological_years[0]})'

            # =========================================================================
            # --- KPI BOXES LAYER ---
            # =========================================================================
            st.markdown(f"### Key Performance Indicators Summary")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

            current_headcount = timeline_summary_df.iloc[-1]['Active_Headcount']
            total_period_attrition = timeline_summary_df['Attrition_Departures'].sum()
            
            total_ever_served = current_headcount + total_period_attrition
            attrition_rate = (total_period_attrition / total_ever_served * 100) if total_ever_served > 0 else 0.0

            with kpi_col1:
                st.metric(label="Active Headcount (Current Window End)", value=f"{int(current_headcount):,} Active Staff")
            with kpi_col2:
                st.metric(label="Total Window Departures", value=f"{int(total_period_attrition):,} Losses")
            with kpi_col3:
                st.metric(label="Calculated Period Attrition Rate", value=f"{attrition_rate:.2f}%")

            st.markdown("---")

            # =========================================================================
            # --- ROW 1: THREE SIDE-BY-SIDE HISTORICAL TRENDS ---
            # =========================================================================
            st.markdown("### Historical Trends Matrix")
            chart_col1, chart_col2, chart_col3 = st.columns(3)

            with chart_col1:
                st.markdown("#### MoM Attrition Rate Velocity")
                attrition_line = alt.Chart(timeline_summary_df).mark_line(
                    color='#ff4b4b', strokeWidth=3, point=True
                ).encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Attrition_Rate_Percent:Q', title='Attrition Rate (%)'),
                    tooltip=['Month_Identifier', alt.Tooltip('Attrition_Rate_Percent:Q', format='.2f', title='Attrition %')]
                ).properties(height=260)
                st.altair_chart(attrition_line, use_container_width=True)

            with chart_col2:
                st.markdown("#### Active Headcount Scaling")
                active_bars = alt.Chart(timeline_summary_df).mark_bar(color='#1f77b4').encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Active_Headcount:Q', title='Workforce Volume'),
                    tooltip=['Month_Identifier', alt.Tooltip('Active_Headcount:Q', title='Active Staff')]
                ).properties(height=260)
                st.altair_chart(active_bars, use_container_width=True)

            with chart_col3:
                st.markdown("#### Volumetric Attrition Losses")
                departures_bars = alt.Chart(timeline_summary_df).mark_bar(color='#e45756').encode(
                    x=alt.X(x_axis_field, sort=x_axis_sort, title=x_axis_title),
                    y=alt.Y('Attrition_Departures:Q', title='Departures Count'),
                    tooltip=['Month_Identifier', alt.Tooltip('Attrition_Departures:Q', title='Losses')]
                ).properties(height=260)
                st.altair_chart(departures_bars, use_container_width=True)

            st.markdown("---")

            # =========================================================================
            # --- ROW 2: SIDE-BY-SIDE CATEGORICAL BREAKDOWNS ---
            # =========================================================================
            st.markdown("### Categorical Attrition Distribution")
            
            if df_departures_period.empty:
                st.info("ℹ️ No departure entries recorded during this timeframe to break down by category.")
            else:
                breakdown_col1, breakdown_col2 = st.columns(2)
                
                with breakdown_col1:
                    st.markdown("#### Primary Root Causes (Reasons)")
                    reason_summary = df_departures_period.groupby('Attrition_Reason', dropna=False).size().reset_index(name='Count')
                    # Fill missing reason tags explicitly for clean presentation
                    reason_summary['Attrition_Reason'] = reason_summary['Attrition_Reason'].fillna('Unspecified')
                    
                    reason_chart = alt.Chart(reason_summary).mark_bar(color='#e45756', cornerRadiusEnd=3).encode(
                        y=alt.Y('Attrition_Reason:N', sort='-x', title='Stated Reason'),
                        x=alt.X('Count:Q', title='Total Losses'),
                        tooltip=['Attrition_Reason:N', 'Count:Q']
                    ).properties(height=280)
                    st.altair_chart(reason_chart, use_container_width=True)
                    
                with breakdown_col2:
                    st.markdown("#### Organizational Impact (Role & Department)")
                    role_summary = df_departures_period.groupby(['Department', 'JobRole']).size().reset_index(name='Count')
                    
                    role_chart = alt.Chart(role_summary).mark_bar(cornerRadiusEnd=3).encode(
                        y=alt.Y('JobRole:N', sort='-x', title='Job Role Category'),
                        x=alt.X('Count:Q', title='Total Losses'),
                        color=alt.Color('Department:N', scale=alt.Scale(scheme='tableau10'), title='Department'),
                        tooltip=['Department:N', 'JobRole:N', 'Count:Q']
                    ).properties(height=280)
                    st.altair_chart(role_chart, use_container_width=True)

            # =========================================================================
            # --- DATA REFERENCE REGISTER ---
            # =========================================================================
            with st.expander("🔍 Operational Overview: Review Monthly Aggregates Table"):
                display_df = timeline_summary_df.copy()
                display_df['Attrition_Rate_Percent'] = display_df['Attrition_Rate_Percent'].map('{:,.2f}%'.format)
                st.dataframe(display_df.drop(columns=['Sort_Key'], errors='ignore'), use_container_width=True)

        st.markdown("---")


# Other tab placeholders remain protected...


elif app_feature == "Attrition Drivers":
    st.header("🎯 Attrition Drivers Analysis")
    st.markdown("---")
    # Matt


elif app_feature == "Wellbeing/Performance":
    st.header("⚖️ Wellbeing & Performance Matrix")
    st.markdown("---")
    # Ruth

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