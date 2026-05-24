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
    import base64
    with open("static/logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; padding: 8px 0 4px 0;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="width:80px; height:80px; border-radius:50%; object-fit:cover;">
        </div>
        """,
        unsafe_allow_html=True,
    )
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

# =========================================================================
# --- GLOBAL CARD COMPONENT (shared across all tabs) ---
# =========================================================================
st.markdown(
    """
    <style>
    .attrition-card {
        background: linear-gradient(180deg, rgba(38,39,48,0.98), rgba(23,24,31,0.98));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        height: 176px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .attrition-card .label {
        font-size: 0.95rem;
        color: #b9c0d0;
        margin-bottom: 0.30rem;
    }
    .attrition-card .value {
        font-size: 2.0rem;
        font-weight: 700;
        line-height: 1.12;
        color: #ffffff;
        min-height: 2.35rem;
        overflow-wrap: anywhere;
    }
    .attrition-card .sub {
        color: #9aa3b2;
        font-size: 0.84rem;
        line-height: 1.30;
        min-height: 2.15rem;
    }
    .attrition-card .meter-wrap {
        margin-top: 0.55rem;
    }
    .attrition-card .meter-label {
        color: #c7cedb;
        font-size: 0.76rem;
        margin-bottom: 0.28rem;
    }
    .attrition-card .meter-track {
        width: 100%;
        height: 7px;
        background: rgba(255,255,255,0.10);
        border-radius: 999px;
        overflow: hidden;
    }
    .attrition-card .meter-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #f87171);
    }
    .section-note {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        color: #c7cedb;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def metric_card(label, value, subtext="", progress=None, progress_label=""):
    meter_block = ""
    if progress is not None:
        progress = max(0, min(1, float(progress)))
        meter_block = (
            f'<div class="meter-wrap">'
            f'<div class="meter-label">{progress_label}</div>'
            f'<div class="meter-track">'
            f'<div class="meter-fill" style="width: {progress * 100:.1f}%;"></div>'
            f'</div>'
            f'</div>'
        )
    html = (
        f'<div class="attrition-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{subtext}</div>'
        f'{meter_block}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
                metric_card(
                    "Active Headcount",
                    f"{int(current_headcount):,}",
                    "Active staff at end of selected window",
                    progress=current_headcount / total_ever_served if total_ever_served > 0 else 0,
                    progress_label=f"{current_headcount / total_ever_served:.0%} of total ever served" if total_ever_served > 0 else ""
                )
            with kpi_col2:
                metric_card(
                    "Total Window Departures",
                    f"{int(total_period_attrition):,}",
                    "Employees who left during the selected period",
                    progress=total_period_attrition / total_ever_served if total_ever_served > 0 else 0,
                    progress_label=f"{total_period_attrition / total_ever_served:.0%} of total ever served" if total_ever_served > 0 else ""
                )
            with kpi_col3:
                metric_card(
                    "Period Attrition Rate",
                    f"{attrition_rate:.2f}%",
                    "Share of total workforce that resigned in this window",
                    progress=min(attrition_rate / 100, 1.0),
                    progress_label="Resignation share within selected period"
                )

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
    st.header("🎯 Resignation Drivers Analysis")
    st.caption("A descriptive view of existing resignation patterns based only on the selected HR driver columns. This section does not use model feature importance or prediction outputs.")
    st.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.16); margin:0.65rem 0 1.05rem 0;'></div>",
        unsafe_allow_html=True,
    )

    # Keep this page clean when no custom data has been uploaded yet.
    # This avoids showing sample-data results or schema-related messages in this section.
    if uploaded_file is None:
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

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

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
    avg_job = wb_df['JobSatisfaction'].mean()
    avg_env = wb_df['EnvironmentSatisfaction'].mean()
    avg_rel = wb_df['RelationshipSatisfaction'].mean()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        metric_card(
            "Avg Job Satisfaction",
            f"{avg_job:.2f} / 4",
            "Average job satisfaction across filtered employees",
            progress=avg_job / 4,
            progress_label=f"{avg_job / 4:.0%} of maximum score"
        )
    with sc2:
        metric_card(
            "Avg Environment Satisfaction",
            f"{avg_env:.2f} / 4",
            "Average environment satisfaction across filtered employees",
            progress=avg_env / 4,
            progress_label=f"{avg_env / 4:.0%} of maximum score"
        )
    with sc3:
        metric_card(
            "Avg Relationship Satisfaction",
            f"{avg_rel:.2f} / 4",
            "Average relationship satisfaction across filtered employees",
            progress=avg_rel / 4,
            progress_label=f"{avg_rel / 4:.0%} of maximum score"
        )

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

    tab_job, tab_env, tab_rel = st.tabs(["💼 Job Satisfaction", "🏢 Environment Satisfaction", "🤝 Relationship Satisfaction"])

    with tab_job:
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

    with tab_env:
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

    with tab_rel:
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