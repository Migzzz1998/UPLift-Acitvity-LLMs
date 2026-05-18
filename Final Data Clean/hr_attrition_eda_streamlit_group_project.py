"""
HR Attrition Project - Streamlit Backend / Terminal Handoff
===========================================================

Install:
    python -m pip install pandas numpy scikit-learn

Run:
    python hr_attrition_eda_streamlit_group_project.py

Output:
    Streamlit-ready DataFrames through run_full_analysis().
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# 1. SETTINGS
# =========================================================
# This section sets the basic file names and column lists used throughout the project.
# It also marks sensitive fields so they can be excluded from modeling and employee review outputs.

DEFAULT_INPUT_FILE = "HR_Attrition_MultiCompany.csv"
FALLBACK_INPUT_FILES = ["HR_Attrition_MultiCompany.csv", "HR_Attrition_Advanced.csv"]

COMPANY_COLUMNS = ["Company", "CompanyName", "Company_Name", "Organization", "Employer", "BusinessUnit", "Business Unit"]

SENSITIVE_COLUMNS = ["Gender", "MaritalStatus", "Age", "Age_Group"]

EDA_COLUMNS = [
    "Company", "OverTime", "BusinessTravel", "Department", "JobRole", "EducationField",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement",
    "RelationshipSatisfaction", "StockOptionLevel", "JobLevel", "PerformanceRating", "TrainingTimesLastYear",
    "Income_Group", "Tenure_Group", "Distance_Group", "Working_Years_Group",
    "Manager_Tenure_Group", "Companies_Worked_Group", "Hire_Year"
]

PATTERN_COLUMNS = [
    "JobRole", "Department", "OverTime", "BusinessTravel", "EducationField", "JobSatisfaction",
    "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction",
    "StockOptionLevel", "JobLevel", "Income_Group", "Tenure_Group", "Distance_Group",
    "Working_Years_Group", "Manager_Tenure_Group", "Companies_Worked_Group", "Hire_Year"
]

SATISFACTION_COLUMNS = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction"]

# =========================================================
# 2. LOAD AND CLEAN DATA
# =========================================================
# This section reads the CSV, removes empty/unnamed columns, cleans text spacing, and standardizes the company column.
# It also creates Employee_Key, converts Attrition into a numeric flag, and prepares date-based columns when available.

# Looks for the HR attrition CSV file to use.
# It first checks the expected filenames, then falls back to any CSV in the folder if needed.
def find_csv_file():
    for filename in FALLBACK_INPUT_FILES:
        if os.path.exists(filename):
            return filename
    csv_files = [file for file in os.listdir() if file.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Place the HR attrition CSV file in the same folder as this script.")
    return sorted(csv_files, key=lambda f: ("attrition" not in f.lower(), "company" not in f.lower(), f.lower()))[0]


# Makes the company column consistent by renaming possible company fields to "Company".
# If no company column exists, it creates a default value so company-level summaries can still run.
def standardize_company_column(df):
    company_col = None
    for col in COMPANY_COLUMNS:
        if col in df.columns:
            company_col = col
            break
    if company_col is None:
        df["Company"] = "All Companies"
    elif company_col != "Company":
        df = df.rename(columns={company_col: "Company"})
    df["Company"] = df["Company"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan}).fillna("Unknown Company")
    return df


# Creates time-based fields from Hire_Date and Exit_Date, such as hire year, exit year, and exit month.
# It also estimates employment length from dates so attrition timing can be summarized later.
def add_time_columns(df):
    df = df.copy()
    if "Hire_Date" in df.columns:
        df["Hire_Year"] = df["Hire_Date"].dt.year
    if "Exit_Date" in df.columns:
        df["Exit_Year"] = df["Exit_Date"].dt.year
        df["Exit_Month"] = df["Exit_Date"].dt.to_period("M").astype(str).replace("NaT", np.nan)

    if "Hire_Date" in df.columns:
        date_sources = [df["Hire_Date"]]
        if "Exit_Date" in df.columns:
            date_sources.append(df["Exit_Date"])
        analysis_date = pd.concat(date_sources).max()
        if pd.isna(analysis_date):
            analysis_date = pd.Timestamp.today().normalize()
        end_date = df["Exit_Date"].where(df["Exit_Date"].notna(), analysis_date) if "Exit_Date" in df.columns else analysis_date
        df["Employment_Days_From_Dates"] = (end_date - df["Hire_Date"]).dt.days
        df.loc[df["Employment_Days_From_Dates"] < 0, "Employment_Days_From_Dates"] = np.nan
        df["Employment_Years_From_Dates"] = (df["Employment_Days_From_Dates"] / 365.25).round(2)
    return df


# Loads the CSV, removes unnecessary columns, cleans text values, and validates the Attrition target column.
# It also creates Employee_Key and Attrition_Flag so the dataset is ready for EDA and modeling.
def load_and_clean_data(input_file=None):
    if input_file is None:
        input_file = find_csv_file()
    df = pd.read_csv(input_file)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = df.columns.astype(str).str.strip()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    df = standardize_company_column(df)
    if "EmployeeNumber" in df.columns:
        df["Employee_Key"] = df["Company"].astype(str) + " - " + df["EmployeeNumber"].astype(str)
    else:
        df["Employee_Key"] = df["Company"].astype(str) + " - Row " + (df.index + 1).astype(str)

    if "Attrition" not in df.columns:
        raise ValueError("Dataset must contain an Attrition column with Yes/No values.")
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.title()
    df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    if df["Attrition_Flag"].isna().any():
        bad_values = df.loc[df["Attrition_Flag"].isna(), "Attrition"].drop_duplicates().tolist()
        raise ValueError(f"Attrition must only contain Yes/No values. Found: {bad_values}")

    for date_col in ["Hire_Date", "Exit_Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return add_time_columns(df)

# =========================================================
# 3. GROUP NUMERIC COLUMNS
# =========================================================
# This section converts continuous numeric fields into readable groups for easier EDA and interpretation.
# Missing or invalid values are kept as "Unknown" so rows are not accidentally dropped from summaries.

# Converts a numeric column into readable groups while safely handling missing or invalid values.
# Values that cannot be grouped are labeled as "Unknown" instead of breaking the analysis.
def safe_group(series, bins, labels):
    values = pd.to_numeric(series, errors="coerce")
    grouped = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return grouped.astype("object").where(grouped.notna(), "Unknown")


# Creates grouped versions of numeric HR fields like income, tenure, distance, and working years.
# These groups make the EDA tables easier to explain than using raw numeric values alone.
def add_grouped_columns(df):
    df = df.copy()
    if "Age" in df.columns:
        df["Age_Group"] = safe_group(df["Age"], [0, 25, 30, 40, 50, np.inf], ["18-25", "26-30", "31-40", "41-50", "51+"])
    if "MonthlyIncome" in df.columns:
        df["Income_Group"] = safe_group(df["MonthlyIncome"], [-1, 3000, 6000, 10000, np.inf], ["<=3000", "3001-6000", "6001-10000", "10001+"])
    if "YearsAtCompany" in df.columns:
        df["Tenure_Group"] = safe_group(df["YearsAtCompany"], [-1, 1, 3, 5, 10, np.inf], ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
    if "DistanceFromHome" in df.columns:
        df["Distance_Group"] = safe_group(df["DistanceFromHome"], [-1, 5, 10, 20, np.inf], ["0-5", "6-10", "11-20", "21+"])
    if "TotalWorkingYears" in df.columns:
        df["Working_Years_Group"] = safe_group(df["TotalWorkingYears"], [-1, 5, 10, 20, np.inf], ["0-5 yrs", "6-10 yrs", "11-20 yrs", "21+ yrs"])
    if "YearsWithCurrManager" in df.columns:
        df["Manager_Tenure_Group"] = safe_group(df["YearsWithCurrManager"], [-1, 1, 3, 7, np.inf], ["0-1 yrs", "2-3 yrs", "4-7 yrs", "8+ yrs"])
    if "NumCompaniesWorked" in df.columns:
        df["Companies_Worked_Group"] = safe_group(df["NumCompaniesWorked"], [-1, 0, 2, 5, np.inf], ["0", "1-2", "3-5", "6+"])
    return df

# =========================================================
# 4. EDA SUMMARY TABLES
# =========================================================
# This section builds the descriptive analysis tables used for KPIs, company comparison, group summaries, reasons, and time patterns.
# It compares each group against the overall baseline using both rate difference and baseline ratio.

# Builds the basic KPI summary for the whole dataset.
# It counts total, active, and resigned employees, then calculates the overall attrition rate.
def make_key_findings(df):
    total = len(df)
    resigned = int(df["Attrition_Flag"].sum())
    active = total - resigned
    rate = df["Attrition_Flag"].mean()
    key_findings = pd.DataFrame({
        "Metric": ["Total employees", "Active employees", "Resigned employees", "Overall attrition rate"],
        "Result": [total, active, resigned, f"{rate:.1%}"]
    })
    target_distribution = df["Attrition"].value_counts().reset_index()
    target_distribution.columns = ["Attrition", "Employees"]
    target_distribution["Percent"] = (target_distribution["Employees"] / total).round(4)
    return key_findings, target_distribution, rate


# Creates an attrition summary for one selected column or factor.
# It compares each group against the baseline using both percentage-point difference and ratio.
def attrition_table(df, column_name, baseline_rate):
    table = df.groupby(column_name, observed=True, dropna=False).agg(
        Employees=("Attrition_Flag", "count"),
        Resigned=("Attrition_Flag", "sum"),
        Attrition_Rate=("Attrition_Flag", "mean")
    ).reset_index()
    table["Stayed"] = table["Employees"] - table["Resigned"]
    table["Baseline_Rate"] = baseline_rate
    table["Difference_From_Baseline"] = table["Attrition_Rate"] - baseline_rate
    table["Compared_to_Baseline"] = np.where(baseline_rate > 0, table["Attrition_Rate"] / baseline_rate, np.nan)
    table[column_name] = table[column_name].astype(str).replace({"nan": "Unknown", "NaT": "Unknown"})
    for col in ["Attrition_Rate", "Baseline_Rate", "Difference_From_Baseline", "Compared_to_Baseline"]:
        table[col] = table[col].round(4)
    return table[[column_name, "Employees", "Stayed", "Resigned", "Attrition_Rate", "Baseline_Rate", "Difference_From_Baseline", "Compared_to_Baseline"]].sort_values(["Attrition_Rate", "Employees"], ascending=[False, False])


# Summarizes attrition per company for company-level comparison.
# This is kept for EDA only and is not used by the model to avoid company-based prediction bias.
def make_company_overview(df):
    overall_rate = df["Attrition_Flag"].mean()
    table = df.groupby("Company", observed=True).agg(
        Employees=("Attrition_Flag", "count"),
        Active_Employees=("Attrition_Flag", lambda s: int((s == 0).sum())),
        Resigned_Employees=("Attrition_Flag", "sum"),
        Attrition_Rate=("Attrition_Flag", "mean")
    ).reset_index()
    table["Resigned_Employees"] = table["Resigned_Employees"].astype(int)
    table["Overall_Baseline_Rate"] = overall_rate
    table["Difference_From_Overall_Baseline"] = table["Attrition_Rate"] - overall_rate
    table["Compared_to_Overall_Baseline"] = np.where(overall_rate > 0, table["Attrition_Rate"] / overall_rate, np.nan)
    for col in ["Attrition_Rate", "Overall_Baseline_Rate", "Difference_From_Overall_Baseline", "Compared_to_Overall_Baseline"]:
        table[col] = table[col].round(4)
    return table.sort_values("Attrition_Rate", ascending=False)


# Loops through the selected EDA columns and combines their attrition tables into one long summary.
# This gives Streamlit one clean table that can be filtered by factor and group.
def make_eda_summary(df, baseline_rate, include_company=True):
    frames = []
    cols = EDA_COLUMNS if include_company else [col for col in EDA_COLUMNS if col != "Company"]
    for col in cols:
        if col in df.columns:
            table = attrition_table(df, col, baseline_rate).rename(columns={col: "Group"})
            table.insert(0, "EDA_Factor", col)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Creates EDA summaries separately for each company.
# This helps compare patterns within companies instead of only looking at the overall dataset.
def make_company_eda_summary(df):
    frames = []
    for company in sorted(df["Company"].astype(str).unique()):
        company_df = df[df["Company"].astype(str) == company]
        table = make_eda_summary(company_df, company_df["Attrition_Flag"].mean(), include_company=False)
        if len(table) > 0:
            table.insert(0, "Company", company)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Summarizes the listed reasons for attrition, but only for employees who already resigned.
# If the dataset has no Attrition_Reason column, it safely returns a blank table.
def make_reason_summary(df):
    if "Attrition_Reason" not in df.columns:
        return pd.DataFrame()
    resigned = df[df["Attrition"] == "Yes"]
    if len(resigned) == 0:
        return pd.DataFrame()
    table = resigned.groupby("Attrition_Reason", dropna=False).agg(Resigned_Employees=("Attrition_Flag", "count")).reset_index()
    table["Percent_of_Resigned"] = (table["Resigned_Employees"] / len(resigned)).round(4)
    return table.sort_values("Resigned_Employees", ascending=False)



# Creates time-based attrition summaries using hire year, exit year, exit month, and tenure at exit.
# This helps show whether attrition is connected to employee cohorts or resignation timing.
def make_time_summary(df):
    frames = []
    if "Hire_Year" in df.columns:
        hired = df.groupby("Hire_Year", dropna=False).agg(
            Employees=("Attrition_Flag", "count"),
            Resigned=("Attrition_Flag", "sum"),
            Attrition_Rate=("Attrition_Flag", "mean")
        ).reset_index().rename(columns={"Hire_Year": "Time_Group"})
        hired.insert(0, "Time_View", "Hire year/cohort")
        hired["Attrition_Rate"] = hired["Attrition_Rate"].round(4)
        frames.append(hired)

    resigned = df[df["Attrition"] == "Yes"].copy()
    if len(resigned) > 0 and "Exit_Year" in resigned.columns:
        exited_year = resigned.groupby("Exit_Year", dropna=False).agg(
            Employees_Exited=("Attrition_Flag", "count")
        ).reset_index().rename(columns={"Exit_Year": "Time_Group"})
        exited_year.insert(0, "Time_View", "Exit year")
        frames.append(exited_year)

    if len(resigned) > 0 and "Exit_Month" in resigned.columns:
        exited_month = resigned.groupby("Exit_Month", dropna=False).agg(
            Employees_Exited=("Attrition_Flag", "count")
        ).reset_index().rename(columns={"Exit_Month": "Time_Group"})
        exited_month.insert(0, "Time_View", "Exit month")
        frames.append(exited_month)

    tenure_source = "Employment_Years_From_Dates" if "Employment_Years_From_Dates" in resigned.columns else "YearsAtCompany" if "YearsAtCompany" in resigned.columns else None
    if len(resigned) > 0 and tenure_source is not None:
        resigned["Tenure_At_Exit_Group"] = safe_group(resigned[tenure_source], [-1, 1, 3, 5, 10, np.inf], ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
        tenure = resigned.groupby("Tenure_At_Exit_Group", dropna=False).agg(
            Employees_Exited=("Attrition_Flag", "count")
        ).reset_index().rename(columns={"Tenure_At_Exit_Group": "Time_Group"})
        tenure.insert(0, "Time_View", "Tenure at exit")
        frames.append(tenure)

    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary["Time_Group"] = summary["Time_Group"].where(summary["Time_Group"].notna(), "Unknown")
    summary["Time_Group"] = summary["Time_Group"].map(lambda value: str(int(value)) if isinstance(value, float) and value.is_integer() else str(value))
    for col in ["Employees", "Resigned", "Employees_Exited"]:
        if col in summary.columns:
            summary[col] = summary[col].astype("Int64")
    return summary

# =========================================================
# 5. HIGH-ATTRITION PATTERNS AND REVIEW SIGNALS
# =========================================================
# This section finds groups with meaningfully higher attrition than the baseline and converts them into matched patterns.
# It also creates employee-level actionable risk factors, positive signals, and adjusted review scores for later prioritization.

# Tries to convert a value into a number for rule checks.
# If conversion fails, it returns NaN so the code can continue safely.
def to_number(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# Placeholder function kept for flexibility.
# It currently does not filter out any high-attrition pattern so the pattern search stays data-driven.
def is_positive_or_unclear_pattern(factor, group):
    return False


# Finds groups with attrition meaningfully higher than the overall baseline.
# It filters out tiny groups, weak differences, and low resignation counts to avoid noisy findings.
def make_high_attrition_patterns(df, baseline_rate, max_rows=20):
    rows = []
    min_group_size = max(10, int(len(df) * 0.02))
    min_resigned = max(3, int(df["Attrition_Flag"].sum() * 0.01))
    for factor in PATTERN_COLUMNS:
        if factor not in df.columns:
            continue
        table = attrition_table(df, factor, baseline_rate)
        for _, item in table.iterrows():
            group = item[factor]
            employees = int(item["Employees"])
            resigned = int(item["Resigned"])
            ratio = float(item["Compared_to_Baseline"])
            diff = float(item["Difference_From_Baseline"])
            if employees < min_group_size or resigned < min_resigned or diff < 0.03 or ratio < 1.20:
                continue
            rows.append({
                "Factor": factor,
                "Group": str(group),
                "High_Attrition_Group": f"{factor} = {group}",
                "Employees": employees,
                "Stayed": int(item["Stayed"]),
                "Resigned": resigned,
                "Attrition_Rate": float(item["Attrition_Rate"]),
                "Difference_From_Baseline": diff,
                "Compared_to_Baseline": ratio
            })
    patterns = pd.DataFrame(rows)
    if len(patterns) == 0:
        return pd.DataFrame(columns=["Factor", "Group", "High_Attrition_Group", "Employees", "Stayed", "Resigned", "Attrition_Rate", "Difference_From_Baseline", "Compared_to_Baseline"])
    return patterns.sort_values(["Compared_to_Baseline", "Resigned"], ascending=False).head(max_rows).reset_index(drop=True)


# Creates simple HR-review flags for each employee based on actionable conditions.
# Examples include overtime, frequent travel, far distance, low income band, or low satisfaction scores.
def make_actionable_flags(row):
    flags = []
    if "OverTime" in row.index and str(row["OverTime"]).lower() == "yes":
        flags.append("Overtime")
    if "BusinessTravel" in row.index and str(row["BusinessTravel"]) == "Travel_Frequently":
        flags.append("Frequent travel")
    if "Distance_Group" in row.index and str(row["Distance_Group"]) == "21+":
        flags.append("Far from home")
    if "MonthlyIncome" in row.index and to_number(row["MonthlyIncome"]) <= 3000:
        flags.append("Low income band")
    if "StockOptionLevel" in row.index and to_number(row["StockOptionLevel"]) == 0:
        flags.append("No stock option")
    for col in SATISFACTION_COLUMNS:
        if col in row.index and to_number(row[col]) <= 2:
            flags.append(f"Low {col}")
    return list(dict.fromkeys(flags))


# Identifies positive or protective signals for each employee.
# These are used to balance the review score so employees are not flagged based only on risks.
def make_positive_signals(row):
    signals = []
    if "OverTime" in row.index and str(row["OverTime"]).lower() == "no":
        signals.append("No overtime")
    if "BusinessTravel" in row.index and str(row["BusinessTravel"]) in ["Non-Travel", "Travel_Rarely"]:
        signals.append("Limited travel")
    for col in SATISFACTION_COLUMNS:
        if col in row.index and to_number(row[col]) >= 3:
            signals.append(f"{col} okay/high")
    return list(dict.fromkeys(signals))


# Checks whether an employee belongs to any of the high-attrition groups found earlier.
# It returns only a few matched labels so the review table stays readable.
def matched_patterns(row, patterns, max_labels=4):
    labels = []
    for _, pattern in patterns.iterrows():
        factor = pattern["Factor"]
        group = pattern["Group"]
        if factor in row.index and str(row[factor]) == str(group):
            labels.append(pattern["High_Attrition_Group"])
        if len(labels) >= max_labels:
            break
    return list(dict.fromkeys(labels))


# Adds review-related columns to the cleaned dataset.
# It combines matched patterns, actionable risk flags, and positive signals into review scores.
def add_review_signals(df, patterns):
    df = df.copy()
    review_scores, adjusted_scores, matched_pattern_text, action_flags, positives = [], [], [], [], []
    action_counts, positive_counts = [], []
    for _, row in df.iterrows():
        pattern_labels = matched_patterns(row, patterns)
        flags = make_actionable_flags(row)
        pos = make_positive_signals(row)
        score = min((len(flags) * 2) + len(pattern_labels), 10)
        review_scores.append(score)
        adjusted_scores.append(max(score - len(pos), 0))
        matched_pattern_text.append("; ".join(pattern_labels))
        action_flags.append("; ".join(flags))
        positives.append("; ".join(pos))
        action_counts.append(len(flags))
        positive_counts.append(len(pos))
    df["Review_Score"] = review_scores
    df["Adjusted_Review_Score"] = adjusted_scores
    df["Matched_High_Attrition_Patterns"] = matched_pattern_text
    df["Actionable_Risk_Factors"] = action_flags
    df["Positive_Signals"] = positives
    df["Actionable_Risk_Count"] = action_counts
    df["Positive_Signal_Count"] = positive_counts
    return df

# =========================================================
# 6. MODEL COMPARISON
# =========================================================
# This section prepares safe model inputs, removes leakage/sensitive fields, and compares Logistic Regression, Decision Tree, and Random Forest.
# The selected model is used to create out-of-sample risk probabilities and feature importance for Streamlit.

# Prepares the model features and target variable.
# It removes leakage, sensitive fields, company identity, and already-created review score columns before modeling.
def prepare_model_inputs(df):
    exclude_cols = [
        "Attrition", "Attrition_Flag", "Attrition_Reason", "Company", "Exit_Date", "Exit_Year", "Exit_Month",
        "Hire_Date", "EmployeeNumber", "Employee_Key", "EmployeeCount", "StandardHours", "Over18",
        "Review_Score", "Adjusted_Review_Score", "Matched_High_Attrition_Patterns", "Actionable_Risk_Factors",
        "Positive_Signals", "Actionable_Risk_Count", "Positive_Signal_Count", "Risk_Score_Type",
        "Employment_Days_From_Dates", "Employment_Years_From_Dates"
    ] + SENSITIVE_COLUMNS
    y = df["Attrition_Flag"].astype(int)
    x = df[[col for col in df.columns if col not in exclude_cols]].copy()
    for col in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[col]):
            x[col] = x[col].map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    for col in x.select_dtypes(include=["object", "string", "category"]).columns:
        x[col] = x[col].astype("object").fillna("Unknown")
    for col in x.select_dtypes(include=[np.number]).columns:
        x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(x[col].median())
    return pd.get_dummies(x, drop_first=True).fillna(0), y


# Creates out-of-sample attrition probabilities using stratified cross-validation.
# Each employee is scored by a model that did not directly train on that employee, making scores less inflated.
def make_out_of_sample_probabilities(model, needs_scale, x, y):
    n_splits = min(5, int(y.value_counts().min()))
    if n_splits < 2:
        return pd.Series(np.nan, index=x.index)
    probabilities = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, holdout_idx in cv.split(x, y):
        fold_model = clone(model)
        if needs_scale:
            fold_scaler = StandardScaler()
            train_x = fold_scaler.fit_transform(x.iloc[train_idx])
            holdout_x = fold_scaler.transform(x.iloc[holdout_idx])
        else:
            train_x = x.iloc[train_idx]
            holdout_x = x.iloc[holdout_idx]
        fold_model.fit(train_x, y.iloc[train_idx])
        probabilities[holdout_idx] = fold_model.predict_proba(holdout_x)[:, 1]
    return pd.Series(probabilities, index=x.index)


# Trains and compares Logistic Regression, Decision Tree, and Random Forest models.
# It selects the best model based on F1, recall, and ROC AUC, then adds risk scores and feature importance.
def train_models(df):
    df = df.copy()
    x, y = prepare_model_inputs(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_all_scaled = scaler.transform(x)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=1), False)
    }
    rows, fitted = [], {}
    for name, (model, needs_scale) in models.items():
        train_x = x_train_scaled if needs_scale else x_train
        test_x = x_test_scaled if needs_scale else x_test
        model.fit(train_x, y_train)
        pred = model.predict(test_x)
        proba = model.predict_proba(test_x)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, proba),
            "True Negative": tn,
            "False Positive": fp,
            "False Negative": fn,
            "True Positive": tp
        })
        fitted[name] = (model, needs_scale)

    model_results = pd.DataFrame(rows).sort_values(["F1 Score", "Recall", "ROC AUC"], ascending=False).reset_index(drop=True)
    model_results.insert(0, "Rank", range(1, len(model_results) + 1))
    best_name = model_results.loc[0, "Model"]
    model_results["Selected_for_App"] = np.where(model_results["Model"] == best_name, "Yes", "No")
    model_results[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]] = model_results[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].round(4)

    best_model, best_needs_scale = fitted[best_name]
    if best_needs_scale:
        best_pred = best_model.predict(x_test_scaled)
    else:
        best_pred = best_model.predict(x_test)
    df["Predicted_Attrition_Risk"] = make_out_of_sample_probabilities(best_model, best_needs_scale, x, y).values
    df["Risk_Score_Type"] = "Out-of-sample cross-validation probability"
    df["Selected_Model"] = best_name

    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    confusion_table = pd.DataFrame(cm, index=["Actual Stayed", "Actual Resigned"], columns=["Predicted Stayed", "Predicted Resigned"]).reset_index().rename(columns={"index": "Confusion Matrix"})
    confusion_table.insert(0, "Selected Model", best_name)

    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_[0])
    else:
        importance = np.zeros(x.shape[1])
    feature_importance = pd.DataFrame({"Feature": x.columns, "Importance": importance}).sort_values("Importance", ascending=False).head(20)
    feature_importance["Importance"] = feature_importance["Importance"].round(4)
    return df, model_results, confusion_table, feature_importance

# =========================================================
# 7. EMPLOYEES TO REVIEW
# =========================================================
# This section creates the active employee review list using model probability plus actionable signals.
# The output is meant as a validation shortlist for HR/context review, not as a final decision tool.

# Builds the active employee review list for HR or manager validation.
# Priority is based on model probability plus actionable signals, not model score alone.
def make_employees_to_review(df):
    active = df[df["Attrition"] == "No"].copy()
    if len(active) == 0:
        return pd.DataFrame()
    high_cutoff = active["Predicted_Attrition_Risk"].quantile(0.90)
    mid_cutoff = active["Predicted_Attrition_Risk"].quantile(0.75)
    has_risk_signal = active["Actionable_Risk_Count"] >= 1
    active["Probability_Level"] = np.where(active["Predicted_Attrition_Risk"] >= high_cutoff, "High model probability", np.where(active["Predicted_Attrition_Risk"] >= mid_cutoff, "Medium model probability", "Lower model probability"))
    active["Review_Priority"] = np.where(
        (active["Predicted_Attrition_Risk"] >= high_cutoff) & (active["Actionable_Risk_Count"] >= 2) & (active["Adjusted_Review_Score"] >= 3),
        "High",
        np.where(has_risk_signal & ((active["Predicted_Attrition_Risk"] >= mid_cutoff) | (active["Adjusted_Review_Score"] >= 2)), "Medium", "Low")
    )
    active["Review_Recommendation"] = np.where(
        active["Review_Priority"] == "High",
        "Review first: high model probability is supported by multiple actionable signals.",
        np.where(active["Review_Priority"] == "Medium", "Monitor only after validating with manager or HR context.", "No immediate review needed from current signals.")
    )
    cols = [
        "Company", "Employee_Key", "EmployeeNumber", "Department", "JobRole", "MonthlyIncome",
        "OverTime", "BusinessTravel", "DistanceFromHome", "JobSatisfaction", "EnvironmentSatisfaction",
        "WorkLifeBalance", "YearsAtCompany", "Employment_Years_From_Dates", "Review_Score", "Adjusted_Review_Score",
        "Actionable_Risk_Count", "Positive_Signal_Count", "Matched_High_Attrition_Patterns", "Actionable_Risk_Factors",
        "Positive_Signals", "Predicted_Attrition_Risk", "Risk_Score_Type", "Selected_Model", "Probability_Level",
        "Review_Priority", "Review_Recommendation"
    ]
    cols = [col for col in cols if col in active.columns]
    review = active[cols].copy()
    review["Predicted_Attrition_Risk"] = review["Predicted_Attrition_Risk"].round(4)
    review["Priority_Order"] = review["Review_Priority"].map({"High": 1, "Medium": 2, "Low": 3})
    return review.sort_values(["Priority_Order", "Actionable_Risk_Count", "Predicted_Attrition_Risk", "Adjusted_Review_Score"], ascending=[True, False, False, False]).drop(columns="Priority_Order")


# Creates the final KPI card summary for Streamlit.
# It includes employee counts, attrition rate, selected model, model metrics, and review priority counts.
def make_kpi_summary(df, employees_to_review, model_results):
    selected = model_results[model_results["Selected_for_App"] == "Yes"].iloc[0]
    priority_counts = employees_to_review["Review_Priority"].value_counts() if len(employees_to_review) else pd.Series(dtype=int)
    return pd.DataFrame([
        {"KPI": "Total Employees", "Value": len(df)},
        {"KPI": "Active Employees", "Value": int((df["Attrition"] == "No").sum())},
        {"KPI": "Resigned Employees", "Value": int(df["Attrition_Flag"].sum())},
        {"KPI": "Overall Attrition Rate", "Value": f"{df['Attrition_Flag'].mean():.1%}"},
        {"KPI": "Selected Model", "Value": selected["Model"]},
        {"KPI": "Selected Model F1 Score", "Value": selected["F1 Score"]},
        {"KPI": "Selected Model Recall", "Value": selected["Recall"]},
        {"KPI": "High Priority Active Employees", "Value": int(priority_counts.get("High", 0))},
        {"KPI": "Medium Priority Active Employees", "Value": int(priority_counts.get("Medium", 0))},
        {"KPI": "Low Priority Active Employees", "Value": int(priority_counts.get("Low", 0))}
    ])

# =========================================================
# 8. MAIN FUNCTION FOR STREAMLIT
# =========================================================
# This section runs the full workflow from cleaning to modeling and returns all tables in one results dictionary.
# Streamlit can import run_full_analysis() and display these returned DataFrames directly.

# Runs the full workflow from data loading to final Streamlit-ready outputs.
# This is the main function the Streamlit app should call to get all tables and chart-ready data.
def run_full_analysis(input_file=None):
    df = load_and_clean_data(input_file)
    df = add_grouped_columns(df)
    key_findings, target_distribution, overall_rate = make_key_findings(df)
    company_overview = make_company_overview(df)
    eda_summary = make_eda_summary(df, overall_rate, include_company=True)
    company_eda_summary = make_company_eda_summary(df)
    reason_summary = make_reason_summary(df)
    time_summary = make_time_summary(df)
    high_attrition_patterns = make_high_attrition_patterns(df, overall_rate)
    df = add_review_signals(df, high_attrition_patterns)
    df, model_results, confusion_matrix_table, feature_importance = train_models(df)
    employees_to_review = make_employees_to_review(df)
    kpi_summary = make_kpi_summary(df, employees_to_review, model_results)
    priority_counts = employees_to_review["Review_Priority"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int).reset_index()
    priority_counts.columns = ["Review_Priority", "Employees"]
    return {
        "cleaned_data": df,
        "key_findings": key_findings,
        "target_distribution": target_distribution,
        "kpi_summary": kpi_summary,
        "company_overview": company_overview,
        "eda_summary": eda_summary,
        "company_eda_summary": company_eda_summary,
        "reason_summary": reason_summary,
        "time_summary": time_summary,
        "high_attrition_patterns": high_attrition_patterns,
        "model_results": model_results,
        "confusion_matrix_table": confusion_matrix_table,
        "feature_importance": feature_importance,
        "employees_to_review": employees_to_review,
        "chart_ready_data": {
            "target_distribution": target_distribution,
            "company_overview": company_overview,
            "priority_counts": priority_counts,
            "time_summary": time_summary,
            "model_comparison": model_results,
            "top_model_factors": feature_importance.head(10),
            "employees_to_review": employees_to_review
        }
    }

# =========================================================
# 9. TERMINAL HANDOFF SUMMARY
# =========================================================
# This section prints a compact summary of the available outputs for the Streamlit teammate.
# It helps confirm the script ran correctly and shows which result keys can be used in the app.

# Prints a compact terminal preview of the main results.
# This helps the Streamlit developer quickly see available outputs without opening every DataFrame manually.
def print_streamlit_handoff_summary(results, top_n=10):
    print("\n" + "=" * 80)
    print("STREAMLIT HANDOFF SUMMARY")
    print("=" * 80)
    print("\nImport in Streamlit:")
    print("from hr_attrition_eda_streamlit_group_project import run_full_analysis")
    print("results = run_full_analysis(uploaded_file_or_csv_path)")
    print("\nMain result keys for Streamlit:")
    for key in ["kpi_summary", "target_distribution", "company_overview", "eda_summary", "company_eda_summary", "time_summary", "high_attrition_patterns", "model_results", "confusion_matrix_table", "feature_importance", "employees_to_review", "chart_ready_data"]:
        value = results.get(key)
        if isinstance(value, pd.DataFrame):
            print(f"- results['{key}'] -> DataFrame shape {value.shape}")
        elif isinstance(value, dict):
            print(f"- results['{key}'] -> dictionary keys {list(value.keys())}")
    print("\nKPI summary:")
    print(results["kpi_summary"].to_string(index=False))
    print("\nCompany overview:")
    print(results["company_overview"].to_string(index=False))
    print("\nModel comparison:")
    model_cols = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Selected_for_App"]
    print(results["model_results"][model_cols].to_string(index=False))
    print(f"\nTop {top_n} employees to review preview:")
    preview_cols = ["Company", "Employee_Key", "EmployeeNumber", "Department", "JobRole", "Review_Priority", "Predicted_Attrition_Risk", "Review_Score", "Actionable_Risk_Factors", "Matched_High_Attrition_Patterns", "Review_Recommendation"]
    preview_cols = [col for col in preview_cols if col in results["employees_to_review"].columns]
    print(results["employees_to_review"][preview_cols].head(top_n).to_string(index=False))
    print("\nSuggested Streamlit sections:")
    print("1. KPI cards: results['kpi_summary']")
    print("2. Company comparison: results['company_overview']")
    print("3. EDA table with filters: results['eda_summary'] or results['company_eda_summary']")
    print("4. Model comparison: results['model_results']")
    print("5. Feature importance chart: results['feature_importance']")
    print("6. Employees to review table: results['employees_to_review']")
    print("\nNo Excel file is created.")

# =========================================================
# 10. RUN THE SCRIPT NORMALLY
# =========================================================
# This section only runs when the file is executed directly from the terminal.
# It calls the full analysis and prints the Streamlit handoff summary.

if __name__ == "__main__":
    results = run_full_analysis()
    print_streamlit_handoff_summary(results, top_n=10)
