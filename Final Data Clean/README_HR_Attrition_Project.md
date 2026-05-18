# HR Attrition Analysis Project

## Project Overview

This project is a simple HR attrition analysis project made for our group work. The goal of the project is to understand employee attrition patterns using basic data cleaning, exploratory data analysis, and machine learning models.

The main idea is to look at the HR dataset, summarize important attrition trends, compare different models, and prepare the results so they can be displayed later in a Streamlit dashboard.

## What the Project Does

The Python file does the main backend work for the project. It:

- Loads and cleans the HR attrition dataset
- Standardizes company names and employee identifiers
- Converts the Attrition column into a numeric target variable
- Groups numeric columns into easier-to-read categories
- Creates summary tables for attrition patterns
- Compares attrition by company and employee-related factors
- Adds a basic time summary using hire and exit dates
- Finds high-attrition patterns compared to the overall baseline
- Compares machine learning models
- Creates out-of-sample attrition risk scores
- Prepares an employees-to-review table for Streamlit

## Main Analysis Included

The project includes several parts of analysis:

1. **Key Findings**  
   Shows total employees, active employees, resigned employees, and the overall attrition rate.

2. **Company Overview**  
   Compares attrition rates across companies while still keeping company identity out of the prediction model.

3. **EDA Summary**  
   Looks at attrition patterns across factors such as overtime, business travel, department, job role, satisfaction scores, income group, tenure group, distance group, and other work-related variables.

4. **Time Summary**  
   Looks at hire year, exit year, exit month, and tenure-at-exit patterns.

5. **High-Attrition Patterns**  
   Identifies groups with attrition rates that are noticeably higher than the overall baseline.

6. **Model Comparison**  
   Compares Logistic Regression, Decision Tree, and Random Forest using metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

7. **Employees to Review**  
   Creates a review list for active employees based on model probability and actionable risk signals. This is only meant for review and validation, not automatic decision-making.

## Models Used

The project compares three models:

- Logistic Regression
- Decision Tree
- Random Forest

The code selects the best model based mainly on F1 score, recall, and ROC AUC.

## Important Notes

Some columns were intentionally excluded from the model to make the analysis safer and less biased. These include sensitive or shortcut columns such as:

- Gender
- Marital Status
- Age
- Age Group
- Company
- Employee identifiers
- Attrition-related columns that would cause data leakage

Company is still used for reporting and EDA, but it is not used as a model input. This is because we wanted the model to focus more on employee and work-related patterns instead of simply learning that one company has higher attrition than another.

## How to Run the Project

Install the needed packages:

```bash
python -m pip install pandas numpy scikit-learn
```

Place the HR attrition CSV file in the same folder as the Python file.

Then run:

```bash
python hr_attrition_eda_streamlit_group_project.py
```

The file will print a short handoff summary in the terminal and return Streamlit-ready DataFrames through:

```python
run_full_analysis()
```

## Files

The main file for this project is:

```text
hr_attrition_eda_streamlit_group_project.py
```

This file contains the data cleaning, EDA, model comparison, and Streamlit-ready outputs.

## Project Limitation

This project is still a student-level prototype. The results should be treated as exploratory and should not be used as the only basis for real HR decisions.

The employees-to-review table is only a shortlist that would still need proper HR context, manager validation, and further checking.

## Group Use

This file is prepared so that the Streamlit developer can easily import the analysis results and display them in the dashboard without needing to redo the data cleaning or modeling process.
