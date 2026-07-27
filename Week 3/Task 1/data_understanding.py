from pathlib import Path
import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

RAW = BASE / "01_Raw_Data"
INSPECTION = BASE / "02_Inspection_Report"
QUALITY = BASE / "03_Data_Quality"
DICTIONARY = BASE / "04_Data_Dictionary"
SCREENSHOTS = BASE / "05_Screenshots"
FINAL = BASE / "06_Final_Report"

for folder in [RAW, INSPECTION, QUALITY, DICTIONARY, SCREENSHOTS, FINAL]:
    folder.mkdir(exist_ok=True)

FILE = RAW / "marketing_campaign.csv"


def save_excel(dataframe, folder, filename):
    dataframe.to_excel(folder / filename, index=False)


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(FILE, sep=None, engine="python", encoding="utf-8")
    encoding = "UTF-8"
except UnicodeDecodeError:
    df = pd.read_csv(FILE, sep=None, engine="python", encoding="latin-1")
    encoding = "Latin-1"

print("Dataset loaded successfully.")
print(df.head())


# Keep original date values for quality checking
original_dates = df["Dt_Customer"].copy()

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y",
    errors="coerce"
)


# ============================================================
# TASK 3: BASIC DATASET INSPECTION
# ============================================================

basic_summary = pd.DataFrame({
    "Property": [
        "Rows",
        "Columns",
        "Dataset Size (KB)",
        "File Type",
        "Encoding"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        round(os.path.getsize(FILE) / 1024, 2),
        FILE.suffix.replace(".", "").upper(),
        encoding
    ]
})

save_excel(
    basic_summary,
    INSPECTION,
    "basic_dataset_summary.xlsx"
)

print("\nBasic Summary")
print(basic_summary)


# ============================================================
# TASK 4 AND TASK 12: VARIABLE INSPECTION / DATA DICTIONARY
# ============================================================

descriptions = {
    "ID": "Unique customer identification number.",
    "Year_Birth": "Customer birth year.",
    "Education": "Customer education level.",
    "Marital_Status": "Customer marital status.",
    "Income": "Annual household income.",
    "Kidhome": "Number of children in the household.",
    "Teenhome": "Number of teenagers in the household.",
    "Dt_Customer": "Date customer joined the company.",
    "Recency": "Days since the last purchase.",
    "MntWines": "Amount spent on wine.",
    "MntFruits": "Amount spent on fruits.",
    "MntMeatProducts": "Amount spent on meat.",
    "MntFishProducts": "Amount spent on fish.",
    "MntSweetProducts": "Amount spent on sweets.",
    "MntGoldProds": "Amount spent on gold products.",
    "NumDealsPurchases": "Purchases made using discounts.",
    "NumWebPurchases": "Purchases made through the website.",
    "NumCatalogPurchases": "Purchases made through catalogues.",
    "NumStorePurchases": "Purchases made in stores.",
    "NumWebVisitsMonth": "Website visits during the month.",
    "AcceptedCmp1": "Accepted marketing campaign 1.",
    "AcceptedCmp2": "Accepted marketing campaign 2.",
    "AcceptedCmp3": "Accepted marketing campaign 3.",
    "AcceptedCmp4": "Accepted marketing campaign 4.",
    "AcceptedCmp5": "Accepted marketing campaign 5.",
    "Complain": "Customer made a complaint.",
    "Z_CostContact": "Constant contact-cost variable.",
    "Z_Revenue": "Constant revenue variable.",
    "Response": "Accepted the latest campaign."
}

business_meaning = {
    "ID": "Customer identification",
    "Year_Birth": "Customer age analysis",
    "Education": "Education-based segmentation",
    "Marital_Status": "Household segmentation",
    "Income": "Purchasing capacity",
    "Dt_Customer": "Customer relationship duration",
    "Recency": "Recent customer activity",
    "Complain": "Customer dissatisfaction",
    "Response": "Latest campaign performance"
}

variable_report = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": [str(df[column].dtype) for column in df.columns],
    "Description": [
        descriptions.get(column, "Customer purchasing or campaign variable.")
        for column in df.columns
    ]
})

save_excel(
    variable_report,
    INSPECTION,
    "variable_inspection.xlsx"
)

data_dictionary = variable_report.copy()

data_dictionary["Business Meaning"] = [
    business_meaning.get(
        column,
        descriptions.get(column, "Supports customer analysis.")
    )
    for column in df.columns
]

save_excel(
    data_dictionary,
    DICTIONARY,
    "data_dictionary.xlsx"
)


# ============================================================
# TASK 5: MISSING VALUES
# ============================================================

missing_report = pd.DataFrame({
    "Variable": df.columns,
    "Missing Values": df.isna().sum().values,
    "Percentage": (
        df.isna().sum().values / len(df) * 100
    ).round(2)
})

save_excel(
    missing_report,
    QUALITY,
    "missing_values.xlsx"
)

columns_without_missing = missing_report[
    missing_report["Missing Values"] == 0
]["Variable"].tolist()


# ============================================================
# TASK 6: DUPLICATE RECORDS
# ============================================================

duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["ID"].duplicated().sum())

duplicate_report = pd.DataFrame({
    "Check": [
        "Duplicate Rows",
        "Duplicate Customer IDs"
    ],
    "Result": [
        duplicate_rows,
        duplicate_ids
    ]
})

save_excel(
    duplicate_report,
    QUALITY,
    "duplicate_records.xlsx"
)


# ============================================================
# TASK 7 AND TASK 13: DATA QUALITY REPORT
# ============================================================

issues = []


def add_issue(issue, variable, description, recommendation):
    issues.append({
        "Issue": issue,
        "Variable": variable,
        "Description": description,
        "Recommendation": recommendation
    })


invalid_births = int(
    ((df["Year_Birth"] < 1900) | (df["Year_Birth"] > 2014)).sum()
)

if invalid_births:
    add_issue(
        "Invalid age",
        "Year_Birth",
        f"{invalid_births} invalid birth years found.",
        "Verify or remove impossible birth years."
    )


negative_income = int((df["Income"] < 0).sum())

if negative_income:
    add_issue(
        "Negative income",
        "Income",
        f"{negative_income} negative income values found.",
        "Verify and correct invalid income values."
    )


income_missing = int(df["Income"].isna().sum())

if income_missing:
    add_issue(
        "Missing values",
        "Income",
        f"{income_missing} income values are missing.",
        "Use median imputation during preprocessing."
    )


invalid_dates = int(
    (original_dates.notna() & df["Dt_Customer"].isna()).sum()
)

if invalid_dates:
    add_issue(
        "Incorrect dates",
        "Dt_Customer",
        f"{invalid_dates} invalid dates found.",
        "Correct or remove invalid dates."
    )


for column in ["Education", "Marital_Status"]:
    empty_count = int(
        df[column].fillna("").astype(str).str.strip().eq("").sum()
    )

    if empty_count:
        add_issue(
            "Empty categories",
            column,
            f"{empty_count} empty values found.",
            "Replace empty categories with missing values."
        )


unexpected_marital = [
    value for value in ["Alone", "Absurd", "YOLO"]
    if value in df["Marital_Status"].unique()
]

if unexpected_marital:
    add_issue(
        "Unexpected categories",
        "Marital_Status",
        "Unexpected values: " + ", ".join(unexpected_marital),
        "Review and combine categories where appropriate."
    )


if "2n Cycle" in df["Education"].unique():
    add_issue(
        "Unclear category",
        "Education",
        "The value '2n Cycle' may be unclear.",
        "Confirm and document its meaning."
    )


for column in ["Z_CostContact", "Z_Revenue"]:
    if df[column].nunique() == 1:
        add_issue(
            "Constant variable",
            column,
            "The column contains only one value.",
            "Consider removing it during preprocessing."
        )


if duplicate_rows:
    add_issue(
        "Duplicate rows",
        "All columns",
        f"{duplicate_rows} duplicate rows found.",
        "Remove confirmed duplicate records."
    )


if duplicate_ids:
    add_issue(
        "Duplicate IDs",
        "ID",
        f"{duplicate_ids} duplicate customer IDs found.",
        "Ensure each customer has a unique ID."
    )


quality_report = pd.DataFrame(issues)

save_excel(
    quality_report,
    QUALITY,
    "data_quality_report.xlsx"
)


# ============================================================
# TASK 8: NUMERICAL VARIABLES
# ============================================================

numerical_columns = df.select_dtypes(include="number").columns

numerical_report = pd.DataFrame({
    "Variable": numerical_columns,
    "Minimum": [df[column].min() for column in numerical_columns],
    "Maximum": [df[column].max() for column in numerical_columns],
    "Mean": [
        round(df[column].mean(), 2)
        for column in numerical_columns
    ],
    "Observation": [
        "Review range and possible outliers."
        for column in numerical_columns
    ]
})

save_excel(
    numerical_report,
    INSPECTION,
    "numerical_variables.xlsx"
)


# ============================================================
# TASK 9: CATEGORICAL VARIABLES
# ============================================================

binary_columns = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Complain",
    "Response"
]

categorical_columns = [
    "Education",
    "Marital_Status"
] + binary_columns

categorical_report = pd.DataFrame({
    "Variable": categorical_columns,
    "Number of Categories": [
        df[column].nunique()
        for column in categorical_columns
    ],
    "Categories": [
        ", ".join(
            map(str, sorted(df[column].dropna().unique()))
        )
        for column in categorical_columns
    ]
})

save_excel(
    categorical_report,
    INSPECTION,
    "categorical_variables.xlsx"
)


# ============================================================
# TASK 10: BUSINESS UNDERSTANDING
# ============================================================

business_text = """
BUSINESS INDICATORS

1. High-Income Customers
Customers with high income have greater purchasing capacity and may respond
well to premium products and exclusive offers.

2. Frequent Buyers
Customers with many web, catalogue, or store purchases are frequent buyers.
They are important for customer retention and loyalty programs.

3. Campaign Responders
Customers who accepted previous campaigns may respond positively to future
marketing offers.

4. Customers with Complaints
Customers with complaints may be dissatisfied. Their issues should be handled
quickly to improve customer satisfaction.

5. Website Visitors
Website visits measure online engagement. Comparing visits with purchases can
help evaluate website conversion.

6. High-Spending Customers
Customers spending large amounts across product categories are valuable and
can be targeted with premium recommendations.

7. Recently Active Customers
Customers with low Recency values purchased recently and may be more likely
to respond to new promotions.
"""

with open(
    INSPECTION / "business_understanding.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(business_text)


# ============================================================
# TASK 11: INITIAL DATA SUMMARY
# ============================================================

summary = f"""
CUSTOMER PERSONALITY ANALYSIS – INITIAL DATA SUMMARY

Dataset Overview:
The dataset contains customer demographics, income, purchasing behavior,
campaign responses, complaints, and website engagement.

Number of Records: {df.shape[0]}
Number of Variables: {df.shape[1]}
Numerical Variables: {len(numerical_columns)}
Categorical Variables: {len(categorical_columns)}
Total Missing Values: {int(df.isna().sum().sum())}
Columns Without Missing Values: {len(columns_without_missing)}
Duplicate Records: {duplicate_rows}
Duplicate Customer IDs: {duplicate_ids}

Major Data Quality Issues:
The main issues include missing Income values, invalid birth years,
possible income outliers, unexpected marital-status categories,
an unclear education category, and constant variables.

Recommendation:
These issues should be reviewed and handled during the preprocessing stage.
No preprocessing has been performed in this task.
"""

with open(
    FINAL / "initial_data_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nTask completed successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", duplicate_rows)
print("Duplicate IDs:", duplicate_ids)
print("\nAll required reports have been saved.")