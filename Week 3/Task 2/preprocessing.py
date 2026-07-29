from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

RAW = BASE / "01_Raw_Data"
QUALITY = BASE / "02_Data_Quality_Assessment"
CLEANED = BASE / "03_Cleaned_Data"
VIZ = BASE / "04_Visualizations"
REPORTS = BASE / "05_Reports"

for folder in [RAW, QUALITY, CLEANED, VIZ, REPORTS]:
    folder.mkdir(exist_ok=True)

RAW_FILE = RAW / "marketing_campaign.csv"
CLEANED_FILE = CLEANED / "customer_personality_cleaned.csv"

CURRENT_YEAR = 2026  # reference year used for age validation

NUMERIC_SPEND_COLS = [
    "Income", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]


def save_excel(dataframe, folder, filename):
    dataframe.to_excel(folder / filename, index=False)


# ============================================================
# ACTIVITY 1: LOAD AND INSPECT THE DATASET
# ============================================================

def load_data(path=RAW_FILE):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    return df


def inspect_data(df):
    print("=" * 60)
    print("ACTIVITY 1: LOAD AND INSPECT THE DATASET")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nLast 10 rows:")
    print(df.tail(10))
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe(include="all").T)

    overview = pd.DataFrame({
        "Property": ["Rows", "Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })
    save_excel(overview, QUALITY, "01_dataset_overview.xlsx")
    df.describe(include="all").T.to_excel(QUALITY / "01_summary_statistics.xlsx")
    return overview


# ============================================================
# ACTIVITY 2: IDENTIFY DATA QUALITY ISSUES
# ============================================================

def assess_data_quality(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 2: DATA QUALITY ASSESSMENT")
    print("=" * 60)

    issues = []

    def add(issue, column, description, severity):
        issues.append({
            "Issue": issue, "Column": column,
            "Description": description, "Severity": severity
        })

    missing_income = int(df["Income"].isna().sum())
    if missing_income:
        add("Missing values", "Income",
            f"{missing_income} missing income values "
            f"({round(missing_income/len(df)*100, 2)}%).", "Medium")

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        add("Duplicate records", "All columns",
            f"{dup_rows} fully duplicated rows.", "Medium")

    dup_ids = int(df["ID"].duplicated().sum())
    if dup_ids:
        add("Duplicate IDs", "ID",
            f"{dup_ids} duplicate customer IDs.", "High")

    add("Incorrect data type", "Dt_Customer",
        "Stored as text/object instead of datetime.", "Medium")

    test_dates = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    invalid_dates = int(test_dates.isna().sum())
    if invalid_dates:
        add("Invalid dates", "Dt_Customer",
            f"{invalid_dates} dates could not be parsed.", "Medium")

    unexpected_marital = [v for v in ["Alone", "Absurd", "YOLO"]
                           if v in df["Marital_Status"].unique()]
    if unexpected_marital:
        add("Inconsistent categories", "Marital_Status",
            f"Unexpected / inconsistent values: {', '.join(unexpected_marital)}.",
            "Low")

    if "2n Cycle" in df["Education"].unique():
        add("Unclear category label", "Education",
            "'2n Cycle' is an ambiguous education label.", "Low")

    invalid_birth = int(((df["Year_Birth"] < 1900) |
                          (df["Year_Birth"] > CURRENT_YEAR)).sum())
    ages_tmp = CURRENT_YEAR - df["Year_Birth"]
    unrealistic_age = int((ages_tmp > 100).sum())
    if invalid_birth or unrealistic_age:
        add("Unrealistic values", "Year_Birth",
            f"{unrealistic_age} customers appear older than 100 years "
            f"based on Year_Birth.", "High")

    negative_income = int((df["Income"] < 0).sum())
    if negative_income:
        add("Unrealistic values", "Income",
            f"{negative_income} negative income values.", "High")

    for col in ["Income"] + [c for c in NUMERIC_SPEND_COLS if c != "Income"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out:
            add("Outliers (IQR method)", col,
                f"{n_out} values fall outside [{round(lower,1)}, {round(upper,1)}].",
                "Medium")

    for col in ["Z_CostContact", "Z_Revenue"]:
        if df[col].nunique() == 1:
            add("Constant variable", col,
                "Column has only one unique value; carries no information.",
                "Low")

    quality_df = pd.DataFrame(issues)
    save_excel(quality_df, QUALITY, "02_data_quality_assessment.xlsx")
    print(quality_df.to_string(index=False))
    return quality_df


# ============================================================
# ACTIVITY 3: HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 3: HANDLE MISSING VALUES")
    print("=" * 60)

    missing_before = df.isna().sum()
    missing_report = pd.DataFrame({
        "Variable": df.columns,
        "Missing Values": missing_before.values,
        "Percentage": (missing_before.values / len(df) * 100).round(2)
    })
    save_excel(missing_report, QUALITY, "03_missing_values_before.xlsx")

    # Income is right-skewed (few very high earners), so the median is a
    # more robust central-tendency estimate than the mean, which would be
    # pulled upward by outliers. Median imputation is therefore used.
    median_income = df["Income"].median()
    n_missing = int(df["Income"].isna().sum())
    df["Income"] = df["Income"].fillna(median_income)

    print(f"Income missing values: {n_missing}")
    print(f"Imputed with median income: {median_income}")

    justification = pd.DataFrame({
        "Item": [
            "Column", "Missing Count", "Missing %", "Method Chosen",
            "Value Used", "Justification"
        ],
        "Details": [
            "Income",
            n_missing,
            round(n_missing / len(df) * 100, 2),
            "Median Imputation",
            median_income,
            "Income is right-skewed with high-earning outliers, so the "
            "median is a more robust and representative estimate than "
            "the mean. The missing percentage (<2%) is too small to "
            "justify dropping records."
        ]
    })
    save_excel(justification, QUALITY, "03_missing_value_justification.xlsx")
    return df


# ============================================================
# ACTIVITY 4: REMOVE DUPLICATE RECORDS
# ============================================================

def remove_duplicates(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 4: REMOVE DUPLICATE RECORDS")
    print("=" * 60)

    before = len(df)
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["ID"].duplicated().sum())

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="ID", keep="first")

    after = len(df)
    removed = before - after

    print(f"Before: {before} | After: {after} | Removed: {removed}")
    print(f"Duplicate rows found: {dup_rows} | Duplicate IDs found: {dup_ids}")

    report = pd.DataFrame({
        "Before": [before],
        "After": [after],
        "Removed": [removed]
    })
    save_excel(report, QUALITY, "04_duplicate_removal.xlsx")
    return df.reset_index(drop=True)


# ============================================================
# ACTIVITY 5: CORRECT DATA TYPES
# ============================================================

def correct_data_types(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 5: CORRECT DATA TYPES")
    print("=" * 60)

    old_types = df.dtypes.astype(str).to_dict()

    df["ID"] = df["ID"].astype(int)
    df["Year_Birth"] = df["Year_Birth"].astype(int)
    df["Income"] = df["Income"].astype(float)
    df["Recency"] = df["Recency"].astype(int)
    df["Kidhome"] = df["Kidhome"].astype(int)
    df["Teenhome"] = df["Teenhome"].astype(int)
    df["Education"] = df["Education"].astype("category")
    df["Marital_Status"] = df["Marital_Status"].astype("category")
    # Dt_Customer is converted to datetime in Activity 6

    binary_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                   "AcceptedCmp4", "AcceptedCmp5", "Complain", "Response"]
    for col in binary_cols:
        df[col] = df[col].astype(int)

    new_types = df.dtypes.astype(str).to_dict()

    type_report = pd.DataFrame({
        "Variable": list(old_types.keys()),
        "Old Type": list(old_types.values()),
        "New Type": [new_types[c] for c in old_types.keys()]
    })
    save_excel(type_report, QUALITY, "05_data_type_correction.xlsx")
    print(type_report.to_string(index=False))
    return df


# ============================================================
# ACTIVITY 6: CONVERT DATE COLUMNS
# ============================================================

def convert_dates(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 6: CONVERT DATE COLUMNS")
    print("=" * 60)

    original = df["Dt_Customer"].copy()
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )

    invalid = int((original.notna() & df["Dt_Customer"].isna()).sum())
    print(f"Invalid dates found: {invalid}")

    if invalid:
        # Drop rows where the enrollment date could not be parsed at all,
        # since it cannot reliably be repaired.
        df = df.dropna(subset=["Dt_Customer"]).reset_index(drop=True)

    df["Enrollment_Year"] = df["Dt_Customer"].dt.year
    df["Enrollment_Month"] = df["Dt_Customer"].dt.month
    df["Enrollment_Day"] = df["Dt_Customer"].dt.day

    summary = pd.DataFrame({
        "Item": ["Invalid Dates Found", "Rows Dropped",
                 "Min Enrollment Date", "Max Enrollment Date"],
        "Value": [invalid, invalid,
                  str(df["Dt_Customer"].min().date()),
                  str(df["Dt_Customer"].max().date())]
    })
    save_excel(summary, QUALITY, "06_date_conversion_summary.xlsx")
    return df


# ============================================================
# ACTIVITY 7: STANDARDIZE CATEGORICAL VARIABLES
# ============================================================

def standardize_categories(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 7: STANDARDIZE CATEGORICAL VARIABLES")
    print("=" * 60)

    before_education = df["Education"].astype(str).unique().tolist()
    before_marital = df["Marital_Status"].astype(str).unique().tolist()

    for col in ["Education", "Marital_Status"]:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    df["Education"] = df["Education"].str.title()
    df["Education"] = df["Education"].replace({
        "2N Cycle": "2n Cycle",
        "Phd": "PhD"
    })

    marital_map = {
        "Single": "Single", "single": "Single", "SINGLE": "Single",
        "Together": "Together", "Married": "Married",
        "Divorced": "Divorced", "Widow": "Widowed", "Widowed": "Widowed",
        "Alone": "Single",     # functionally equivalent to Single
        "Absurd": "Other",     # not a meaningful marital status
        "YOLO": "Other"        # not a meaningful marital status
    }
    df["Marital_Status"] = df["Marital_Status"].str.title()
    df["Marital_Status"] = df["Marital_Status"].replace({
        k.title(): v for k, v in marital_map.items()
    })

    df["Education"] = df["Education"].astype("category")
    df["Marital_Status"] = df["Marital_Status"].astype("category")

    after_education = df["Education"].astype(str).unique().tolist()
    after_marital = df["Marital_Status"].astype(str).unique().tolist()

    report = pd.DataFrame({
        "Variable": ["Education", "Marital_Status"],
        "Before": [", ".join(sorted(before_education)),
                   ", ".join(sorted(before_marital))],
        "After": [", ".join(sorted(after_education)),
                  ", ".join(sorted(after_marital))]
    })
    save_excel(report, QUALITY, "07_categorical_standardization.xlsx")
    print(report.to_string(index=False))
    return df


# ============================================================
# ACTIVITY 8: DETECT UNREALISTIC CUSTOMER AGES
# ============================================================

def detect_unrealistic_ages(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 8: DETECT UNREALISTIC CUSTOMER AGES")
    print("=" * 60)

    df["Age"] = CURRENT_YEAR - df["Year_Birth"]

    invalid_birth_year = df[df["Year_Birth"] > CURRENT_YEAR]
    negative_age = df[df["Age"] < 0]
    too_old = df[df["Age"] > 100]

    n_invalid_birth = len(invalid_birth_year)
    n_negative = len(negative_age)
    n_too_old = len(too_old)

    print(f"Birth year > current year: {n_invalid_birth}")
    print(f"Negative age: {n_negative}")
    print(f"Age > 100: {n_too_old}")

    before = len(df)
    # Records older than 100 years are almost certainly data-entry errors
    # (e.g. birth year 1893, 1899, 1900) and are removed since the true
    # age cannot be recovered.
    df = df[df["Age"] <= 100].reset_index(drop=True)
    after = len(df)

    report = pd.DataFrame({
        "Check": ["Birth Year > Current Year", "Negative Age",
                  "Age > 100 (Removed)", "Records Before", "Records After"],
        "Count": [n_invalid_birth, n_negative, n_too_old, before, after]
    })
    save_excel(report, QUALITY, "08_age_validation_report.xlsx")
    print(report.to_string(index=False))
    return df


# ============================================================
# ACTIVITY 9: DETECT INCOME AND SPENDING OUTLIERS (IQR)
# ============================================================

def detect_outliers(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 9: DETECT INCOME AND SPENDING OUTLIERS")
    print("=" * 60)

    outlier_summary = []
    bounds = {}

    for col in NUMERIC_SPEND_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        bounds[col] = (lower, upper)
        outlier_summary.append({
            "Variable": col,
            "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
            "Lower Bound": round(lower, 2), "Upper Bound": round(upper, 2),
            "Number of Outliers": n_out
        })

    outlier_df = pd.DataFrame(outlier_summary)
    save_excel(outlier_df, QUALITY, "09_outlier_summary.xlsx")
    print(outlier_df.to_string(index=False))

    # Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_SPEND_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="#4C72B0")
        axes[i].set_title(col)
    for j in range(len(NUMERIC_SPEND_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Boxplots — Income & Spending Variables (Before Outlier Handling)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(VIZ / "boxplots_before_outlier_handling.png", dpi=150)
    plt.close()

    return outlier_df, bounds


# ============================================================
# ACTIVITY 10: HANDLE OUTLIERS (CAPPING / WINSORIZATION)
# ============================================================

def handle_outliers(df, bounds):
    print("\n" + "=" * 60)
    print("ACTIVITY 10: HANDLE OUTLIERS")
    print("=" * 60)

    decisions = []
    for col, (lower, upper) in bounds.items():
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if col == "Income":
            # A small number of very high incomes (up to ~666k) are
            # plausible for real high-net-worth customers, so instead of
            # removing them we cap (winsorize) to the IQR upper bound to
            # reduce their leverage on models while keeping the records.
            df[col] = np.where(df[col] > upper, upper, df[col])
            method, reason = "Capped (Winsorization)", \
                "Extreme but plausible values; capped to preserve records " \
                "while limiting leverage on downstream models."
        else:
            # Spending columns are heavily right-skewed by nature (most
            # customers spend little, a few spend a lot on a category).
            # These are kept as-is but flagged, since capping would
            # distort genuine high-value customer behaviour that is
            # valuable for segmentation.
            method, reason = "Kept (No Change)", \
                "Right-skewed spending is expected customer behaviour, " \
                "not an error; retained for accurate segmentation " \
                "analysis, but flagged for awareness in EDA."
        decisions.append({
            "Variable": col, "Outliers Detected": n_out,
            "Method": method, "Reason": reason
        })

    decisions_df = pd.DataFrame(decisions)
    save_excel(decisions_df, QUALITY, "10_outlier_handling_decisions.xlsx")
    print(decisions_df.to_string(index=False))

    # Boxplots after handling
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_SPEND_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="#55A868")
        axes[i].set_title(col)
    for j in range(len(NUMERIC_SPEND_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Boxplots — Income & Spending Variables (After Outlier Handling)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(VIZ / "boxplots_after_outlier_handling.png", dpi=150)
    plt.close()

    return df


# ============================================================
# ACTIVITY 11: VALIDATE THE CLEANED DATASET
# ============================================================

def validate_dataset(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 11: VALIDATE THE CLEANED DATASET")
    print("=" * 60)

    checks = []

    missing_total = int(df.isna().sum().sum())
    checks.append(("Missing Values", "PASS" if missing_total == 0 else "CHECK",
                    f"{missing_total} missing values remain."))

    dup_total = int(df.duplicated().sum())
    checks.append(("Duplicates", "PASS" if dup_total == 0 else "CHECK",
                    f"{dup_total} duplicate rows remain."))

    dtype_ok = (
        pd.api.types.is_datetime64_any_dtype(df["Dt_Customer"]) and
        pd.api.types.is_float_dtype(df["Income"])
    )
    checks.append(("Data Types", "PASS" if dtype_ok else "CHECK",
                    "Dt_Customer is datetime; Income is float."))

    dates_ok = df["Dt_Customer"].isna().sum() == 0
    checks.append(("Dates", "PASS" if dates_ok else "CHECK",
                    "All Dt_Customer values are valid dates."))

    valid_marital = {"Single", "Together", "Married", "Divorced", "Widowed", "Other"}
    categories_ok = set(df["Marital_Status"].unique()).issubset(valid_marital)
    checks.append(("Categories", "PASS" if categories_ok else "CHECK",
                    "Marital_Status standardized to consistent labels."))

    age_ok = df["Age"].between(0, 100).all()
    checks.append(("Age Range", "PASS" if age_ok else "CHECK",
                    "All ages fall within 0-100 years."))

    income_ok = (df["Income"] >= 0).all()
    checks.append(("Outliers / Income Range", "PASS" if income_ok else "CHECK",
                    "No negative income values remain."))

    validation_df = pd.DataFrame(checks, columns=["Validation Check", "Status", "Notes"])
    save_excel(validation_df, QUALITY, "11_validation_checklist.xlsx")
    print(validation_df.to_string(index=False))
    return validation_df


# ============================================================
# ACTIVITY 12: SAVE THE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df):
    print("\n" + "=" * 60)
    print("ACTIVITY 12: SAVE THE CLEANED DATASET")
    print("=" * 60)
    df.to_csv(CLEANED_FILE, index=False)
    print(f"Cleaned dataset saved to: {CLEANED_FILE}")
    print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return CLEANED_FILE


# ============================================================
# ACTIVITY 14: BEFORE-AND-AFTER COMPARISON
# ============================================================

def before_after_comparison(df_before, df_after, dup_before, invalid_dates_before,
                             outliers_income_before, age_invalid_before):
    print("\n" + "=" * 60)
    print("ACTIVITY 14: BEFORE-AND-AFTER COMPARISON")
    print("=" * 60)

    dtype_issues_before = 1  # Dt_Customer stored as text before cleaning

    comparison = pd.DataFrame({
        "Metric": ["Rows", "Missing Values", "Duplicate Records",
                   "Invalid Data Types", "Income Outliers (IQR)", "Date Errors",
                   "Unrealistic Ages (>100 yrs)"],
        "Before Cleaning": [
            len(df_before),
            int(df_before.isna().sum().sum()),
            dup_before,
            dtype_issues_before,
            outliers_income_before,
            invalid_dates_before,
            age_invalid_before
        ],
        "After Cleaning": [
            len(df_after),
            int(df_after.isna().sum().sum()),
            int(df_after.duplicated().sum()),
            0,
            int((df_after["Income"] > df_after["Income"].quantile(0.75) +
                 1.5 * (df_after["Income"].quantile(0.75) - df_after["Income"].quantile(0.25))).sum()),
            0,
            0
        ]
    })
    save_excel(comparison, REPORTS, "14_before_after_comparison.xlsx")
    print(comparison.to_string(index=False))
    return comparison


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    df_raw = load_data()
    df_before = df_raw.copy()

    inspect_data(df_raw)
    assess_data_quality(df_raw)

    dup_before = int(df_raw.duplicated().sum())
    test_dates = pd.to_datetime(df_raw["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    invalid_dates_before = int((df_raw["Dt_Customer"].notna() & test_dates.isna()).sum())
    q1, q3 = df_raw["Income"].quantile(0.25), df_raw["Income"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    outliers_income_before = int((df_raw["Income"] > upper).sum())
    age_before_tmp = CURRENT_YEAR - df_raw["Year_Birth"]
    age_invalid_before = int((age_before_tmp > 100).sum())

    df = handle_missing_values(df_raw)
    df = remove_duplicates(df)
    df = correct_data_types(df)
    df = convert_dates(df)
    df = standardize_categories(df)
    df = detect_unrealistic_ages(df)
    outlier_df, bounds = detect_outliers(df)
    df = handle_outliers(df, bounds)
    validate_dataset(df)
    save_cleaned_dataset(df)
    before_after_comparison(df_before, df, dup_before, invalid_dates_before,
                             outliers_income_before, age_invalid_before)

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    run_pipeline()
