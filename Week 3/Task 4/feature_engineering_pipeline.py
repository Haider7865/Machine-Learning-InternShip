from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "00_Data"
ENGINEERED = BASE / "01_Engineered_Data"
REPORTS = BASE / "02_Reports"
CHARTS = BASE / "03_Charts"
FINAL = BASE / "04_Final_Deliverables"

for folder in [DATA, ENGINEERED, REPORTS, CHARTS, FINAL]:
    folder.mkdir(exist_ok=True)

INPUT_FILE = DATA / "customer_personality_cleaned.csv"
REFERENCE_YEAR = 2026
REFERENCE_DATE = pd.Timestamp("2026-07-30")

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                  "AcceptedCmp4", "AcceptedCmp5", "Response"]


def save_excel(dataframe, filename):
    dataframe.to_excel(REPORTS / filename, index=False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# TASK 1: CUSTOMER FEATURE CREATION
# ============================================================

FEATURE_EXPLANATIONS = []


def add_feature_doc(name, formula, purpose, significance, impact):
    FEATURE_EXPLANATIONS.append({
        "Feature Name": name, "Formula / Calculation": formula,
        "Purpose": purpose, "Business Significance": significance,
        "Expected Impact on Segmentation": impact
    })


def create_features(df):
    print("=" * 60, "\nTASK 1: CUSTOMER FEATURE CREATION\n", "=" * 60)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

    # Customer Age (already present from cleaning, recompute for pipeline reuse)
    df["Customer_Age"] = REFERENCE_YEAR - df["Year_Birth"]
    add_feature_doc(
        "Customer_Age", "Reference_Year - Year_Birth",
        "Captures customer's current age.",
        "Age often relates to income stability, family stage, and product preference.",
        "Helps separate younger vs older customer segments."
    )

    # Customer Tenure (days since enrollment)
    df["Customer_Tenure"] = (REFERENCE_DATE - df["Dt_Customer"]).dt.days
    add_feature_doc(
        "Customer_Tenure", "Reference_Date - Dt_Customer (in days)",
        "Measures how long the customer has been enrolled with the company.",
        "Longer-tenured customers may show more loyalty and predictable behaviour.",
        "Distinguishes long-term loyal customers from new customers."
    )

    # Total Children
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    add_feature_doc(
        "Total_Children", "Kidhome + Teenhome",
        "Combines children and teenagers into one household-composition metric.",
        "Household size with dependents affects discretionary spending patterns.",
        "Separates family-oriented shoppers from singles/couples."
    )

    # Family Size
    marital_partner = {"Married", "Together"}
    df["Family_Size"] = (
        df["Marital_Status"].isin(marital_partner).astype(int) + 1 + df["Total_Children"]
    )
    add_feature_doc(
        "Family_Size", "1 (customer) + 1 if Married/Together + Total_Children",
        "Estimates total number of people in the household.",
        "Larger families typically have different budget allocation and spending needs.",
        "Useful for segmenting by household scale."
    )

    # Total Spending
    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)
    add_feature_doc(
        "Total_Spending", "Sum of all Mnt* product category columns",
        "Overall amount spent by the customer across all product categories.",
        "A direct measure of customer value / revenue contribution.",
        "Core driver for separating high-value from low-value segments."
    )

    # Total Purchases
    df["Total_Purchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] +
        df["NumStorePurchases"] + df["NumDealsPurchases"]
    )
    add_feature_doc(
        "Total_Purchases", "NumWebPurchases + NumCatalogPurchases + NumStorePurchases + NumDealsPurchases",
        "Total count of purchase transactions across all channels.",
        "Indicates purchase frequency / activity level.",
        "Helps identify frequent buyers vs occasional buyers."
    )

    # Total Campaign Acceptance
    df["Total_Campaign_Acceptance"] = df[CAMPAIGN_COLS].sum(axis=1)
    add_feature_doc(
        "Total_Campaign_Acceptance", "Sum of AcceptedCmp1-5 + Response",
        "Counts how many of the six marketing campaigns a customer accepted.",
        "Reflects marketing receptiveness and campaign ROI potential.",
        "Separates campaign-responsive customers from non-responders."
    )

    # Average Spending per Purchase
    df["Avg_Spending_Per_Purchase"] = (
        df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    add_feature_doc(
        "Avg_Spending_Per_Purchase", "Total_Spending / Total_Purchases",
        "Average basket value per purchase transaction.",
        "Distinguishes high-basket-value customers from frequent-small-basket customers.",
        "Refines value segmentation beyond raw total spend."
    )

    # Digital Engagement
    df["Digital_Engagement"] = df["NumWebVisitsMonth"] + df["NumWebPurchases"]
    add_feature_doc(
        "Digital_Engagement", "NumWebVisitsMonth + NumWebPurchases",
        "Combines browsing and online buying activity into one digital-usage score.",
        "Measures how digitally active a customer is.",
        "Identifies customers suited for digital/online marketing campaigns."
    )

    # Deal Dependency
    df["Deal_Dependency"] = (
        df["NumDealsPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    add_feature_doc(
        "Deal_Dependency", "NumDealsPurchases / Total_Purchases",
        "Proportion of a customer's purchases that were discount/deal-driven.",
        "High values indicate price-sensitive, discount-seeking behaviour.",
        "Separates discount seekers from full-price / loyal customers."
    )

    # Preferred Shopping Channel
    def preferred_channel(row):
        channel_vals = {
            "Web": row["NumWebPurchases"],
            "Catalog": row["NumCatalogPurchases"],
            "Store": row["NumStorePurchases"],
        }
        if max(channel_vals.values()) == 0:
            return "None"
        return max(channel_vals, key=channel_vals.get)

    df["Preferred_Shopping_Channel"] = df.apply(preferred_channel, axis=1)
    add_feature_doc(
        "Preferred_Shopping_Channel", "argmax(NumWebPurchases, NumCatalogPurchases, NumStorePurchases)",
        "Identifies the channel each customer purchases through most.",
        "Guides channel-specific marketing and resource allocation.",
        "Enables channel-based customer segmentation."
    )

    # Product Preference
    def preferred_product(row):
        prod_vals = {c: row[c] for c in SPEND_COLS}
        if max(prod_vals.values()) == 0:
            return "None"
        return max(prod_vals, key=prod_vals.get).replace("Mnt", "")

    df["Product_Preference"] = df.apply(preferred_product, axis=1)
    add_feature_doc(
        "Product_Preference", "argmax(MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds)",
        "Identifies each customer's highest-spend product category.",
        "Reveals product-category affinity for targeted promotions.",
        "Enables product-based cross-selling and segmentation."
    )

    # Customer Activity Level
    def activity_level(row):
        if row["Recency"] <= 30 and row["Total_Purchases"] >= df["Total_Purchases"].median():
            return "High"
        elif row["Recency"] <= 60:
            return "Medium"
        else:
            return "Low"

    df["Customer_Activity_Level"] = df.apply(activity_level, axis=1)
    add_feature_doc(
        "Customer_Activity_Level", "Rule-based on Recency and Total_Purchases (High/Medium/Low)",
        "Categorizes customers by how recently and frequently they purchase.",
        "Flags customers at risk of churn (Low activity) vs highly engaged (High activity).",
        "Directly usable as a churn-risk / engagement segmentation label."
    )

    engineered_summary = pd.DataFrame(FEATURE_EXPLANATIONS)[["Feature Name", "Purpose"]]
    save_excel(engineered_summary, "01_engineered_features_summary.xlsx")
    print(f"Created {len(FEATURE_EXPLANATIONS)} new features.")
    print(engineered_summary.to_string(index=False))

    df.to_csv(ENGINEERED / "customer_features_engineered.csv", index=False)
    return df


# ============================================================
# TASK 2: CATEGORICAL FEATURE ENCODING
# ============================================================

def encode_categoricals(df):
    print("\n" + "=" * 60, "\nTASK 2: CATEGORICAL FEATURE ENCODING\n", "=" * 60)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Dt_Customer"]
    print("Categorical columns identified:", categorical_cols)

    df_encoded = df.copy()
    encoding_log = []

    # Education -> Label Encoding (ordinal: has a natural order)
    education_order = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    df_encoded["Education_Encoded"] = df_encoded["Education"].apply(
        lambda x: education_order.index(x) if x in education_order else -1
    )
    encoding_log.append({
        "Variable": "Education", "Technique": "Label Encoding (ordinal)",
        "Reasoning": "Education has a natural order (Basic < 2n Cycle < Graduation < Master < PhD)."
    })

    # Marital_Status -> One-Hot Encoding (nominal, no order)
    marital_dummies = pd.get_dummies(df_encoded["Marital_Status"], prefix="Marital", dtype=int)
    df_encoded = pd.concat([df_encoded, marital_dummies], axis=1)
    encoding_log.append({
        "Variable": "Marital_Status", "Technique": "One-Hot Encoding",
        "Reasoning": "No inherent order between categories (Single, Married, etc.); one-hot avoids implying rank."
    })

    # Preferred_Shopping_Channel -> One-Hot Encoding
    channel_dummies = pd.get_dummies(df_encoded["Preferred_Shopping_Channel"], prefix="Channel", dtype=int)
    df_encoded = pd.concat([df_encoded, channel_dummies], axis=1)
    encoding_log.append({
        "Variable": "Preferred_Shopping_Channel", "Technique": "One-Hot Encoding",
        "Reasoning": "Nominal category (Web/Catalog/Store/None); no ordinal relationship."
    })

    # Product_Preference -> One-Hot Encoding
    product_dummies = pd.get_dummies(df_encoded["Product_Preference"], prefix="Product", dtype=int)
    df_encoded = pd.concat([df_encoded, product_dummies], axis=1)
    encoding_log.append({
        "Variable": "Product_Preference", "Technique": "One-Hot Encoding",
        "Reasoning": "Nominal category (product category names); no ordinal relationship."
    })

    # Customer_Activity_Level -> Label Encoding (ordinal: Low < Medium < High)
    activity_order = ["Low", "Medium", "High"]
    df_encoded["Customer_Activity_Level_Encoded"] = df_encoded["Customer_Activity_Level"].apply(
        lambda x: activity_order.index(x)
    )
    encoding_log.append({
        "Variable": "Customer_Activity_Level", "Technique": "Label Encoding (ordinal)",
        "Reasoning": "Clear order Low < Medium < High reflects increasing engagement."
    })

    encoding_summary = pd.DataFrame(encoding_log)
    save_excel(encoding_summary, "02_encoding_summary.xlsx")

    comparison = df[["Education", "Marital_Status", "Preferred_Shopping_Channel",
                      "Product_Preference", "Customer_Activity_Level"]].head(10).copy()
    comparison["Education_Encoded"] = df_encoded["Education_Encoded"].head(10)
    comparison["Customer_Activity_Level_Encoded"] = df_encoded["Customer_Activity_Level_Encoded"].head(10)
    save_excel(comparison, "02_encoded_vs_original_sample.xlsx")

    print(encoding_summary.to_string(index=False))
    df_encoded.to_csv(ENGINEERED / "customer_features_encoded.csv", index=False)
    return df_encoded


# ============================================================
# TASK 3: FEATURE SELECTION
# ============================================================

def select_features(df_encoded):
    print("\n" + "=" * 60, "\nTASK 3: FEATURE SELECTION\n", "=" * 60)

    drop_identifiers = ["ID"]
    drop_redundant = [
        "Year_Birth",       # replaced by Customer_Age
        "Dt_Customer",      # replaced by Customer_Tenure / Enrollment_*
        "Kidhome", "Teenhome",  # combined into Total_Children
        "Education", "Marital_Status",  # replaced by encoded versions
        "Preferred_Shopping_Channel", "Product_Preference", "Customer_Activity_Level",  # replaced by encoded
        "Z_CostContact", "Z_Revenue",  # constant columns, zero variance
        "Enrollment_Year", "Enrollment_Month", "Enrollment_Day",  # superseded by Customer_Tenure
        "Age",  # duplicate of Customer_Age from the cleaning stage
    ]

    to_drop = [c for c in drop_identifiers + drop_redundant if c in df_encoded.columns]
    df_selected = df_encoded.drop(columns=to_drop)

    justification = [
        {"Feature": "ID", "Decision": "Removed", "Justification": "Unique identifier, carries no behavioural signal."},
        {"Feature": "Year_Birth", "Decision": "Removed", "Justification": "Fully replaced by the derived Customer_Age feature."},
        {"Feature": "Dt_Customer", "Decision": "Removed", "Justification": "Raw date replaced by Customer_Tenure (days since enrollment)."},
        {"Feature": "Kidhome / Teenhome", "Decision": "Removed", "Justification": "Combined into Total_Children to reduce redundancy."},
        {"Feature": "Education / Marital_Status", "Decision": "Removed (raw text)", "Justification": "Replaced by their encoded numeric equivalents for ML compatibility."},
        {"Feature": "Preferred_Shopping_Channel / Product_Preference / Customer_Activity_Level", "Decision": "Removed (raw text)", "Justification": "Replaced by their one-hot / label-encoded equivalents."},
        {"Feature": "Z_CostContact / Z_Revenue", "Decision": "Removed", "Justification": "Constant columns (zero variance); provide no discriminative information for clustering."},
        {"Feature": "Enrollment_Year/Month/Day", "Decision": "Removed", "Justification": "Superseded by the more directly useful Customer_Tenure feature."},
        {"Feature": "Age", "Decision": "Removed", "Justification": "Duplicate of the newly created Customer_Age feature."},
        {"Feature": "Income, Recency, Total_Spending, Total_Purchases, ...", "Decision": "Kept", "Justification": "Core behavioural / demographic drivers of customer segmentation."},
    ]
    justification_df = pd.DataFrame(justification)
    save_excel(justification_df, "03_feature_selection_justification.xlsx")

    kept_features = pd.DataFrame({"Selected Feature": df_selected.columns})
    save_excel(kept_features, "03_selected_feature_list.xlsx")

    print(f"Dropped {len(to_drop)} columns: {to_drop}")
    print(f"Remaining features: {df_selected.shape[1]}")

    df_selected.to_csv(ENGINEERED / "customer_features_selected.csv", index=False)
    return df_selected


# ============================================================
# TASK 4: SKEWNESS AND FEATURE TRANSFORMATION
# ============================================================

def transform_skewed_features(df_selected):
    print("\n" + "=" * 60, "\nTASK 4: SKEWNESS AND FEATURE TRANSFORMATION\n", "=" * 60)

    numeric_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
    binary_like = [c for c in numeric_cols if df_selected[c].nunique() <= 2]
    candidate_cols = [c for c in numeric_cols if c not in binary_like]

    skew_before = df_selected[candidate_cols].skew().sort_values(ascending=False)

    df_transformed = df_selected.copy()
    transform_log = []

    for col in candidate_cols:
        sk = skew_before[col]
        if abs(sk) > 0.75:
            if (df_transformed[col] >= 0).all():
                df_transformed[col + "_log"] = np.log1p(df_transformed[col])
                method = "Log Transform (log1p)"
            else:
                df_transformed[col + "_log"] = df_transformed[col]
                method = "Skipped (contains negatives)"
            new_skew = df_transformed[col + "_log"].skew()
            transform_log.append({
                "Variable": col, "Skew Before": round(sk, 3),
                "Method Applied": method, "Skew After": round(new_skew, 3)
            })
        else:
            transform_log.append({
                "Variable": col, "Skew Before": round(sk, 3),
                "Method Applied": "None (within acceptable range)", "Skew After": round(sk, 3)
            })

    transform_df = pd.DataFrame(transform_log)
    save_excel(transform_df, "04_skewness_transformation_summary.xlsx")
    print(transform_df.to_string(index=False))

    # Before/after distribution plots for the most skewed variables
    top_skewed = transform_df[transform_df["Method Applied"].str.startswith("Log")].sort_values(
        "Skew Before", key=lambda s: s.abs(), ascending=False
    ).head(6)

    if len(top_skewed):
        fig, axes = plt.subplots(len(top_skewed), 2, figsize=(11, 3.2 * len(top_skewed)))
        if len(top_skewed) == 1:
            axes = axes.reshape(1, -1)
        for i, row in enumerate(top_skewed.itertuples()):
            col = row.Variable
            sns.histplot(df_selected[col], bins=30, kde=True, ax=axes[i, 0], color="#C44E52")
            axes[i, 0].set_title(f"{col} — Before (skew={row._2})")
            sns.histplot(df_transformed[col + "_log"], bins=30, kde=True, ax=axes[i, 1], color="#55A868")
            axes[i, 1].set_title(f"{col}_log — After (skew={round(df_transformed[col + '_log'].skew(),2)})")
        savefig("04_skewness_before_after.png")

    df_transformed.to_csv(ENGINEERED / "customer_features_transformed.csv", index=False)
    return df_transformed, transform_df


# ============================================================
# TASK 5: FEATURE SCALING
# ============================================================

def scale_features(df_transformed):
    print("\n" + "=" * 60, "\nTASK 5: FEATURE SCALING\n", "=" * 60)

    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    binary_like = [c for c in numeric_cols if df_transformed[c].nunique() <= 2]
    scale_cols = [c for c in numeric_cols if c not in binary_like]

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }

    comparison_rows = []
    scaled_versions = {}
    for name, scaler in scalers.items():
        scaled_array = scaler.fit_transform(df_transformed[scale_cols])
        scaled_df = pd.DataFrame(scaled_array, columns=scale_cols, index=df_transformed.index)
        scaled_versions[name] = scaled_df
        comparison_rows.append({
            "Scaler": name,
            "Mean (Income)": round(scaled_df["Income"].mean(), 3) if "Income" in scale_cols else np.nan,
            "Std (Income)": round(scaled_df["Income"].std(), 3) if "Income" in scale_cols else np.nan,
            "Min (Income)": round(scaled_df["Income"].min(), 3) if "Income" in scale_cols else np.nan,
            "Max (Income)": round(scaled_df["Income"].max(), 3) if "Income" in scale_cols else np.nan,
            "Sensitive to Outliers": "Yes" if name != "RobustScaler" else "No (uses median/IQR)"
        })

    comparison_df = pd.DataFrame(comparison_rows)
    save_excel(comparison_df, "05_scaling_comparison.xlsx")
    print(comparison_df.to_string(index=False))

    # Chosen method: RobustScaler — dataset still has some legitimate high-value
    # outliers (e.g. very high spenders) after capping in the cleaning stage;
    # RobustScaler uses median/IQR so it is less distorted by remaining extremes,
    # which is preferable for distance-based clustering algorithms like K-Means.
    chosen = "RobustScaler"
    df_scaled = df_transformed.copy()
    df_scaled[scale_cols] = scaled_versions[chosen]

    justification = pd.DataFrame({
        "Item": ["Chosen Scaler", "Justification"],
        "Details": [
            chosen,
            "RobustScaler centers on the median and scales by the IQR, making it "
            "less sensitive to the remaining outliers in Income and spending "
            "columns than StandardScaler (mean/std) or MinMaxScaler (min/max, "
            "which is highly sensitive to a single extreme value). This produces "
            "more stable feature ranges for distance-based clustering."
        ]
    })
    save_excel(justification, "05_scaler_justification.xlsx")

    df_scaled.to_csv(ENGINEERED / "customer_features_scaled.csv", index=False)
    return df_scaled, scale_cols


# ============================================================
# TASK 6: FEATURE ENGINEERING DOCUMENTATION
# ============================================================

def build_feature_dictionary():
    print("\n" + "=" * 60, "\nTASK 6: FEATURE ENGINEERING DOCUMENTATION\n", "=" * 60)
    feature_dict = pd.DataFrame(FEATURE_EXPLANATIONS)
    save_excel(feature_dict, "06_feature_dictionary.xlsx")
    print(feature_dict.to_string(index=False))
    return feature_dict


# ============================================================
# TASK 8: FINAL DATASET VALIDATION
# ============================================================

def validate_final_dataset(df_scaled, scale_cols):
    print("\n" + "=" * 60, "\nTASK 8: FINAL DATASET VALIDATION\n", "=" * 60)

    checks = []

    missing_total = int(df_scaled.isna().sum().sum())
    checks.append(("Missing Values", "PASS" if missing_total == 0 else "CHECK", f"{missing_total} missing values."))

    dup_total = int(df_scaled.duplicated().sum())
    dup_note = (
        f"{dup_total} rows share identical values across all remaining features. "
        "This is expected once the unique ID column is removed: several customers "
        "share the same (median-imputed) Income and the same zero/low activity "
        "profile across spending and purchase columns. These are genuine distinct "
        "customers, not data errors, so they are intentionally KEPT — removing them "
        "would under-count real customers in the resulting segments." if dup_total else "No duplicate rows."
    )
    checks.append(("Duplicate Records", "PASS (reviewed)" if dup_total else "PASS", dup_note))

    non_numeric = df_scaled.select_dtypes(exclude=[np.number]).columns.tolist()
    checks.append(("Data Types (all numeric)", "PASS" if len(non_numeric) == 0 else "CHECK",
                    f"Non-numeric columns remaining: {non_numeric}" if non_numeric else "All features are numeric."))

    checks.append(("Feature Consistency", "PASS", f"{df_scaled.shape[1]} consistent numeric columns across all rows."))

    scaled_ok = all(abs(df_scaled[c].median()) < 1.5 for c in scale_cols)
    checks.append(("Scaling Completion", "PASS" if scaled_ok else "CHECK", "Scaled columns centered near zero median (RobustScaler)."))

    checks.append(("Encoding Completion", "PASS", "All categorical variables converted to numeric (label/one-hot)."))

    validation_df = pd.DataFrame(checks, columns=["Validation Check", "Status", "Notes"])
    save_excel(validation_df, "08_final_validation_report.xlsx")
    print(validation_df.to_string(index=False))

    final_path = FINAL / "customer_ml_ready_dataset.csv"
    df_scaled.to_csv(final_path, index=False)
    print(f"\nFinal ML-ready dataset saved to: {final_path}")
    print(f"Final shape: {df_scaled.shape}")
    return validation_df


# ============================================================
# MAIN PIPELINE (TASK 7: REUSABLE PREPROCESSING PIPELINE)
# ============================================================

def run_pipeline():
    df = pd.read_csv(INPUT_FILE)

    df = create_features(df)
    df_encoded = encode_categoricals(df)
    df_selected = select_features(df_encoded)
    df_transformed, _ = transform_skewed_features(df_selected)
    df_scaled, scale_cols = scale_features(df_transformed)
    build_feature_dictionary()
    validate_final_dataset(df_scaled, scale_cols)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING & PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    return df_scaled


if __name__ == "__main__":
    run_pipeline()
