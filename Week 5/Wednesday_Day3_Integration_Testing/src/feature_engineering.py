"""
Feature Engineering Module
===========================
Reusable feature-creation logic shared between model training (Module 05)
and live prediction (the Streamlit dashboard). Keeping this logic in one
place guarantees the same transformations are applied at both training
and inference time.
"""

from datetime import datetime

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026
REFERENCE_DATE = pd.Timestamp("2026-07-30")

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                 "AcceptedCmp4", "AcceptedCmp5", "Response"]

# The exact 12 features the segmentation model was trained on (Module 06),
# in this exact order.
CLUSTER_FEATURES = [
    "Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
    "Family_Size", "Total_Children", "Total_Purchases",
    "Total_Campaign_Acceptance", "NumWebPurchases", "NumStorePurchases",
    "NumCatalogPurchases",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full Module-05 feature-engineering logic to a raw/cleaned
    customer dataframe. Missing optional columns are filled with sensible
    defaults so this also works on partial (single-customer form) input.
    """
    df = df.copy()

    # Ensure required raw columns exist with safe defaults
    defaults = {
        "Year_Birth": REFERENCE_YEAR - 45, "Income": 0, "Kidhome": 0, "Teenhome": 0,
        "Recency": 0, "Marital_Status": "Single",
        **{c: 0 for c in SPEND_COLS}, **{c: 0 for c in CHANNEL_COLS},
        "NumDealsPurchases": 0, "NumWebVisitsMonth": 0,
        **{c: 0 for c in CAMPAIGN_COLS},
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
        df["Customer_Tenure"] = (REFERENCE_DATE - df["Dt_Customer"]).dt.days
        df["Customer_Tenure"] = df["Customer_Tenure"].fillna(365 * 3)  # ~3yr default for new customers
    elif "Customer_Tenure" not in df.columns:
        df["Customer_Tenure"] = 365 * 3

    df["Customer_Age"] = REFERENCE_YEAR - df["Year_Birth"]
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    marital_partner = {"Married", "Together"}
    df["Family_Size"] = df["Marital_Status"].isin(marital_partner).astype(int) + 1 + df["Total_Children"]

    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)
    df["Total_Purchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] +
        df["NumStorePurchases"] + df["NumDealsPurchases"]
    )
    df["Total_Campaign_Acceptance"] = df[CAMPAIGN_COLS].sum(axis=1)
    df["Avg_Spending_Per_Purchase"] = (
        df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)
    df["Digital_Engagement"] = df["NumWebVisitsMonth"] + df["NumWebPurchases"]
    df["Deal_Dependency"] = (
        df["NumDealsPurchases"] / df["Total_Purchases"].replace(0, np.nan)
    ).fillna(0)

    return df


def get_cluster_feature_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 12 features the model expects, in the correct order."""
    engineered = create_features(df)
    return engineered[CLUSTER_FEATURES]
