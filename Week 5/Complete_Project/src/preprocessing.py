"""
Preprocessing Module
=====================
Data-cleaning and validation logic shared across the pipeline (mirrors
Module 03 - Data Cleaning and Preprocessing). Used both for batch dataset
loading and for validating single-customer form input in the dashboard.
"""

import numpy as np
import pandas as pd


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw dataset and apply the core Module-03 cleaning steps."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

    # Income: median imputation (right-skewed distribution)
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    # Duplicates
    df = df.drop_duplicates()
    if "ID" in df.columns:
        df = df.drop_duplicates(subset="ID", keep="first")

    # Dates
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["Dt_Customer"])

    # Standardize categories
    for col in ["Education", "Marital_Status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()

    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace({
            "Alone": "Single", "Absurd": "Other", "Yolo": "Other"
        })

    # Unrealistic ages
    if "Year_Birth" in df.columns:
        age = 2026 - df["Year_Birth"]
        df = df[age <= 100].reset_index(drop=True)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Validation used by the dashboard's "Individual Customer" input form
# ---------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when user-supplied input fails validation."""


def validate_customer_input(values: dict) -> dict:
    """
    Validate a dict of raw form values for a single customer.
    Raises ValidationError with a clear message on the first problem found.
    Returns a cleaned dict of numeric values on success.
    """
    required = ["age", "income", "total_spending", "web_purchases",
                "store_purchases", "recency"]
    for field in required:
        if field not in values or values[field] in (None, ""):
            raise ValidationError(f"'{field.replace('_', ' ').title()}' is required.")

    cleaned = {}
    for field in required:
        raw = values[field]
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValidationError(
                f"'{field.replace('_', ' ').title()}' must be a number — got '{raw}'."
            )
        cleaned[field] = num

    if not (0 <= cleaned["age"] <= 120):
        raise ValidationError("Age must be between 0 and 120.")
    if cleaned["income"] < 0:
        raise ValidationError("Income cannot be negative.")
    if cleaned["income"] > 5_000_000:
        raise ValidationError("Income value looks unrealistically large (> 5,000,000). Please check the input.")
    if cleaned["total_spending"] < 0:
        raise ValidationError("Total spending cannot be negative.")
    if cleaned["web_purchases"] < 0 or cleaned["store_purchases"] < 0:
        raise ValidationError("Purchase counts cannot be negative.")
    if cleaned["recency"] < 0 or cleaned["recency"] > 999:
        raise ValidationError("Recency must be between 0 and 999 days.")

    return cleaned
