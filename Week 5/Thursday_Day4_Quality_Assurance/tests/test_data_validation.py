"""Task 9 — Data Validation Testing"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "customer_segments_with_strategy.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_dataset_loads(df):
    assert df is not None
    assert len(df) > 0


def test_row_count(df):
    assert df.shape[0] == 2237, f"Expected 2237 rows, got {df.shape[0]}"


def test_column_count(df):
    assert df.shape[1] >= 60, f"Expected at least 60 columns, got {df.shape[1]}"


def test_data_types(df):
    assert pd.api.types.is_numeric_dtype(df["Income"])
    assert pd.api.types.is_numeric_dtype(df["Total_Spending"])
    assert pd.api.types.is_integer_dtype(df["Cluster"]) or pd.api.types.is_numeric_dtype(df["Cluster"])


def test_no_missing_values_in_key_columns(df):
    key_cols = ["ID", "Income", "Total_Spending", "Cluster", "Segment_Name"]
    for col in key_cols:
        assert df[col].isna().sum() == 0, f"Column {col} has missing values"


def test_no_duplicate_ids(df):
    assert df["ID"].duplicated().sum() == 0


def test_outliers_handled(df):
    # Income should not exceed the capped upper bound established in Module 03/05
    assert df["Income"].max() < 200000, "Income outliers appear uncapped"


def test_feature_values_valid(df):
    assert (df["Customer_Age"] >= 0).all() and (df["Customer_Age"] <= 100).all()
    assert (df["Income"] >= 0).all()
    assert (df["Total_Spending"] >= 0).all()
    assert df["Cluster"].nunique() == 4


def test_segment_names_valid(df):
    valid_segments = {
        "High-Value Customers", "Premium / Loyal Buyers",
        "Discount Seekers / Budget Customers", "New / Developing Customers"
    }
    assert set(df["Segment_Name"].unique()) == valid_segments
