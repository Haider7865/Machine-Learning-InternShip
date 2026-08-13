"""Task 11 — Feature Engineering Testing
Verifies each engineered feature's calculation, and that the SAME
feature-engineering logic is used for both training data and live
prediction input (src.feature_engineering.create_features)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import create_features, CLUSTER_FEATURES


@pytest.fixture
def sample_customer():
    return pd.DataFrame([{
        "Year_Birth": 1990, "Income": 60000, "Recency": 20,
        "MntWines": 500, "MntFruits": 30, "MntMeatProducts": 200,
        "MntFishProducts": 40, "MntSweetProducts": 20, "MntGoldProds": 50,
        "NumWebPurchases": 5, "NumCatalogPurchases": 2, "NumStorePurchases": 8,
        "NumDealsPurchases": 3, "NumWebVisitsMonth": 4,
        "Marital_Status": "Married", "Kidhome": 1, "Teenhome": 0,
        "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 1,
        "AcceptedCmp4": 0, "AcceptedCmp5": 0, "Response": 1,
        "Dt_Customer": "2022-01-15",
    }])


def test_age_calculation(sample_customer):
    result = create_features(sample_customer)
    assert result.loc[0, "Customer_Age"] == 2026 - 1990


def test_family_size_calculation(sample_customer):
    result = create_features(sample_customer)
    # Married (+1) + self (+1) + 1 child = 3
    assert result.loc[0, "Family_Size"] == 3


def test_total_children(sample_customer):
    result = create_features(sample_customer)
    assert result.loc[0, "Total_Children"] == 1  # Kidhome(1) + Teenhome(0)


def test_total_spending(sample_customer):
    result = create_features(sample_customer)
    expected = 500 + 30 + 200 + 40 + 20 + 50
    assert result.loc[0, "Total_Spending"] == expected


def test_total_purchases(sample_customer):
    result = create_features(sample_customer)
    expected = 5 + 2 + 8 + 3  # web + catalog + store + deals
    assert result.loc[0, "Total_Purchases"] == expected


def test_campaign_acceptance(sample_customer):
    result = create_features(sample_customer)
    assert result.loc[0, "Total_Campaign_Acceptance"] == 2  # Cmp3 + Response


def test_digital_engagement(sample_customer):
    result = create_features(sample_customer)
    assert result.loc[0, "Digital_Engagement"] == 4 + 5  # web visits + web purchases


def test_deal_dependency(sample_customer):
    result = create_features(sample_customer)
    total_purchases = 5 + 2 + 8 + 3
    assert abs(result.loc[0, "Deal_Dependency"] - (3 / total_purchases)) < 1e-6


def test_customer_tenure_computed(sample_customer):
    result = create_features(sample_customer)
    assert result.loc[0, "Customer_Tenure"] > 0


def test_zero_purchase_customer_no_division_error():
    """Boundary case: a customer with zero purchases should not raise
    a division-by-zero error (Deal_Dependency / Avg_Spending_Per_Purchase)."""
    df = pd.DataFrame([{
        "Year_Birth": 1990, "Income": 0, "Recency": 0,
        "NumWebPurchases": 0, "NumCatalogPurchases": 0,
        "NumStorePurchases": 0, "NumDealsPurchases": 0,
    }])
    result = create_features(df)
    assert result.loc[0, "Deal_Dependency"] == 0
    assert result.loc[0, "Avg_Spending_Per_Purchase"] == 0


def test_all_cluster_features_present(sample_customer):
    result = create_features(sample_customer)
    for feature in CLUSTER_FEATURES:
        assert feature in result.columns, f"Missing required feature: {feature}"


def test_feature_engineering_consistent_across_calls(sample_customer):
    """Same logic must produce identical output on repeated calls (determinism
    required for training/prediction consistency)."""
    r1 = create_features(sample_customer)
    r2 = create_features(sample_customer)
    pd.testing.assert_frame_equal(r1[CLUSTER_FEATURES], r2[CLUSTER_FEATURES])
