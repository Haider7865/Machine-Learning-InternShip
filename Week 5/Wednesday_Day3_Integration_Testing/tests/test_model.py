"""Task 12 — Model Testing"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prediction import load_pipeline, SegmentationPipeline


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


def test_model_loads(pipeline):
    assert isinstance(pipeline, SegmentationPipeline)
    assert pipeline.model is not None
    assert pipeline.scaler is not None


def test_feature_list_matches_training(pipeline):
    expected = [
        "Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
        "Family_Size", "Total_Children", "Total_Purchases",
        "Total_Campaign_Acceptance", "NumWebPurchases", "NumStorePurchases",
        "NumCatalogPurchases",
    ]
    assert pipeline.feature_list == expected


def test_k_equals_4(pipeline):
    assert pipeline.k == 4


def test_prediction_returns_valid_cluster(pipeline):
    record = {
        "Year_Birth": 1985, "Income": 60000, "Recency": 20,
        "MntWines": 500, "MntMeatProducts": 200, "NumWebPurchases": 5,
        "NumStorePurchases": 8, "NumCatalogPurchases": 2, "NumDealsPurchases": 2,
        "Marital_Status": "Married", "Kidhome": 1, "Teenhome": 0,
    }
    result = pipeline.predict_from_raw(record)
    assert result["cluster"] in [0, 1, 2, 3]
    assert result["segment_name"] != ""


def test_prediction_consistency(pipeline):
    """The same input must always produce the same cluster (deterministic
    K-Means prediction with a fixed, already-trained model)."""
    record = {
        "Year_Birth": 1970, "Income": 90000, "Recency": 10,
        "MntWines": 900, "MntMeatProducts": 400, "NumWebPurchases": 4,
        "NumStorePurchases": 10, "NumCatalogPurchases": 6,
    }
    r1 = pipeline.predict_from_raw(record)
    r2 = pipeline.predict_from_raw(record)
    assert r1["cluster"] == r2["cluster"]


def test_high_income_high_spend_predicts_high_value(pipeline):
    """A customer with clearly high income/spend/campaign-engagement should
    be assigned to the High-Value segment (sanity/regression check)."""
    record = {
        "Year_Birth": 1965, "Income": 95000, "Recency": 10,
        "MntWines": 950, "MntMeatProducts": 450, "MntGoldProds": 90,
        "MntFishProducts": 60, "MntFruits": 40, "MntSweetProducts": 30,
        "NumWebPurchases": 4, "NumStorePurchases": 10, "NumCatalogPurchases": 7,
        "NumDealsPurchases": 1, "AcceptedCmp5": 1, "Response": 1,
    }
    result = pipeline.predict_from_raw(record)
    assert result["segment_name"] == "High-Value Customers"


def test_correct_feature_order_enforced(pipeline):
    """Passing features out of order in the source dict must not change the
    prediction, since prediction.py always selects by the trained
    feature_list order rather than dict/column order."""
    record_a = {"Year_Birth": 1990, "Income": 40000, "Recency": 30,
                "MntWines": 50, "NumStorePurchases": 3, "NumWebPurchases": 2}
    r1 = pipeline.predict_from_raw(record_a)
    # Re-run with same values in a differently-ordered dict
    record_b = {"NumWebPurchases": 2, "NumStorePurchases": 3, "Recency": 30,
                "Income": 40000, "Year_Birth": 1990, "MntWines": 50}
    r2 = pipeline.predict_from_raw(record_b)
    assert r1["cluster"] == r2["cluster"]
