"""
Prediction Module
===================
Loads the saved production pipeline (scaler + K-Means model, Module 06) and
assigns new customers to a segment using the SAME feature-engineering logic
used during training (src/feature_engineering.py) — satisfying the
requirement that training and inference share identical logic.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import create_features, CLUSTER_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Cluster -> business segment metadata, derived from Module 07/08 analysis.
SEGMENT_INFO = {
    3: {
        "name": "High-Value Customers",
        "value": "High",
        "risk": "Medium",
        "avg_income": 76705, "avg_spend": 1430, "avg_recency": 50,
        "top_product": "Wines", "preferred_channel": "Store",
        "recommended_action": "VIP loyalty program, premium/exclusive product bundles, early access",
        "recommended_channel": "Email + personal outreach",
        "discount": "5-10% (low discount need, premium pricing candidate)",
        "campaign_response_rate": 35.0,
    },
    1: {
        "name": "Premium / Loyal Buyers",
        "value": "Medium-High",
        "risk": "Low",
        "avg_income": 59927, "avg_spend": 814, "avg_recency": 48,
        "top_product": "Wines", "preferred_channel": "Store",
        "recommended_action": "Cross-channel loyalty points, bundle discounts, cross-sell across channels",
        "recommended_channel": "Email + App/Web push",
        "discount": "10-15% on bundles",
        "campaign_response_rate": 9.7,
    },
    0: {
        "name": "Discount Seekers / Budget Customers",
        "value": "Low",
        "risk": "High",
        "avg_income": 42574, "avg_spend": 131, "avg_recency": 50,
        "top_product": "Wines", "preferred_channel": "Store",
        "recommended_action": "Family-size bundle discounts, value-led SMS/Email promotions",
        "recommended_channel": "SMS + Email",
        "discount": "15-25% (highest discount sensitivity)",
        "campaign_response_rate": 7.0,
    },
    2: {
        "name": "New / Developing Customers",
        "value": "Low (growth potential)",
        "risk": "High",
        "avg_income": 31891, "avg_spend": 105, "avg_recency": 49,
        "top_product": "Wines", "preferred_channel": "Store",
        "recommended_action": "Welcome discount, starter bundle, onboarding email/SMS series",
        "recommended_channel": "Email + SMS onboarding series",
        "discount": "10-20% welcome discount",
        "campaign_response_rate": 9.9,
    },
}


class SegmentationPipeline:
    """Loads the trained scaler + K-Means model once and exposes .predict()."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        pipeline_path = models_dir / "production_pipeline.pkl"
        artifacts = joblib.load(pipeline_path)
        self.feature_list = artifacts["feature_list"]
        self.scaler = artifacts["scaler"]
        self.model = artifacts["model"]
        self.k = artifacts["k"]

    def predict_from_raw(self, raw_record: dict) -> dict:
        """
        Accepts a dict of raw customer fields (as entered in the dashboard
        form or looked up from the dataset), engineers the required
        features, scales them, predicts the cluster, and returns a full
        result dict including business segment metadata.
        """
        df = pd.DataFrame([raw_record])
        engineered = create_features(df)
        X = engineered[self.feature_list]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_list)
        cluster = int(self.model.predict(X_scaled)[0])
        distances = self.model.transform(X_scaled)[0]
        confidence = float(1 - (distances[cluster] / (distances.sum() + 1e-9)))

        info = SEGMENT_INFO.get(cluster, {})
        return {
            "cluster": cluster,
            "segment_name": info.get("name", f"Cluster {cluster}"),
            "customer_value": info.get("value", "Unknown"),
            "retention_risk": info.get("risk", "Unknown"),
            "recommended_action": info.get("recommended_action", ""),
            "recommended_channel": info.get("recommended_channel", ""),
            "recommended_discount": info.get("discount", ""),
            "top_product": info.get("top_product", ""),
            "confidence_proxy": round(confidence, 3),
            "engineered_features": {f: float(engineered.iloc[0][f]) for f in self.feature_list},
        }


def load_pipeline() -> SegmentationPipeline:
    return SegmentationPipeline()
