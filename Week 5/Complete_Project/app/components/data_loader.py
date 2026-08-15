"""Data loading utilities for the Streamlit dashboard, with caching."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "customer_segments_with_strategy.csv"

SEGMENT_COLORS = {
    "High-Value Customers": "#C44E52",
    "Premium / Loyal Buyers": "#4C72B0",
    "Discount Seekers / Budget Customers": "#8172B2",
    "New / Developing Customers": "#55A868",
}

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]


@st.cache_data(show_spinner="Loading customer data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df
