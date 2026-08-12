"""
Customer Segmentation Dashboard — Application Entry Point
=============================================================
Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dashboard import render_dashboard

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Customer Personality Analysis — Segmentation Dashboard")
st.caption(
    "AI Lab 99 Internship Program · Customer Personality Analysis & Segmentation Project · "
    "Live K-Means model (K=4) trained on 2,237 customers"
)

try:
    render_dashboard()
except Exception as e:
    st.error(f"⚠️ The application encountered an unexpected error: {e}")
    st.info("Please check that the data and model files exist in their expected locations "
            "(data/processed/ and models/) and reload the page.")
