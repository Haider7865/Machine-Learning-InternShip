"""Customer lookup (by existing ID) and individual customer segmentation
(new customer form, using the live trained model)."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing import validate_customer_input, ValidationError
from src.prediction import load_pipeline


def render_customer_lookup(df):
    st.subheader("🔍 Individual Customer Profile — Lookup by ID")
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g. 5524")

    if st.button("Look Up Customer"):
        if not customer_id.strip():
            st.error("Please enter a Customer ID.")
            return
        try:
            cid = int(customer_id)
        except ValueError:
            st.error("Customer ID must be a number.")
            return

        match = df[df["ID"] == cid]
        if match.empty:
            st.warning(f"No customer found with ID {cid}. Try one from the table below.")
            st.dataframe(df[["ID", "Segment_Name"]].head(10), use_container_width=True)
            return

        row = match.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Segment", row["Segment_Name"])
        col2.metric("Income", f"${row['Income']:,.0f}")
        col3.metric("Total Spending", f"${row['Total_Spending']:,.0f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Age", f"{row['Customer_Age']:.0f}")
        col5.metric("Recency (days)", f"{row['Recency']:.0f}")
        col6.metric("Total Purchases", f"{row['Total_Purchases']:.0f}")

        st.markdown("**Demographics**")
        st.write(f"Education: {row.get('Education', 'N/A')} | "
                 f"Marital Status: {row.get('Marital_Status', 'N/A')} | "
                 f"Family Size: {row.get('Family_Size', 'N/A')}")

        st.markdown("**Recommended Marketing Action**")
        from src.prediction import SEGMENT_INFO
        info = SEGMENT_INFO.get(int(row["Cluster"]), {})
        st.info(f"**{info.get('recommended_action', 'N/A')}**  \n"
                f"Channel: {info.get('recommended_channel', 'N/A')}  \n"
                f"Discount strategy: {info.get('discount', 'N/A')}")


def render_new_customer_prediction():
    st.subheader("🧮 Analyze a New Customer")
    st.caption("Enter customer details below. The trained K-Means model will assign a segment live.")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.text_input("Age", value="35")
            income = st.text_input("Income ($)", value="50000")
        with c2:
            total_spending = st.text_input("Total Spending ($)", value="1200")
            web_purchases = st.text_input("Web Purchases", value="5")
        with c3:
            store_purchases = st.text_input("Store Purchases", value="8")
            recency = st.text_input("Recency (days)", value="20")

        submitted = st.form_submit_button("Analyze Customer", type="primary")

    if not submitted:
        return

    raw_values = {
        "age": age, "income": income, "total_spending": total_spending,
        "web_purchases": web_purchases, "store_purchases": store_purchases,
        "recency": recency,
    }

    try:
        cleaned = validate_customer_input(raw_values)
    except ValidationError as e:
        st.error(f"⚠️ Input error: {e}")
        return

    pipeline = load_pipeline()
    year_birth = 2026 - int(cleaned["age"])
    record = {
        "Year_Birth": year_birth,
        "Income": cleaned["income"],
        "Recency": cleaned["recency"],
        "MntWines": cleaned["total_spending"] * 0.55,
        "MntMeatProducts": cleaned["total_spending"] * 0.25,
        "MntFruits": cleaned["total_spending"] * 0.05,
        "MntFishProducts": cleaned["total_spending"] * 0.06,
        "MntSweetProducts": cleaned["total_spending"] * 0.05,
        "MntGoldProds": cleaned["total_spending"] * 0.04,
        "NumWebPurchases": cleaned["web_purchases"],
        "NumStorePurchases": cleaned["store_purchases"],
        "NumCatalogPurchases": 1,
        "NumDealsPurchases": 1,
        "NumWebVisitsMonth": 4,
        "Marital_Status": "Married",
        "Kidhome": 0, "Teenhome": 0,
        "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0,
        "AcceptedCmp4": 0, "AcceptedCmp5": 0, "Response": 0,
    }

    result = pipeline.predict_from_raw(record)

    st.success("Customer analyzed successfully.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customer Segment", result["segment_name"])
    col2.metric("Customer Value", result["customer_value"])
    col3.metric("Retention Risk", result["retention_risk"])

    st.markdown("**Recommended Action**")
    st.info(f"{result['recommended_action']}")
    col4, col5 = st.columns(2)
    col4.markdown(f"**Recommended Channel:** {result['recommended_channel']}")
    col5.markdown(f"**Recommended Product Focus:** {result['top_product']}")
    st.caption(f"Model confidence proxy: {result['confidence_proxy']:.1%} "
               f"(relative closeness to assigned cluster centroid)")
