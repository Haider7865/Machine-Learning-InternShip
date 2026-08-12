"""Sidebar filter controls. Every filter here updates the filtered dataframe
used by all dashboard sections."""

import streamlit as st


def render_filters(df):
    st.sidebar.header("🔎 Filters")

    segments = sorted(df["Segment_Name"].unique())
    selected_segments = st.sidebar.multiselect(
        "Customer Segment", segments, default=segments
    )

    age_min, age_max = int(df["Customer_Age"].min()), int(df["Customer_Age"].max())
    age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    income_min, income_max = float(df["Income"].min()), float(df["Income"].max())
    income_range = st.sidebar.slider(
        "Income Range ($)", income_min, income_max, (income_min, income_max)
    )

    education_opts = sorted(df["Education"].unique()) if "Education" in df.columns else []
    selected_education = st.sidebar.multiselect(
        "Education", education_opts, default=education_opts
    )

    marital_opts = sorted(df["Marital_Status"].unique()) if "Marital_Status" in df.columns else []
    selected_marital = st.sidebar.multiselect(
        "Marital Status", marital_opts, default=marital_opts
    )

    product_map = {
        "Wine": "MntWines", "Fruits": "MntFruits", "Meat": "MntMeatProducts",
        "Fish": "MntFishProducts", "Sweets": "MntSweetProducts", "Gold": "MntGoldProds",
    }
    selected_product = st.sidebar.selectbox(
        "Highlight Product Category (spend > $0)", ["All"] + list(product_map.keys())
    )

    channel_map = {"Web": "NumWebPurchases", "Store": "NumStorePurchases", "Catalog": "NumCatalogPurchases"}
    selected_channel = st.sidebar.selectbox(
        "Highlight Purchase Channel (purchases > 0)", ["All"] + list(channel_map.keys())
    )

    campaign_resp = st.sidebar.selectbox(
        "Campaign Response (latest)", ["All", "Responded", "Did Not Respond"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("↺ Reset All Filters"):
        st.rerun()

    filtered = df[
        df["Segment_Name"].isin(selected_segments)
        & df["Customer_Age"].between(age_range[0], age_range[1])
        & df["Income"].between(income_range[0], income_range[1])
    ]
    if selected_education:
        filtered = filtered[filtered["Education"].isin(selected_education)]
    if selected_marital:
        filtered = filtered[filtered["Marital_Status"].isin(selected_marital)]
    if selected_product != "All":
        filtered = filtered[filtered[product_map[selected_product]] > 0]
    if selected_channel != "All":
        filtered = filtered[filtered[channel_map[selected_channel]] > 0]
    if campaign_resp == "Responded":
        filtered = filtered[filtered["Response"] == 1]
    elif campaign_resp == "Did Not Respond":
        filtered = filtered[filtered["Response"] == 0]

    st.sidebar.markdown(f"**{len(filtered):,}** customers match current filters "
                         f"({len(filtered)/len(df)*100:.1f}% of total)")

    return filtered
