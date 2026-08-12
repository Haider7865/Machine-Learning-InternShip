"""Main dashboard: assembles all 10 required sections into a navigable
Streamlit app."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.components.data_loader import load_data, SEGMENT_COLORS, SPEND_COLS
from app.components.filters import render_filters
from app.components import charts
from app.components.customer_lookup import render_customer_lookup, render_new_customer_prediction
from src.prediction import SEGMENT_INFO


SECTIONS = [
    "1. Executive Overview",
    "2. Customer Demographics",
    "3. Spending Analysis",
    "4. Product Analysis",
    "5. Purchase Channel Analysis",
    "6. Campaign Response",
    "7. Customer Segments",
    "8. Segment Comparison",
    "9. Individual Customer Profile",
    "10. Marketing Recommendations",
]


def kpi_card(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


def section_executive_overview(df):
    st.header("1. Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total Customers", f"{len(df):,}")
    kpi_card(c2, "Total Segments", df["Segment_Name"].nunique())
    kpi_card(c3, "Avg Spending", f"${df['Total_Spending'].mean():,.0f}")
    kpi_card(c4, "Total Spending", f"${df['Total_Spending'].sum():,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    kpi_card(c5, "Avg Income", f"${df['Income'].mean():,.0f}")
    kpi_card(c6, "Avg Purchase Frequency", f"{df['Total_Purchases'].mean():.1f}")
    kpi_card(c7, "Campaign Response Rate", f"{df['Response'].mean()*100:.1f}%")
    kpi_card(c8, "Avg Recency", f"{df['Recency'].mean():.0f} days")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Customer Segment Distribution**")
        st.plotly_chart(charts.segment_distribution_pie(df), use_container_width=True)
    with col2:
        st.markdown("**Revenue Contribution by Segment**")
        st.plotly_chart(charts.spending_by_segment_bar(df), use_container_width=True)


def section_demographics(df):
    st.header("2. Customer Demographics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Age Distribution**")
        st.plotly_chart(charts.age_distribution_hist(df), use_container_width=True)
    with col2:
        st.markdown("**Income Distribution by Segment**")
        st.plotly_chart(charts.income_by_segment_box(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Education Level**")
        edu_counts = df["Education"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Customers"]
        st.bar_chart(edu_counts.set_index("Education"))
    with col4:
        st.markdown("**Marital Status**")
        mar_counts = df["Marital_Status"].value_counts().reset_index()
        mar_counts.columns = ["Marital Status", "Customers"]
        st.bar_chart(mar_counts.set_index("Marital Status"))

    st.markdown("**Household Information**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Family Size", f"{df['Family_Size'].mean():.1f}")
    c2.metric("Avg Children per Household", f"{df['Total_Children'].mean():.1f}")
    c3.metric("Households with Children", f"{(df['Total_Children'] > 0).mean()*100:.0f}%")


def section_spending(df):
    st.header("3. Spending Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spending", f"${df['Total_Spending'].sum():,.0f}")
    c2.metric("Average Spending", f"${df['Total_Spending'].mean():,.0f}")
    c3.metric("Median Spending", f"${df['Total_Spending'].median():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Spending by Product Category**")
        st.plotly_chart(charts.product_category_spend_bar(df), use_container_width=True)
    with col2:
        st.markdown("**Average Spending by Segment**")
        st.plotly_chart(charts.spending_by_segment_bar(df), use_container_width=True)

    st.markdown("**Purchase Frequency by Segment**")
    freq = df.groupby("Segment_Name")["Total_Purchases"].mean().round(1).reset_index()
    st.dataframe(freq, use_container_width=True, hide_index=True)


def section_product(df):
    st.header("4. Product Analysis")
    labels = {"MntWines": "Wine", "MntFruits": "Fruit", "MntMeatProducts": "Meat",
              "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"}
    cols = st.columns(3)
    for i, (col_name, label) in enumerate(labels.items()):
        with cols[i % 3]:
            st.metric(f"{label} — Total Spend", f"${df[col_name].sum():,.0f}")

    st.markdown("**Spending by Product Category**")
    st.plotly_chart(charts.product_category_spend_bar(df), use_container_width=True)

    st.markdown("**Product Preference by Segment (avg spend, $)**")
    pref = df.groupby("Segment_Name")[SPEND_COLS].mean().round(1)
    pref.columns = [labels[c] for c in pref.columns]
    st.dataframe(pref, use_container_width=True)


def section_channel(df):
    st.header("5. Purchase Channel Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Purchases by Channel**")
        st.plotly_chart(charts.purchase_channel_pie(df), use_container_width=True)
    with col2:
        st.markdown("**Web Engagement vs Recency**")
        st.plotly_chart(charts.customer_activity_scatter(df), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Web Purchases", f"{df['NumWebPurchases'].mean():.1f}")
    c2.metric("Avg Store Purchases", f"{df['NumStorePurchases'].mean():.1f}")
    c3.metric("Avg Web Visits/Month", f"{df['NumWebVisitsMonth'].mean():.1f}")


def section_campaign(df):
    st.header("6. Campaign Response")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Campaign Acceptance", f"{df['Response'].mean()*100:.1f}%")
    c2.metric("Complaint Rate", f"{df['Complain'].mean()*100:.2f}%")
    c3.metric("Avg Campaigns Accepted", f"{df['Total_Campaign_Acceptance'].mean():.2f}")

    st.markdown("**Campaign Acceptance Rate (all 6 campaigns)**")
    st.plotly_chart(charts.campaign_response_bar(df), use_container_width=True)

    st.markdown("**Campaign Performance by Segment**")
    seg_resp = (df.groupby("Segment_Name")["Response"].mean() * 100).round(1).reset_index()
    seg_resp.columns = ["Segment", "Latest Campaign Response Rate (%)"]
    st.dataframe(seg_resp, use_container_width=True, hide_index=True)


def section_segments(df):
    st.header("7. Customer Segments")
    seg_summary = df.groupby("Segment_Name").agg(
        Customers=("ID", "count"), Avg_Income=("Income", "mean"),
        Avg_Spending=("Total_Spending", "mean"), Avg_Recency=("Recency", "mean"),
    ).round(1)
    seg_summary["Percentage"] = (seg_summary["Customers"] / len(df) * 100).round(1)
    st.dataframe(seg_summary, use_container_width=True)

    st.markdown("**Segment Characteristics**")
    for seg in sorted(df["Segment_Name"].unique()):
        cluster_id = df.loc[df["Segment_Name"] == seg, "Cluster"].iloc[0]
        info = SEGMENT_INFO.get(int(cluster_id), {})
        with st.expander(f"{seg} ({(df['Segment_Name']==seg).sum():,} customers)"):
            st.write(f"**Customer Value:** {info.get('value')}  |  **Retention Risk:** {info.get('risk')}")
            st.write(f"**Top Product:** {info.get('top_product')}  |  **Preferred Channel:** {info.get('preferred_channel')}")
            st.write(f"**Avg Income:** ${info.get('avg_income'):,}  |  **Avg Spend:** ${info.get('avg_spend'):,}")


def section_comparison(df):
    st.header("8. Segment Comparison")
    segments = sorted(df["Segment_Name"].unique())
    col1, col2 = st.columns(2)
    seg_a = col1.selectbox("Segment A", segments, index=0)
    seg_b = col2.selectbox("Segment B", segments, index=min(1, len(segments) - 1))

    st.plotly_chart(charts.segment_comparison_radar(df, seg_a, seg_b), use_container_width=True)

    metrics = ["Income", "Total_Spending", "Total_Purchases", "Recency", "Total_Campaign_Acceptance"]
    comp = df.groupby("Segment_Name")[metrics].mean().round(1)
    comp = comp.loc[[seg_a, seg_b]].T
    comp.columns = [seg_a, seg_b]
    st.markdown("**Side-by-Side Metrics**")
    st.dataframe(comp, use_container_width=True)

    csv = comp.to_csv().encode("utf-8")
    st.download_button("⬇ Download Comparison as CSV", csv, "segment_comparison.csv", "text/csv")


def section_individual(df):
    st.header("9. Individual Customer Profile")
    tab1, tab2 = st.tabs(["🔍 Look Up Existing Customer", "🧮 Analyze New Customer"])
    with tab1:
        render_customer_lookup(df)
    with tab2:
        render_new_customer_prediction()


def section_recommendations(df):
    st.header("10. Marketing Recommendations")
    st.caption("Recommendations generated in Module 08, grounded in each segment's actual behaviour.")
    for cluster_id, info in sorted(SEGMENT_INFO.items(), key=lambda x: -x[1]["avg_spend"]):
        with st.container(border=True):
            st.subheader(info["name"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Customer Value", info["value"])
            c2.metric("Retention Risk", info["risk"])
            c3.metric("Campaign Response Rate", f"{info['campaign_response_rate']}%")
            st.markdown(f"**Recommended Action:** {info['recommended_action']}")
            st.markdown(f"**Channel:** {info['recommended_channel']}  |  **Discount Strategy:** {info['discount']}")


def render_dashboard():
    df_full = load_data()
    st.sidebar.title("📊 Navigation")
    section = st.sidebar.radio("Go to section", SECTIONS, label_visibility="collapsed")
    st.sidebar.markdown("---")

    df = render_filters(df_full)

    if df.empty:
        st.warning("No customers match the current filter selection. Try widening your filters.")
        return

    section_map = {
        SECTIONS[0]: section_executive_overview,
        SECTIONS[1]: section_demographics,
        SECTIONS[2]: section_spending,
        SECTIONS[3]: section_product,
        SECTIONS[4]: section_channel,
        SECTIONS[5]: section_campaign,
        SECTIONS[6]: section_segments,
        SECTIONS[7]: section_comparison,
        SECTIONS[8]: section_individual,
        SECTIONS[9]: section_recommendations,
    }
    section_map[section](df)
