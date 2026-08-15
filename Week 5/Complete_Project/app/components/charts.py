"""Plotly chart builders. Every chart here exists to answer a specific
business question (per the module brief's 'no charts just to have charts'
rule) — each function docstring states the question it answers."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from app.components.data_loader import SEGMENT_COLORS, SPEND_COLS, CAMPAIGN_COLS


def segment_distribution_pie(df):
    """How many customers fall into each segment?"""
    counts = df["Segment_Name"].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig = px.pie(counts, names="Segment", values="Customers", hole=0.55,
                 color="Segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
    return fig


def spending_by_segment_bar(df):
    """Which segment spends the most, on average?"""
    avg = df.groupby("Segment_Name")["Total_Spending"].mean().reset_index()
    fig = px.bar(avg, x="Segment_Name", y="Total_Spending", color="Segment_Name",
                 color_discrete_map=SEGMENT_COLORS, text_auto=".0f")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Avg Total Spending ($)",
                       margin=dict(t=10, b=10), height=340)
    return fig


def income_by_segment_box(df):
    """How does income distribution differ across segments?"""
    fig = px.box(df, x="Segment_Name", y="Income", color="Segment_Name",
                 color_discrete_map=SEGMENT_COLORS)
    fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=10, b=10), height=340)
    return fig


def age_distribution_hist(df):
    """What is the age profile of the current filtered customer base?"""
    fig = px.histogram(df, x="Customer_Age", nbins=25, color_discrete_sequence=["#4C72B0"])
    fig.update_layout(xaxis_title="Age", yaxis_title="Customers", margin=dict(t=10, b=10), height=320)
    return fig


def product_category_spend_bar(df):
    """Which product categories generate the most revenue?"""
    totals = df[SPEND_COLS].sum().sort_values(ascending=False).reset_index()
    totals.columns = ["Category", "Total Spend"]
    totals["Category"] = totals["Category"].str.replace("Mnt", "")
    fig = px.bar(totals, x="Category", y="Total Spend", color="Category",
                 color_discrete_sequence=px.colors.qualitative.Set2, text_auto=".2s")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=340)
    return fig


def purchase_channel_pie(df):
    """Which purchase channel is most used by the current customer base?"""
    channels = {"Web": df["NumWebPurchases"].sum(), "Store": df["NumStorePurchases"].sum(),
                "Catalog": df["NumCatalogPurchases"].sum(), "Deals": df["NumDealsPurchases"].sum()}
    fig = px.pie(names=list(channels.keys()), values=list(channels.values()), hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(margin=dict(t=10, b=10), height=320)
    return fig


def campaign_response_bar(df):
    """How did acceptance rates differ across the six campaigns?"""
    rates = (df[CAMPAIGN_COLS].mean() * 100).reset_index()
    rates.columns = ["Campaign", "Acceptance Rate (%)"]
    rates["Campaign"] = ["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Latest"]
    fig = px.bar(rates, x="Campaign", y="Acceptance Rate (%)", color="Campaign",
                 color_discrete_sequence=px.colors.qualitative.Bold, text_auto=".1f")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=320)
    return fig


def recency_by_segment_bar(df):
    """Which segments are most at risk of disengagement (high Recency)?"""
    avg = df.groupby("Segment_Name")["Recency"].mean().reset_index()
    fig = px.bar(avg, x="Segment_Name", y="Recency", color="Segment_Name",
                 color_discrete_map=SEGMENT_COLORS, text_auto=".0f")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Avg Recency (days)",
                       margin=dict(t=10, b=10), height=320)
    return fig


def segment_comparison_radar(df, seg_a, seg_b):
    """How do two chosen segments compare across key behavioural dimensions?"""
    cols = ["Income", "Total_Spending", "Recency", "Total_Purchases", "Total_Campaign_Acceptance"]
    labels = ["Income", "Spending", "Recency", "Purchases", "Campaign Accept."]
    means = df.groupby("Segment_Name")[cols].mean()
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)

    fig = go.Figure()
    for seg in [seg_a, seg_b]:
        if seg in norm.index:
            vals = norm.loc[seg, cols].tolist()
            vals += vals[:1]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels + [labels[0]], fill='toself', name=seg,
                line_color=SEGMENT_COLORS.get(seg, "#333")
            ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       showlegend=True, margin=dict(t=30, b=10), height=420)
    return fig


def customer_activity_scatter(df):
    """Is there a relationship between website engagement and recency (activity)?"""
    fig = px.scatter(df, x="NumWebVisitsMonth", y="Recency", color="Segment_Name",
                      color_discrete_map=SEGMENT_COLORS, opacity=0.5,
                      labels={"NumWebVisitsMonth": "Web Visits/Month", "Recency": "Recency (days)"})
    fig.update_layout(margin=dict(t=10, b=10), height=340)
    return fig
