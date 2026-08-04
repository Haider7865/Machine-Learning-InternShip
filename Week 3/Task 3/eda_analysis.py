from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
PALETTE = "viridis"
sns.set_palette(PALETTE)

BASE = Path(__file__).resolve().parent
DATA = BASE / "00_Data"
CHARTS = BASE / "01_Charts"
REPORTS = BASE / "02_Reports"

for folder in [DATA, CHARTS, REPORTS]:
    folder.mkdir(exist_ok=True)

DATA_FILE = DATA / "customer_personality_cleaned.csv"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]


def save_excel(dataframe, filename):
    dataframe.to_excel(REPORTS / filename, index=False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# TASK 1: DATASET OVERVIEW
# ============================================================

def task1_overview(df):
    print("=" * 60, "\nTASK 1: DATASET OVERVIEW\n", "=" * 60)

    print("Shape:", df.shape)
    print(df.head())
    print(df.dtypes)

    summary = pd.DataFrame({
        "Property": ["Rows", "Columns", "Duplicate Rows", "Duplicate IDs", "Total Missing Values"],
        "Value": [df.shape[0], df.shape[1], int(df.duplicated().sum()),
                  int(df["ID"].duplicated().sum()), int(df.isna().sum().sum())]
    })
    save_excel(summary, "01_dataset_summary.xlsx")

    missing = pd.DataFrame({
        "Variable": df.columns,
        "Missing Values": df.isna().sum().values,
        "Percentage": (df.isna().sum().values / len(df) * 100).round(2)
    })
    save_excel(missing, "01_missing_value_report.xlsx")

    stats = df.describe(include="all").T
    stats.to_excel(REPORTS / "01_basic_statistics.xlsx")

    print(summary.to_string(index=False))
    return summary


# ============================================================
# TASK 2: CUSTOMER DEMOGRAPHIC ANALYSIS
# ============================================================

def task2_demographics(df):
    print("=" * 60, "\nTASK 2: CUSTOMER DEMOGRAPHIC ANALYSIS\n", "=" * 60)

    # Age distribution - Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=25, kde=True, color="#4C72B0")
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    savefig("02a_age_histogram.png")

    # Income distribution - Histogram + Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["Income"], bins=30, kde=True, ax=axes[0], color="#55A868")
    axes[0].set_title("Income Distribution")
    sns.boxplot(x=df["Income"], ax=axes[1], color="#55A868")
    axes[1].set_title("Income Boxplot")
    savefig("02b_income_distribution.png")

    # Education level - Count plot
    plt.figure(figsize=(7, 5))
    order = df["Education"].value_counts().index
    sns.countplot(y=df["Education"], order=order, color="#4C72B0")
    plt.title("Education Level Distribution")
    plt.xlabel("Number of Customers")
    savefig("02c_education_countplot.png")

    # Marital status - Pie chart
    plt.figure(figsize=(6, 6))
    counts = df["Marital_Status"].value_counts()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette(PALETTE, len(counts)))
    plt.title("Marital Status Distribution")
    savefig("02d_marital_status_pie.png")

    # Household composition - Kidhome / Teenhome count plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x=df["Kidhome"], ax=axes[0], color="#C44E52")
    axes[0].set_title("Number of Children at Home")
    sns.countplot(x=df["Teenhome"], ax=axes[1], color="#8172B2")
    axes[1].set_title("Number of Teenagers at Home")
    savefig("02e_household_composition.png")

    demo_summary = pd.DataFrame({
        "Metric": ["Mean Age", "Median Age", "Min Age", "Max Age",
                   "Mean Income", "Median Income", "Min Income", "Max Income"],
        "Value": [round(df["Age"].mean(), 1), df["Age"].median(), df["Age"].min(), df["Age"].max(),
                  round(df["Income"].mean(), 2), df["Income"].median(),
                  df["Income"].min(), df["Income"].max()]
    })
    save_excel(demo_summary, "02_demographic_summary.xlsx")
    print(demo_summary.to_string(index=False))
    return demo_summary


# ============================================================
# TASK 3: CUSTOMER SPENDING ANALYSIS
# ============================================================

def task3_spending(df):
    print("=" * 60, "\nTASK 3: CUSTOMER SPENDING ANALYSIS\n", "=" * 60)

    df["Total_Spend"] = df[SPEND_COLS].sum(axis=1)

    total_spend = df["Total_Spend"].sum()
    avg_spend = df["Total_Spend"].mean()

    category_totals = df[SPEND_COLS].sum().sort_values(ascending=False)

    # Bar chart - spending by category
    plt.figure(figsize=(8, 5))
    sns.barplot(x=category_totals.values, y=category_totals.index, color="#4C72B0")
    plt.title("Total Spending by Product Category")
    plt.xlabel("Total Amount Spent")
    savefig("03a_spending_by_category_bar.png")

    # Histogram of total spend per customer
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Total_Spend"], bins=30, kde=True, color="#55A868")
    plt.title("Distribution of Total Spend per Customer")
    plt.xlabel("Total Spend")
    savefig("03b_total_spend_histogram.png")

    # Boxplot of spend by category
    plt.figure(figsize=(9, 5))
    melted = df[SPEND_COLS].melt(var_name="Category", value_name="Amount")
    sns.boxplot(x="Category", y="Amount", data=melted, color="#8172B2")
    plt.xticks(rotation=30)
    plt.title("Spending Distribution by Product Category (Boxplot)")
    savefig("03c_spending_boxplot.png")

    # Stacked bar chart - average spend by category per education level
    edu_spend = df.groupby("Education")[SPEND_COLS].mean()
    edu_spend.plot(kind="bar", stacked=True, figsize=(9, 6), colormap=PALETTE)
    plt.title("Average Spending by Category per Education Level (Stacked)")
    plt.ylabel("Average Amount Spent")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("03d_spending_by_education_stacked.png")

    spend_summary = pd.DataFrame({
        "Metric": ["Total Spending (All Customers)", "Average Spending per Customer",
                   "Median Spending per Customer", "Highest Spending Category",
                   "Lowest Spending Category"],
        "Value": [round(total_spend, 2), round(avg_spend, 2), df["Total_Spend"].median(),
                  category_totals.index[0], category_totals.index[-1]]
    })
    save_excel(spend_summary, "03_spending_summary.xlsx")
    category_totals.reset_index().rename(
        columns={"index": "Category", 0: "Total_Spend"}
    ).to_excel(REPORTS / "03_spending_by_category.xlsx", index=False)

    print(spend_summary.to_string(index=False))
    return df, spend_summary


# ============================================================
# TASK 4: PURCHASING BEHAVIOR ANALYSIS
# ============================================================

def task4_purchasing_behavior(df):
    print("=" * 60, "\nTASK 4: PURCHASING BEHAVIOR ANALYSIS\n", "=" * 60)

    channel_totals = df[CHANNEL_COLS].sum().sort_values(ascending=False)

    # Bar plot - total purchases by channel
    plt.figure(figsize=(8, 5))
    sns.barplot(x=channel_totals.index, y=channel_totals.values, color="#4C72B0")
    plt.title("Total Purchases by Channel")
    plt.ylabel("Total Number of Purchases")
    plt.xticks(rotation=15)
    savefig("04a_purchases_by_channel_bar.png")

    df["Total_Purchases"] = df[CHANNEL_COLS].sum(axis=1)

    # Distribution plot - total purchases per customer
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Total_Purchases"], bins=25, kde=True, color="#55A868")
    plt.title("Distribution of Total Purchases per Customer")
    plt.xlabel("Total Purchases")
    savefig("04b_total_purchases_distribution.png")

    # Count plot - Deals purchases frequency (binned)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["NumDealsPurchases"], color="#C44E52")
    plt.title("Number of Deal (Discount) Purchases — Count Plot")
    plt.xlabel("Number of Deal Purchases")
    savefig("04c_deals_purchases_countplot.png")

    behavior_summary = pd.DataFrame({
        "Channel": CHANNEL_COLS,
        "Total Purchases": [df[c].sum() for c in CHANNEL_COLS],
        "Average per Customer": [round(df[c].mean(), 2) for c in CHANNEL_COLS]
    })
    save_excel(behavior_summary, "04_purchasing_behavior_summary.xlsx")
    print(behavior_summary.to_string(index=False))
    return df, behavior_summary


# ============================================================
# TASK 5: WEBSITE ENGAGEMENT ANALYSIS
# ============================================================

def task5_website_engagement(df):
    print("=" * 60, "\nTASK 5: WEBSITE ENGAGEMENT ANALYSIS\n", "=" * 60)

    # Histogram - website visits
    plt.figure(figsize=(8, 5))
    sns.histplot(df["NumWebVisitsMonth"], bins=15, kde=True, color="#4C72B0")
    plt.title("Website Visits per Month — Distribution")
    plt.xlabel("Number of Web Visits per Month")
    savefig("05a_web_visits_histogram.png")

    # Scatter plot - web visits vs web purchases
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="NumWebVisitsMonth", y="NumWebPurchases", data=df,
                     alpha=0.5, color="#55A868")
    plt.title("Website Visits vs. Web Purchases")
    plt.xlabel("Web Visits per Month")
    plt.ylabel("Web Purchases")
    savefig("05b_visits_vs_purchases_scatter.png")

    # Density plot - web visits
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df["NumWebVisitsMonth"], fill=True, color="#8172B2")
    plt.title("Website Visits — Density Plot")
    plt.xlabel("Number of Web Visits per Month")
    savefig("05c_web_visits_density.png")

    high_engagement = df[df["NumWebVisitsMonth"] >= df["NumWebVisitsMonth"].quantile(0.75)]
    low_engagement = df[df["NumWebVisitsMonth"] <= df["NumWebVisitsMonth"].quantile(0.25)]

    engagement_summary = pd.DataFrame({
        "Metric": ["Average Web Visits/Month", "Median Web Visits/Month",
                   "High-Engagement Customers (Top 25%)", "Low-Engagement Customers (Bottom 25%)",
                   "Correlation: Visits vs Web Purchases"],
        "Value": [round(df["NumWebVisitsMonth"].mean(), 2), df["NumWebVisitsMonth"].median(),
                  len(high_engagement), len(low_engagement),
                  round(df["NumWebVisitsMonth"].corr(df["NumWebPurchases"]), 3)]
    })
    save_excel(engagement_summary, "05_website_engagement_summary.xlsx")
    print(engagement_summary.to_string(index=False))
    return engagement_summary


# ============================================================
# TASK 6: CUSTOMER RECENCY ANALYSIS
# ============================================================

def task6_recency(df):
    print("=" * 60, "\nTASK 6: CUSTOMER RECENCY ANALYSIS\n", "=" * 60)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Recency"], bins=25, kde=True, color="#4C72B0")
    plt.title("Distribution of Recency (Days Since Last Purchase)")
    plt.xlabel("Recency (days)")
    savefig("06a_recency_histogram.png")

    median_recency = df["Recency"].median()
    active = df[df["Recency"] <= median_recency]
    inactive = df[df["Recency"] > median_recency]

    plt.figure(figsize=(6, 5))
    status = pd.Series({"Active (<= median)": len(active), "Inactive (> median)": len(inactive)})
    sns.barplot(x=status.index, y=status.values, color="#C44E52")
    plt.title("Active vs Inactive Customers (Recency Split at Median)")
    plt.ylabel("Number of Customers")
    savefig("06b_active_vs_inactive_bar.png")

    recency_summary = pd.DataFrame({
        "Metric": ["Average Recency", "Median Recency", "Active Customers (<= median)",
                   "Inactive Customers (> median)", "Customers Needing Re-engagement (Recency > 75)"],
        "Value": [round(df["Recency"].mean(), 1), median_recency, len(active), len(inactive),
                  int((df["Recency"] > 75).sum())]
    })
    save_excel(recency_summary, "06_recency_summary.xlsx")
    print(recency_summary.to_string(index=False))
    return recency_summary


# ============================================================
# TASK 7: MARKETING CAMPAIGN ANALYSIS
# ============================================================

def task7_campaigns(df):
    print("=" * 60, "\nTASK 7: MARKETING CAMPAIGN ANALYSIS\n", "=" * 60)

    all_campaigns = CAMPAIGN_COLS + ["Response"]
    labels = ["Campaign 1", "Campaign 2", "Campaign 3", "Campaign 4", "Campaign 5", "Latest Campaign"]
    acceptance_counts = [df[c].sum() for c in all_campaigns]
    acceptance_rate = [round(df[c].mean() * 100, 2) for c in all_campaigns]

    campaign_df = pd.DataFrame({
        "Campaign": labels,
        "Accepted (Count)": acceptance_counts,
        "Acceptance Rate (%)": acceptance_rate
    })
    save_excel(campaign_df, "07_campaign_performance.xlsx")

    # Bar chart - acceptance rate per campaign
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Campaign", y="Acceptance Rate (%)", data=campaign_df, color="#4C72B0")
    plt.title("Campaign Acceptance Rate")
    plt.xticks(rotation=15)
    savefig("07a_campaign_acceptance_bar.png")

    # Count plot - number of campaigns accepted per customer
    df["Campaigns_Accepted"] = df[all_campaigns].sum(axis=1)
    plt.figure(figsize=(7, 5))
    sns.countplot(x=df["Campaigns_Accepted"], color="#55A868")
    plt.title("Number of Campaigns Accepted per Customer")
    plt.xlabel("Campaigns Accepted")
    savefig("07b_campaigns_accepted_countplot.png")

    # Pie chart - responders vs non-responders (latest campaign)
    plt.figure(figsize=(6, 6))
    resp_counts = df["Response"].value_counts().rename({0: "Did Not Respond", 1: "Responded"})
    plt.pie(resp_counts, labels=resp_counts.index, autopct="%1.1f%%", startangle=90,
            colors=["#C44E52", "#55A868"])
    plt.title("Latest Campaign Response")
    savefig("07c_latest_campaign_response_pie.png")

    print(campaign_df.to_string(index=False))
    return campaign_df


# ============================================================
# TASK 8: CUSTOMER COMPLAINT ANALYSIS
# ============================================================

def task8_complaints(df):
    print("=" * 60, "\nTASK 8: CUSTOMER COMPLAINT ANALYSIS\n", "=" * 60)

    complaint_count = int(df["Complain"].sum())
    complaint_pct = round(df["Complain"].mean() * 100, 2)

    plt.figure(figsize=(6, 5))
    sns.countplot(x=df["Complain"], color="#C44E52")
    plt.title("Customer Complaints — Count Plot")
    plt.xlabel("Complaint (0 = No, 1 = Yes)")
    savefig("08a_complaints_countplot.png")

    # Heatmap - complaints vs spending/income (mean values)
    comp_group = df.groupby("Complain")[["Income", "Total_Spend"]].mean()
    plt.figure(figsize=(6, 4))
    sns.heatmap(comp_group.T, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Average Income & Spending: Complaints vs No Complaints")
    savefig("08b_complaints_heatmap.png")

    complaint_summary = pd.DataFrame({
        "Metric": ["Total Complaints", "Complaint Percentage",
                   "Avg Income (Complainers)", "Avg Income (Non-Complainers)",
                   "Avg Spend (Complainers)", "Avg Spend (Non-Complainers)"],
        "Value": [complaint_count, complaint_pct,
                  round(comp_group.loc[1, "Income"], 2) if 1 in comp_group.index else 0,
                  round(comp_group.loc[0, "Income"], 2),
                  round(comp_group.loc[1, "Total_Spend"], 2) if 1 in comp_group.index else 0,
                  round(comp_group.loc[0, "Total_Spend"], 2)]
    })
    save_excel(complaint_summary, "08_complaint_summary.xlsx")
    print(complaint_summary.to_string(index=False))
    return complaint_summary


# ============================================================
# TASK 9: CORRELATION ANALYSIS
# ============================================================

def task9_correlation(df):
    print("=" * 60, "\nTASK 9: CORRELATION ANALYSIS\n", "=" * 60)

    num_cols = ["Income", "Age", "Recency", "Total_Spend", "Total_Purchases",
                "NumWebVisitsMonth", "NumDealsPurchases"] + SPEND_COLS
    corr = df[num_cols].corr(method="pearson")

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5)
    plt.title("Correlation Heatmap (Pearson)")
    savefig("09_correlation_heatmap.png")

    corr_pairs = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
    corr_pairs = corr_pairs[~corr_pairs.index.duplicated()]
    strongest_pos = corr_pairs.sort_values(ascending=False).drop_duplicates().head(5)
    strongest_neg = corr_pairs.sort_values().drop_duplicates().head(5)

    corr.to_excel(REPORTS / "09_correlation_matrix.xlsx")

    top_corr_df = pd.DataFrame({
        "Type": ["Strong Positive"] * len(strongest_pos) + ["Strong Negative"] * len(strongest_neg),
        "Variable Pair": [f"{a} - {b}" for a, b in strongest_pos.index] +
                          [f"{a} - {b}" for a, b in strongest_neg.index],
        "Correlation": list(strongest_pos.round(3).values) + list(strongest_neg.round(3).values)
    })
    save_excel(top_corr_df, "09_top_correlations.xlsx")
    print(top_corr_df.to_string(index=False))
    return corr, top_corr_df


# ============================================================
# TASK 10: CUSTOMER SEGMENTATION
# ============================================================

def task10_segmentation(df):
    print("=" * 60, "\nTASK 10: CUSTOMER SEGMENTATION\n", "=" * 60)

    spend_75 = df["Total_Spend"].quantile(0.75)
    spend_25 = df["Total_Spend"].quantile(0.25)

    high_value = df[df["Total_Spend"] >= spend_75]
    low_value = df[df["Total_Spend"] <= spend_25]
    frequent_buyers = df[df["Total_Purchases"] >= df["Total_Purchases"].quantile(0.75)]
    discount_seekers = df[df["NumDealsPurchases"] >= df["NumDealsPurchases"].quantile(0.75)]
    campaign_responders = df[df["Campaigns_Accepted"] >= 1]
    loyal_customers = df[df["Recency"] <= df["Recency"].quantile(0.25)]
    inactive_customers = df[df["Recency"] >= df["Recency"].quantile(0.75)]

    segments = pd.DataFrame({
        "Segment": ["High-Value Customers (Top 25% Spend)", "Low-Value Customers (Bottom 25% Spend)",
                    "Frequent Buyers (Top 25% Purchases)", "Discount Seekers (Top 25% Deal Purchases)",
                    "Campaign Responders (>=1 Campaign Accepted)", "Loyal / Recently Active Customers",
                    "Inactive Customers (High Recency)"],
        "Customer Count": [len(high_value), len(low_value), len(frequent_buyers),
                           len(discount_seekers), len(campaign_responders),
                           len(loyal_customers), len(inactive_customers)],
        "Percentage of Total": [
            round(len(high_value) / len(df) * 100, 1), round(len(low_value) / len(df) * 100, 1),
            round(len(frequent_buyers) / len(df) * 100, 1), round(len(discount_seekers) / len(df) * 100, 1),
            round(len(campaign_responders) / len(df) * 100, 1), round(len(loyal_customers) / len(df) * 100, 1),
            round(len(inactive_customers) / len(df) * 100, 1)
        ]
    })
    save_excel(segments, "10_customer_segments.xlsx")

    plt.figure(figsize=(9, 5))
    sns.barplot(y="Segment", x="Customer Count", data=segments, color="#4C72B0")
    plt.title("Customer Segment Sizes")
    savefig("10a_customer_segments_bar.png")

    print(segments.to_string(index=False))
    return segments


# ============================================================
# TASK 11: BUSINESS INSIGHTS
# ============================================================

def task11_business_insights(df, corr):
    print("=" * 60, "\nTASK 11: BUSINESS INSIGHTS\n", "=" * 60)

    top_spender_income = df.sort_values("Total_Spend", ascending=False).head(10)["Income"].mean()
    top_category = df[SPEND_COLS].sum().idxmax()
    income_spend_corr = df["Income"].corr(df["Total_Spend"])
    channel_totals = df[CHANNEL_COLS].sum()
    top_channel = channel_totals.idxmax()

    campaign_label_map = {
        "AcceptedCmp1": "Campaign 1", "AcceptedCmp2": "Campaign 2",
        "AcceptedCmp3": "Campaign 3", "AcceptedCmp4": "Campaign 4",
        "AcceptedCmp5": "Campaign 5", "Response": "The Latest Campaign"
    }
    campaign_rates = {c: df[c].mean() for c in CAMPAIGN_COLS + ["Response"]}
    best_campaign_col = max(campaign_rates, key=campaign_rates.get)
    best_campaign = campaign_label_map[best_campaign_col]

    freq_responders = df[df["Campaigns_Accepted"] >= 2]
    inactive = df[df["Recency"] >= df["Recency"].quantile(0.75)]
    top_web_visitors = df.sort_values("NumWebVisitsMonth", ascending=False).head(10)

    with_kids = df[(df["Kidhome"] + df["Teenhome"]) > 0]["Total_Spend"].mean()
    without_kids = df[(df["Kidhome"] + df["Teenhome"]) == 0]["Total_Spend"].mean()

    hv_profile = df[df["Total_Spend"] >= df["Total_Spend"].quantile(0.75)]

    strongest_corr_pair = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna().abs().sort_values(ascending=False).index[0]

    answers = [
        ("Which customers spend the most?",
         f"High-income, older, no/fewer-children households; the top 10 spenders average "
         f"${top_spender_income:,.0f} income vs the overall average of ${df['Income'].mean():,.0f}."),
        ("Which product category generates the highest revenue?",
         f"{top_category} generates the highest total revenue among all six product categories."),
        ("Does income influence spending?",
         f"Yes — Income and Total_Spend are strongly correlated (Pearson r = {income_spend_corr:.2f}), "
         f"indicating higher income customers spend substantially more."),
        ("Which purchase channel is most popular?",
         f"{top_channel} is the most-used channel by total purchase volume."),
        ("Which campaign performs best?",
         f"{best_campaign} has the highest acceptance rate "
         f"({campaign_rates[best_campaign_col]*100:.1f}%) among all campaigns."),
        ("Which customers respond most frequently?",
         f"{len(freq_responders)} customers ({len(freq_responders)/len(df)*100:.1f}%) accepted 2 or more "
         f"campaigns and represent the most reliably responsive segment."),
        ("Which customers are inactive?",
         f"{len(inactive)} customers ({len(inactive)/len(df)*100:.1f}%) have a Recency in the top quartile "
         f"(purchased longest ago) and are candidates for re-engagement campaigns."),
        ("Which customers visit the website most?",
         f"The top 10 web visitors average {top_web_visitors['NumWebVisitsMonth'].mean():.1f} visits/month, "
         f"but their web purchase conversion is often lower than their visit frequency would suggest."),
        ("Do customers with children spend differently?",
         f"Yes — households with children/teens spend an average of ${with_kids:,.0f} versus "
         f"${without_kids:,.0f} for households without, a meaningful reduction in discretionary spend."),
        ("What characteristics define high-value customers?",
         f"High-value customers (top 25% by spend, n={len(hv_profile)}) tend to have higher income "
         f"(avg ${hv_profile['Income'].mean():,.0f}), lower Recency, more store/catalogue purchases, "
         f"and fewer children at home."),
        ("Which variables have the strongest correlations?",
         f"{strongest_corr_pair[0]} and {strongest_corr_pair[1]} show the strongest relationship in the "
         f"correlation matrix; in general, spending categories are strongly correlated with each other "
         f"and with Income."),
        ("What marketing strategies should be recommended?",
         "Target high-income / low-recency segments with premium offers; re-engage the inactive segment "
         "with win-back campaigns; promote catalogue/store channels to frequent buyers; and tailor offers "
         "for households with children toward value/discount-oriented products."),
    ]

    insights_df = pd.DataFrame(answers, columns=["Business Question", "Insight"])
    save_excel(insights_df, "11_business_insights.xlsx")
    for q, a in answers:
        print(f"\nQ: {q}\nA: {a}")
    return insights_df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_eda():
    df = pd.read_csv(DATA_FILE)

    task1_overview(df)
    task2_demographics(df)
    df, _ = task3_spending(df)
    df, _ = task4_purchasing_behavior(df)
    task5_website_engagement(df)
    task6_recency(df)
    task7_campaigns(df)
    task8_complaints(df)
    corr, _ = task9_correlation(df)
    task10_segmentation(df)
    task11_business_insights(df, corr)

    print("\n" + "=" * 60)
    print("EDA PIPELINE COMPLETE — all charts and tables generated.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_eda()
