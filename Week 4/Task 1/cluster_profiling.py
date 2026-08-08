from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
PALETTE = "tab10"

BASE = Path(__file__).resolve().parent
DATA = BASE / "00_Data"
REPORTS = BASE / "02_Reports"
CHARTS = BASE / "03_Charts"
FINAL = BASE / "04_Final_Deliverables"

for folder in [DATA, REPORTS, CHARTS, FINAL]:
    folder.mkdir(exist_ok=True)

INPUT_FILE = DATA / "customer_segments_with_categories.csv"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]

CLUSTER_ORDER = None  # set after loading, sorted by cluster id
CLUSTER_COLORS = {}


def save_excel(dataframe, filename):
    dataframe.to_excel(REPORTS / filename, index=False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()


def load_data():
    df = pd.read_csv(INPUT_FILE)
    global CLUSTER_ORDER, CLUSTER_COLORS
    CLUSTER_ORDER = sorted(df["Cluster"].unique())
    palette = sns.color_palette(PALETTE, len(CLUSTER_ORDER))
    CLUSTER_COLORS = {c: palette[i] for i, c in enumerate(CLUSTER_ORDER)}
    return df


def cluster_label(df, c):
    name = df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0]
    return f"Cluster {c}: {name}"


# ============================================================
# ACTIVITY 1: CLUSTER STATISTICS & DEMOGRAPHIC ANALYSIS
# ============================================================

def activity1_demographics(df):
    print("=" * 60, "\nACTIVITY 1: CLUSTER STATISTICS & DEMOGRAPHIC ANALYSIS\n", "=" * 60)

    sizes = df["Cluster"].value_counts().sort_index()
    cluster_stats = pd.DataFrame({
        "Cluster": sizes.index,
        "Segment Name": [df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0] for c in sizes.index],
        "Size": sizes.values,
        "Percentage": (sizes.values / len(df) * 100).round(2),
    })
    save_excel(cluster_stats, "01_cluster_statistics.xlsx")
    print(cluster_stats.to_string(index=False))

    numeric_cols = ["Customer_Age", "Income", "Family_Size", "Total_Children"]
    demo_avg = df.groupby("Cluster")[numeric_cols].mean().round(1)
    demo_avg["Size"] = sizes
    save_excel(demo_avg.reset_index(), "01_demographic_averages.xlsx")

    education_dist = pd.crosstab(df["Cluster"], df["Education"], normalize="index").round(3) * 100
    save_excel(education_dist.reset_index(), "01_education_distribution.xlsx")

    marital_dist = pd.crosstab(df["Cluster"], df["Marital_Status"], normalize="index").round(3) * 100
    save_excel(marital_dist.reset_index(), "01_marital_status_distribution.xlsx")

    # Charts
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sns.boxplot(x="Cluster", y="Customer_Age", data=df, ax=axes[0, 0], palette=PALETTE)
    axes[0, 0].set_title("Customer Age by Cluster")
    sns.boxplot(x="Cluster", y="Income", data=df, ax=axes[0, 1], palette=PALETTE)
    axes[0, 1].set_title("Income by Cluster")
    sns.boxplot(x="Cluster", y="Family_Size", data=df, ax=axes[1, 0], palette=PALETTE)
    axes[1, 0].set_title("Family Size by Cluster")
    sns.boxplot(x="Cluster", y="Total_Children", data=df, ax=axes[1, 1], palette=PALETTE)
    axes[1, 1].set_title("Number of Children by Cluster")
    savefig("01a_demographic_boxplots.png")

    plt.figure(figsize=(10, 5))
    education_dist.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title("Education Level Distribution by Cluster (%)")
    plt.ylabel("Percentage")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("01b_education_by_cluster.png")

    plt.figure(figsize=(10, 5))
    marital_dist.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="plasma")
    plt.title("Marital Status Distribution by Cluster (%)")
    plt.ylabel("Percentage")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("01c_marital_status_by_cluster.png")

    print("\nDemographic comparison charts and reports saved.")
    return cluster_stats, demo_avg


# ============================================================
# ACTIVITY 2: SPENDING BEHAVIOR ANALYSIS
# ============================================================

def activity2_spending(df):
    print("\n" + "=" * 60, "\nACTIVITY 2: SPENDING BEHAVIOR ANALYSIS\n", "=" * 60)

    spend_avg = df.groupby("Cluster")[SPEND_COLS + ["Total_Spending"]].mean().round(1)
    save_excel(spend_avg.reset_index(), "02_spending_comparison.xlsx")
    print(spend_avg.to_string())

    highest_spend_cluster = spend_avg["Total_Spending"].idxmax()
    lowest_spend_cluster = spend_avg["Total_Spending"].idxmin()

    # Premium buyers: high spend on Wine + Gold (premium categories)
    premium_score = spend_avg["MntWines"].rank(pct=True) + spend_avg["MntGoldProds"].rank(pct=True)
    premium_cluster = premium_score.idxmax()
    budget_cluster = spend_avg["Total_Spending"].rank(pct=True).idxmin()

    product_preference = spend_avg[SPEND_COLS].idxmax(axis=1)

    spend_summary = pd.DataFrame({
        "Item": ["Highest Spending Cluster", "Lowest Spending Cluster",
                 "Premium Product Buyers (Wine+Gold)", "Budget-Conscious Cluster"],
        "Cluster": [highest_spend_cluster, lowest_spend_cluster, premium_cluster, budget_cluster],
        "Segment Name": [cluster_label(df, c).split(": ")[1] for c in
                          [highest_spend_cluster, lowest_spend_cluster, premium_cluster, budget_cluster]]
    })
    save_excel(spend_summary, "02_spending_extremes.xlsx")
    print("\n", spend_summary.to_string(index=False))

    pref_df = pd.DataFrame({"Cluster": product_preference.index,
                             "Top Product Category": product_preference.values})
    save_excel(pref_df, "02_product_preference_summary.xlsx")

    # Charts
    plt.figure(figsize=(9, 5))
    spend_avg["Total_Spending"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in spend_avg.index])
    plt.title("Average Total Spending by Cluster")
    plt.ylabel("Average Spend ($)")
    savefig("02a_total_spending_by_cluster.png")

    plt.figure(figsize=(10, 6))
    spend_avg[SPEND_COLS].plot(kind="bar", ax=plt.gca(), colormap="Set2")
    plt.title("Average Spending by Product Category and Cluster")
    plt.ylabel("Average Spend ($)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("02b_product_spending_by_cluster.png")

    plt.figure(figsize=(8, 6))
    spend_share = spend_avg[SPEND_COLS].div(spend_avg[SPEND_COLS].sum(axis=1), axis=0)
    sns.heatmap(spend_share, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Product Spending Share by Cluster (Row-Normalized)")
    savefig("02c_product_spending_heatmap.png")

    return spend_avg, product_preference


# ============================================================
# ACTIVITY 3: SHOPPING CHANNEL & ENGAGEMENT ANALYSIS
# ============================================================

def activity3_channels(df):
    print("\n" + "=" * 60, "\nACTIVITY 3: SHOPPING CHANNEL & CUSTOMER ENGAGEMENT ANALYSIS\n", "=" * 60)

    channel_avg = df.groupby("Cluster")[CHANNEL_COLS + ["NumWebVisitsMonth", "Recency"]].mean().round(1)
    save_excel(channel_avg.reset_index(), "03_channel_engagement_comparison.xlsx")
    print(channel_avg.to_string())

    digital_first = channel_avg["NumWebPurchases"].idxmax()
    store_oriented = channel_avg["NumStorePurchases"].idxmax()
    catalog_oriented = channel_avg["NumCatalogPurchases"].idxmax()
    deal_seeking = channel_avg["NumDealsPurchases"].idxmax()
    most_active = channel_avg["Recency"].idxmin()
    least_active = channel_avg["Recency"].idxmax()

    channel_id_summary = pd.DataFrame({
        "Behavior Type": ["Digital-First", "Store-Oriented", "Catalog-Oriented",
                           "Deal-Seeking", "Most Active (lowest Recency)", "Least Active (highest Recency)"],
        "Cluster": [digital_first, store_oriented, catalog_oriented, deal_seeking, most_active, least_active],
        "Segment Name": [cluster_label(df, c).split(": ")[1] for c in
                          [digital_first, store_oriented, catalog_oriented, deal_seeking, most_active, least_active]]
    })
    save_excel(channel_id_summary, "03_behavior_identification.xlsx")
    print("\n", channel_id_summary.to_string(index=False))

    # Charts
    plt.figure(figsize=(10, 6))
    channel_avg[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]].plot(
        kind="bar", ax=plt.gca(), colormap="Set1")
    plt.title("Average Purchases by Channel and Cluster")
    plt.ylabel("Average Number of Purchases")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("03a_channel_comparison.png")

    plt.figure(figsize=(8, 5))
    channel_avg["Recency"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in channel_avg.index])
    plt.title("Average Recency by Cluster (Lower = More Active)")
    plt.ylabel("Days Since Last Purchase")
    savefig("03b_recency_by_cluster.png")

    plt.figure(figsize=(8, 5))
    channel_avg["NumWebVisitsMonth"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in channel_avg.index])
    plt.title("Average Website Visits per Month by Cluster")
    savefig("03c_web_visits_by_cluster.png")

    return channel_avg


# ============================================================
# ACTIVITY 4: MARKETING CAMPAIGN ANALYSIS
# ============================================================

def activity4_campaigns(df):
    print("\n" + "=" * 60, "\nACTIVITY 4: MARKETING CAMPAIGN ANALYSIS\n", "=" * 60)

    campaign_rates = (df.groupby("Cluster")[CAMPAIGN_COLS].mean() * 100).round(1)
    campaign_rates["Complaint Rate (%)"] = (df.groupby("Cluster")["Complain"].mean() * 100).round(1)
    save_excel(campaign_rates.reset_index(), "04_campaign_effectiveness.xlsx")
    print(campaign_rates.to_string())

    overall_response_rate = campaign_rates["Response"].mean()
    campaign_responsive = campaign_rates["Response"].idxmax()
    marketing_resistant = campaign_rates[CAMPAIGN_COLS].mean(axis=1).idxmin()
    needs_reengagement = df.groupby("Cluster")["Recency"].mean().idxmax()

    marketing_insights = pd.DataFrame({
        "Insight": ["Most Campaign-Responsive Cluster", "Most Marketing-Resistant Cluster",
                    "Cluster Needing Re-engagement (highest Recency)"],
        "Cluster": [campaign_responsive, marketing_resistant, needs_reengagement],
        "Segment Name": [cluster_label(df, c).split(": ")[1] for c in
                          [campaign_responsive, marketing_resistant, needs_reengagement]]
    })
    save_excel(marketing_insights, "04_marketing_insights.xlsx")
    print("\n", marketing_insights.to_string(index=False))

    plt.figure(figsize=(10, 6))
    campaign_rates[CAMPAIGN_COLS].plot(kind="bar", ax=plt.gca(), colormap="coolwarm")
    plt.title("Campaign Acceptance Rate by Cluster (%)")
    plt.ylabel("Acceptance Rate (%)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("04a_campaign_response_by_cluster.png")

    plt.figure(figsize=(8, 5))
    campaign_rates["Complaint Rate (%)"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in campaign_rates.index])
    plt.title("Complaint Rate by Cluster (%)")
    savefig("04b_complaint_rate_by_cluster.png")

    return campaign_rates


# ============================================================
# ACTIVITY 5: BUSINESS SEGMENTATION & CLUSTER NAMING
# ============================================================

def activity5_naming(df, demo_avg, spend_avg, channel_avg, campaign_rates):
    print("\n" + "=" * 60, "\nACTIVITY 5: BUSINESS SEGMENTATION & CLUSTER NAMING\n", "=" * 60)

    naming_records = []
    strategy_map = {
        "High-Value Customers": "Premium bundles, early access, loyalty/VIP perks to protect this high-margin segment.",
        "Premium / Loyal Buyers": "Cross-sell across web, store and catalogue; loyalty rewards to reinforce repeat purchase.",
        "Discount Seekers / Budget Customers": "Value bundles, family-size promotions, deal/discount-led campaigns.",
        "New / Developing Customers": "Onboarding journeys, welcome incentives, and educational content to grow spend over time.",
    }

    for c in CLUSTER_ORDER:
        name = df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0]
        justification = (
            f"Income ${demo_avg.loc[c, 'Income']:,.0f}, Total Spend ${spend_avg.loc[c, 'Total_Spending']:,.0f}, "
            f"Recency {channel_avg.loc[c, 'Recency']:.0f} days, Family Size {demo_avg.loc[c, 'Family_Size']:.1f}, "
            f"Campaign Acceptance {campaign_rates.loc[c, CAMPAIGN_COLS].mean():.1f}% avg."
        )
        naming_records.append({
            "Cluster": c, "Business Name": name, "Justification": justification,
            "Commercial Usefulness": "High" if "High-Value" in name or "Premium" in name else
                                     ("Medium" if "New" in name else "Medium-Low"),
            "Recommended Strategy": strategy_map.get(name, "Tailor offers based on segment profile.")
        })

    naming_df = pd.DataFrame(naming_records)
    save_excel(naming_df, "05_cluster_naming_document.xlsx")
    print(naming_df.to_string(index=False))

    interpretation = pd.DataFrame({
        "Cluster": CLUSTER_ORDER,
        "Segment Name": [df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0] for c in CLUSTER_ORDER],
        "Defining Characteristics": [
            "; ".join([
                f"Income: {'High' if demo_avg.loc[c,'Income'] > demo_avg['Income'].median() else 'Low'}",
                f"Spend: {'High' if spend_avg.loc[c,'Total_Spending'] > spend_avg['Total_Spending'].median() else 'Low'}",
                f"Family Size: {'Large' if demo_avg.loc[c,'Family_Size'] > demo_avg['Family_Size'].median() else 'Small'}",
                f"Campaign response: {'Above avg' if campaign_rates.loc[c,'Response'] > campaign_rates['Response'].mean() else 'Below avg'}"
            ]) for c in CLUSTER_ORDER
        ]
    })
    save_excel(interpretation, "05_cluster_interpretation_report.xlsx")
    print("\n", interpretation.to_string(index=False))

    return naming_df


# ============================================================
# ACTIVITY 6: CUSTOMER PERSONA DEVELOPMENT
# ============================================================

def activity6_personas(df, demo_avg, spend_avg, channel_avg, campaign_rates, product_preference):
    print("\n" + "=" * 60, "\nACTIVITY 6: CUSTOMER PERSONA DEVELOPMENT\n", "=" * 60)

    persona_names = {
        "High-Value Customers": "Victoria the Premium Patron",
        "Premium / Loyal Buyers": "Marcus the Multi-Channel Regular",
        "Discount Seekers / Budget Customers": "The Rodriguez Family (Budget-Conscious Household)",
        "New / Developing Customers": "Jamie the New Explorer",
    }

    personas = []
    for c in CLUSTER_ORDER:
        name = df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0]
        age_mean = demo_avg.loc[c, "Customer_Age"]
        income_mean = demo_avg.loc[c, "Income"]
        family = demo_avg.loc[c, "Family_Size"]
        top_channel = channel_avg.loc[c, ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].idxmax()
        top_channel_name = {"NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog", "NumStorePurchases": "Store"}[top_channel]
        response_rate = campaign_rates.loc[c, "Response"]
        recency = channel_avg.loc[c, "Recency"]
        top_product = product_preference.loc[c].replace("Mnt", "")
        clv_estimate = spend_avg.loc[c, "Total_Spending"] * 3  # simple 3-year retention proxy

        challenge = (
            "Low marketing responsiveness; needs differentiated re-engagement offers." if response_rate < campaign_rates["Response"].mean()
            else "Risk of over-saturation from frequent campaigns; needs relevance over frequency."
        )

        personas.append({
            "Persona Name": persona_names.get(name, f"Customer Segment {c}"),
            "Cluster": c, "Segment Name": name,
            "Age Range": f"{int(age_mean-8)}-{int(age_mean+8)} yrs (avg {age_mean:.0f})",
            "Income Level": f"${income_mean:,.0f}/year",
            "Family Status": f"Avg family size {family:.1f}",
            "Shopping Habits": f"~{channel_avg.loc[c].sum():.0f} purchases/period across channels; "
                                f"Recency {recency:.0f} days",
            "Preferred Product Category": top_product,
            "Preferred Purchasing Channel": top_channel_name,
            "Marketing Responsiveness": f"{response_rate:.1f}% latest-campaign acceptance",
            "Customer Challenges": challenge,
            "Recommended Marketing Strategy": {
                "High-Value Customers": "Premium bundles, early access, VIP loyalty perks.",
                "Premium / Loyal Buyers": "Cross-channel cross-sell and loyalty rewards.",
                "Discount Seekers / Budget Customers": "Value bundles and family-size discount promotions.",
                "New / Developing Customers": "Onboarding journeys and welcome incentives.",
            }.get(name, "Tailored offer based on segment profile."),
            "Estimated Customer Lifetime Value (3yr proxy)": f"${clv_estimate:,.0f}",
        })

    persona_df = pd.DataFrame(personas)
    save_excel(persona_df, "06_customer_persona_report.xlsx")
    print(persona_df.to_string(index=False))
    return persona_df


# ============================================================
# ACTIVITY 7: CLUSTER VISUALIZATION & FINAL DOCUMENTATION
# ============================================================

def activity7_visualization(df, demo_avg, spend_avg, channel_avg, campaign_rates):
    print("\n" + "=" * 60, "\nACTIVITY 7: CLUSTER VISUALIZATION & FINAL DOCUMENTATION\n", "=" * 60)

    # Cluster distribution
    sizes = df["Cluster"].value_counts().sort_index()
    plt.figure(figsize=(7, 6))
    plt.pie(sizes, labels=[cluster_label(df, c) for c in sizes.index], autopct="%1.1f%%",
            colors=[CLUSTER_COLORS[c] for c in sizes.index], startangle=90)
    plt.title("Cluster Distribution")
    savefig("07a_cluster_distribution.png")

    # Income comparison
    plt.figure(figsize=(8, 5))
    demo_avg["Income"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in demo_avg.index])
    plt.title("Average Income by Cluster")
    savefig("07b_income_comparison.png")

    # Spending comparison (reuse Activity 2 chart concept, regenerate for completeness)
    plt.figure(figsize=(8, 5))
    spend_avg["Total_Spending"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in spend_avg.index])
    plt.title("Average Total Spending by Cluster")
    savefig("07c_spending_comparison.png")

    # Product preference heatmap (reuse structure)
    plt.figure(figsize=(8, 6))
    spend_share = spend_avg[SPEND_COLS].div(spend_avg[SPEND_COLS].sum(axis=1), axis=0)
    sns.heatmap(spend_share, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Product Preference Heatmap by Cluster")
    savefig("07d_product_preference_heatmap.png")

    # Purchasing channel comparison
    plt.figure(figsize=(9, 5))
    channel_avg[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].plot(kind="bar", ax=plt.gca(), colormap="Set2")
    plt.title("Purchasing Channel Comparison by Cluster")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("07e_channel_comparison.png")

    # Campaign response comparison
    plt.figure(figsize=(9, 5))
    campaign_rates[CAMPAIGN_COLS].plot(kind="bar", ax=plt.gca(), colormap="coolwarm")
    plt.title("Campaign Response Comparison by Cluster")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("07f_campaign_response_comparison.png")

    # Recency comparison
    plt.figure(figsize=(8, 5))
    channel_avg["Recency"].plot(kind="bar", color=[CLUSTER_COLORS[c] for c in channel_avg.index])
    plt.title("Recency Comparison by Cluster")
    savefig("07g_recency_comparison.png")

    # Radar chart of cluster characteristics
    radar_cols = ["Income", "Total_Spending", "Recency", "Family_Size", "Total_Campaign_Acceptance",
                  "NumWebPurchases", "NumStorePurchases"]
    radar_source = df.groupby("Cluster")[radar_cols].mean()
    radar_norm = (radar_source - radar_source.min()) / (radar_source.max() - radar_source.min() + 1e-9)
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in radar_norm.index:
        values = radar_norm.loc[c].tolist()
        values += values[:1]
        ax.plot(angles, values, label=cluster_label(df, c), color=CLUSTER_COLORS[c])
        ax.fill(angles, values, alpha=0.1, color=CLUSTER_COLORS[c])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_cols)
    ax.set_title("Radar Chart of Cluster Characteristics")
    ax.legend(bbox_to_anchor=(1.3, 1.1))
    savefig("07h_radar_chart.png")

    # PCA visualization (2D) colored by cluster
    numeric_features = ["Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
                         "Family_Size", "Total_Purchases", "Total_Campaign_Acceptance",
                         "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"]
    X = df[numeric_features].copy()
    X = (X - X.mean()) / X.std()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))
    for c in CLUSTER_ORDER:
        mask = df["Cluster"] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], alpha=0.6, s=15,
                    color=CLUSTER_COLORS[c], label=cluster_label(df, c))
    plt.title(f"PCA Cluster Visualization (Explained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("07i_pca_visualization.png")

    print("All Activity 7 visualizations saved (9 charts).")

    # Segment comparison table (compiled)
    segment_comparison = pd.concat([
        demo_avg[["Customer_Age", "Income", "Family_Size", "Total_Children", "Size"]],
        spend_avg[["Total_Spending"]],
        channel_avg[["Recency", "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases", "NumDealsPurchases"]],
        campaign_rates[["Response", "Complaint Rate (%)"]]
    ], axis=1)
    segment_comparison.insert(0, "Segment Name", [df.loc[df["Cluster"] == c, "Segment_Name"].iloc[0] for c in segment_comparison.index])
    save_excel(segment_comparison.reset_index(), "07_segment_comparison_table.xlsx")
    print("\nSegment Comparison Table:")
    print(segment_comparison.to_string())

    return segment_comparison


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_profiling():
    df = load_data()

    cluster_stats, demo_avg = activity1_demographics(df)
    spend_avg, product_preference = activity2_spending(df)
    channel_avg = activity3_channels(df)
    campaign_rates = activity4_campaigns(df)
    naming_df = activity5_naming(df, demo_avg, spend_avg, channel_avg, campaign_rates)
    persona_df = activity6_personas(df, demo_avg, spend_avg, channel_avg, campaign_rates, product_preference)
    segment_comparison = activity7_visualization(df, demo_avg, spend_avg, channel_avg, campaign_rates)

    df.to_csv(FINAL / "customer_profiles_final.csv", index=False)

    print("\n" + "=" * 60)
    print("CLUSTER EVALUATION & CUSTOMER PROFILING COMPLETE")
    print("=" * 60)
    return df, naming_df, persona_df, segment_comparison


if __name__ == "__main__":
    run_profiling()
