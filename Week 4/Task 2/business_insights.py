from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE = Path(__file__).resolve().parent
DATA = BASE / "00_Data"
REPORTS = BASE / "02_Reports"
CHARTS = BASE / "03_Charts"
DASHBOARD = BASE / "04_Dashboard"
FINAL = BASE / "05_Final_Deliverables"

for folder in [DATA, REPORTS, CHARTS, DASHBOARD, FINAL]:
    folder.mkdir(exist_ok=True)

INPUT_FILE = DATA / "customer_profiles_final.csv"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CAMPAIGN_COLS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]

SEGMENT_COLORS = {
    "High-Value Customers": "#C44E52",
    "Premium / Loyal Buyers": "#4C72B0",
    "Discount Seekers / Budget Customers": "#8172B2",
    "New / Developing Customers": "#55A868",
}


def save_excel(dataframe, filename):
    dataframe.to_excel(REPORTS / filename, index=False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()


def load_data():
    return pd.read_csv(INPUT_FILE)


# ============================================================
# TASK 1: REVIEW CUSTOMER SEGMENTS
# ============================================================

def task1_segment_summary(df):
    print("=" * 60, "\nTASK 1: REVIEW CUSTOMER SEGMENTS\n", "=" * 60)

    counts = df["Segment_Name"].value_counts()
    summary = pd.DataFrame({
        "Cluster": [df.loc[df["Segment_Name"] == s, "Cluster"].iloc[0] for s in counts.index],
        "Business Name": counts.index,
        "Customers": counts.values,
        "Percentage": (counts.values / len(df) * 100).round(1),
    }).sort_values("Cluster").reset_index(drop=True)

    save_excel(summary, "01_customer_segment_summary.xlsx")
    print(summary.to_string(index=False))
    return summary


# ============================================================
# SHARED: PER-SEGMENT METRIC TABLE (used across many tasks)
# ============================================================

def build_segment_metrics(df):
    metrics = df.groupby("Segment_Name").agg(
        Customers=("Segment_Name", "count"),
        Avg_Income=("Income", "mean"),
        Avg_Total_Spending=("Total_Spending", "mean"),
        Avg_Purchases=("Total_Purchases", "mean"),
        Avg_Recency=("Recency", "mean"),
        Avg_Web_Purchases=("NumWebPurchases", "mean"),
        Avg_Store_Purchases=("NumStorePurchases", "mean"),
        Avg_Catalog_Purchases=("NumCatalogPurchases", "mean"),
        Avg_Deal_Purchases=("NumDealsPurchases", "mean"),
        Deal_Dependency=("Deal_Dependency", "mean"),
        Campaign_Response_Rate=("Response", "mean"),
        Avg_Campaigns_Accepted=("Total_Campaign_Acceptance", "mean"),
        Complaint_Rate=("Complain", "mean"),
    ).round(2)

    metrics["Revenue_Contribution"] = df.groupby("Segment_Name")["Total_Spending"].sum()
    metrics["Revenue_Share_Pct"] = (metrics["Revenue_Contribution"] / metrics["Revenue_Contribution"].sum() * 100).round(1)

    # Churn-risk proxy: normalized blend of high Recency + low campaign response + low purchase frequency
    r_norm = (metrics["Avg_Recency"] - metrics["Avg_Recency"].min()) / (metrics["Avg_Recency"].max() - metrics["Avg_Recency"].min() + 1e-9)
    resp_norm = 1 - (metrics["Campaign_Response_Rate"] - metrics["Campaign_Response_Rate"].min()) / (metrics["Campaign_Response_Rate"].max() - metrics["Campaign_Response_Rate"].min() + 1e-9)
    freq_norm = 1 - (metrics["Avg_Purchases"] - metrics["Avg_Purchases"].min()) / (metrics["Avg_Purchases"].max() - metrics["Avg_Purchases"].min() + 1e-9)
    churn_score = (r_norm * 0.4 + resp_norm * 0.3 + freq_norm * 0.3)
    metrics["Churn_Risk_Score"] = churn_score.round(3)
    metrics["Churn_Risk_Level"] = pd.cut(churn_score, bins=[-0.01, 0.33, 0.66, 1.01],
                                          labels=["Low", "Medium", "High"])

    # Product preference per segment
    top_products = df.groupby("Segment_Name")[SPEND_COLS].mean().idxmax(axis=1)
    metrics["Top_Product_Category"] = top_products.str.replace("Mnt", "")

    # Preferred channel
    channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    top_channel = df.groupby("Segment_Name")[channel_cols].mean().idxmax(axis=1)
    metrics["Preferred_Channel"] = top_channel.map({
        "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog", "NumStorePurchases": "Store"
    })

    # Estimated 3-year CLV proxy
    metrics["Estimated_CLV_3yr"] = (metrics["Avg_Total_Spending"] * 3).round(0)

    return metrics.reset_index()


# ============================================================
# TASK 2: DEVELOP CUSTOMER PERSONAS
# ============================================================

def task2_personas(df, metrics):
    print("\n" + "=" * 60, "\nTASK 2: DEVELOP CUSTOMER PERSONAS\n", "=" * 60)

    personas = []
    for _, row in metrics.iterrows():
        seg = row["Segment_Name"]
        income_level = "High" if row["Avg_Income"] > metrics["Avg_Income"].median() else "Low-Medium"
        spend_level = "High" if row["Avg_Total_Spending"] > metrics["Avg_Total_Spending"].median() else "Low"
        freq_level = "High" if row["Avg_Purchases"] > metrics["Avg_Purchases"].median() else "Low"
        activity = "Active" if row["Avg_Recency"] <= metrics["Avg_Recency"].median() else "Less Active"

        personas.append({
            "Segment": seg,
            "Income Level": income_level,
            "Spending Behavior": f"{spend_level} (avg ${row['Avg_Total_Spending']:,.0f})",
            "Purchase Frequency": f"{freq_level} (avg {row['Avg_Purchases']:.1f} purchases)",
            "Preferred Products": row["Top_Product_Category"],
            "Campaign Response": f"{row['Campaign_Response_Rate']*100:.1f}% latest-campaign acceptance",
            "Shopping Channel": row["Preferred_Channel"],
            "Deal Dependency": f"{row['Deal_Dependency']*100:.1f}% of purchases are deal-driven",
            "Customer Activity": f"{activity} (avg Recency {row['Avg_Recency']:.0f} days)",
            "Risk Level": row["Churn_Risk_Level"],
            "Lifetime Value": f"${row['Estimated_CLV_3yr']:,.0f} (3-yr proxy)",
        })

    persona_df = pd.DataFrame(personas)
    save_excel(persona_df, "02_customer_personas.xlsx")
    print(persona_df.to_string(index=False))
    return persona_df


# ============================================================
# TASK 3: ANALYZE BUSINESS OPPORTUNITIES
# ============================================================

OPPORTUNITY_CONTENT = {
    "High-Value Customers": {
        "value": "Highest income and spend per customer; largest revenue share despite being a mid-sized segment; strongest campaign responsiveness.",
        "opportunity": "Upsell premium/exclusive product lines and bundles; grow share-of-wallet via loyalty tiers.",
        "risk": "Losing even a few of these customers has outsized revenue impact; risk of campaign fatigue from over-targeting.",
        "objective": "Maximize retention and share-of-wallet through premium, low-frequency, high-relevance campaigns.",
    },
    "Premium / Loyal Buyers": {
        "value": "Second-highest spend, most active across every purchase channel (web, store, catalogue, deals) — the most engaged segment.",
        "opportunity": "Cross-sell across channels they already use; convert high activity into higher basket value.",
        "risk": "Currently under-targeted by campaigns relative to their engagement level — a missed-revenue risk, not a churn risk.",
        "objective": "Increase campaign investment to match engagement and lift average order value.",
    },
    "Discount Seekers / Budget Customers": {
        "value": "Largest households (family size 3.7) with steady, price-sensitive demand; a stable transaction-volume base.",
        "opportunity": "Family-size bundles and value promotions can lift basket size without competing on premium price.",
        "risk": "Lowest campaign response rate of all segments and above-average Recency — highest risk of disengagement.",
        "objective": "Re-engage with value-led, high-relevance offers rather than generic premium campaigns.",
    },
    "New / Developing Customers": {
        "value": "Largest segment by customer count (30.2%) — the biggest pool of future growth potential.",
        "opportunity": "Nurture into higher-value segments via onboarding and habit-building offers; long growth runway given youngest average age.",
        "risk": "Currently lowest income and spend; risk of never developing into a higher-value segment without intervention.",
        "objective": "Build engagement and purchase frequency through onboarding journeys and habit-forming incentives.",
    },
}


def task3_opportunities(metrics):
    print("\n" + "=" * 60, "\nTASK 3: ANALYZE BUSINESS OPPORTUNITIES\n", "=" * 60)

    rows = []
    for seg in metrics["Segment_Name"]:
        c = OPPORTUNITY_CONTENT[seg]
        rows.append({
            "Segment": seg,
            "What Makes This Segment Valuable": c["value"],
            "Business Opportunity": c["opportunity"],
            "Business Risk": c["risk"],
            "Marketing Objective": c["objective"],
        })
    opp_df = pd.DataFrame(rows)
    save_excel(opp_df, "03_business_opportunity_report.xlsx")
    print(opp_df.to_string(index=False))
    return opp_df


# ============================================================
# TASK 4: MARKETING STRATEGY DESIGN
# ============================================================

STRATEGY_CONTENT = {
    "High-Value Customers": {
        "message": "You've earned exclusive access — discover our Premium Collection.",
        "tone": "Exclusive, appreciative, VIP",
        "offer": "Early access to new/premium products + loyalty tier perks",
        "channel": "Email + personal outreach",
        "timing": "Product launches and milestone anniversaries",
        "frequency": "Monthly (low-frequency, high-relevance)",
    },
    "Premium / Loyal Buyers": {
        "message": "More of what you love, across every way you shop.",
        "tone": "Warm, rewarding, recognition-focused",
        "offer": "Cross-channel loyalty points + bundle discounts",
        "channel": "Email + App/Web push",
        "timing": "Aligned to their regular purchase cycle",
        "frequency": "Bi-weekly",
    },
    "Discount Seekers / Budget Customers": {
        "message": "Save more on the family favorites you already buy.",
        "tone": "Practical, value-focused, family-oriented",
        "offer": "Family-size bundle discounts and deal alerts",
        "channel": "SMS + Email",
        "timing": "Paydays / start of month, seasonal sales",
        "frequency": "Bi-weekly",
    },
    "New / Developing Customers": {
        "message": "Welcome! Here's a taste of what you can explore with us.",
        "tone": "Friendly, welcoming, educational",
        "offer": "Welcome discount + starter bundle",
        "channel": "Email + SMS onboarding series",
        "timing": "First 30-60 days after acquisition",
        "frequency": "Weekly (onboarding window), then monthly",
    },
}


def task4_marketing_strategy(metrics):
    print("\n" + "=" * 60, "\nTASK 4: MARKETING STRATEGY DESIGN\n", "=" * 60)

    rows = []
    for seg in metrics["Segment_Name"]:
        c = STRATEGY_CONTENT[seg]
        rows.append({
            "Segment": seg, "Marketing Message": c["message"], "Tone": c["tone"],
            "Personalized Offer": c["offer"], "Preferred Channel": c["channel"],
            "Best Campaign Timing": c["timing"], "Campaign Frequency": c["frequency"],
        })
    strat_df = pd.DataFrame(rows)
    save_excel(strat_df, "04_marketing_strategy_design.xlsx")
    print(strat_df.to_string(index=False))
    return strat_df


# ============================================================
# TASK 5: PRODUCT RECOMMENDATION STRATEGY
# ============================================================

def task5_product_recommendations(df, metrics):
    print("\n" + "=" * 60, "\nTASK 5: PRODUCT RECOMMENDATION STRATEGY\n", "=" * 60)

    rows = []
    for _, row in metrics.iterrows():
        seg = row["Segment_Name"]
        seg_spend = df.loc[df["Segment_Name"] == seg, SPEND_COLS].mean().sort_values(ascending=False)
        primary = seg_spend.index[0].replace("Mnt", "")
        secondary = seg_spend.index[1].replace("Mnt", "")
        cross_sell = seg_spend.index[2].replace("Mnt", "")
        upsell = "Gold Products / Premium tier of primary category"
        bundle = f"{primary} + {secondary} starter/family bundle"
        rows.append({
            "Segment": seg, "Primary Product Category": primary, "Secondary Product Category": secondary,
            "Cross-Selling Products": cross_sell, "Upselling Products": upsell,
            "Bundle Recommendation": bundle,
        })
    prod_df = pd.DataFrame(rows)
    save_excel(prod_df, "05_product_recommendation_table.xlsx")
    print(prod_df.to_string(index=False))
    return prod_df


# ============================================================
# TASK 6: PRICING AND DISCOUNT STRATEGY
# ============================================================

DISCOUNT_CONTENT = {
    "High-Value Customers": {
        "discount": "5-10% (low discount need)", "coupon": "Exclusive/invite-only coupons, not mass discounts",
        "promo": "Early access + free premium samples/gifts", "seasonal": "VIP pre-sale access before public launch",
        "premium_pricing": "Strong candidate for premium/limited-edition pricing tiers",
    },
    "Premium / Loyal Buyers": {
        "discount": "10-15% on bundles", "coupon": "Loyalty-points-redeemable coupons",
        "promo": "Buy-more-save-more multi-channel promos", "seasonal": "Seasonal loyalty-tier bonus points",
        "premium_pricing": "Moderate opportunity — tiered pricing for bundle upgrades",
    },
    "Discount Seekers / Budget Customers": {
        "discount": "15-25% (highest discount sensitivity)", "coupon": "Broad, easily redeemable coupons",
        "promo": "Family-size / multi-buy promotional pricing", "seasonal": "Major seasonal sales (back-to-school, holidays)",
        "premium_pricing": "Low — avoid premium pricing pushes for this segment",
    },
    "New / Developing Customers": {
        "discount": "10-20% welcome discount", "coupon": "First-purchase and referral coupons",
        "promo": "Starter bundle promotional pricing", "seasonal": "Anniversary-of-signup reactivation offers",
        "premium_pricing": "Low initially — build trust before premium upsell",
    },
}


def task6_discount_strategy(metrics):
    print("\n" + "=" * 60, "\nTASK 6: PRICING AND DISCOUNT STRATEGY\n", "=" * 60)

    rows = []
    for seg in metrics["Segment_Name"]:
        c = DISCOUNT_CONTENT[seg]
        rows.append({
            "Segment": seg, "Discount %": c["discount"], "Coupon Strategy": c["coupon"],
            "Promotional Offers": c["promo"], "Seasonal Offers": c["seasonal"],
            "Premium Pricing Opportunity": c["premium_pricing"],
        })
    discount_df = pd.DataFrame(rows)
    save_excel(discount_df, "06_discount_recommendation_matrix.xlsx")
    print(discount_df.to_string(index=False))
    return discount_df


# ============================================================
# TASK 7: CUSTOMER RETENTION STRATEGY
# ============================================================

RETENTION_CONTENT = {
    "High-Value Customers": {
        "loyalty": "VIP tier with exclusive perks and early access", "campaign": "Quarterly appreciation campaign",
        "followup": "Personal follow-up within 48h of any support issue", "support": "Highest priority / dedicated support",
        "membership": "Invite-only VIP membership", "referral": "High-value referral bonus (cash or premium credit)",
    },
    "Premium / Loyal Buyers": {
        "loyalty": "Points-based loyalty program across channels", "campaign": "Monthly loyalty rewards campaign",
        "followup": "Standard follow-up within 72h", "support": "High priority",
        "membership": "Standard loyalty membership tier", "referral": "Moderate referral bonus",
    },
    "Discount Seekers / Budget Customers": {
        "loyalty": "Punch-card style / cumulative-purchase discounts", "campaign": "Bi-weekly value/deal reminders",
        "followup": "Follow-up within 1 week", "support": "Standard priority",
        "membership": "Free basic loyalty membership", "referral": "Family/friend referral discount",
    },
    "New / Developing Customers": {
        "loyalty": "Onboarding rewards for first 3 purchases", "campaign": "30-60 day onboarding campaign series",
        "followup": "Proactive check-in after first purchase", "support": "Standard priority, extra guidance",
        "membership": "Free trial of loyalty membership", "referral": "New-customer-friendly referral incentive",
    },
}


def task7_retention_strategy(metrics):
    print("\n" + "=" * 60, "\nTASK 7: CUSTOMER RETENTION STRATEGY\n", "=" * 60)

    rows = []
    for seg in metrics["Segment_Name"]:
        c = RETENTION_CONTENT[seg]
        rows.append({
            "Segment": seg, "Loyalty Rewards": c["loyalty"], "Retention Campaign": c["campaign"],
            "Follow-up Frequency": c["followup"], "Support Priority": c["support"],
            "Membership Plan": c["membership"], "Referral Incentive": c["referral"],
        })
    retention_df = pd.DataFrame(rows)
    save_excel(retention_df, "07_retention_strategy_document.xlsx")
    print(retention_df.to_string(index=False))
    return retention_df


# ============================================================
# TASK 8: CUSTOMER REACTIVATION STRATEGY
# ============================================================

def task8_reactivation(df, metrics):
    print("\n" + "=" * 60, "\nTASK 8: CUSTOMER REACTIVATION STRATEGY\n", "=" * 60)

    recency_threshold = df["Recency"].quantile(0.75)
    needs_reactivation = df[df["Recency"] >= recency_threshold]
    by_segment = needs_reactivation["Segment_Name"].value_counts()

    reactivation_summary = pd.DataFrame({
        "Segment": by_segment.index,
        "Customers Needing Reactivation": by_segment.values,
        "% of Segment": [
            round(by_segment[s] / (df["Segment_Name"] == s).sum() * 100, 1) for s in by_segment.index
        ]
    })
    save_excel(reactivation_summary, "08_customers_needing_reactivation.xlsx")
    print(reactivation_summary.to_string(index=False))

    reactivation_plan = pd.DataFrame({
        "Segment": metrics["Segment_Name"],
        "Win-Back Offer": [
            "Exclusive high-value comeback offer + personal outreach" if s == "High-Value Customers" else
            "Loyalty-points bonus to resume regular purchasing" if s == "Premium / Loyal Buyers" else
            "Deep discount / bundle deal to re-attract price-sensitive spend" if s == "Discount Seekers / Budget Customers" else
            "Simplified welcome-back discount + product education"
            for s in metrics["Segment_Name"]
        ],
        "Reminder Campaign": [
            "\"We miss you\" premium email with curated picks" if s == "High-Value Customers" else
            "Cross-channel reminder highlighting new arrivals" if s == "Premium / Loyal Buyers" else
            "SMS deal reminder with time-limited discount" if s == "Discount Seekers / Budget Customers" else
            "Friendly re-introduction email with getting-started tips"
            for s in metrics["Segment_Name"]
        ],
        "Personalized Recommendation": metrics["Top_Product_Category"] + " favorites based on past purchases",
        "Reactivation Timeline": [
            "Immediate (within 7 days of inactivity threshold)" if s == "High-Value Customers" else
            "Within 14 days" if s == "Premium / Loyal Buyers" else
            "Within 30 days, tied to seasonal promotions" if s == "Discount Seekers / Budget Customers" else
            "Within 30-45 days, paired with onboarding follow-up"
            for s in metrics["Segment_Name"]
        ],
    })
    save_excel(reactivation_plan, "08_customer_reactivation_plan.xlsx")
    print("\n", reactivation_plan.to_string(index=False))
    return reactivation_summary, reactivation_plan


# ============================================================
# TASK 9: CAMPAIGN ACTION PLAN
# ============================================================

def task9_campaign_action_plan(metrics):
    print("\n" + "=" * 60, "\nTASK 9: CAMPAIGN ACTION PLAN\n", "=" * 60)

    campaigns = [
        {"Campaign": "VIP Premium Rewards", "Target Segment": "High-Value Customers",
         "Objective": "Increase loyalty & share-of-wallet", "Marketing Channel": "Email + Personal outreach",
         "Budget Priority": "High", "Expected Outcome": "Higher repeat purchase rate, reduced attrition",
         "KPI": "Repeat Purchase Rate, Revenue per Customer"},
        {"Campaign": "Cross-Channel Loyalty Boost", "Target Segment": "Premium / Loyal Buyers",
         "Objective": "Increase basket value & channel cross-use", "Marketing Channel": "Email + App/Web push",
         "Budget Priority": "Medium-High", "Expected Outcome": "Higher average order value",
         "KPI": "Average Order Value, Purchase Frequency"},
        {"Campaign": "Family Value Bundles", "Target Segment": "Discount Seekers / Budget Customers",
         "Objective": "Increase engagement & re-purchase", "Marketing Channel": "SMS + Email",
         "Budget Priority": "Medium", "Expected Outcome": "Higher campaign acceptance, reduced churn risk",
         "KPI": "Campaign Acceptance Rate, Conversion Rate"},
        {"Campaign": "Welcome & Onboarding Series", "Target Segment": "New / Developing Customers",
         "Objective": "Build engagement & grow spend", "Marketing Channel": "Email + SMS",
         "Budget Priority": "Medium", "Expected Outcome": "Increased purchase frequency and category exploration",
         "KPI": "30/60/90-day Retention Rate, Avg. Spend Growth"},
    ]
    plan_df = pd.DataFrame(campaigns)
    save_excel(plan_df, "09_campaign_action_plan.xlsx")
    print(plan_df.to_string(index=False))
    return plan_df


# ============================================================
# TASK 10: MARKETING STRATEGY MATRIX
# ============================================================

def task10_strategy_matrix(metrics, strat_df, prod_df, discount_df, retention_df, reactivation_plan):
    print("\n" + "=" * 60, "\nTASK 10: MARKETING STRATEGY MATRIX\n", "=" * 60)

    matrix = metrics[["Segment_Name"]].copy()
    matrix = matrix.merge(strat_df[["Segment", "Marketing Message", "Preferred Channel", "Campaign Frequency"]],
                           left_on="Segment_Name", right_on="Segment").drop(columns="Segment")
    matrix = matrix.merge(prod_df[["Segment", "Primary Product Category", "Cross-Selling Products"]].rename(
        columns={"Primary Product Category": "Product Recommendation", "Cross-Selling Products": "Cross-Selling"}),
        left_on="Segment_Name", right_on="Segment").drop(columns="Segment")
    matrix = matrix.merge(discount_df[["Segment", "Discount %"]].rename(columns={"Discount %": "Discount Strategy"}),
                           left_on="Segment_Name", right_on="Segment").drop(columns="Segment")
    matrix = matrix.merge(retention_df[["Segment", "Loyalty Rewards"]].rename(columns={"Loyalty Rewards": "Loyalty Strategy"}),
                           left_on="Segment_Name", right_on="Segment").drop(columns="Segment")
    matrix["Upselling"] = "Premium/Gold tier upgrade of primary category"
    matrix = matrix.merge(reactivation_plan[["Segment", "Win-Back Offer"]].rename(columns={"Win-Back Offer": "Reactivation Strategy"}),
                           left_on="Segment_Name", right_on="Segment").drop(columns="Segment")

    matrix = matrix.rename(columns={"Segment_Name": "Segment Name"})
    save_excel(matrix, "10_marketing_strategy_matrix.xlsx")
    print(matrix.to_string(index=False))
    return matrix


# ============================================================
# TASK 11: BUSINESS DASHBOARD
# ============================================================

def task11_dashboard(df, metrics):
    print("\n" + "=" * 60, "\nTASK 11: BUSINESS DASHBOARD\n", "=" * 60)

    order = metrics.sort_values("Customers", ascending=False)["Segment_Name"].tolist()
    colors = [SEGMENT_COLORS[s] for s in order]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # 1. Customers per segment
    sizes = metrics.set_index("Segment_Name").loc[order, "Customers"]
    axes[0, 0].pie(sizes, labels=order, autopct="%1.1f%%", colors=colors, startangle=90,
                    textprops={"fontsize": 9})
    axes[0, 0].set_title("Customers per Segment", fontsize=13)

    # 2. Revenue contribution by segment
    rev = metrics.set_index("Segment_Name").loc[order, "Revenue_Contribution"]
    axes[0, 1].bar(range(len(order)), rev.values, color=colors)
    axes[0, 1].set_xticks(range(len(order)))
    axes[0, 1].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[0, 1].set_title("Revenue Contribution by Segment ($)", fontsize=13)

    # 3. Campaign priority (budget priority proxy via response rate)
    resp = metrics.set_index("Segment_Name").loc[order, "Campaign_Response_Rate"] * 100
    axes[0, 2].bar(range(len(order)), resp.values, color=colors)
    axes[0, 2].set_xticks(range(len(order)))
    axes[0, 2].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[0, 2].set_title("Campaign Response Rate (%) — Priority Signal", fontsize=13)

    # 4. Churn risk by segment
    risk_map = {"Low": 1, "Medium": 2, "High": 3}
    risk_vals = metrics.set_index("Segment_Name").loc[order, "Churn_Risk_Level"].map(risk_map)
    risk_colors = ["#55A868" if v == 1 else "#DD8452" if v == 2 else "#C44E52" for v in risk_vals]
    axes[1, 0].bar(range(len(order)), risk_vals.values, color=risk_colors)
    axes[1, 0].set_xticks(range(len(order)))
    axes[1, 0].set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    axes[1, 0].set_yticks([1, 2, 3])
    axes[1, 0].set_yticklabels(["Low", "Medium", "High"])
    axes[1, 0].set_title("Churn Risk Level by Segment", fontsize=13)

    # 5. Recommended marketing channels
    channel_counts = metrics["Preferred_Channel"].value_counts()
    axes[1, 1].bar(channel_counts.index, channel_counts.values, color="#4C72B0")
    axes[1, 1].set_title("Recommended Marketing Channels (Segment Count)", fontsize=13)

    # 6. Product preferences
    prod_counts = metrics["Top_Product_Category"].value_counts()
    axes[1, 2].bar(prod_counts.index, prod_counts.values, color="#8172B2")
    axes[1, 2].set_title("Top Product Preferences (Segment Count)", fontsize=13)
    axes[1, 2].tick_params(axis="x", rotation=20)

    plt.suptitle("Business Insights Dashboard — Customer Segmentation", fontsize=17, y=1.02)
    plt.tight_layout()
    plt.savefig(DASHBOARD / "business_insights_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Dashboard saved to 04_Dashboard/business_insights_dashboard.png")


if __name__ == "__main__":
    df = load_data()
    seg_summary = task1_segment_summary(df)
    metrics = build_segment_metrics(df)
    save_excel(metrics, "00_segment_metrics_master.xlsx")

    persona_df = task2_personas(df, metrics)
    opp_df = task3_opportunities(metrics)
    strat_df = task4_marketing_strategy(metrics)
    prod_df = task5_product_recommendations(df, metrics)
    discount_df = task6_discount_strategy(metrics)
    retention_df = task7_retention_strategy(metrics)
    reactivation_summary, reactivation_plan = task8_reactivation(df, metrics)
    campaign_plan = task9_campaign_action_plan(metrics)
    strategy_matrix = task10_strategy_matrix(metrics, strat_df, prod_df, discount_df, retention_df, reactivation_plan)
    task11_dashboard(df, metrics)

    df.to_csv(FINAL / "customer_segments_with_strategy.csv", index=False)

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS & MARKETING RECOMMENDATIONS COMPLETE")
    print("=" * 60)
