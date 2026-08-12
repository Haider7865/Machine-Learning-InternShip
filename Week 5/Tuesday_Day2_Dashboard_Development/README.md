# Day 2 — Dashboard Development

## Task 3: Interactive Dashboard
Built with Streamlit. Contains: sidebar navigation, 8 live filters, interactive
Plotly charts, KPI cards, tables, customer search, segment selection, and a
CSV download option (Segment Comparison section).

## Task 3.1: Filters (app/components/filters.py)
Implemented filters: Customer Segment, Age Group, Income Range, Education,
Marital Status, Product Category, Purchase Channel, Campaign Response.
Changing any filter updates all charts and tables on the page immediately
(no "Apply" button — Streamlit's reactive execution model handles this).

## Task 4: Visualization Development (app/components/charts.py)
10 chart functions, each with a docstring stating the specific business
question it answers, per the module rule: "Do not create charts simply to
increase the number of charts."

| Chart | Business Question Answered |
|---|---|
| segment_distribution_pie | How many customers fall into each segment? |
| spending_by_segment_bar | Which segment spends the most, on average? |
| income_by_segment_box | How does income differ across segments? |
| age_distribution_hist | What is the age profile of the customer base? |
| product_category_spend_bar | Which product categories generate the most revenue? |
| purchase_channel_pie | Which purchase channel is most used? |
| campaign_response_bar | How did acceptance rates differ across campaigns? |
| recency_by_segment_bar | Which segments are most at risk of disengagement? |
| segment_comparison_radar | How do two chosen segments compare? |
| customer_activity_scatter | Is there a relationship between web engagement and recency? |

See the `screenshots/` folder for all 10 dashboard sections rendered with
live data.
