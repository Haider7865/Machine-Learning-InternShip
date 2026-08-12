# Day 3 — Model/Application Integration & Initial Testing

## Task 5: Connect ML Model to Dashboard (src/prediction.py)
- Loads the saved production_pipeline.pkl (scaler + trained K-Means + feature list).
- Accepts raw customer input, applies the SAME feature-engineering logic used
  during training (src/feature_engineering.py), scales it, and assigns a
  segment — then attaches business-recommendation metadata (SEGMENT_INFO).

## Task 6: Individual Customer Prediction Interface
Implemented in app/components/customer_lookup.py — see
screenshots/09b_new_customer_prediction.png for the live interface:
Age / Income / Total Spending / Web Purchases / Store Purchases / Recency
input fields, an "Analyze Customer" button, and an output showing Customer
Segment, Customer Value, Retention Risk, Recommended Action, Channel, and
Product.

## Task 7: Error Handling (src/preprocessing.py — validate_customer_input)
Tested invalid inputs: empty Customer ID, negative income, negative
spending, invalid/negative/out-of-range age, missing required field, text
in a numeric field, and extremely large values. Every case returns a clear
error message instead of crashing the application (see
tests/test_preprocessing.py for the automated versions of these checks).

## Task 8: Initial Test Plan (reports/10_test_plan.xlsx)
20 test cases (TC-001 to TC-020) covering data, model, prediction,
dashboard, and input validation — see reports/10_test_plan.xlsx.
