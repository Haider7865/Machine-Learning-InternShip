# Customer Segmentation Project

**AI Lab 99 Internship Program — Customer Personality Analysis & Segmentation**

An end-to-end machine learning application that cleans customer data, engineers behavioural
features, trains a K-Means segmentation model, and serves live predictions and business
recommendations through an interactive Streamlit dashboard.

## Project Structure

```
customer-segmentation-project/
├── app/                    # Streamlit dashboard application
│   ├── app.py               # Entry point (streamlit run app/app.py)
│   ├── dashboard.py          # Assembles all 10 dashboard sections
│   └── components/           # Filters, charts, customer lookup/prediction
├── data/
│   ├── raw/                 # Original marketing_campaign.csv
│   └── processed/            # Final cleaned, engineered, segmented dataset
├── models/                  # Saved scaler + K-Means model (production_pipeline.pkl)
├── src/                     # Reusable pipeline logic (shared by training & the app)
│   ├── preprocessing.py       # Cleaning + input validation
│   ├── feature_engineering.py # Feature creation (same logic train & predict)
│   └── prediction.py          # Loads model, assigns segment + recommendations
├── tests/                   # Pytest suite (40 tests, data/preprocessing/FE/model)
├── reports/                  # QA reports (test plan, defect report, UAT, etc.)
├── notebooks/                # Key Jupyter notebooks from Modules 5-8
├── documentation/            # Final report, technical docs, user manual, deployment guide
├── presentation/              # Final slides + demonstration script
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard opens at `http://localhost:8501`.

## Running Tests

```bash
pytest tests/ -v
```

## Model Summary

- **Algorithm:** K-Means (K=4), selected over Hierarchical, GMM, and DBSCAN
- **Features:** 12 behavioural/demographic features, StandardScaler-normalized
- **Segments:** High-Value Customers, Premium/Loyal Buyers, Discount Seekers/Budget
  Customers, New/Developing Customers
- **Validation:** Silhouette 0.169, Davies-Bouldin 1.890, stability ARI > 0.95

See `documentation/Final_Project_Report.pdf` for the complete project write-up.
