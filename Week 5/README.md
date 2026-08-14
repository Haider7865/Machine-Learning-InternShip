# Week 5: Modules 9, 10 & 11 — Dashboard, QA, Documentation & Final Presentation

**Customer Segmentation Project — AI Lab 99 Internship Program**
**Target Due: 2026-08-14**

This submission is organized by day, following the weekly workflow table in
the module brief, plus one `Complete_Project/` folder containing the entire,
integrated application (this is the folder to actually run).

```
Week1_Modules_9_10_11/
│
├── Monday_Day1_Dashboard_Planning/         (Module 9 — Task 1, 1.1, 2)
├── Tuesday_Day2_Dashboard_Development/     (Module 9 — Task 3, 3.1, 4)
├── Wednesday_Day3_Integration_Testing/     (Module 9 Task 5-7 + Module 10 Task 8)
├── Thursday_Day4_Quality_Assurance/        (Module 10 — Task 9-20)
├── Friday_Day5_Documentation_Presentation/ (Module 11 — Task 21-28)
│
└── Complete_Project/                       (The full, integrated, runnable application)
```

## Day-by-Day Summary

| Day | Focus | Key Outputs |
|---|---|---|
| **Monday** | Dashboard planning & architecture | Architecture plan, project structure, identified data/model assets |
| **Tuesday** | Dashboard development | Working Streamlit app (app/), 8 filters, 10 chart functions, screenshots |
| **Wednesday** | Model integration & initial testing | src/ pipeline modules, live prediction UI, error handling, initial test plan |
| **Thursday** | Quality assurance | 40/40 pytest tests passing, 13 QA reports, defect + regression reports |
| **Friday** | Documentation & presentation | 13-chapter final report, technical docs, user manual, deployment guide, 15-slide deck, demo script |

## Running the Application

The daily folders contain copies of the relevant files for review purposes.
To actually **run** the application, use the `Complete_Project/` folder:

```bash
cd Complete_Project
pip install -r requirements.txt
streamlit run app/app.py
```

## Running the Tests

```bash
cd Complete_Project
pytest tests/ -v
```

All 40 tests pass. See `Thursday_Day4_Quality_Assurance/` for the full QA
report set and `Complete_Project/reports/pytest_report.html` for the raw
pytest output.
