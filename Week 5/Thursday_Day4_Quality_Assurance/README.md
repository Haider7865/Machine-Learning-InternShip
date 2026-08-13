# Day 4 — Quality Assurance

All QA activities (Tasks 9-20) are grounded in the actual pytest suite
(40/40 tests passing — see reports/pytest_report.html) plus manual
dashboard/UAT testing performed during development.

| Task | Report |
|---|---|
| 9. Data Validation | 11_data_validation_report.xlsx |
| 10. Preprocessing Testing | 12_preprocessing_test_report.xlsx |
| 11. Feature Engineering Testing | 13_feature_engineering_test_report.xlsx |
| 12. Model Testing | 14_model_validation_report.xlsx |
| 13. Dashboard Testing | 15_dashboard_test_report.xlsx |
| 14. User Input Testing | 16_user_input_test_report.xlsx |
| 15. Performance Testing | 17_performance_test_report.xlsx |
| 16. Code Quality Review | 18_code_quality_review.xlsx |
| 17. Integration Testing | 19_integration_test_report.xlsx |
| 18. User Acceptance Testing | 20_uat_report.xlsx |
| 19. Defect Report | 21_defect_report.xlsx |
| 20. Regression Testing | 22_regression_test_report.xlsx |

## Summary
- **40/40** automated pytest tests passing across data validation,
  preprocessing, feature engineering, and model test suites.
- **3 defects** found and fixed during development (see defect report);
  all confirmed via regression testing.
- **0 open defects** at final QA sign-off.

## Re-running the tests
```bash
pip install -r ../../Complete_Project/requirements.txt
pytest tests/ -v
```
