

## Day 3 - Multi-Model Tuning and Validation

Run `python src/tune_models.py` to tune SVM, random forest, and decision tree candidates with five-fold cross-validation. The script stores the tuned champion at `models/churn_v3.joblib`, comparison diagnostics at `model_comparison_v3.csv`, and learning-curve data at `learning_curve_v3.csv`.
