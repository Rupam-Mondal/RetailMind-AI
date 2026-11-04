# RetailMind AI

RetailMind AI predicts customer churn using the Telco Customer Churn dataset. The maintained training workflow is `notebooks/day4_feature_engineering_ensembles.ipynb`.

## Dataset

The project uses `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle's [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The dataset is already included in this repository.

## Day 4 — Feature Engineering, PCA & Ensembles — Completed

The Day 4 notebook keeps the test set isolated, applies imputing, encoding, scaling, PCA fitting, and hyperparameter search only to training folds, and uses a fixed random state (`42`).

### Engineered features

- Tenure buckets: new (0–6 months), short-term (7–24), medium-term (25–48), and long-term (49+).
- Charge signals: tenure-adjusted average monthly charge, monthly-to-total-charge ratio, charge/tenure interaction, and charge-versus-average gap.
- Contract and loyalty signals: month-to-month and long-term contract flags, loyalty score, automatic payment, electronic-check, and paperless-electronic-check flags.
- Service signals: fiber-optic flag, service count, and security-or-support flag.

`TotalCharges` is coerced safely to numeric; missing numerical and categorical values are imputed within sklearn pipelines.

### PCA

Categorical fields are one-hot encoded and all numerical fields are standardized before PCA. The training split has 62 encoded features; 20 principal components retain at least 95% of its variance. The explained-variance table and cumulative-variance plot are in `models/day4_pca_explained_variance.csv` and `models/day4_pca_cumulative_variance.png`.

### Validation and held-out test comparison

The champion is selected by 3-fold stratified validation ROC-AUC, then validation F1 and recall. Values below are held-out test results except the `CV ROC-AUC` column.

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| PCA Logistic Regression | 0.8488 | 0.7480 | 0.5166 | 0.7914 | 0.6251 | 0.8487 |
| XGBoost | 0.8478 | 0.7459 | 0.5138 | 0.7941 | 0.6239 | 0.8433 |
| Random Forest | 0.8464 | 0.7622 | 0.5356 | 0.7834 | 0.6363 | 0.8456 |
| Existing baseline (Logistic Regression) | 0.8458 | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |

Random Forest uses bagging (`250` trees, depth tuned between 8 and unrestricted). XGBoost uses boosting with class-imbalance weighting; its tuned configuration uses 200 trees, depth 3, learning rate 0.05, subsampling 0.8, and column subsampling 0.8.

### Final model and persistence

The selected model is **PCA Logistic Regression**: it has the strongest validation ROC-AUC (0.8488), while maintaining 0.7914 recall on the held-out test set. The notebook also writes complete classification reports and confusion matrices to metadata and an image artifact.

- Champion pipeline: `models/day4_churn_champion.joblib`
- Preprocessor: `models/day4_preprocessor.joblib`
- PCA object: `models/day4_pca.joblib`
- Metadata, feature names, reports, and confusion matrices: `models/day4_metadata.json`
- Model comparison: `models/day4_model_comparison.csv`

The persisted champion contains standard sklearn preprocessing, PCA, and classifier objects and can be loaded with `joblib.load`. For raw Telco records, first run the deterministic `TelcoFeatureEngineer` cell in the Day 4 notebook, then call `predict` / `predict_proba` on the loaded pipeline.

## Run

```powershell
python -m pip install -r requirements.txt
jupyter notebook notebooks/day4_feature_engineering_ensembles.ipynb
```

Run the notebook top-to-bottom. It trains all candidates, writes the artifacts above, reloads the champion, and runs a sample prediction from the test split.

## Milestones

- Day 1: data preparation and baseline churn pipeline
- Day 2: multi-model comparison
- Day 3: cross-validation and model tuning
- Day 4: feature engineering, PCA, Random Forest, XGBoost, model selection, and persistence — completed
