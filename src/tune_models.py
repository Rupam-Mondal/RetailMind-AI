import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_v3.joblib"
COMPARISON_PATH = BASE_DIR / "model_comparison_v3.csv"
LEARNING_CURVE_PATH = BASE_DIR / "learning_curve_v3.csv"
SUMMARY_PATH = BASE_DIR / "models" / "churn_v3_summary.json"
RANDOM_STATE = 42


def build_preprocessor(X):
    numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])


def build_pipeline(preprocessor, classifier):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )


def main():
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "SVM": (
            build_pipeline(preprocessor, SVC(probability=True, random_state=RANDOM_STATE)),
            {
                "classifier__C": [0.1, 1],
                "classifier__gamma": ["scale"],
                "classifier__kernel": ["rbf"]
            }
        ),
        "Random Forest": (
            build_pipeline(preprocessor, RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [None, 8],
                "classifier__min_samples_leaf": [3]
            }
        ),
        "Decision Tree": (
            build_pipeline(preprocessor, DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {
                "classifier__max_depth": [3, 8],
                "classifier__min_samples_split": [10],
                "classifier__min_samples_leaf": [3]
            }
        )
    }

    results = []
    searches = {}
    for name, (pipeline, parameter_grid) in candidates.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
        search.fit(X_train, y_train)
        diagnostics = cross_validate(
            search.best_estimator_,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=True
        )
        results.append({
            "model": name,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            "grid_train_roc_auc": search.cv_results_["mean_train_score"][search.best_index_],
            "cv_validation_roc_auc": diagnostics["test_score"].mean(),
            "cv_validation_std": diagnostics["test_score"].std(),
            "bias_variance_gap": diagnostics["train_score"].mean() - diagnostics["test_score"].mean()
        })
        searches[name] = search

    comparison = pd.DataFrame(results).sort_values(
        "cv_validation_roc_auc", ascending=False
    ).reset_index(drop=True)
    champion_name = comparison.loc[0, "model"]
    champion = searches[champion_name].best_estimator_

    train_sizes, train_scores, validation_scores = learning_curve(
        champion,
        X_train,
        y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    curve = pd.DataFrame({
        "training_examples": train_sizes,
        "train_roc_auc_mean": train_scores.mean(axis=1),
        "train_roc_auc_std": train_scores.std(axis=1),
        "validation_roc_auc_mean": validation_scores.mean(axis=1),
        "validation_roc_auc_std": validation_scores.std(axis=1)
    })

    champion.fit(X_train, y_train)
    y_pred = champion.predict(X_test)
    y_probability = champion.predict_proba(X_test)[:, 1]
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_probability)
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODEL_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)
    curve.to_csv(LEARNING_CURVE_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps({
        "champion": champion_name,
        "best_parameters": searches[champion_name].best_params_,
        **test_metrics
    }, indent=2), encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"Champion: {champion_name}")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
