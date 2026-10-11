"""
model.py
--------
Trains two models on the catchment dataset:
  1. RandomForestClassifier  -> flood probability (predict_proba)
  2. RandomForestRegressor   -> predicted runoff (mm)

Both share the same preprocessing pipeline (features.py) and are wrapped
end-to-end with sklearn.pipeline.Pipeline so raw feature rows go straight in.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features import ALL_FEATURES, build_preprocessor


def train_flood_classifier(df: pd.DataFrame, random_state: int = 42):
    X = df[ALL_FEATURES]
    y = df["flood_occurred"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, target_names=["No Flood", "Flood"])
    auc = roc_auc_score(y_test, y_proba)

    metrics = {"classification_report": report, "roc_auc": auc}
    return clf, metrics, (X_test, y_test, y_proba)


def train_runoff_regressor(df: pd.DataFrame, random_state: int = 42):
    X = df[ALL_FEATURES]
    y = df["runoff_mm"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    reg = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1,
        )),
    ])
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae_mm": mae, "rmse_mm": rmse, "r2": r2}
    return reg, metrics, (X_test, y_test, y_pred)


def get_feature_importance(pipeline: Pipeline) -> pd.Series:
    """Extract feature importances mapped back to human-readable names."""
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["land_use", "soil_type"]))
    from features import NUMERIC_FEATURES
    names = NUMERIC_FEATURES + cat_names
    importances = pipeline.named_steps["model"].feature_importances_
    return pd.Series(importances, index=names).sort_values(ascending=False)


def save_models(clf, reg, out_dir="outputs"):
    joblib.dump(clf, f"{out_dir}/flood_classifier.joblib")
    joblib.dump(reg, f"{out_dir}/runoff_regressor.joblib")
