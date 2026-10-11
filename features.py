"""
features.py
------------
Feature engineering + preprocessing pipeline shared by the classifier
(flood probability) and regressor (runoff).
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "elevation_m",
    "slope",
    "imperviousness_pct",
    "drainage_capacity_mmhr",
    "rainfall_mm_24h",
]
CATEGORICAL_FEATURES = ["land_use", "soil_type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical (land use, soil)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
