"""Tam veri işleme ve modelleme pipeline'ı."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
from typing import List, Optional


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    imputer_strategy: str = "median",
    scaler: str = "standard",
) -> ColumnTransformer:
    """Ön işleme pipeline'ı oluşturur."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler() if scaler == "standard" else "passthrough"),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def create_full_pipeline(
    preprocessor: ColumnTransformer,
    model,
) -> Pipeline:
    """Ön işleme ve modeli içeren tam pipeline."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)
