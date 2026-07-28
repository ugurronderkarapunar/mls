"""Modelleme fonksiyonları."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib
from typing import Dict, Any, Tuple, List
import shap


MODEL_DICT = {
    "classification": {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    },
    "regression": {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "SVM": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    },
}


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.0, random_state: int = 42
) -> Tuple:
    """Veriyi eğitim, doğrulama (opsiyonel) ve test olarak böler."""
    if val_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_size + test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state, stratify=y_temp if y_temp.dtype == 'object' else None
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None
        )
        return X_train, None, X_test, y_train, None, y_test


def evaluate_model(
    model, X_train, y_train, X_test, y_test, task_type: str
) -> Dict[str, float]:
    """Modeli eğitir ve test metriklerini döndürür."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        if y_pred_proba is not None:
            metrics["ROC AUC"] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
    return metrics


def hyperparameter_tuning(
    model, param_grid: Dict[str, list], X_train, y_train, cv: int = 5, n_jobs: int = -1, method: str = "grid"
) -> Any:
    """Grid veya Random Search ile hiperparametre optimizasyonu."""
    if method == "grid":
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy' if hasattr(model, "predict_proba") else 'neg_mean_squared_error', n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=cv, scoring='accuracy' if hasattr(model, "predict_proba") else 'neg_mean_squared_error', n_jobs=n_jobs, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def shap_analysis(model, X_sample: pd.DataFrame) -> None:
    """SHAP değerleri ile model yorumu (grafik çizmek için)."""
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    # Streamlit için figure döndürebiliriz, burada pass.
    return shap_values


def save_model(model, path: str) -> None:
    """Modeli diske kaydeder."""
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Modeli diskten yükler."""
    return joblib.load(path)
