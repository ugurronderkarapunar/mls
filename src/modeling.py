"""Modelleme fonksiyonları – XGBoost, LightGBM ve Optuna destekli."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
from typing import Dict, Any, Tuple, List, Optional
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------
# Model sözlüğü (sınıflandırma ve regresyon için genişletilmiş)
# ---------------------------------------------------------------------
MODEL_DICT = {
    "classification": {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    },
    "regression": {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "SVM": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    },
}


# ---------------------------------------------------------------------
# Veri bölme
# ---------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> Tuple:
    """Veriyi eğitim, doğrulama (opsiyonel) ve test olarak böler.

    Returns:
        X_train, X_val (None olabilir), X_test, y_train, y_val, y_test
    """
    if val_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=val_size + test_size,
            random_state=random_state,
            stratify=y if y.dtype == 'object' else None,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
            stratify=y_temp if y_temp.dtype == 'object' else None,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.dtype == 'object' else None,
        )
        return X_train, None, X_test, y_train, None, y_test


# ---------------------------------------------------------------------
# Model değerlendirme
# ---------------------------------------------------------------------
def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> Dict[str, float]:
    """Modeli eğitir ve test seti üzerinde performans metriklerini döndürür."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        # Olasılık tahmini yapabilen modeller için ROC AUC
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        if roc_auc is not None:
            metrics["ROC AUC"] = roc_auc
    else:
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
    return metrics


# ---------------------------------------------------------------------
# Grid/Random Search (eski)
# ---------------------------------------------------------------------
def hyperparameter_tuning(
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_jobs: int = -1,
    method: str = "grid",
) -> Tuple[Any, Dict[str, Any]]:
    """Grid veya Random Search ile hiperparametre optimizasyonu.

    Returns:
        (best_model, best_params)
    """
    if method == "grid":
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='accuracy' if hasattr(model, "predict_proba") else 'neg_mean_squared_error',
            n_jobs=n_jobs,
        )
    else:
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='accuracy' if hasattr(model, "predict_proba") else 'neg_mean_squared_error',
            n_jobs=n_jobs,
            random_state=42,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------
# Optuna tabanlı hiperparametre optimizasyonu (Yeni)
# ---------------------------------------------------------------------
def optuna_optimize(
    model_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    timeout: int = 120,
) -> Tuple[Any, Dict[str, Any]]:
    """Optuna ile belirtilen model için hiperparametre optimizasyonu yapar.

    Args:
        model_name: MODEL_DICT'teki anahtar isim (örn. "Random Forest").
        task_type: "classification" veya "regression".
        X_train, y_train: Eğitim verisi.
        n_trials: Deneme sayısı.
        timeout: Saniye cinsinden zaman aşımı.

    Returns:
        (best_model, best_params)
    """

    if task_type not in MODEL_DICT or model_name not in MODEL_DICT[task_type]:
        raise ValueError(f"Model {model_name} bulunamadı.")

    def objective(trial):
        if model_name == "Random Forest":
            if task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    random_state=42,
                )

        elif model_name == "XGBoost":
            if task_type == "classification":
                model = XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                )
            else:
                model = XGBRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    random_state=42,
                )

        elif model_name == "LightGBM":
            if task_type == "classification":
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 20, 150),
                    random_state=42,
                    verbose=-1,
                )
            else:
                model = LGBMRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 20, 150),
                    random_state=42,
                    verbose=-1,
                )

        elif model_name == "Logistic Regression":
            model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 10, log=True),
                max_iter=1000,
                random_state=42,
            )

        elif model_name == "SVM":
            model = SVC(
                C=trial.suggest_float("C", 0.01, 10, log=True),
                gamma=trial.suggest_float("gamma", 0.001, 1, log=True),
                probability=True,
                random_state=42,
            )

        elif model_name == "Decision Tree":
            if task_type == "classification":
                model = DecisionTreeClassifier(
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    random_state=42,
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=trial.suggest_int("max_depth", 3, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    random_state=42,
                )
        else:
            # Fallback: temel model
            model = MODEL_DICT[task_type][model_name]

        # Değerlendirme
        if task_type == "classification":
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
        else:
            score = -cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_squared_error").mean()  # MSE
        return score

    study = optuna.create_study(direction="maximize" if task_type == "classification" else "minimize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    # En iyi modeli yeniden eğit
    if model_name in ["Random Forest", "XGBoost", "LightGBM"]:
        if task_type == "classification":
            if model_name == "Random Forest":
                best_model = RandomForestClassifier(**best_params, random_state=42)
            elif model_name == "XGBoost":
                best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
            elif model_name == "LightGBM":
                best_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        else:
            if model_name == "Random Forest":
                best_model = RandomForestRegressor(**best_params, random_state=42)
            elif model_name == "XGBoost":
                best_model = XGBRegressor(**best_params, random_state=42)
            elif model_name == "LightGBM":
                best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    elif model_name == "Logistic Regression":
        best_model = LogisticRegression(C=best_params["C"], max_iter=1000, random_state=42)
    elif model_name == "SVM":
        best_model = SVC(C=best_params["C"], gamma=best_params["gamma"], probability=True, random_state=42)
    elif model_name == "Decision Tree":
        if task_type == "classification":
            best_model = DecisionTreeClassifier(**best_params, random_state=42)
        else:
            best_model = DecisionTreeRegressor(**best_params, random_state=42)
    else:
        best_model = MODEL_DICT[task_type][model_name]

    best_model.fit(X_train, y_train)
    return best_model, best_params


# ---------------------------------------------------------------------
# Model kaydetme / yükleme
# ---------------------------------------------------------------------
def save_model(model: Any, path: str) -> None:
    """Modeli diske kaydeder."""
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Modeli diskten yükler."""
    return joblib.load(path)


# ---------------------------------------------------------------------
# SHAP analizi (opsiyonel)
# ---------------------------------------------------------------------
def shap_analysis(model: Any, X_sample: pd.DataFrame):
    """SHAP değerleri ile model yorumu (matplotlib figürü döndürür)."""
    import shap
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    return shap_values
