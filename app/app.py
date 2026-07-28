"""CRISP-DM Veri Bilimi Asistanı – Senior Sürüm."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import DBSCAN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
import joblib
import base64
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt
import shap
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
@st.cache_data
def cached_describe(df):
    return df.describe(include="all").T

# ---------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        raise ValueError("Desteklenmeyen dosya türü.")

# ---------------------------------------------------------------------
# Temizleme
# ---------------------------------------------------------------------
def drop_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df, strategy='median', fill_value=None, columns=None):
    if columns is None:
        columns = df.columns.tolist()
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    elif strategy in ['knn', 'mice']:
        numeric_cols = [c for c in columns if df[c].dtype in [np.float64, np.int64]]
        if numeric_cols:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = IterativeImputer(random_state=RANDOM_STATE)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        cat_cols = [c for c in columns if c not in numeric_cols]
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
    else:
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif strategy == 'constant':
                    df[col].fillna(fill_value if fill_value is not None else 0, inplace=True)
            else:
                if strategy in ['mean','median']:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
                elif strategy == 'constant':
                    df[col].fillna(fill_value if fill_value is not None else 'Bilinmiyor', inplace=True)
    return df

def remove_outliers(df, method='iqr', threshold=1.5, columns=None, action='remove'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            if action == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            elif action == 'cap':
                df[col] = df[col].clip(lower, upper)
    elif method == 'zscore':
        for col in columns:
            z = np.abs(stats.zscore(df[col].dropna()))
            if action == 'remove':
                df = df[(z < threshold)]
            elif action == 'cap':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - threshold * std, mean + threshold * std)
    elif method == 'isolation_forest':
        iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
        outlier_labels = iso.fit_predict(df[columns].dropna())
        df = df[outlier_labels == 1]
    elif method == 'dbscan':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[columns].dropna())
        db = DBSCAN(eps=0.5, min_samples=5)
        labels = db.fit_predict(X_scaled)
        df = df[labels != -1]
    return df

def fix_data_types(df, conversions):
    for col, dtype in conversions.items():
        try:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str":
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(dtype)
        except Exception:
            pass
    return df

# ---------------------------------------------------------------------
# Özellik mühendisliği
# ---------------------------------------------------------------------
def extract_date_features(df, date_col, drop_original=False):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[f"{date_col}_year"] = df[date_col].dt.year
    df[f"{date_col}_month"] = df[date_col].dt.month
    df[f"{date_col}_day"] = df[date_col].dt.day
    df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
    df[f"{date_col}_quarter"] = df[date_col].dt.quarter
    if drop_original:
        df.drop(columns=[date_col], inplace=True)
    return df

def scale_numeric(df, columns, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Geçersiz ölçeklendirme yöntemi.")
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df

def encode_categorical(df, columns, method='onehot', drop_first=True):
    from sklearn.preprocessing import OrdinalEncoder
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'ordinal':
        for col in columns:
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]])
    else:
        raise ValueError("Geçersiz kodlama yöntemi.")
    return df

def add_polynomial_features(df, columns, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_data = poly.fit_transform(df[columns])
    poly_cols = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=df.index)
    df = df.drop(columns=columns)
    return pd.concat([df, poly_df], axis=1)

def add_interaction_features(df, col_pairs):
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df

def feature_selection(X, y, method='selectkbest', k=10, estimator=None):
    """Eksik değerleri geçici olarak doldurur, sonra seçim yapar."""
    X_temp = X.copy()
    for col in X_temp.columns:
        if X_temp[col].isnull().any():
            if X_temp[col].dtype in [np.float64, np.int64]:
                X_temp[col].fillna(X_temp[col].median(), inplace=True)
            else:
                X_temp[col].fillna('missing', inplace=True)

    if method == 'selectkbest':
        if y.dtype in [np.float64, np.int64] and y.nunique() > 10:
            selector = SelectKBest(score_func=f_regression, k=min(k, X_temp.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_temp.shape[1]))
        selector.fit(X_temp, y)
        cols = X_temp.columns[selector.get_support()]
        return cols.tolist()
    elif method == 'rfe':
        if estimator is None:
            estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        selector = RFE(estimator, n_features_to_select=min(k, X_temp.shape[1]))
        selector.fit(X_temp, y)
        cols = X_temp.columns[selector.support_]
        return cols.tolist()
    elif method == 'vif':
        vif_data = pd.DataFrame({
            'feature': X_temp.columns,
            'VIF': [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
        })
        selected = vif_data[vif_data['VIF'] < 10]['feature'].tolist()
        return selected
    return X_temp.columns.tolist()

# ---------------------------------------------------------------------
# Modelleme
# ---------------------------------------------------------------------
MODEL_DICT = {
    "classification": {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "KNN": KNeighborsClassifier(),
    },
    "regression": {
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Linear Regression": LinearRegression(),
        "SVM": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        "KNN": KNeighborsRegressor(),
    },
}

def split_data(X, y, test_size=0.2, val_size=0.0, random_state=RANDOM_STATE):
    if val_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_size+test_size, random_state=random_state,
            stratify=y if y.dtype == 'object' else None)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(val_size+test_size), random_state=random_state,
            stratify=y_temp if y_temp.dtype == 'object' else None)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if y.dtype == 'object' else None)
        return X_train, None, X_test, y_train, None, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, task_type):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if task_type == "classification":
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics["ROC AUC"] = roc_auc_score(y_test, y_prob[:, 1])
            else:
                metrics["ROC AUC (ovr)"] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    else:
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
    return metrics, model

def compare_models(task_type, X_train, y_train, X_test, y_test):
    # NaN temizliği
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if X_train[col].dtype in [np.float64, np.int64]:
                med = X_train[col].median()
                X_train[col].fillna(med, inplace=True)
                X_test[col].fillna(med, inplace=True)
            else:
                X_train[col].fillna('missing', inplace=True)
                X_test[col].fillna('missing', inplace=True)
    results = {}
    models = MODEL_DICT[task_type]
    progress = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task_type == "classification":
            results[name] = accuracy_score(y_test, y_pred)
        else:
            results[name] = r2_score(y_test, y_pred)
        progress.progress((i+1)/len(models))
    progress.empty()
    return results

def optuna_optimize(model_name, task_type, X_train, y_train, n_trials=30):
    if model_name not in MODEL_DICT[task_type]:
        return None, {}
    def objective(trial):
        if model_name == "Random Forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
            model = RandomForestClassifier(**params, random_state=RANDOM_STATE) if task_type=="classification" else RandomForestRegressor(**params, random_state=RANDOM_STATE)
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            }
            model = XGBClassifier(**params, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss') if task_type=="classification" else XGBRegressor(**params, random_state=RANDOM_STATE)
        elif model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            }
            model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1) if task_type=="classification" else LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
        else:
            return 0.0
        if task_type == "classification":
            return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy'))
        else:
            return -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))
    study = optuna.create_study(direction='maximize' if task_type=='classification' else 'minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    if model_name == "Random Forest":
        best_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE) if task_type=="classification" else RandomForestRegressor(**best_params, random_state=RANDOM_STATE)
    elif model_name == "XGBoost":
        best_model = XGBClassifier(**best_params, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss') if task_type=="classification" else XGBRegressor(**best_params, random_state=RANDOM_STATE)
    elif model_name == "LightGBM":
        best_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1) if task_type=="classification" else LGBMRegressor(**best_params, random_state=RANDOM_STATE, verbose=-1)
    best_model.fit(X_train, y_train)
    return best_model, best_params

# ---------------------------------------------------------------------
# Görselleştirme yardımcıları
# ---------------------------------------------------------------------
def plot_qq(df, col):
    data = df[col].dropna()
    theoretical = stats.norm.ppf((np.arange(len(data))+0.5)/len(data))
    theoretical = np.sort(theoretical)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=np.sort(data), mode='markers', name='QQ'))
    fig.add_trace(go.Scatter(x=theoretical, y=theoretical*np.std(data)/np.std(theoretical)+np.mean(data), mode='lines', name='Doğru'))
    fig.update_layout(title=f"{col} QQ Plot", xaxis_title="Teorik", yaxis_title="Örneklem")
    return fig

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Tahmin", y="Gerçek"), x=labels, y=labels)
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_roc_curve(model, X_test, y_test, task_type):
    if task_type != "classification": return None
    y_prob = model.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=model.classes_[1])
        fig = px.area(x=fpr, y=tpr, title="ROC Eğrisi", labels={"x":"False Positive Rate", "y":"True Positive Rate"})
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        return fig
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=model.classes_)
        fig = go.Figure()
        for i in range(y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=model.classes_[i]))
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.update_layout(title="ROC Eğrileri")
        return fig

def plot_precision_recall_curve(model, X_test, y_test, task_type):
    if task_type != "classification": return None
    y_prob = model.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label=model.classes_[1])
        fig = px.area(x=recall, y=precision, title="Precision-Recall Eğrisi", labels={"x":"Recall", "y":"Precision"})
        return fig
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=model.classes_)
        fig = go.Figure()
        for i in range(y_bin.shape[1]):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=model.classes_[i]))
        fig.update_layout(title="PR Eğrileri")
        return fig

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, name='Eğitim Skoru'))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, name='Doğrulama Skoru'))
    fig.update_layout(title="Öğrenme Eğrisi", xaxis_title="Eğitim Boyutu", yaxis_title="Skor")
    return fig

def shap_summary_plot(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Önem Düzeyi")
    ax.set_title("Özellik Önem Sıralaması")
    plt.tight_layout()
    return fig

def get_top_features(model, feature_names, top_n=3):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []
    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in indices]

# ---------------------------------------------------------------------
# PDF Rapor
# ---------------------------------------------------------------------
class EDAReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "CRISP-DM Veri Analizi Raporu", 0, 1, "C")
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", 0, 0, "C")

def generate_pdf_report(df, target, model=None, metrics=None, top_feats=None):
    pdf = EDAReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    from datetime import datetime
    pdf.write(5, f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    pdf.write(5, f"Veri boyutu: {df.shape[0]} satir, {df.shape[1]} sutun\n")
    pdf.write(5, f"Hedef degisken: {target}\n\n")
    pdf.write(5, "Temel Istatistikler:\n")
    desc = df.describe(include='all').to_string()
    pdf.set_font("Courier", size=8)
    pdf.multi_cell(0, 4, desc)
    if model and metrics:
        pdf.set_font("Arial", size=11)
        pdf.write(5, "\n\nModel Metrikleri:\n")
        for k, v in metrics.items():
            pdf.write(5, f"{k}: {v}\n")
        if top_feats:
            pdf.write(5, "\nEn Onemli Degiskenler:\n")
            for feat, imp in top_feats:
                pdf.write(5, f"{feat} ({imp:.4f})\n")
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------------------
# EDA yardımcıları
# ---------------------------------------------------------------------
def descriptive_stats(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].describe(include="all").T

def check_skewness(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].skew().sort_values(ascending=False)

def correlation_matrix(df, method="pearson"):
    return df.corr(method=method, numeric_only=True)

def vif_analysis(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    temp_df = df[columns].fillna(df[columns].median())
    vif_data = pd.DataFrame({
        "Değişken": columns,
        "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(len(columns))]
    })
    return vif_data.sort_values("VIF", ascending=False)

def get_skewness_insight(skew_val):
    if abs(skew_val) < 0.5:
        return "✅ Simetrik dağılım."
    elif skew_val > 1:
        return "⚠️ Sağa çarpık. Log/karekök dönüşümü önerilir."
    elif skew_val < -1:
        return "⚠️ Sola çarpık. Kare/Box-Cox dönüşümü önerilir."
    else:
        return "ℹ️ Orta düzey çarpıklık."

def get_vif_insight(vif_val):
    if vif_val < 5:
        return "✅ Çoklu bağlantı yok."
    elif vif_val < 10:
        return "⚠️ Orta düzey çoklu bağlantı."
    else:
        return "❌ Yüksek çoklu bağlantı!"

def auto_hypothesis_test(df, target, alpha=0.05):
    results = []
    for col in df.columns.drop(target):
        if df[col].dtype in [np.float64, np.int64]:
            if df[target].dtype in [np.float64, np.int64] and df[target].nunique() > 10:
                corr, p = stats.pearsonr(df[col].dropna(), df[target].dropna())
                results.append((col, 'Pearson Korelasyon', f'r={corr:.3f}, p={p:.4f}'))
            else:
                groups = [g[col].dropna().values for _, g in df.groupby(target)]
                if len(groups) == 2:
                    stat, p = stats.ttest_ind(groups[0], groups[1])
                    test_name = 'Bağımsız t-test'
                else:
                    stat, p = stats.f_oneway(*groups)
                    test_name = 'ANOVA'
                sig = 'Anlamlı fark var' if p < alpha else 'Fark yok'
                results.append((col, test_name, f'stat={stat:.3f}, p={p:.4f} ({sig})'))
        else:
            if df[target].dtype not in [np.float64, np.int64]:
                try:
                    table = pd.crosstab(df[col], df[target])
                    chi2, p, dof, _ = stats.chi2_contingency(table)
                    sig = 'Bağımlılık var' if p < alpha else 'Bağımsız'
                    results.append((col, 'Ki-kare', f'chi2={chi2:.3f}, p={p:.4f} ({sig})'))
                except:
                    pass
    return results

# ---------------------------------------------------------------------
# Streamlit arayüz
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="CRISP-DM Asistanı")
st.title("📊 CRISP-DM Veri Bilimi Asistanı – Senior")

for key, default in [
    ("df", None), ("target", None), ("X_train", None), ("X_test", None),
    ("y_train", None), ("y_test", None), ("task", None), ("model", None),
    ("model_trained", False), ("uploaded_file_name", None), ("top_features", []),
    ("pipeline", None), ("baseline_metrics", None), ("optimized_metrics", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

tabs = st.tabs(["📂 Veri Yükleme", "🧹 Temizleme", "📊 EDA", "⚙️ Özellik Müh.", "🤖 Modelleme", "📄 Rapor", "🚀 Tahmin"])

with tabs[0]:
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV veya Excel dosyası yükleyin", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        if (st.session_state.df is None) or (st.session_state.uploaded_file_name != uploaded_file.name):
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.toast("Veri yüklendi!", icon="✅")
        else:
            df = st.session_state.df

        st.dataframe(df.head())
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sütun Tipleri")
            st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Sütun", 0: "Tip"}))
        with col2:
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Sütun", "Eksik"]
            missing["%"] = (missing["Eksik"] / len(df)) * 100
            st.subheader("Eksik Değerler")
            st.dataframe(missing)
        target = st.selectbox("Hedef sütunu", df.columns)
        st.session_state.target = target
        st.success(f"Hedef: {target}")

        cols_to_drop = st.multiselect("Silmek istediğiniz sütunlar", df.columns)
        if st.button("Seçili Sütunları Sil"):
            if target in cols_to_drop:
                st.error("Hedef sütunu silemezsiniz!")
            else:
                df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.df = df
                st.toast(f"{len(cols_to_drop)} sütun silindi.", icon="🗑️")
                st.rerun()

if st.session_state.df is not None:
    df = st.session_state.df
    target = st.session_state.target

    with tabs[1]:
        st.header("Veri Temizleme")
        dup_count = df.duplicated().sum()
        st.write(f"Tekrar eden satır sayısı: **{dup_count}**")
        if st.button("Tekrarlanan Satırları Sil"):
            df = drop_duplicates(df)
            st.session_state.df = df
            st.toast(f"{dup_count} tekrar silindi.", icon="🧹")

        st.subheader("Eksik Veri")
        miss_cols = st.multiselect("Sütun (boş=tümü)", df.columns)
        strategy = st.selectbox("Strateji", ["median", "mean", "mode", "constant", "drop", "knn", "mice"])
        fill_val = None
        if strategy == "constant":
            fill_val = st.text_input("Sabit değer", "Bilinmiyor")
        if st.button("Eksikleri Doldur/Sil"):
            before = df.isnull().sum().sum()
            df = handle_missing_values(df, strategy, fill_val, miss_cols if miss_cols else None)
            after = df.isnull().sum().sum()
            st.session_state.df = df
            st.toast(f"İşlem tamamlandı. Eksik: {before} -> {after}", icon="✅")

        st.subheader("Aykırı Değerler")
        out_cols = st.multiselect("Sayısal sütunlar", df.select_dtypes(include=np.number).columns)
        method = st.radio("Yöntem", ["iqr", "zscore", "isolation_forest", "dbscan"])
        action = st.radio("İşlem", ["remove", "cap (sınırlandır)"])
        thresh = st.number_input("Eşik (IQR/Z için)", 1.0, 5.0, 1.5)
        if st.button("Aykırıları İşle"):
            act = 'remove' if action.startswith("remove") else 'cap'
            df = remove_outliers(df, method, thresh, out_cols if out_cols else None, action=act)
            st.session_state.df = df
            st.toast("İşlem tamamlandı.", icon="✨")

        st.subheader("Veri Tipi Düzeltme")
        col_to_fix = st.multiselect("Tipini değiştir", df.columns)
        new_type = st.selectbox("Yeni tip", ["int", "float", "str", "datetime"])
        if st.button("Tipi Dönüştür"):
            conversions = {c: new_type for c in col_to_fix}
            df = fix_data_types(df, conversions)
            st.session_state.df = df
            st.toast("Tipler güncellendi.", icon="🔄")

    with tabs[2]:
        st.header("Keşifsel Veri Analizi")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.subheader("Eksik Veri Haritası")
        fig_missing = px.imshow(df.isnull(), color_continuous_scale=['green', 'red'], aspect='auto')
        st.plotly_chart(fig_missing, use_container_width=True)

        if num_cols:
            st.subheader("Tanımlayıcı İstatistikler")
            st.dataframe(descriptive_stats(df))
            col_sel = st.selectbox("Box/Violin/QQ için sütun", num_cols)
            fig_box = px.box(df, y=col_sel)
            fig_violin = px.violin(df, y=col_sel, box=True)
            fig_qq = plot_qq(df, col_sel)
            st.plotly_chart(fig_box, use_container_width=True)
            st.plotly_chart(fig_violin, use_container_width=True)
            st.plotly_chart(fig_qq, use_container_width=True)
            st.subheader("Çarpıklık")
            skew_df = check_skewness(df).to_frame("Çarpıklık")
            skew_df["Yorum"] = skew_df["Çarpıklık"].apply(get_skewness_insight)
            st.dataframe(skew_df)
            st.subheader("Korelasyon Matrisi")
            fig_corr = px.imshow(correlation_matrix(df), text_auto=".2f", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)
            if len(num_cols) > 1:
                st.subheader("VIF")
                vif_df = vif_analysis(df)
                vif_df["Yorum"] = vif_df["VIF"].apply(get_vif_insight)
                st.dataframe(vif_df)

        if cat_cols:
            st.subheader("Kategorik Değişkenler")
            cat_sel = st.selectbox("Count Plot", cat_cols)
            fig_count = px.histogram(df, x=cat_sel, color=target if target in df.columns else None)
            st.plotly_chart(fig_count, use_container_width=True)

        if target and target in df.columns:
            st.subheader(f"Hedef: {target}")
            if df[target].dtype in [np.int64, np.float64] and df[target].nunique() > 10:
                fig_target = px.histogram(df, x=target, marginal="box")
            else:
                fig_target = px.histogram(df, x=target)
            st.plotly_chart(fig_target, use_container_width=True)

        st.subheader("Otomatik Hipotez Testleri")
        test_results = auto_hypothesis_test(df, target)
        if test_results:
            st.table(pd.DataFrame(test_results, columns=["Değişken", "Test", "Sonuç"]))
        else:
            st.info("Test yapılabilecek uygun değişken bulunamadı.")

    with tabs[3]:
        st.header("Özellik Mühendisliği")
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if date_cols:
            date_sel = st.selectbox("Tarih sütunu", date_cols)
            if st.button("Tarih özellikleri çıkar"):
                df = extract_date_features(df, date_sel)
                st.session_state.df = df
                st.toast("Eklendi.", icon="📅")
        else:
            st.info("Datetime sütunu yok.")
        st.subheader("Ölçeklendirme")
        scale_cols = st.multiselect("Sütunlar", df.select_dtypes(include=np.number).columns)
        scale_method = st.selectbox("Yöntem", ["standard", "minmax", "robust"])
        if st.button("Ölçeklendir"):
            df = scale_numeric(df, scale_cols, scale_method)
            st.session_state.df = df
            st.toast("Ölçeklendi.", icon="⚖️")
        st.subheader("Kategorik Kodlama")
        cat_cols_fe = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols_fe:
            cat_sel_fe = st.multiselect("Kodlanacak", cat_cols_fe)
            enc_method = st.selectbox("Yöntem", ["onehot", "label"])
            if st.button("Kodla"):
                df = encode_categorical(df, cat_sel_fe, enc_method)
                st.session_state.df = df
                st.toast("Kodlandı.", icon="🔢")
        st.subheader("Polinomal Özellikler")
        poly_cols = st.multiselect("Derece alınacak sayısal sütunlar", df.select_dtypes(include=np.number).columns)
        poly_degree = st.slider("Derece", 2, 4, 2)
        if st.button("Polinomal Ekle"):
            df = add_polynomial_features(df, poly_cols, poly_degree)
            st.session_state.df = df
            st.toast("Polinomal özellikler eklendi.", icon="📈")
        st.subheader("Etkileşim Özellikleri")
        int_col1 = st.selectbox("Birinci sütun", df.select_dtypes(include=np.number).columns, key="int1")
        int_col2 = st.selectbox("İkinci sütun", df.select_dtypes(include=np.number).columns, key="int2")
        if st.button("Çarpım Ekle"):
            df = add_interaction_features(df, [(int_col1, int_col2)])
            st.session_state.df = df
            st.toast(f"{int_col1} x {int_col2} eklendi.", icon="✖️")
        st.subheader("Özellik Seçimi")
        y_fe = df[target]
        X_fe = df.drop(columns=[target])
        X_fe = pd.get_dummies(X_fe, drop_first=True)
        sel_method = st.selectbox("Yöntem", ["selectkbest", "rfe", "vif"])
        k = st.slider("Seçilecek özellik sayısı", 5, min(50, X_fe.shape[1]), 10)
        if st.button("Özellik Seçimi Uygula"):
            selected = feature_selection(X_fe, y_fe, method=sel_method, k=k)
            st.success(f"Seçilen {len(selected)} özellik: {selected}")

    with tabs[4]:
        st.header("Modelleme")
        if target is None:
            st.warning("Hedef seçilmedi.")
        else:
            y = df[target]
            X = df.drop(columns=[target])
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype in [np.int64, np.float64] and y.nunique() > 10:
                task = "regression"
            else:
                task = "classification"
                y = y.astype(str)
            st.session_state.task = task
            st.info(f"Görev: {task}")

            use_smote = False
            if task == "classification":
                class_counts = y.value_counts()
                st.write("Sınıf dağılımı:", class_counts)
                if st.checkbox("SMOTE ile dengele"):
                    use_smote = True
                    smote = SMOTE(random_state=RANDOM_STATE)
                    X, y = smote.fit_resample(X, y)
                    st.success("SMOTE uygulandı. Yeni boyut:", X.shape)
                    st.write(pd.Series(y).value_counts())

            test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
            val_size = st.slider("Doğrulama oranı (%)", 0, 20, 0) / 100
            model_name = st.selectbox("Model", list(MODEL_DICT[task].keys()))
            if st.button("Modeli Eğit"):
                with st.spinner("Eğitiliyor..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size, val_size)
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    model = MODEL_DICT[task][model_name]
                    metrics, model = evaluate_model(model, X_train, y_train, X_test, y_test, task)
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    st.session_state.baseline_metrics = metrics
                    top_feats = get_top_features(model, X.columns, top_n=3)
                    st.session_state.top_features = top_feats
                st.toast("Eğitim tamam!", icon="🎯")
                st.subheader("Metrikler (Baseline)")
                st.json(metrics)
                if hasattr(model, "feature_importances_"):
                    st.subheader("Özellik Önemi")
                    fig_imp = plot_feature_importance(model, X.columns)
                    st.pyplot(fig_imp)
                try:
                    st.subheader("SHAP")
                    fig_shap = shap_summary_plot(model, X_train[:100])
                    st.pyplot(fig_shap)
                except:
                    st.warning("SHAP çalışmadı.")
                if task == "classification":
                    st.subheader("Confusion Matrix")
                    y_pred = model.predict(X_test)
                    fig_cm = plot_confusion_matrix(y_test, y_pred, labels=model.classes_)
                    st.plotly_chart(fig_cm)
                    st.subheader("ROC Eğrisi")
                    fig_roc = plot_roc_curve(model, X_test, y_test, task)
                    if fig_roc: st.plotly_chart(fig_roc)
                    st.subheader("Precision-Recall Eğrisi")
                    fig_pr = plot_precision_recall_curve(model, X_test, y_test, task)
                    if fig_pr: st.plotly_chart(fig_pr)
                else:
                    st.subheader("Residual Plot")
                    y_pred = model.predict(X_test)
                    residuals = y_test - y_pred
                    fig_res = px.scatter(x=y_pred, y=residuals, title="Hatalar")
                    fig_res.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig_res)
                st.subheader("Öğrenme Eğrisi")
                fig_lc = plot_learning_curve(model, X_train, y_train)
                st.plotly_chart(fig_lc)
                if st.button("Modeli Kaydet"):
                    joblib.dump(model, f"models/{model_name}.joblib")
                    st.toast("Kaydedildi.", icon="💾")

            st.subheader("Model Karşılaştırması")
            if st.button("Tüm Modelleri Karşılaştır"):
                with st.spinner("Karşılaştırılıyor..."):
                    # Veriyi bölmeden önce NaN temizliği yapalım
                    X_temp = X.copy()
                    y_temp = y.copy()
                    for col in X_temp.columns:
                        if X_temp[col].isnull().any():
                            if X_temp[col].dtype in [np.float64, np.int64]:
                                X_temp[col].fillna(X_temp[col].median(), inplace=True)
                            else:
                                X_temp[col].fillna('missing', inplace=True)
                    X_train, _, X_test, y_train, _, y_test = split_data(X_temp, y_temp, test_size=0.2)
                    results = compare_models(task, X_train, y_train, X_test, y_test)
                    st.write(pd.DataFrame(results.items(), columns=["Model", "Skor"]).sort_values("Skor", ascending=False))

            st.subheader("Hiperparametre Optimizasyonu")
            if st.button("Optuna ile Optimize Et"):
                if not st.session_state.model_trained:
                    st.error("Önce model eğitin.")
                else:
                    with st.spinner("Optuna arıyor..."):
                        best_model, best_params = optuna_optimize(model_name, task, X_train, y_train, n_trials=30)
                    metrics_opt, _ = evaluate_model(best_model, X_train, y_train, X_test, y_test, task)
                    st.session_state.optimized_metrics = metrics_opt
                    st.session_state.model = best_model
                    top_feats = get_top_features(best_model, X.columns, top_n=3)
                    st.session_state.top_features = top_feats
                st.success(f"En iyi: {best_params}")
                col_b, col_o = st.columns(2)
                with col_b:
                    st.subheader("Baseline Metrikler")
                    st.json(st.session_state.baseline_metrics)
                with col_o:
                    st.subheader("Optimize Metrikler")
                    st.json(metrics_opt)

    with tabs[5]:
        st.header("PDF Rapor")
        if st.button("Rapor Oluştur"):
            with st.spinner("Rapor hazırlanıyor..."):
                metrics = st.session_state.get("optimized_metrics") or st.session_state.get("baseline_metrics")
                top_feats = st.session_state.get("top_features", [])
                pdf_bytes = generate_pdf_report(df, target, st.session_state.get("model"), metrics, top_feats)
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="rapor.pdf">📥 İndir</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.toast("Rapor hazır!", icon="📄")

    with tabs[6]:
        st.header("Tahmin")
        if not st.session_state.model_trained:
            st.warning("Lütfen önce bir model eğitin.")
        else:
            tahmin_yontemi = st.radio("Tahmin yöntemi", ["CSV Yükle", "Manuel Girdi"])
            if tahmin_yontemi == "CSV Yükle":
                uploaded_pred = st.file_uploader("Tahmin CSV", type="csv")
                if uploaded_pred and st.button("Tahmin Yap"):
                    pred_df = load_data(uploaded_pred)
                    pred_processed = pd.get_dummies(pred_df, drop_first=True)
                    missing_cols = set(st.session_state.X_train.columns) - set(pred_processed.columns)
                    for c in missing_cols:
                        pred_processed[c] = 0
                    pred_processed = pred_processed[st.session_state.X_train.columns]
                    preds = st.session_state.model.predict(pred_processed)
                    if task == "classification":
                        class_labels = st.session_state.y_train.unique()
                        label_desc = {0: class_labels[0], 1: class_labels[1]} if len(class_labels)==2 else {i:lbl for i,lbl in enumerate(class_labels)}
                        readable = [label_desc.get(p, p) for p in preds]
                        st.write("Tahminler:", readable)
                    else:
                        st.write("Tahminler:", preds)
            else:
                st.subheader("En Önemli 3 Değişken ile Tahmin")
                if not st.session_state.top_features:
                    st.warning("Önce model eğitin.")
                else:
                    top_feats = st.session_state.top_features
                    cols = st.columns(3)
                    user_input = {}
                    for i, (feat, _) in enumerate(top_feats):
                        with cols[i]:
                            val = st.number_input(f"{feat}", value=0.0, step=0.1, key=f"manual_{feat}")
                            user_input[feat] = val
                    if st.button("Manuel Tahmin Yap"):
                        input_df = pd.DataFrame([user_input])
                        for c in st.session_state.X_train.columns:
                            if c not in input_df.columns:
                                input_df[c] = 0.0
                        input_df = input_df[st.session_state.X_train.columns]
                        pred = st.session_state.model.predict(input_df)[0]
                        if task == "classification":
                            class_labels = st.session_state.y_train.unique()
                            label_desc = {0: class_labels[0], 1: class_labels[1]} if len(class_labels)==2 else {i:lbl for i,lbl in enumerate(class_labels)}
                            pred_label = label_desc.get(pred, pred)
                            st.success(f"Tahmin: **{pred_label}**")
                            if pred == 0:
                                st.info("0 sınıfı: " + str(label_desc[0]))
                            else:
                                st.info("1 sınıfı: " + str(label_desc[1]))
                        else:
                            st.success(f"Tahmin: **{pred}**")
