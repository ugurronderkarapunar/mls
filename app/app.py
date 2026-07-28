"""CRISP-DM Veri Bilimi Asistanı – Gelişmiş Sürüm."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import base64
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Geçersiz ölçeklendirme yöntemi.")
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df

def encode_categorical(df, columns, method='onehot', drop_first=True):
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
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
    # Drop original columns to avoid duplication (user can choose)
    df = df.drop(columns=columns)
    return pd.concat([df, poly_df], axis=1)

def add_interaction_features(df, col_pairs):
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df

# ---------------------------------------------------------------------
# Modelleme
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna

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

def split_data(X, y, test_size=0.2, val_size=0.0, random_state=42):
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
            metrics["ROC AUC"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }
    return metrics

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def optuna_optimize(model_name, task_type, X_train, y_train, n_trials=30):
    if model_name == "Random Forest":
        if task_type == "classification":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
                model = RandomForestClassifier(**params, random_state=42)
                return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy'))
        else:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
                model = RandomForestRegressor(**params, random_state=42)
                return -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))
    elif model_name == "XGBoost":
        if task_type == "classification":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
                return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy'))
        else:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = XGBRegressor(**params, random_state=42)
                return -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))
    elif model_name == "LightGBM":
        if task_type == "classification":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                }
                model = LGBMClassifier(**params, random_state=42, verbose=-1)
                return np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy'))
        else:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                }
                model = LGBMRegressor(**params, random_state=42, verbose=-1)
                return -np.mean(cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error'))
    else:
        return MODEL_DICT[task_type][model_name], {}
    study = optuna.create_study(direction='maximize' if task_type=='classification' else 'minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    if model_name == "Random Forest":
        if task_type == "classification":
            best_model = RandomForestClassifier(**best_params, random_state=42)
        else:
            best_model = RandomForestRegressor(**best_params, random_state=42)
    elif model_name == "XGBoost":
        if task_type == "classification":
            best_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            best_model = XGBRegressor(**best_params, random_state=42)
    elif model_name == "LightGBM":
        if task_type == "classification":
            best_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        else:
            best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    best_model.fit(X_train, y_train)
    return best_model, best_params

# ---------------------------------------------------------------------
# Model yorumlama
# ---------------------------------------------------------------------
import shap

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

def generate_pdf_report(df, target, model=None):
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
    if model:
        pdf.set_font("Arial", size=11)
        pdf.write(5, "\n\nModel basariyla egitildi.")
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

# ============= Streamlit Arayüzü =============
st.set_page_config(layout="wide", page_title="CRISP-DM Asistanı")
st.title("📊 CRISP-DM Veri Bilimi Asistanı")

# Session state
for key, default in [
    ("df", None), ("target", None), ("X_train", None), ("X_test", None),
    ("y_train", None), ("y_test", None), ("task", None), ("model", None),
    ("model_trained", False), ("uploaded_file_name", None), ("top_features", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

tabs = st.tabs(["📂 Veri Yükleme", "🧹 Temizleme", "📊 EDA", "⚙️ Özellik Müh.", "🤖 Modelleme", "📄 Rapor", "🚀 Tahmin"])

# -------------------- 1. VERİ YÜKLEME --------------------
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

    # -------------------- 2. TEMİZLEME --------------------
    with tabs[1]:
        st.header("Veri Temizleme")
        # Tekrar eden satırlar
        dup_count = df.duplicated().sum()
        st.write(f"Tekrar eden satır sayısı: **{dup_count}**")
        if st.button("Tekrarlanan Satırları Sil"):
            df = drop_duplicates(df)
            st.session_state.df = df
            st.toast(f"{dup_count} tekrar silindi.", icon="🧹")

        # Eksik veri yönetimi
        st.subheader("Eksik Veri")
        miss_cols = st.multiselect("Sütun (boş=tümü)", df.columns)
        strategy = st.selectbox("Strateji", ["median", "mean", "mode", "constant", "drop"])
        fill_val = None
        if strategy == "constant":
            fill_val = st.text_input("Sabit değer", "Bilinmiyor")
        if st.button("Eksikleri Doldur/Sil"):
            before = df.isnull().sum().sum()
            df = handle_missing_values(df, strategy, fill_val, miss_cols if miss_cols else None)
            after = df.isnull().sum().sum()
            st.session_state.df = df
            st.toast(f"İşlem tamamlandı. Eksik: {before} -> {after}", icon="✅")

        # Aykırı değerler
        st.subheader("Aykırı Değerler")
        out_cols = st.multiselect("Sayısal sütunlar", df.select_dtypes(include=np.number).columns)
        method = st.radio("Yöntem", ["iqr", "zscore"])
        action = st.radio("İşlem", ["remove", "cap (sınırlandır)"])
        thresh = st.number_input("Eşik", 1.0, 5.0, 1.5)
        if st.button("Aykırıları İşle"):
            act = 'remove' if action.startswith("remove") else 'cap'
            df = remove_outliers(df, method, thresh, out_cols if out_cols else None, action=act)
            st.session_state.df = df
            st.toast("İşlem tamamlandı.", icon="✨")

        # Veri tipi düzeltme
        st.subheader("Veri Tipi Düzeltme")
        col_to_fix = st.multiselect("Tipini değiştirmek istediğiniz sütunlar", df.columns)
        new_type = st.selectbox("Yeni tip", ["int", "float", "str", "datetime"])
        if st.button("Tipi Dönüştür"):
            conversions = {c: new_type for c in col_to_fix}
            df = fix_data_types(df, conversions)
            st.session_state.df = df
            st.toast("Tipler güncellendi.", icon="🔄")

    # -------------------- 3. EDA --------------------
    with tabs[2]:
        st.header("Keşifsel Veri Analizi")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Eksik veri ısı haritası
        st.subheader("Eksik Veri Haritası")
        missing_data = df.isnull()
        fig_missing = px.imshow(missing_data, color_continuous_scale=['green', 'red'], aspect='auto')
        st.plotly_chart(fig_missing, use_container_width=True)

        if num_cols:
            st.subheader("Tanımlayıcı İstatistikler")
            st.dataframe(descriptive_stats(df))
            # Box plot
            col_sel = st.selectbox("Box Plot için sütun seçin", num_cols)
            fig_box = px.box(df, y=col_sel, title=f"{col_sel} Box Plot")
            st.plotly_chart(fig_box, use_container_width=True)
            # Violin plot
            fig_violin = px.violin(df, y=col_sel, box=True, title=f"{col_sel} Violin Plot")
            st.plotly_chart(fig_violin, use_container_width=True)

            st.subheader("Çarpıklık")
            skew_df = check_skewness(df).to_frame("Çarpıklık")
            skew_df["Yorum"] = skew_df["Çarpıklık"].apply(get_skewness_insight)
            st.dataframe(skew_df)

            st.subheader("Korelasyon Matrisi")
            corr = correlation_matrix(df)
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

            if len(num_cols) > 1:
                st.subheader("VIF")
                vif_df = vif_analysis(df)
                vif_df["Yorum"] = vif_df["VIF"].apply(get_vif_insight)
                st.dataframe(vif_df)

        if cat_cols:
            st.subheader("Kategorik Değişkenler")
            cat_sel = st.selectbox("Count Plot için sütun seçin", cat_cols)
            fig_count = px.histogram(df, x=cat_sel, color=target if target in df.columns else None)
            st.plotly_chart(fig_count, use_container_width=True)

        if target and target in df.columns:
            st.subheader(f"Hedef: {target}")
            if df[target].dtype in [np.int64, np.float64]:
                fig_target = px.histogram(df, x=target, marginal="box")
            else:
                fig_target = px.histogram(df, x=target)
            st.plotly_chart(fig_target, use_container_width=True)

    # -------------------- 4. ÖZELLİK MÜHENDİSLİĞİ --------------------
    with tabs[3]:
        st.header("Özellik Mühendisliği")
        # Tarih
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if date_cols:
            date_sel = st.selectbox("Tarih sütunu", date_cols)
            if st.button("Tarih özellikleri çıkar"):
                df = extract_date_features(df, date_sel)
                st.session_state.df = df
                st.toast("Eklendi.", icon="📅")
        else:
            st.info("Datetime sütunu yok.")

        # Ölçeklendirme
        st.subheader("Ölçeklendirme")
        scale_cols = st.multiselect("Sütunlar", df.select_dtypes(include=np.number).columns)
        scale_method = st.selectbox("Yöntem", ["standard", "minmax", "robust"])
        if st.button("Ölçeklendir"):
            df = scale_numeric(df, scale_cols, scale_method)
            st.session_state.df = df
            st.toast("Ölçeklendi.", icon="⚖️")

        # Kategorik kodlama
        st.subheader("Kategorik Kodlama")
        cat_cols_fe = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols_fe:
            cat_sel_fe = st.multiselect("Kodlanacak", cat_cols_fe)
            enc_method = st.selectbox("Yöntem", ["onehot", "label"])
            if st.button("Kodla"):
                df = encode_categorical(df, cat_sel_fe, enc_method)
                st.session_state.df = df
                st.toast("Kodlandı.", icon="🔢")

        # Polinomal özellikler
        st.subheader("Polinomal Özellikler")
        poly_cols = st.multiselect("Derece alınacak sayısal sütunlar", df.select_dtypes(include=np.number).columns)
        poly_degree = st.slider("Derece", 2, 4, 2)
        if st.button("Polinomal Özellik Ekle"):
            df = add_polynomial_features(df, poly_cols, poly_degree)
            st.session_state.df = df
            st.toast("Polinomal özellikler eklendi.", icon="📈")

        # Etkileşim (çarpım)
        st.subheader("Etkileşim Özellikleri")
        int_col1 = st.selectbox("Birinci sütun", df.select_dtypes(include=np.number).columns, key="int1")
        int_col2 = st.selectbox("İkinci sütun", df.select_dtypes(include=np.number).columns, key="int2")
        if st.button("Çarpım Ekle"):
            df = add_interaction_features(df, [(int_col1, int_col2)])
            st.session_state.df = df
            st.toast(f"{int_col1} x {int_col2} eklendi.", icon="✖️")

    # -------------------- 5. MODELLEME --------------------
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
            test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
            val_size = st.slider("Doğrulama oranı (%)", 0, 20, 0) / 100
            model_name = st.selectbox("Model", list(MODEL_DICT[task].keys()))
            if st.button("Modeli Eğit"):
                with st.spinner("Eğitiliyor..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size, val_size)
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    model = MODEL_DICT[task][model_name]
                    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, task)
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    top_feats = get_top_features(model, X.columns, top_n=3)
                    st.session_state.top_features = top_feats
                st.toast("Eğitim tamam!", icon="🎯")
                st.subheader("Metrikler")
                st.json(metrics)
                if hasattr(model, "feature_importances_"):
                    st.subheader("Özellik Önemi")
                    fig = plot_feature_importance(model, X.columns)
                    st.pyplot(fig)
                try:
                    st.subheader("SHAP")
                    fig_shap = shap_summary_plot(model, X_train[:100])
                    st.pyplot(fig_shap)
                except Exception:
                    st.warning("SHAP çalışmadı.")
                if st.button("Modeli Kaydet"):
                    save_model(model, f"models/{model_name}.joblib")
                    st.toast("Kaydedildi.", icon="💾")

            st.subheader("Hiperparametre Optimizasyonu")
            if st.button("Optuna ile Optimize Et"):
                if not st.session_state.model_trained:
                    st.error("Önce model eğitin.")
                else:
                    with st.spinner("Optuna arıyor..."):
                        best_model, best_params = optuna_optimize(
                            model_name, task,
                            st.session_state.X_train, st.session_state.y_train, n_trials=30)
                    st.success(f"En iyi: {best_params}")
                    metrics = evaluate_model(best_model, st.session_state.X_train, st.session_state.y_train,
                                             st.session_state.X_test, st.session_state.y_test, task)
                    st.json(metrics)
                    st.session_state.model = best_model
                    top_feats = get_top_features(best_model, st.session_state.X_train.columns, top_n=3)
                    st.session_state.top_features = top_feats

    # -------------------- 6. RAPOR --------------------
    with tabs[5]:
        st.header("PDF Rapor")
        if st.button("Rapor Oluştur"):
            with st.spinner("Rapor hazırlanıyor..."):
                pdf_bytes = generate_pdf_report(df, target, st.session_state.get("model"))
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="rapor.pdf">📥 İndir</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.toast("Rapor hazır!", icon="📄")

    # -------------------- 7. TAHMİN --------------------
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
                    if st.session_state.task == "classification":
                        # Map predictions to original labels if possible
                        try:
                            label_map = {str(i): label for i, label in enumerate(st.session_state.y_train.unique())}
                            preds_readable = [label_map.get(str(p), str(p)) for p in preds]
                        except:
                            preds_readable = preds
                        st.write("Tahminler (sınıflar):", preds_readable)
                    else:
                        st.write("Tahminler:", preds)
            else:
                st.subheader("En Önemli 3 Değişken ile Tahmin")
                if "top_features" not in st.session_state or not st.session_state.top_features:
                    st.warning("Model eğitildikten sonra önemli değişkenler belirlenecek.")
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
                        if st.session_state.task == "classification":
                            # Label mapping
                            try:
                                label_map = {str(i): label for i, label in enumerate(st.session_state.y_train.unique())}
                                pred_label = label_map.get(str(pred), str(pred))
                            except:
                                pred_label = pred
                            st.success(f"Tahmin: **{pred_label}**")
                        else:
                            st.success(f"Tahmin: **{pred}**")
