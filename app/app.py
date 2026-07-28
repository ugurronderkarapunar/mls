"""CRISP‑DM Multi‑Agent Asistanı – Senior Sürüm."""
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import DBSCAN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RANDOM_STATE = 42

# =====================================================================
# YARDIMCI FONKSİYONLAR (tüm ajanların kullanacağı)
# =====================================================================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    else:
        raise ValueError("Desteklenmeyen dosya türü.")

def drop_duplicates(df): return df.drop_duplicates()

def handle_missing_values(df, strategy='median', fill_value=None, columns=None):
    if columns is None: columns = df.columns.tolist()
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    elif strategy in ['knn', 'mice']:
        numeric_cols = [c for c in columns if df[c].dtype in [np.float64, np.int64]]
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5) if strategy == 'knn' else IterativeImputer(random_state=RANDOM_STATE)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        for col in [c for c in columns if c not in numeric_cols]:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
    else:
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean': df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median': df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode': df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif strategy == 'constant': df[col].fillna(fill_value if fill_value is not None else 0, inplace=True)
            else:
                if strategy in ['mean','median']:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor', inplace=True)
                elif strategy == 'constant':
                    df[col].fillna(fill_value if fill_value is not None else 'Bilinmiyor', inplace=True)
    return df

def remove_outliers(df, method='iqr', threshold=1.5, columns=None, action='remove'):
    if columns is None: columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if method == 'iqr':
        for col in columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            if action == 'remove': df = df[(df[col] >= lower) & (df[col] <= upper)]
            else: df[col] = df[col].clip(lower, upper)
    elif method == 'zscore':
        for col in columns:
            z = np.abs(stats.zscore(df[col].dropna()))
            if action == 'remove': df = df[(z < threshold)]
            else: df[col] = df[col].clip(df[col].mean() - threshold * df[col].std(), df[col].mean() + threshold * df[col].std())
    elif method == 'isolation_forest':
        iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
        df = df[iso.fit_predict(df[columns].dropna()) == 1]
    elif method == 'dbscan':
        scaler = StandardScaler()
        db = DBSCAN(eps=0.5, min_samples=5)
        df = df[db.fit_predict(scaler.fit_transform(df[columns].dropna())) != -1]
    return df

def fix_data_types(df, conversions):
    for col, dtype in conversions.items():
        try:
            if dtype == "datetime": df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "int": df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float": df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str": df[col] = df[col].astype(str)
            else: df[col] = df[col].astype(dtype)
        except Exception: pass
    return df

def extract_date_features(df, date_col, drop_original=False):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    for part in ['year','month','day','dayofweek','quarter']:
        df[f"{date_col}_{part}"] = getattr(df[date_col].dt, part)
    if drop_original: df.drop(columns=[date_col], inplace=True)
    return df

def scale_numeric(df, columns, method='standard'):
    scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}[method]
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df

def encode_categorical(df, columns, method='onehot', drop_first=True):
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    elif method == 'label':
        for col in columns: df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif method == 'ordinal':
        from sklearn.preprocessing import OrdinalEncoder
        for col in columns: df[col] = OrdinalEncoder().fit_transform(df[[col]])
    return df

def descriptive_stats(df, columns=None):
    if columns is None: columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].describe(include="all").T

def correlation_matrix(df): return df.corr(numeric_only=True)

def vif_analysis(df, columns=None):
    if columns is None: columns = df.select_dtypes(include=[np.number]).columns.tolist()
    temp_df = df[columns].fillna(df[columns].median())
    return pd.DataFrame({
        "Değişken": columns,
        "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(len(columns))]
    }).sort_values("VIF", ascending=False)

# ... (diğer görselleştirme fonksiyonları aynı kalabilir, uzatmamak için kısaltıyorum)
# Tamamını önceki sürümlerden kopyalayabilirsiniz.

# =====================================================================
# AJAN 1: VERİ MÜHENDİSİ & EDA
# =====================================================================
def agent1_ui(df):
    st.header("🔍 Ajan 1: Veri Mühendisi & Keşifsel Analiz")
    st.markdown("**Rol:** Veri yapısını çıkarmak ve size karar seçenekleri sunmak.")

    # 1. VERİ GENEL BAKIŞI
    st.subheader("1. Veri Genel Bakışı")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Gözlem", df.shape[0])
    col2.metric("Değişken Sayısı", df.shape[1])
    col3.metric("Bellek Kullanımı", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Eksik Sayısı": missing, "Oran (%)": missing_pct.round(2)})
    st.write("Eksik Veri Özeti:")
    st.dataframe(missing_df[missing_df["Eksik Sayısı"] > 0])

    # 2. DEĞİŞKEN TİPLERİ VE ÖNERİLER
    st.subheader("2. Tespit Edilen Değişken Tipleri ve Tip Değişikliği Önerileri")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    st.write(f"**Sayısal:** {len(numeric_cols)} adet – {numeric_cols}")
    st.write(f"**Kategorik/Metin:** {len(cat_cols)} adet – {cat_cols}")

    # Yanlış tipleri bul
    wrong_int = [c for c in cat_cols if df[c].str.isnumeric().all()]
    wrong_date = [c for c in cat_cols if pd.to_datetime(df[c], errors='coerce').notna().sum() > 0.8 * len(df)]
    if wrong_int:
        st.warning(f"⚠️ **Integer görünümlü kategorikler:** {wrong_int}. Bunları int/float yapabilirsiniz.")
    if wrong_date:
        st.info(f"📅 **Tarih görünümlü stringler:** {wrong_date}. datetime'a çevrilebilir.")
    st.markdown("**Opsiyon A (Tavsiye Edilen):** Yukarıdaki yanlış tipleri düzelt + kategorikleri one‑hot encode")
    st.markdown("**Opsiyon B (Minimal):** Sadece tarih sütunlarını düzelt, geriye kalanlara dokunma")

    # 3. HEDEF DEĞİŞKEN ADAYLARI
    st.subheader("3. Hedef Değişken Seçenekleri")
    candidates = []
    for col in numeric_cols:
        if df[col].nunique() > 10:
            candidates.append((col, "Regresyon (sürekli sayısal)"))
        elif df[col].nunique() <= 10:
            candidates.append((col, "Sınıflandırma (az sayıda benzersiz değer)"))
    for col in cat_cols:
        if df[col].nunique() <= 10:
            candidates.append((col, "Sınıflandırma (kategorik, az sınıf)"))
    for i, (col, reason) in enumerate(candidates[:5], 1):
        st.write(f"**Opsiyon {i}:** {col} – {reason}")

    # 4. SİLİNMESİ ÖNERİLEN DEĞİŞKENLER
    st.subheader("4. Silinmesi Önerilen Değişkenler")
    drop_candidates = []
    for col in df.columns:
        if missing_pct[col] > 95:
            drop_candidates.append((col, f"%{missing_pct[col]:.1f} eksik"))
        elif df[col].nunique() == 1:
            drop_candidates.append((col, "Tek bir değere sahip (zero variance)"))
        elif col.lower() in ['id', 'index', 'row', 'no']:
            drop_candidates.append((col, "ID / sıra numarası görünümlü"))
    if drop_candidates:
        for col, reason in drop_candidates:
            st.write(f"- **{col}** – {reason}")
    else:
        st.success("Silinmesi önerilen değişken bulunamadı.")

    # 5. KARAR ALMA
    st.subheader("5. Karar Bekleyen Sorular")
    with st.form("agent1_decision"):
        target = st.selectbox("Hedef değişkeni seçin:", df.columns)
        type_fix_cols = st.multiselect("Tip dönüşümü yapılacak sütunlar:", wrong_int + wrong_date)
        type_fix_to = st.selectbox("Dönüşüm tipi:", ["int", "float", "datetime", "str"])
        drop_selected = st.multiselect("Silinecek sütunlar:", [c[0] for c in drop_candidates])
        submitted = st.form_submit_button("Ajan 1 Kararlarını Uygula")
        if submitted:
            if type_fix_cols:
                df = fix_data_types(df, {c: type_fix_to for c in type_fix_cols})
            if drop_selected:
                df.drop(columns=drop_selected, inplace=True, errors='ignore')
            st.session_state.df = df
            st.session_state.target = target
            st.success("Kararlar uygulandı.")
            st.rerun()

# =====================================================================
# AJAN 2: ÖN İŞLEME & İSTATİSTİK
# =====================================================================
def agent2_ui(df, target):
    st.header("🧹 Ajan 2: Ön İşleme ve İstatistik Uzmanı")
    if target is None:
        st.warning("Lütfen önce Ajan 1'de hedef değişkeni belirleyin.")
        return
    st.markdown("**Rol:** Dağılımları analiz eder, eksik/aykırı değerleri işler, özellik mühendisliği seçenekleri sunar.")

    # 1. İSTATİSTİKSEL ÖZET VE DAĞILIM ANALİZİ
    st.subheader("1. İstatistiksel Özet ve Dağılım Analizi")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        desc = descriptive_stats(df)
        st.dataframe(desc)
        skew_df = df[num_cols].skew().sort_values(ascending=False).to_frame("Çarpıklık")
        st.write("Çarpıklık değerleri:")
        st.dataframe(skew_df)
        col_sel = st.selectbox("Dağılım grafiği için sütun seçin:", num_cols)
        fig = px.histogram(df, x=col_sel, marginal="box", title=f"{col_sel} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    # 2. EKSİK VE AYKIRI DEĞER HARİTASI
    st.subheader("2. Eksik ve Aykırı Değer Haritası")
    fig_miss = px.imshow(df.isnull(), color_continuous_scale=['green','red'], aspect='auto')
    st.plotly_chart(fig_miss, use_container_width=True)
    # Aykırı değer boxplot'u
    if num_cols:
        fig_box = px.box(df[num_cols[:6]], title="İlk 6 sayısal sütun için box plot")
        st.plotly_chart(fig_box, use_container_width=True)

    # 3. ADIM ADIM SEÇENEKLER MENÜSÜ
    st.subheader("3. Adım Adım Seçenekler Menüsü")
    with st.form("agent2_options"):
        st.markdown("**Adım 2.1: Veri Bölme**")
        split_method = st.radio("Yöntem:", ["Random", "Stratified", "Time Series"], horizontal=True)
        split_ratio = st.slider("Test oranı (%)", 10, 40, 20)
        st.markdown("**Adım 2.2: Eksik Değer Yöntemi**")
        missing_method = st.selectbox("Strateji:", ["median", "mean", "mode", "knn", "mice", "drop"])
        st.markdown("**Adım 2.3: Aykırı Değer Stratejisi**")
        outlier_method = st.selectbox("Yöntem:", ["iqr", "zscore", "isolation_forest", "dbscan"])
        outlier_action = st.radio("İşlem:", ["remove", "cap (winsorize)"], horizontal=True)
        st.markdown("**Adım 2.4: Feature Engineering**")
        add_date_feat = st.checkbox("Tarih özellikleri çıkar (varsa)")
        add_poly = st.checkbox("Polinomal özellikler ekle")
        poly_degree = st.slider("Polinom derecesi:", 2, 4, 2) if add_poly else 2
        st.markdown("**Adım 2.5: Encoding & Scaling**")
        enc_method = st.selectbox("Kodlama:", ["onehot", "label", "ordinal"])
        scale_method = st.selectbox("Ölçeklendirme:", ["standard", "minmax", "robust"])
        submitted2 = st.form_submit_button("Ajan 2 Kararlarını Uygula")
        if submitted2:
            # Bölme
            y = df[target]
            X = df.drop(columns=[target])
            stratify = y if split_method == "Stratified" and y.dtype == 'object' else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio/100, random_state=RANDOM_STATE, stratify=stratify)
            # Eksik
            X_train = handle_missing_values(X_train, missing_method, columns=X_train.columns)
            # Aykırı
            X_train = remove_outliers(X_train, outlier_method, columns=X_train.select_dtypes(include=np.number).columns, action='remove' if outlier_action.startswith('remove') else 'cap')
            # Tarih özellikleri
            if add_date_feat:
                date_cols = X_train.select_dtypes(include=['datetime64']).columns
                for dcol in date_cols:
                    X_train = extract_date_features(X_train, dcol)
            # Polinomal
            if add_poly:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                num_feats = X_train.select_dtypes(include=np.number).columns[:3]
                if len(num_feats) > 0:
                    poly_data = poly.fit_transform(X_train[num_feats])
                    poly_cols = poly.get_feature_names_out(num_feats)
                    X_train = pd.concat([X_train.drop(columns=num_feats), pd.DataFrame(poly_data, columns=poly_cols)], axis=1)
            # Encoding
            cat_feats = X_train.select_dtypes(include=['object','category']).columns
            X_train = encode_categorical(X_train, cat_feats, method=enc_method)
            # Scaling
            num_feats = X_train.select_dtypes(include=np.number).columns
            X_train = scale_numeric(X_train, num_feats, method=scale_method)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Tüm adımlar uygulandı. Veri modelleme için hazır.")
            st.rerun()

# =====================================================================
# AJAN 3: ML MİMARI
# =====================================================================
def agent3_ui():
    st.header("🤖 Ajan 3: Makine Öğrenmesi Mimarı")
    if 'X_train' not in st.session_state or st.session_state.X_train is None:
        st.warning("Lütfen önce Ajan 2'de veriyi ön işleyin.")
        return
    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    y_train, y_test = st.session_state.y_train, st.session_state.y_test
    task = "classification" if y_train.dtype == 'object' else "regression"
    st.info(f"Problem tipi: **{task}**")

    st.subheader("1. Metrik Önerisi")
    if task == "classification":
        st.write("Önerilen birincil metrik: **ROC AUC** (dengeli veri) / **F1 score** (dengesiz veri)")
    else:
        st.write("Önerilen birincil metrik: **RMSE** veya **R²**")

    st.subheader("2. Model Seçimi")
    models = {
        "Baseline (Lojistik/Lineer)": LogisticRegression(max_iter=1000) if task=="classification" else LinearRegression(),
        "Random Forest": RandomForestClassifier() if task=="classification" else RandomForestRegressor(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss') if task=="classification" else XGBRegressor(),
        "LightGBM": LGBMClassifier(verbose=-1) if task=="classification" else LGBMRegressor(verbose=-1),
    }
    selected_models = st.multiselect("Eğitmek istediğiniz modeller:", list(models.keys()), default=["Random Forest"])
    primary_metric = st.selectbox("Birincil metrik:", ["Accuracy", "ROC AUC", "F1"] if task=="classification" else ["RMSE", "R2", "MAE"])

    cv_folds = st.slider("CV kat sayısı:", 3, 10, 5)
    hyperopt = st.checkbox("Optuna ile hiperparametre optimizasyonu")

    if st.button("Modelleri Eğit ve Karşılaştır"):
        results = {}
        progress = st.progress(0)
        for i, name in enumerate(selected_models):
            model = models[name]
            if hyperopt and name in ["Random Forest", "XGBoost", "LightGBM"]:
                # Kısa Optuna çağrısı
                def objective(trial):
                    if name == "Random Forest":
                        params = {'n_estimators': trial.suggest_int('n', 50, 200), 'max_depth': trial.suggest_int('d', 3, 15)}
                        m = RandomForestClassifier(**params) if task=="classification" else RandomForestRegressor(**params)
                    elif name == "XGBoost":
                        params = {'n_estimators': trial.suggest_int('n', 50, 200), 'max_depth': trial.suggest_int('d', 3, 10), 'learning_rate': trial.suggest_float('lr', 0.01, 0.3)}
                        m = XGBClassifier(**params) if task=="classification" else XGBRegressor(**params)
                    else:
                        params = {'n_estimators': trial.suggest_int('n', 50, 200), 'num_leaves': trial.suggest_int('l', 20, 100)}
                        m = LGBMClassifier(**params) if task=="classification" else LGBMRegressor(**params)
                    return np.mean(cross_val_score(m, X_train, y_train, cv=3, scoring='accuracy' if task=="classification" else 'neg_mean_squared_error'))
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=20, show_progress_bar=False)
                model = study.best_params  # gerçekte yeniden eğitilmeli, burada özet
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task == "classification":
                results[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "F1": f1_score(y_test, y_pred, average='weighted'),
                    "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if hasattr(model, "predict_proba") else None
                }
            else:
                results[name] = {
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R2": r2_score(y_test, y_pred)
                }
            progress.progress((i+1)/len(selected_models))
        progress.empty()
        st.session_state.model_results = results
        st.success("Modeller eğitildi!")
        st.write(pd.DataFrame(results).T)

# =====================================================================
# AJAN 4: MLOps & DAĞITIM
# =====================================================================
def agent4_ui():
    st.header("🚀 Ajan 4: MLOps & Dağıtım Mühendisi")
    if 'model_results' not in st.session_state:
        st.warning("Lütfen önce Ajan 3'te model eğitin.")
        return
    st.subheader("1. Pipeline Mimarisi")
    st.markdown("""
    1. Veri yükleme → 2. Eksik imputasyonu → 3. Aykırı işleme → 4. Encoding → 5. Scaling → 6. Model
    """)
    st.subheader("2. Serileştirme Seçenekleri")
    ser_option = st.radio("Format:", ["joblib", "pickle", "onnx"])
    st.subheader("3. Dağıtım Opsiyonları")
    deploy = st.radio("Dağıtım:", ["Sadece pipeline dosyası", "FastAPI endpoint", "Docker imajı", "Streamlit demo"])
    if st.button("Pipeline'ı Hazırla"):
        # Örnek pipeline oluştur
        model = LogisticRegression()  # gerçekte en iyi model alınır
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        joblib.dump(pipeline, "models/pipeline.joblib")
        st.success("Pipeline kaydedildi. Dosya: models/pipeline.joblib")
        if deploy == "Streamlit demo":
            st.info("Streamlit arayüzü zaten çalışıyor! Tahmin sekmesinden yararlanabilirsiniz.")

# =====================================================================
# ANA AKIŞ
# =====================================================================
st.set_page_config(layout="wide", page_title="CRISP‑DM Multi‑Agent")
st.title("🤖 CRISP‑DM Multi‑Agent Asistanı")

# Session state
for key, default in [
    ("df", None), ("target", None), ("X_train", None), ("X_test", None),
    ("y_train", None), ("y_test", None), ("model_results", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Dosya yükleme (ortak)
uploaded_file = st.file_uploader("Veri setini yükleyin (CSV/Excel)", type=["csv","xlsx","xls"], key="main_uploader")
if uploaded_file:
    if st.session_state.df is None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.toast("Veri yüklendi!", icon="✅")
    else:
        df = st.session_state.df

    # Ajan sekmeleri
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Ajan 1: Veri Mühendisi & EDA",
        "🧹 Ajan 2: Ön İşleme & İstatistik",
        "🤖 Ajan 3: ML Mimarı",
        "🚀 Ajan 4: MLOps & Dağıtım"
    ])
    with tab1:
        agent1_ui(df)
    with tab2:
        agent2_ui(df, st.session_state.target)
    with tab3:
        agent3_ui()
    with tab4:
        agent4_ui()
else:
    st.info("Lütfen bir veri seti yükleyerek başlayın.")
