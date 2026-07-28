"""CRISP-DM Streamlit Uygulaması."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import cross_val_score
import os
import sys

# src dizinini sys.path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_data, get_basic_info
from src.cleaning import drop_duplicates, handle_missing_values, remove_outliers, fix_data_types
from src.eda import descriptive_stats, check_skewness, correlation_matrix, vif_analysis, normality_test, compare_groups
from src.feature_engineering import extract_date_features, scale_numeric, encode_categorical
from src.modeling import MODEL_DICT, split_data, evaluate_model, hyperparameter_tuning, save_model
from src.pipeline import create_preprocessing_pipeline, create_full_pipeline, save_pipeline

st.set_page_config(layout="wide")
st.title("📊 CRISP-DM Veri Bilimi Asistanı")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None

# ------ Veri Yükleme Sekmesi ------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📂 Veri Yükleme", "🧹 Temizleme", "📊 EDA", "⚙️ Özellik Müh.", "🤖 Modelleme", "🚀 Tahmin"])

with tab1:
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV veya Excel dosyası yükleyin", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Veri başarıyla yüklendi!")
        st.subheader("İlk 5 Satır")
        st.dataframe(df.head())
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sütun Tipleri")
            st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Sütun", 0: "Tip"}))
        with col2:
            st.subheader("Eksik Değerler")
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Sütun", "Eksik"]
            missing["%"] = (missing["Eksik"] / len(df)) * 100
            st.dataframe(missing)
        st.subheader("Hedef Değişkeni Seçin")
        target = st.selectbox("Hedef sütunu", df.columns)
        st.session_state.target = target
        st.write(f"Hedef değişken: **{target}**")
        st.write(f"Proje tipini daha sonra belirleyeceksiniz.")

# Diğer sekmeler yalnızca veri yüklendiğinde aktif
if st.session_state.df is not None:
    df = st.session_state.df
    target = st.session_state.target

    with tab2:
        st.header("Veri Temizleme")
        if st.button("Tekrarlanan Satırları Sil"):
            df = drop_duplicates(df)
            st.success("Tekrarlanan satırlar silindi.")
            st.session_state.df = df

        st.subheader("Eksik Veri Yönetimi")
        miss_cols = st.multiselect("Eksik işlemi yapılacak sütunlar (boş bırakılırsa tümü)", df.columns)
        strategy = st.selectbox("Strateji", ["median", "mean", "mode", "constant", "drop"])
        if strategy == "constant":
            fill_val = st.text_input("Sabit değer", "Bilinmiyor")
        else:
            fill_val = None
        if st.button("Eksik Değerleri Doldur/Sil"):
            df = handle_missing_values(df, strategy=strategy, fill_value=fill_val, columns=None if not miss_cols else miss_cols)
            st.session_state.df = df
            st.success("İşlem tamamlandı.")

        st.subheader("Aykırı Değerler")
        out_cols = st.multiselect("Sayısal sütunlar (boşsa tüm sayısallar)", df.select_dtypes(include=np.number).columns)
        method = st.radio("Yöntem", ["iqr", "zscore"])
        thresh = st.number_input("Eşik", 1.0, 5.0, 1.5)
        if st.button("Aykırı Değerleri Temizle"):
            df = remove_outliers(df, method=method, threshold=thresh, columns=None if not out_cols else out_cols)
            st.session_state.df = df
            st.success("Aykırı değerler temizlendi.")

    with tab3:
        st.header("Keşifsel Veri Analizi")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            st.subheader("Tanımlayıcı İstatistikler")
            st.dataframe(descriptive_stats(df))
            st.subheader("Çarpıklık")
            st.dataframe(check_skewness(df).to_frame("Skewness"))
            st.subheader("Korelasyon Matrisi")
            corr = correlation_matrix(df)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("VIF Analizi")
            if len(num_cols) > 1:
                vif_df = vif_analysis(df)
                st.dataframe(vif_df)

        if target and target in df.columns:
            st.subheader(f"Hedef Değişken: {target}")
            if df[target].dtype in [np.int64, np.float64]:
                fig = px.histogram(df, x=target, marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(df, x=target)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Özellik Mühendisliği")
        # Tarih özellikleri
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if not date_cols:
            st.info("Veride datetime sütunu bulunamadı.")
        else:
            date_sel = st.selectbox("Tarih sütunu", date_cols)
            if st.button("Tarih özellikleri çıkar"):
                df = extract_date_features(df, date_sel)
                st.session_state.df = df
                st.success("Özellikler eklendi.")
        st.subheader("Ölçeklendirme")
        scale_cols = st.multiselect("Ölçeklenecek sütunlar", df.select_dtypes(include=np.number).columns)
        scale_method = st.selectbox("Ölçeklendirme yöntemi", ["standard", "minmax", "robust"])
        if st.button("Ölçeklendir"):
            df = scale_numeric(df, scale_cols, scale_method)
            st.session_state.df = df
            st.success("Ölçeklendirme tamamlandı.")
        st.subheader("Kategorik Kodlama")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            cat_sel = st.multiselect("Kodlanacak sütunlar", cat_cols)
            enc_method = st.selectbox("Kodlama yöntemi", ["onehot", "label"])
            if st.button("Kodla"):
                df = encode_categorical(df, cat_sel, method=enc_method)
                st.session_state.df = df
                st.success("Kodlama tamamlandı.")

    with tab5:
        st.header("Modelleme")
        if target is None:
            st.warning("Hedef değişken seçilmedi.")
        else:
            y = df[target]
            X = df.drop(columns=[target])
            # Kategorik hala varsa one-hot yap (basit)
            X = pd.get_dummies(X, drop_first=True)
            # Hedef tipine göre task belirle
            if y.dtype in [np.int64, np.float64] and y.nunique() > 10:
                task = "regression"
            else:
                task = "classification"
                y = y.astype(str)
            st.write(f"Tespit edilen görev: **{task}**")
            test_size = st.slider("Test seti boyutu (%)", 10, 40, 20)
            val_size = st.slider("Doğrulama seti boyutu (%)", 0, 20, 0)
            model_name = st.selectbox("Model seçin", list(MODEL_DICT[task].keys()))
            if st.button("Modeli Eğit ve Değerlendir"):
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=test_size/100, val_size=val_size/100)
                model = MODEL_DICT[task][model_name]
                metrics = evaluate_model(model, X_train, y_train, X_test, y_test, task)
                st.subheader("Test Metrikleri")
                st.json(metrics)
                # Kaydetme opsiyonu
                if st.button("Modeli Kaydet"):
                    save_model(model, f"models/{model_name.replace(' ', '_')}_{task}.joblib")
                    st.success("Model kaydedildi.")
            # Hiperparametre optimizasyonu (basit grid)
            if st.checkbox("Hiperparametre Optimizasyonu"):
                param_grid = {}
                if model_name == "Random Forest":
                    param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
                elif model_name == "Logistic Regression" or model_name == "Linear Regression":
                    param_grid = {"C": [0.1, 1, 10]} if task=="classification" else {}
                if param_grid:
                    if st.button("Optimize Et"):
                        model = MODEL_DICT[task][model_name]
                        best_model, best_params = hyperparameter_tuning(model, param_grid, X_train, y_train)
                        st.write("En iyi parametreler:", best_params)
                        metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test, task)
                        st.json(metrics)
