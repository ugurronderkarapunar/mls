"""CRISP-DM Veri Bilimi Asistanı – Full Özellikli Sürüm."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import joblib
import base64
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt
import time
import sys
import os

# app.py'nin bulunduğu dizini (proje kökü) Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# src modülleri
from src.data_loader import load_data
from src.cleaning import drop_duplicates, handle_missing_values, remove_outliers
from src.feature_engineering import extract_date_features, scale_numeric, encode_categorical
from src.modeling import MODEL_DICT, split_data, evaluate_model, save_model, hyperparameter_tuning
from src.interpretability import shap_summary_plot, plot_feature_importance
from src.reporting import generate_pdf_report

st.set_page_config(layout="wide", page_title="CRISP-DM Asistanı")
st.title("📊 CRISP-DM Veri Bilimi Asistanı")

# --- Yardımcı Fonksiyonlar ---
def add_insights(stat_name, value, insight_dict):
    """İstatistiksel değer için otomatik yorum ve öneri üretir."""
    if stat_name in insight_dict:
        thresholds = insight_dict[stat_name]
        for condition, msg in thresholds.items():
            if condition == "default":
                return msg
            if isinstance(condition, str) and eval(condition.format(val=value)):
                return msg
    return ""

def get_skewness_insight(skew_val):
    if abs(skew_val) < 0.5:
        return "✅ Dağılım yaklaşık simetrik, normallik varsayımı için uygundur."
    elif skew_val > 1:
        return "⚠️ Sağa çarpık dağılım. Log dönüşümü veya karekök dönüşümü önerilir. Aykırı değer kontrolü yapın."
    elif skew_val < -1:
        return "⚠️ Sola çarpık dağılım. Kare dönüşümü veya Box-Cox dönüşümü deneyebilirsiniz."
    else:
        return "ℹ️ Orta düzey çarpıklık, normallik testi ile kontrol edin."

def get_vif_insight(vif_val):
    if vif_val < 5:
        return "✅ Çoklu bağlantı sorunu yok."
    elif vif_val < 10:
        return "⚠️ Orta düzey çoklu bağlantı; dikkatle izleyin."
    else:
        return "❌ Yüksek çoklu bağlantı! Değişkeni çıkarmayı veya PCA uygulamayı düşünün."

# --- EDA fonksiyonları (app.py içinde gömülü) ---
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

# Session state initialization
for key, default in [
    ("df", None), ("target", None), ("X_train", None), ("X_test", None),
    ("y_train", None), ("y_test", None), ("task", None), ("cleaning_history", []),
    ("model_trained", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Ana Sekmeler ---
tabs = st.tabs(["📂 Veri Yükleme", "🧹 Temizleme", "📊 EDA", "⚙️ Özellik Müh.", "🤖 Modelleme", "📄 Rapor", "🚀 Tahmin"])

# ==================== 1. VERİ YÜKLEME ====================
with tabs[0]:
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV veya Excel dosyası yükleyin", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        with st.spinner("Veri yükleniyor..."):
            df = load_data(uploaded_file)
        st.session_state.df = df
        st.toast("Veri başarıyla yüklendi!", icon="✅")
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
        st.success(f"Hedef değişken: **{target}**")

        # Değişken silme arayüzü
        st.subheader("Gereksiz Değişkenleri Sil")
        cols_to_drop = st.multiselect("Silmek istediğiniz sütunları seçin", df.columns)
        if st.button("Seçili Sütunları Sil"):
            if target in cols_to_drop:
                st.error("Hedef değişkeni silemezsiniz!")
            else:
                df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.df = df
                st.toast(f"{len(cols_to_drop)} sütun silindi.", icon="🗑️")
                st.rerun()

if st.session_state.df is not None:
    df = st.session_state.df
    target = st.session_state.target

    # ==================== 2. TEMİZLEME ====================
    with tabs[1]:
        st.header("Veri Temizleme")
        if st.button("Tekrarlanan Satırları Sil"):
            df = drop_duplicates(df)
            st.session_state.df = df
            st.toast("Tekrarlanan satırlar silindi.", icon="🧹")

        st.subheader("Eksik Veri Yönetimi")
        miss_cols = st.multiselect("Sütun seç (boş=tümü)", df.columns)
        strategy = st.selectbox("Strateji", ["median", "mean", "mode", "constant", "drop"])
        fill_val = None
        if strategy == "constant":
            fill_val = st.text_input("Sabit değer", "Bilinmiyor")
        if st.button("Eksik Değerleri Doldur/Sil"):
            with st.spinner("Eksik veriler işleniyor..."):
                df = handle_missing_values(df, strategy=strategy, fill_value=fill_val,
                                           columns=None if not miss_cols else miss_cols)
            st.session_state.df = df
            st.toast("Eksik veri işlemi tamamlandı.", icon="✅")

        st.subheader("Aykırı Değerler")
        out_cols = st.multiselect("Sayısal sütunlar", df.select_dtypes(include=np.number).columns)
        method = st.radio("Yöntem", ["iqr", "zscore"])
        thresh = st.number_input("Eşik", 1.0, 5.0, 1.5)
        if st.button("Aykırı Değerleri Temizle"):
            with st.spinner("Aykırı değerler temizleniyor..."):
                df = remove_outliers(df, method=method, threshold=thresh,
                                     columns=None if not out_cols else out_cols)
            st.session_state.df = df
            st.toast("Aykırı değerler temizlendi.", icon="✨")

    # ==================== 3. EDA ====================
    with tabs[2]:
        st.header("Keşifsel Veri Analizi (EDA)")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            st.subheader("Tanımlayıcı İstatistikler")
            desc = descriptive_stats(df)
            st.dataframe(desc)
            with st.expander("📘 İstatistikler Ne Anlama Geliyor?"):
                st.markdown("""
                - **Ortalama (mean)**: Verinin aritmetik ortalaması. Uç değerlerden etkilenir.
                - **Medyan (%50)**: Veriyi ortadan ikiye bölen değer. Aykırı değerlere karşı dirençlidir.
                - **Standart Sapma**: Verinin ortalamadan ne kadar saptığını gösterir. Yüksekse veri yayılımı fazladır.
                - **Min / Max**: En küçük ve en büyük değerler.
                - **%25 / %75 (Çeyrekler)**: Verinin alt ve üst çeyrek sınırları.
                """)

            st.subheader("Çarpıklık (Skewness)")
            skew_vals = check_skewness(df)
            skew_df = skew_vals.to_frame("Çarpıklık")
            skew_df["Yorum / Öneri"] = skew_df["Çarpıklık"].apply(get_skewness_insight)
            st.dataframe(skew_df)
            with st.expander("📘 Çarpıklık Nedir?"):
                st.markdown("""
                **Çarpıklık**, bir dağılımın simetriden ne kadar saptığını ölçer.
                - 0: Simetrik (normal dağılım)
                - Pozitif (>0): Sağa çarpık, tepe solda, kuyruk sağda. Gelir dağılımı buna örnektir.
                - Negatif (<0): Sola çarpık, tepe sağda, kuyruk solda.
                """)

            st.subheader("Korelasyon Matrisi")
            corr = correlation_matrix(df)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📘 Korelasyon Nedir?"):
                st.markdown("""
                **Korelasyon**, iki sayısal değişken arasındaki doğrusal ilişkinin gücünü ve yönünü gösterir.
                - +1: Mükemmel pozitif ilişki (biri artarken diğeri artar)
                - -1: Mükemmel negatif ilişki (biri artarken diğeri azalır)
                - 0: Doğrusal ilişki yok.
                Modellemede yüksek korelasyon (>0.8) çoklu bağlantı sorununa yol açabilir.
                """)

            st.subheader("VIF (Çoklu Bağlantı) Analizi")
            if len(num_cols) > 1:
                vif_df = vif_analysis(df)
                vif_df["Yorum"] = vif_df["VIF"].apply(get_vif_insight)
                st.dataframe(vif_df)
                with st.expander("📘 VIF Nedir?"):
                    st.markdown("""
                    **Varyans Büyütme Faktörü (VIF)**, bir bağımsız değişkenin diğer bağımsız değişkenlerle ne kadar ilişkili olduğunu ölçer.
                    - VIF = 1: Hiç bağlantı yok.
                    - VIF > 5: Orta düzey bağlantı.
                    - VIF > 10: Yüksek çoklu bağlantı, regresyon katsayılarını güvensiz yapar.
                    """)

        if target and target in df.columns:
            st.subheader(f"Hedef Değişken: {target}")
            if df[target].dtype in [np.int64, np.float64]:
                fig = px.histogram(df, x=target, marginal="box", title="Hedef Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(df, x=target, title="Hedef Sınıf Dağılımı")
                st.plotly_chart(fig, use_container_width=True)

    # ==================== 4. ÖZELLİK MÜHENDİSLİĞİ ====================
    with tabs[3]:
        st.header("Özellik Mühendisliği")
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if date_cols:
            date_sel = st.selectbox("Tarih sütunu", date_cols)
            if st.button("Tarih özellikleri çıkar"):
                df = extract_date_features(df, date_sel)
                st.session_state.df = df
                st.toast("Tarih özellikleri eklendi.", icon="📅")
        else:
            st.info("Datetime sütunu bulunamadı.")

        st.subheader("Ölçeklendirme")
        scale_cols = st.multiselect("Sütunlar", df.select_dtypes(include=np.number).columns)
        scale_method = st.selectbox("Yöntem", ["standard", "minmax", "robust"])
        if st.button("Ölçeklendir"):
            df = scale_numeric(df, scale_cols, scale_method)
            st.session_state.df = df
            st.toast("Ölçeklendirme tamamlandı.", icon="⚖️")

        st.subheader("Kategorik Kodlama")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            cat_sel = st.multiselect("Kodlanacak sütunlar", cat_cols)
            enc_method = st.selectbox("Yöntem", ["onehot", "label"])
            if st.button("Kodla"):
                df = encode_categorical(df, cat_sel, method=enc_method)
                st.session_state.df = df
                st.toast("Kodlama tamamlandı.", icon="🔢")

    # ==================== 5. MODELLEME ====================
    with tabs[4]:
        st.header("Modelleme")
        if target is None:
            st.warning("Hedef değişken seçilmedi.")
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
            st.info(f"Görev tipi: **{task}**")

            test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
            val_size = st.slider("Doğrulama oranı (%)", 0, 20, 0) / 100
            model_name = st.selectbox("Model", list(MODEL_DICT[task].keys()))

            if st.button("Modeli Eğit"):
                with st.spinner("Model eğitiliyor..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                        X, y, test_size=test_size, val_size=val_size
                    )
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    model = MODEL_DICT[task][model_name]
                    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, task)
                    st.session_state.model = model
                    st.session_state.model_trained = True
                st.toast("Model eğitimi tamamlandı!", icon="🎯")
                st.subheader("Test Metrikleri")
                st.json(metrics)

                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    st.subheader("Özellik Önem Düzeyleri")
                    fig_imp = plot_feature_importance(model, X.columns)
                    st.pyplot(fig_imp)
                # SHAP
                st.subheader("SHAP Özet Grafiği")
                try:
                    shap_fig = shap_summary_plot(model, X_train[:100])
                    st.pyplot(shap_fig)
                except Exception as e:
                    st.warning(f"SHAP görselleştirilemedi: {e}")

                if st.button("Modeli Kaydet"):
                    save_model(model, f"models/{model_name.replace(' ', '_')}.joblib")
                    st.toast("Model kaydedildi.", icon="💾")

            if st.checkbox("Hiperparametre Optimizasyonu"):
                param_grid = {}
                if model_name == "Random Forest":
                    param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
                if param_grid and st.button("Optimize Et"):
                    with st.spinner("Optimum parametreler aranıyor..."):
                        best_model, best_params = hyperparameter_tuning(
                            MODEL_DICT[task][model_name], param_grid,
                            st.session_state.X_train, st.session_state.y_train
                        )
                    st.success(f"En iyi parametreler: {best_params}")
                    metrics = evaluate_model(best_model, st.session_state.X_train, st.session_state.y_train,
                                             st.session_state.X_test, st.session_state.y_test, task)
                    st.json(metrics)

    # ==================== 6. RAPOR ====================
    with tabs[5]:
        st.header("PDF Raporlama")
        if st.button("PDF Rapor Oluştur"):
            with st.spinner("Rapor hazırlanıyor..."):
                pdf_bytes = generate_pdf_report(df, target, st.session_state.get("model"))
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="eda_rapor.pdf">📥 Raporu İndir</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.toast("Rapor hazır!", icon="📄")

    # ==================== 7. TAHMİN ====================
    with tabs[6]:
        st.header("Yeni Veri ile Tahmin")
        if st.session_state.model_trained:
            st.success("Eğitilmiş model hazır.")
            uploaded_pred = st.file_uploader("Tahmin için CSV yükleyin", type="csv")
            if uploaded_pred and st.button("Tahmin Yap"):
                pred_df = load_data(uploaded_pred)
                # Aynı ön işleme adımları (basitçe)
                pred_processed = pd.get_dummies(pred_df, drop_first=True)
                # Eksik sütunları ekle (modelin eğitildiği sütunlar)
                missing_cols = set(st.session_state.X_train.columns) - set(pred_processed.columns)
                for c in missing_cols:
                    pred_processed[c] = 0
                pred_processed = pred_processed[st.session_state.X_train.columns]
                predictions = st.session_state.model.predict(pred_processed)
                st.write("Tahminler:", predictions)
        else:
            st.warning("Lütfen önce bir model eğitin.")
