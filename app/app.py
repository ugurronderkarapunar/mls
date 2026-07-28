"""Veri Analisti Dashboard – Profil Odaklı Bağımsız Uygulama"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ydata_profiling import ProfileReport
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------
st.set_page_config(page_title="Veri Analisti Dashboard", layout="wide")
st.title("📋 Veri Analisti Dashboard")
st.markdown("Veri setinizi yükleyin, otomatik profil raporu ve etkileşimli analizlerle hemen tanıyın.")

# ---------------------------------------------------------------------
# Sidebar – Veri Yükleme
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV veya Excel", type=["csv", "xlsx", "xls"])
    
if uploaded_file is not None:
    # Veriyi oku
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # -----------------------------------------------------------------
    # Üst Kartlar
    # -----------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gözlem Sayısı", df.shape[0])
    col2.metric("Değişken Sayısı", df.shape[1])
    missing_total = df.isnull().sum().sum()
    missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
    col3.metric("Eksik Hücre", missing_total, f"%{missing_pct:.1f}")
    col4.metric("Tekrar Eden Satır", df.duplicated().sum())
    
    # -----------------------------------------------------------------
    # Profil Raporu (Ana Öğe)
    # -----------------------------------------------------------------
    st.header("📄 Otomatik Profil Raporu")
    st.markdown("Veri setinizin kapsamlı profili. Kaydırarak tüm bölümleri inceleyebilirsiniz.")
    
    if st.button("Profil Raporu Oluştur / Yenile", key="profile_btn"):
        with st.spinner("Profil raporu hazırlanıyor... (büyük verilerde biraz zaman alabilir)"):
            profile = ProfileReport(
                df,
                title="Veri Profil Raporu",
                explorative=True,
                minimal=False,
                progress_bar=False,
            )
            # Doğrudan HTML string olarak al
            html_str = profile.to_html()
        # Streamlit içinde göster
        components.html(html_str, height=800, scrolling=True)
    else:
        st.info("Raporu oluşturmak için yukarıdaki butona tıklayın.")
    
    # -----------------------------------------------------------------
    # Eksik Veri Haritası
    # -----------------------------------------------------------------
    st.header("🗺️ Eksik Veri Isı Haritası")
    st.markdown("Kırmızı = eksik, Yeşil = dolu. Hangi sütunlarda eksiklik var, anında görün.")
    fig_missing = px.imshow(df.isnull(), color_continuous_scale=['green', 'red'], aspect='auto')
    st.plotly_chart(fig_missing, use_container_width=True)
    
    # -----------------------------------------------------------------
    # Dağılım Paneli
    # -----------------------------------------------------------------
    st.header("📊 Dağılım Paneli")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if num_cols:
        dist_col = st.selectbox("Sayısal sütun seçin", num_cols, key="dist_num")
        # Histogram + Box
        fig_dist = px.histogram(df, x=dist_col, marginal="box", title=f"{dist_col} Dağılımı")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # QQ Plot
        data = df[dist_col].dropna()
        if len(data) > 2:
            theoretical = stats.norm.ppf((np.arange(len(data)) + 0.5) / len(data))
            theoretical = np.sort(theoretical)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=theoretical, y=np.sort(data), mode='markers', name='QQ'))
            fig_qq.add_trace(go.Scatter(x=theoretical, y=theoretical * np.std(data) / np.std(theoretical) + np.mean(data),
                                        mode='lines', name='Doğru'))
            fig_qq.update_layout(title=f"{dist_col} Q-Q Plot")
            st.plotly_chart(fig_qq, use_container_width=True)
    
    if cat_cols:
        st.subheader("Kategorik Değişken")
        cat_sel = st.selectbox("Kategorik sütun seçin", cat_cols, key="dist_cat")
        fig_cat = px.histogram(df, x=cat_sel, title=f"{cat_sel} Frekansı")
        st.plotly_chart(fig_cat, use_container_width=True)
    
    # -----------------------------------------------------------------
    # Korelasyon Isı Haritası
    # -----------------------------------------------------------------
    st.header("🔗 Korelasyon Isı Haritası")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Korelasyon için en az 2 sayısal sütun gereklidir.")
    
    # -----------------------------------------------------------------
    # VIF Analizi (opsiyonel)
    # -----------------------------------------------------------------
    if len(num_cols) > 1:
        with st.expander("📈 Varyans Büyütme Faktörü (VIF) Analizi"):
            try:
                temp_df = df[num_cols].fillna(df[num_cols].median())
                vif_data = pd.DataFrame({
                    "Değişken": num_cols,
                    "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(len(num_cols))]
                }).sort_values("VIF", ascending=False)
                st.dataframe(vif_data)
                st.markdown("*VIF > 10: yüksek çoklu bağlantı.*")
            except Exception as e:
                st.warning(f"VIF hesaplanamadı: {e}")
    
    # -----------------------------------------------------------------
    # Hızlı Hipotez Testi
    # -----------------------------------------------------------------
    st.header("🧪 Hızlı Hipotez Testi")
    if cat_cols and num_cols:
        group_col = st.selectbox("Gruplandırma sütunu (kategorik)", cat_cols, key="ht_group")
        value_col = st.selectbox("Test edilecek sayısal sütun", num_cols, key="ht_value")
        alpha = st.slider("Anlamlılık düzeyi (α)", 0.01, 0.10, 0.05)
        
        if st.button("Testi Çalıştır"):
            groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
            if len(groups) >= 2:
                # Normallik testi
                normal = True
                for g in groups:
                    if len(g) >= 3:
                        _, p = stats.shapiro(g)
                        if p < alpha:
                            normal = False
                            break
                if len(groups) == 2:
                    if normal:
                        stat, p = stats.ttest_ind(groups[0], groups[1])
                        test_name = "Bağımsız t-testi"
                    else:
                        stat, p = stats.mannwhitneyu(groups[0], groups[1])
                        test_name = "Mann-Whitney U"
                else:
                    if normal:
                        stat, p = stats.f_oneway(*groups)
                        test_name = "ANOVA"
                    else:
                        stat, p = stats.kruskal(*groups)
                        test_name = "Kruskal-Wallis"
                
                st.write(f"**{test_name}** sonucu:")
                st.write(f"İstatistik = {stat:.4f}, p-değeri = {p:.4f}")
                if p < alpha:
                    st.error(f"Gruplar arasında anlamlı fark var (p < {alpha}).")
                else:
                    st.success(f"Gruplar arasında anlamlı fark yok (p >= {alpha}).")
            else:
                st.warning("En az 2 grup olmalı.")
    else:
        st.info("Hipotez testi için en az bir kategorik ve bir sayısal sütun gerekir.")
    
else:
    st.info("👈 Lütfen sol kenardan bir veri seti yükleyin.")
