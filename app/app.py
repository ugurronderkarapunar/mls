"""Veri Analisti Dashboard – Tam Teşekküllü Sürüm"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportions_ztest
from ydata_profiling import ProfileReport
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import missingno as msno
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fpdf import FPDF
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Sayfa konfigürasyonu
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Veri Analisti Dashboard", layout="wide")
st.title("📋 Veri Analisti Dashboard")
st.markdown("Veri setinizi yükleyin, analiz edin, içgörüleri keşfedin.")

# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------
def detect_outliers_iqr(series, threshold=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
    return series[(series < lower) | (series > upper)]

def get_outlier_summary(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    summary = []
    for col in columns:
        outliers = detect_outliers_iqr(df[col].dropna())
        summary.append({
            "Sütun": col,
            "Aykırı Sayısı": len(outliers),
            "Aykırı Oranı (%)": round(100 * len(outliers) / max(len(df[col].dropna()), 1), 2),
        })
    return pd.DataFrame(summary)

def normality_test_table(df, columns=None, alpha=0.05):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) < 3:
            results.append({"Sütun": col, "Test": "Yetersiz veri", "p-değeri": np.nan, "Normal mi?": "Belirsiz"})
            continue
        stat, p = stats.shapiro(data)
        results.append({"Sütun": col, "p-değeri": p, "Normal mi?": "Evet" if p > alpha else "Hayır"})
    return pd.DataFrame(results)

def natural_language_summary(df):
    """Otomatik metin özeti üretir."""
    lines = []
    lines.append(f"Veri seti {df.shape[0]} gözlem ve {df.shape[1]} değişkenden oluşuyor.")
    missing_total = df.isnull().sum().sum()
    lines.append(f"Toplam {missing_total} eksik hücre bulunuyor (%{100*missing_total/df.size:.1f}).")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        max_corr = corr.unstack().sort_values(ascending=False)
        max_pair = max_corr[max_corr < 1].index[0]
        lines.append(f"En yüksek korelasyon {max_pair[0]} ile {max_pair[1]} arasında: {corr.loc[max_pair[0], max_pair[1]]:.2f}.")
    skew_vals = df[num_cols].skew().sort_values(ascending=False)
    if not skew_vals.empty:
        most_skewed = skew_vals.index[0]
        lines.append(f"En çarpık sütun: {most_skewed} (çarpıklık = {skew_vals[0]:.2f}).")
    outlier_summary = get_outlier_summary(df)
    if not outlier_summary.empty:
        worst = outlier_summary.sort_values("Aykırı Oranı (%)", ascending=False).iloc[0]
        lines.append(f"En çok aykırı değer içeren sütun: {worst['Sütun']} (%{worst['Aykırı Oranı (%)']:.1f}).")
    return " ".join(lines)

def ab_test_analysis(df, group_col, target_col, control_val, treatment_val):
    """İki grup arasında dönüşüm oranı testi."""
    control = df[df[group_col] == control_val][target_col]
    treatment = df[df[group_col] == treatment_val][target_col]
    if target_col.dtype in [np.int64, np.float64]:
        # sayısal -> t-test
        stat, p = stats.ttest_ind(control.dropna(), treatment.dropna())
        test_name = "Bağımsız t-testi"
    else:
        # kategorik -> 0/1 dönüşüm
        control_success = (control == treatment_val).sum()
        treatment_success = (treatment == treatment_val).sum()
        stat, p = proportions_ztest([control_success, treatment_success], [len(control), len(treatment)])
        test_name = "Oran testi (z-test)"
    return test_name, stat, p

def cohort_analysis(df, date_col, user_col, period='M'):
    """Basit kohort analizi: aylık elde tutma."""
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['cohort'] = df.groupby(user_col)[date_col].transform('min').dt.to_period(period)
    df['period'] = df[date_col].dt.to_period(period)
    cohort_pivot = df.groupby(['cohort', 'period']).agg(n_users=(user_col, 'nunique')).reset_index()
    cohort_pivot['period_number'] = (cohort_pivot['period'] - cohort_pivot['cohort']).apply(lambda x: x.n)
    cohort_table = cohort_pivot.pivot_table(index='cohort', columns='period_number', values='n_users', aggfunc='sum')
    cohort_table = cohort_table.div(cohort_table.iloc[:,0], axis=0)
    return cohort_table

def forecast_simple(df, date_col, value_col, periods=7):
    """Basit üssel düzeltme ile tahmin."""
    df = df.sort_values(date_col)
    series = df[value_col].dropna()
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(series).fit()
    forecast = model.forecast(periods)
    return forecast

def detect_anomalies_zscore(series, threshold=3):
    """Z-score tabanlı anomali tespiti."""
    z = np.abs(stats.zscore(series.dropna()))
    return np.where(z > threshold)[0]

# -----------------------------------------------------------------------------
# Sidebar – Veri Yükleme ve Veri Sözlüğü
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Veri Kaynağı")
    uploaded_file = st.file_uploader("CSV, Excel veya Parquet", type=["csv", "xlsx", "xls", "parquet"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state['df'] = df
    else:
        if 'df' in st.session_state:
            df = st.session_state['df']
        else:
            df = None

    if df is not None:
        st.header("🔗 Veri Birleştirme")
        merge_file = st.file_uploader("Birleştirilecek ikinci dosya", type=["csv", "xlsx"], key="merge")
        if merge_file is not None:
            if merge_file.name.endswith(".csv"):
                df2 = pd.read_csv(merge_file)
            else:
                df2 = pd.read_excel(merge_file)
            common_cols = list(set(df.columns) & set(df2.columns))
            if common_cols:
                join_col = st.selectbox("Birleştirme anahtarı", common_cols)
                join_type = st.selectbox("Join tipi", ["inner", "left", "right", "outer"])
                if st.button("Verileri Birleştir"):
                    df = df.merge(df2, on=join_col, how=join_type)
                    st.session_state['df'] = df
                    st.success(f"Birleştirildi: {df.shape}")
            else:
                st.warning("Ortak sütun yok.")

        st.header("📖 Veri Sözlüğü")
        if 'data_dict' not in st.session_state:
            st.session_state['data_dict'] = {col: "" for col in df.columns}
        dict_col = st.selectbox("Sütun seç", df.columns)
        desc = st.text_area("Açıklama", value=st.session_state['data_dict'].get(dict_col, ""))
        if st.button("Kaydet"):
            st.session_state['data_dict'][dict_col] = desc
            st.success(f"{dict_col} açıklaması güncellendi.")

        st.header("⚙️ Kontroller")
        sample_pct = st.slider("Rastgele örneklem (%)", 10, 100, 100)
        if sample_pct < 100:
            df = df.sample(frac=sample_pct/100, random_state=42)
        st.session_state['df'] = df

# -----------------------------------------------------------------------------
# Ana Ekran
# -----------------------------------------------------------------------------
if df is None:
    st.info("👈 Lütfen sol kenardan bir veri seti yükleyin.")
else:
    # Üst Kartlar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gözlem", df.shape[0])
    col2.metric("Değişken", df.shape[1])
    missing_total = df.isnull().sum().sum()
    col3.metric("Eksik Hücre", missing_total, f"%{100*missing_total/df.size:.1f}")
    col4.metric("Tekrar", df.duplicated().sum())

    # Doğal Dil Özeti
    st.subheader("🧠 Otomatik İçgörü Özeti")
    st.info(natural_language_summary(df))

    tabs = st.tabs([
        "📄 Profil", "🔍 Kalite", "📊 Dağılım", "🧩 Segmentasyon",
        "🧪 A/B Testi", "📈 Tahmin", "👥 Kohort", "📑 Rapor"
    ])

    # ---------- Profil ----------
    with tabs[0]:
        if st.button("Profil Raporu Oluştur"):
            with st.spinner("Profil raporu oluşturuluyor..."):
                profile = ProfileReport(df, title="Profil", explorative=True, minimal=False, progress_bar=False)
                components.html(profile.to_html(), height=800, scrolling=True)

    # ---------- Kalite ----------
    with tabs[1]:
        st.subheader("Eksik Veri Deseni")
        fig, ax = plt.subplots(figsize=(10, 4))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)
        st.subheader("Aykırı Değer Özeti")
        st.dataframe(get_outlier_summary(df))
        low_var = [c for c in df.columns if df[c].nunique() <= 1]
        if low_var:
            st.warning(f"Sabit sütunlar: {', '.join(low_var)}")

    # ---------- Dağılım ----------
    with tabs[2]:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if num_cols:
            st.subheader("Normallik Testleri")
            st.dataframe(normality_test_table(df))
            dist_col = st.selectbox("Sütun", num_cols, key="dist")
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "QQ"))
            fig.add_trace(go.Histogram(x=df[dist_col]), row=1, col=1)
            data_qq = df[dist_col].dropna()
            theo = stats.norm.ppf((np.arange(len(data_qq))+0.5)/len(data_qq))
            fig.add_trace(go.Scatter(x=np.sort(theo), y=np.sort(data_qq), mode='markers'), row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Korelasyon")
            st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

    # ---------- Segmentasyon ----------
    with tabs[3]:
        st.subheader("RFM Analizi")
        if st.checkbox("RFM etkinleştir"):
            cust_col = st.selectbox("Müşteri ID", df.columns)
            date_col = st.selectbox("Tarih", df.columns)
            amount_col = st.selectbox("Tutar", num_cols)
            if st.button("RFM Hesapla"):
                rfm = df.copy()
                rfm[date_col] = pd.to_datetime(rfm[date_col], errors='coerce')
                ref_date = rfm[date_col].max()
                rfm_table = rfm.groupby(cust_col).agg(
                    Recency=pd.NamedAgg(column=date_col, aggfunc=lambda x: (ref_date - x.max()).days),
                    Frequency=pd.NamedAgg(column=date_col, aggfunc='count'),
                    Monetary=pd.NamedAgg(column=amount_col, aggfunc='sum')
                ).reset_index()
                st.dataframe(rfm_table.head(20))
        st.subheader("K-Means")
        if st.checkbox("KMeans etkinleştir"):
            clust_cols = st.multiselect("Sayısal sütunlar", num_cols, default=num_cols[:2])
            n = st.slider("Küme sayısı", 2, 10, 3)
            if st.button("Kümele"):
                scaler = StandardScaler()
                X = scaler.fit_transform(df[clust_cols].dropna())
                clusters = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(X)
                df['cluster'] = clusters
                st.plotly_chart(px.scatter(df, x=clust_cols[0], y=clust_cols[1], color='cluster'), use_container_width=True)

    # ---------- A/B Testi ----------
    with tabs[4]:
        st.subheader("A/B Testi")
        group_col = st.selectbox("Grup sütunu", cat_cols) if cat_cols else None
        target_col = st.selectbox("Hedef sütun", df.columns)
        if group_col and target_col:
            unique_vals = df[group_col].unique()
            if len(unique_vals) >= 2:
                control = st.selectbox("Kontrol grubu", unique_vals, index=0)
                treatment = st.selectbox("Deney grubu", unique_vals, index=1)
                if st.button("Testi Çalıştır"):
                    test_name, stat, p = ab_test_analysis(df, group_col, target_col, control, treatment)
                    st.write(f"**{test_name}**: istatistik={stat:.4f}, p={p:.4f}")
                    if p < 0.05:
                        st.success("Anlamlı fark var.")
                    else:
                        st.info("Anlamlı fark yok.")

    # ---------- Tahmin ----------
    with tabs[5]:
        st.subheader("Basit Tahmin (Üssel Düzeltme)")
        date_col = st.selectbox("Tarih sütunu", df.columns, key="fc_date")
        val_col = st.selectbox("Değer sütunu", num_cols, key="fc_val")
        periods = st.slider("Tahmin dönemi", 1, 30, 7)
        if st.button("Tahmin Yap"):
            forecast = forecast_simple(df, date_col, val_col, periods)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df[val_col].dropna().values, name='Gerçek'))
            fig.add_trace(go.Scatter(x=list(range(len(df), len(df)+periods)), y=forecast, name='Tahmin'))
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Kohort ----------
    with tabs[6]:
        st.subheader("Kohort Analizi")
        date_col = st.selectbox("Tarih", df.columns, key="co_date")
        user_col = st.selectbox("Kullanıcı ID", df.columns, key="co_user")
        period = st.selectbox("Periyot", ['M', 'W', 'D'])
        if st.button("Kohort Oluştur"):
            cohort = cohort_analysis(df, date_col, user_col, period)
            st.dataframe(cohort.style.background_gradient(cmap='Blues'), use_container_width=True)

    # ---------- Rapor ----------
    with tabs[7]:
        st.subheader("PDF Raporu")
        if st.button("PDF Oluştur"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 10, "Veri Analisti Raporu", ln=True)
            pdf.multi_cell(0, 5, f"Gözlem: {df.shape[0]}, Değişken: {df.shape[1]}")
            pdf.ln(5)
            pdf.set_font("Courier", size=8)
            pdf.multi_cell(0, 4, df.describe(include='all').to_string())
            # Veri sözlüğü ekle
            if 'data_dict' in st.session_state:
                pdf.add_page()
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 10, "Veri Sözlüğü", ln=True)
                for col, desc in st.session_state['data_dict'].items():
                    if desc:
                        pdf.set_font("Arial", size=10)
                        pdf.cell(0, 8, f"{col}: {desc}", ln=True)
            b64 = base64.b64encode(pdf.output(dest='S')).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="rapor.pdf">📥 PDF İndir</a>', unsafe_allow_html=True)

        st.subheader("Paylaşım")
        st.markdown("Bu dashboard'un bağlantısını kopyalayarak iş arkadaşlarınızla paylaşabilirsiniz.")
        st.code(window.location.href, language="text")
