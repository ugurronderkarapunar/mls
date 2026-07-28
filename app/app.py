"""Veri Analisti Dashboard – İnteraktif, PDF Hatasız Sürüm"""
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
import sweetviz as sv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import missingno as msno
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Sayfa yapılandırması
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Veri Analisti Dashboard", layout="wide")
st.title("📋 Veri Analisti Dashboard")
st.markdown("Veri setinizi yükleyin, analiz edin, içgörüleri keşfedin.")

# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------
def sweetviz_html(df):
    """Sweetviz raporunu HTML string olarak döndürür."""
    report = sv.analyze(df, pairwise_analysis='off')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        report.show_html(filepath=f.name, open_browser=False)
        html_path = f.name
    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
    os.unlink(html_path)
    return html_str

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
            results.append({"Sütun": col, "p-değeri": np.nan, "Normal mi?": "Belirsiz"})
            continue
        stat, p = stats.shapiro(data)
        results.append({"Sütun": col, "p-değeri": p, "Normal mi?": "Evet" if p > alpha else "Hayır"})
    return pd.DataFrame(results)

def natural_language_summary(df):
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
    control = df[df[group_col] == control_val][target_col]
    treatment = df[df[group_col] == treatment_val][target_col]
    if pd.api.types.is_numeric_dtype(df[target_col]):
        stat, p = stats.ttest_ind(control.dropna(), treatment.dropna())
        test_name = "Bağımsız t-testi"
    else:
        control_success = (control == treatment_val).sum()
        treatment_success = (treatment == treatment_val).sum()
        stat, p = proportions_ztest([control_success, treatment_success], [len(control), len(treatment)])
        test_name = "Oran testi (z-test)"
    return test_name, stat, p

def cohort_analysis(df, date_col, user_col, period='M'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['cohort'] = df.groupby(user_col)[date_col].transform('min').dt.to_period(period)
    df['period'] = df[date_col].dt.to_period(period)
    cohort_pivot = df.groupby(['cohort', 'period']).agg(n_users=(user_col, 'nunique')).reset_index()
    cohort_pivot['period_number'] = (cohort_pivot['period'] - cohort_pivot['cohort']).apply(lambda x: x.n)
    cohort_table = cohort_pivot.pivot_table(index='cohort', columns='period_number', values='n_users', aggfunc='sum')
    cohort_table = cohort_table.div(cohort_table.iloc[:,0], axis=0)
    return cohort_table

def forecast_simple(df, date_col, value_col, periods=7):
    df = df.sort_values(date_col)
    series = df[value_col].dropna()
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(series).fit()
    forecast = model.forecast(periods)
    return forecast

# -----------------------------------------------------------------------------
# Sidebar – tüm kontroller burada
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
        df = st.session_state.get('df', None)

    if df is not None:
        st.header("🔗 Veri Birleştirme")
        merge_file = st.file_uploader("İkinci dosya", type=["csv", "xlsx"], key="merge")
        if merge_file is not None:
            df2 = pd.read_csv(merge_file) if merge_file.name.endswith(".csv") else pd.read_excel(merge_file)
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
        if st.button("Açıklamayı Kaydet"):
            st.session_state['data_dict'][dict_col] = desc
            st.success(f"{dict_col} açıklaması güncellendi.")

        st.header("⚙️ Kontroller")
        sample_pct = st.slider("Rastgele örneklem (%)", 10, 100, 100)
        if sample_pct < 100:
            df = df.sample(frac=sample_pct/100, random_state=42)
        st.session_state['df'] = df

# -----------------------------------------------------------------------------
# Ana ekran
# -----------------------------------------------------------------------------
if df is None:
    st.info("👈 Lütfen sol kenardan bir veri seti yükleyin.")
else:
    # Özet kartlar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gözlem", df.shape[0])
    col2.metric("Değişken", df.shape[1])
    missing_total = df.isnull().sum().sum()
    col3.metric("Eksik Hücre", missing_total, f"%{100*missing_total/df.size:.1f}")
    col4.metric("Tekrar", df.duplicated().sum())

    # Otomatik içgörü
    st.subheader("🧠 Otomatik İçgörü Özeti")
    st.info(natural_language_summary(df))

    # Ana sekmeler
    tabs = st.tabs([
        "📄 Profil", "🔍 Kalite", "📊 Dağılım", "🧩 Segmentasyon",
        "🧪 A/B Testi", "📈 Tahmin", "👥 Kohort", "📑 Dışa Aktar"
    ])

    with tabs[0]:
        st.header("📄 Profil Raporu (Sweetviz)")
        if st.button("Profil Raporu Oluştur"):
            with st.spinner("Rapor hazırlanıyor..."):
                html_str = sweetviz_html(df)
            components.html(html_str, height=800, scrolling=True)

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

    with tabs[4]:
        st.subheader("A/B Testi")
        if cat_cols:
            group_col = st.selectbox("Grup sütunu", cat_cols)
            target_col = st.selectbox("Hedef sütun", df.columns)
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

    with tabs[5]:
        st.subheader("Basit Tahmin (Üssel Düzeltme)")
        date_col_ts = st.selectbox("Tarih sütunu", df.columns, key="fc_date")
        val_col_ts = st.selectbox("Değer sütunu", num_cols, key="fc_val")
        periods = st.slider("Tahmin dönemi", 1, 30, 7)
        if st.button("Tahmin Yap"):
            forecast = forecast_simple(df, date_col_ts, val_col_ts, periods)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df[val_col_ts].dropna().values, name='Gerçek'))
            fig.add_trace(go.Scatter(x=list(range(len(df), len(df)+periods)), y=forecast, name='Tahmin'))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.subheader("Kohort Analizi")
        date_col_co = st.selectbox("Tarih", df.columns, key="co_date")
        user_col_co = st.selectbox("Kullanıcı ID", df.columns, key="co_user")
        period = st.selectbox("Periyot", ['M', 'W', 'D'])
        if st.button("Kohort Oluştur"):
            cohort = cohort_analysis(df, date_col_co, user_col_co, period)
            st.dataframe(cohort.style.background_gradient(cmap='Blues'), use_container_width=True)

    with tabs[7]:
        st.header("📑 Dışa Aktar")
        st.subheader("Profil Raporunu İndir (HTML)")
        if st.button("HTML Raporu Oluştur"):
            html_str = sweetviz_html(df)
            b64 = base64.b64encode(html_str.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="profil_raporu.html">📥 HTML İndir</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.subheader("Veri Özetini Excel Olarak İndir")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.describe(include='all').to_excel(writer, sheet_name='İstatistikler')
            get_outlier_summary(df).to_excel(writer, sheet_name='Aykırılar', index=False)
        st.download_button("📥 Excel İndir", data=output.getvalue(), file_name="veri_ozeti.xlsx")

        st.subheader("Dashboard Bağlantısı")
        st.code("Bu sayfanın URL'sini kopyalayarak paylaşabilirsiniz.")
