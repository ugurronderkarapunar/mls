"""Keşifsel Veri Analizi (EDA) ve istatistiksel testler."""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple, List, Optional


def descriptive_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Sayısal sütunlar için merkezi eğilim ve yayılım istatistikleri.

    Args:
        df: DataFrame.
        columns: İstatistikleri alınacak sütunlar (None ise tüm sayısal sütunlar).

    Returns:
        Her sütun için count, mean, std, min, %25, %50, %75, max bilgileri.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].describe(include="all").T


def check_skewness(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.Series:
    """Çarpıklık (skewness) değerlerini döndürür."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].skew().sort_values(ascending=False)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Korelasyon matrisi."""
    return df.corr(method=method, numeric_only=True)


def vif_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Varyans Büyütme Faktörü (VIF) analizi."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    temp_df = df[columns].fillna(df[columns].median())
    vif_data = pd.DataFrame({
        "Değişken": columns,
        "VIF": [variance_inflation_factor(temp_df.values, i) for i in range(len(columns))]
    })
    return vif_data.sort_values("VIF", ascending=False)


def normality_test(df: pd.DataFrame, column: str, alpha: float = 0.05) -> Tuple[str, float]:
    """Shapiro-Wilk normallik testi."""
    data = df[column].dropna()
    if len(data) < 3:
        return "Yetersiz veri", np.nan
    stat, p = stats.shapiro(data)
    if p > alpha:
        return "Normal dağılıma uygun (H0 reddedilemez)", p
    else:
        return "Normal dağılıma uygun değil (H0 reddedilir)", p


def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    test_type: str = "auto",
) -> Tuple[str, float, float, str]:
    """İki veya daha fazla grup için uygun hipotez testi yapar."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return "En az 2 grup gerekli", np.nan, np.nan, ""

    if test_type == "auto":
        is_normal = True
        for g in groups:
            if len(g) >= 3:
                _, p_norm = stats.shapiro(g)
                if p_norm < 0.05:
                    is_normal = False
                    break
        if is_normal and len(groups) == 2:
            test_type = "t-test"
        elif is_normal and len(groups) > 2:
            test_type = "anova"
        elif len(groups) == 2:
            test_type = "mannwhitney"
        else:
            test_type = "kruskal"

    if test_type == "t-test":
        stat, p = stats.ttest_ind(groups[0], groups[1])
        test_name = "Bağımsız t-testi"
    elif test_type == "anova":
        stat, p = stats.f_oneway(*groups)
        test_name = "ANOVA"
    elif test_type == "mannwhitney":
        stat, p = stats.mannwhitneyu(groups[0], groups[1])
        test_name = "Mann-Whitney U"
    elif test_type == "kruskal":
        stat, p = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis"
    else:
        return "Geçersiz test tipi", np.nan, np.nan, ""

    result = "Anlamlı fark var (p<0.05)" if p < 0.05 else "Anlamlı fark yok (p>=0.05)"
    return test_name, stat, p, result
