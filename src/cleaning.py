"""Veri temizleme işlemleri."""
import pandas as pd
import numpy as np
from typing import Optional, List, Union


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Tekrarlanan satırları siler."""
    return df.drop_duplicates()


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_value: Optional[Union[str, float]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Eksik değerleri doldurur veya siler.

    Args:
        df: DataFrame.
        strategy: 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: 'constant' için doldurulacak değer.
        columns: İşlem yapılacak sütunlar (None = tümü).

    Returns:
        Temizlenmiş DataFrame.
    """
    if columns is None:
        columns = df.columns.tolist()

    if strategy == "drop":
        df = df.dropna(subset=columns)
    else:
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                elif strategy == "constant":
                    df[col].fillna(fill_value if fill_value is not None else 0, inplace=True)
            else:
                # Kategorik için sadece mode veya constant
                if strategy in ["mean", "median"]:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Bilinmiyor", inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Bilinmiyor", inplace=True)
                elif strategy == "constant":
                    df[col].fillna(fill_value if fill_value is not None else "Bilinmiyor", inplace=True)
    return df


def remove_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aykırı değerleri siler veya sınırlandırır.

    Args:
        df: DataFrame.
        method: 'iqr' veya 'zscore'.
        threshold: IQR çarpanı veya Z-score eşiği.
        columns: İşlem yapılacak sayısal sütunlar.

    Returns:
        Aykırı değerlerden arındırılmış DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "iqr":
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif method == "zscore":
        from scipy import stats
        for col in columns:
            z = np.abs(stats.zscore(df[col].dropna()))
            df = df[(z < threshold)]
    return df


def fix_data_types(
    df: pd.DataFrame, conversions: dict
) -> pd.DataFrame:
    """Sütun veri tiplerini belirtilen dönüşümlere göre düzeltir.

    Args:
        df: DataFrame.
        conversions: {'sütun_adı': 'float/int/str/datetime'} sözlüğü.

    Returns:
        Düzeltilmiş DataFrame.
    """
    for col, dtype in conversions.items():
        try:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
        except Exception:
            pass
    return df
