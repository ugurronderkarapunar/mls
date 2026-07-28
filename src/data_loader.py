"""Veri yükleme ve ilk tanıma fonksiyonları."""
import pandas as pd
from typing import Tuple


def load_data(file) -> pd.DataFrame:
    """Yüklenen dosyayı pandas DataFrame'e çevirir.

    Args:
        file: Streamlit yükleme objesi (BytesIO).

    Returns:
        pd.DataFrame: Yüklenen veri.
    """
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Desteklenmeyen dosya türü. Lütfen CSV veya Excel yükleyin.")
    return df


def get_basic_info(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Veri hakkında temel bilgileri döndürür.

    Returns:
        Tuple: (ilk 5 satır, sütun bilgileri, eksik değer sayıları)
    """
    head = df.head()
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["Sütun", "Veri Tipi"]
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Sütun", "Eksik Sayısı"]
    missing["Eksik Oranı (%)"] = (missing["Eksik Sayısı"] / len(df)) * 100
    return head, dtypes, missing
