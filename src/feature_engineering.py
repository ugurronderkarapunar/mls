"""Özellik mühendisliği işlemleri."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from typing import List, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def extract_date_features(df: pd.DataFrame, date_col: str, drop_original: bool = False) -> pd.DataFrame:
    """Tarih sütunundan yıl, ay, gün, haftanın günü gibi özellikler türetir."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[f"{date_col}_year"] = df[date_col].dt.year
    df[f"{date_col}_month"] = df[date_col].dt.month
    df[f"{date_col}_day"] = df[date_col].dt.day
    df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
    df[f"{date_col}_quarter"] = df[date_col].dt.quarter
    if drop_original:
        df.drop(columns=[date_col], inplace=True)
    return df


def create_interaction_features(
    df: pd.DataFrame, col_pairs: List[tuple]
) -> pd.DataFrame:
    """Belirtilen sayısal sütun çiftleri için çarpım/interaction özellikleri ekler."""
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df


def scale_numeric(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> pd.DataFrame:
    """Sayısal sütunları ölçeklendirir. (inplace)"""
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Geçersiz ölçeklendirme yöntemi.")
    df[columns] = scaler.fit_transform(df[columns].astype(float))
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    drop_first: bool = True,
) -> pd.DataFrame:
    """Kategorik değişkenleri kodlar.

    Args:
        df: DataFrame.
        columns: Kodlanacak sütun listesi.
        method: 'onehot', 'label', 'ordinal'.
        drop_first: One-hot için ilk kategoriyi düşür.

    Returns:
        Kodlanmış DataFrame (eski sütunlar düşer).
    """
    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    elif method == "label":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "ordinal":
        # Sıralı kodlama için sütunların sırası varsayılan olarak unique sıralaması
        for col in columns:
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]])
    else:
        raise ValueError("Geçersiz kodlama yöntemi.")
    return df
