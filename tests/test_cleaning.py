import pandas as pd
import numpy as np
from src.cleaning import drop_duplicates, handle_missing_values, remove_outliers

def test_drop_duplicates():
    df = pd.DataFrame({'A': [1,2,1]})
    result = drop_duplicates(df)
    assert len(result) == 2

def test_handle_missing_median(sample_df):
    df = sample_df.copy()
    df = handle_missing_values(df, strategy='median', columns=['sayisal1'])
    assert df['sayisal1'].isnull().sum() == 0
    assert df['sayisal1'].iloc[2] == 3.0  # median of [1,2,4,5] = 3.0

def test_handle_missing_drop(sample_df):
    df = sample_df.copy()
    df = handle_missing_values(df, strategy='drop', columns=['sayisal1'])
    assert len(df) == 4

def test_remove_outliers_iqr(sample_df):
    df = sample_df.copy()
    df = remove_outliers(df, method='iqr', threshold=1.5, columns=['sayisal2'])
    # 400 outlier olacak
    assert len(df) == 4
