import pandas as pd
from src.feature_engineering import extract_date_features, scale_numeric, encode_categorical

def test_extract_date_features(sample_df):
    df = sample_df.copy()
    df = extract_date_features(df, 'tarih')
    assert 'tarih_year' in df.columns
    assert df['tarih_year'].iloc[0] == 2020

def test_scale_numeric(sample_df):
    df = sample_df.copy()
    df = scale_numeric(df, ['sayisal1'], method='standard')
    assert abs(df['sayisal1'].mean()) < 0.5

def test_encode_categorical_onehot(sample_df):
    df = sample_df.copy()
    df = encode_categorical(df, ['kategori'], method='onehot')
    assert 'kategori_B' in df.columns
