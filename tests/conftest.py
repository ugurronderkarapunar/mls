import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'sayisal1': [1, 2, np.nan, 4, 5],
        'sayisal2': [10, 20, 30, 400, 50],
        'kategori': ['A', 'B', 'A', 'B', None],
        'tarih': pd.to_datetime(['2020-01-01', '2020-02-01', None, '2020-04-01', '2020-05-01']),
        'hedef': [0, 1, 0, 1, 0]
    })
