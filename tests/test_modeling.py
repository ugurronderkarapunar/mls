import pandas as pd
import numpy as np
from src.modeling import split_data, evaluate_model, MODEL_DICT

def test_split_data():
    X = pd.DataFrame({'a': range(100)})
    y = pd.Series([0]*50 + [1]*50)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.1)
    assert len(X_train) == 70
    assert len(X_val) == 10
    assert len(X_test) == 20

def test_evaluate_model_classification():
    X = pd.DataFrame(np.random.randn(100, 3))
    y = pd.Series(np.random.choice([0,1], 100))
    model = MODEL_DICT['classification']['Decision Tree']
    metrics = evaluate_model(model, X, y, X, y, 'classification')
    assert 'Accuracy' in metrics
    assert 0 <= metrics['Accuracy'] <= 1
