"""Model yorumlama araçları (SHAP, Feature Importance)."""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def shap_summary_plot(model, X_sample):
    """SHAP özet grafiğini döndürür."""
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig

def plot_feature_importance(model, feature_names):
    """Özellik önem düzeylerini çubuk grafiği olarak verir."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Önem Düzeyi")
    ax.set_title("Özellik Önem Sıralaması")
    plt.tight_layout()
    return fig
