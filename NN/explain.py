"""
Explainability tools for the multi-task network.

Two complementary lenses:

* SHAP (DeepExplainer), per head — the gold standard for local + global
  attribution on deep models. It can be version-fragile, so the head wrapper
  and shape normalisation below handle both old and new SHAP outputs.

* Permutation importance, per head — model-agnostic, directly comparable to
  the Random Forest baseline's importance scores, and measures the quantity
  that matters at inference: drop in PR-AUC when a feature is shuffled.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.metrics import average_precision_score


# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------
class HeadWrapper(torch.nn.Module):
    """
    Wraps the multi-task model to expose a single-output branch.

    SHAP's DeepExplainer expects a single scalar output per sample; our model
    returns a tuple (logits_accum, logits_income). This wrapper selects one
    head and returns a sigmoid probability with a trailing output dimension.
    """

    def __init__(self, base, head_name):
        super().__init__()
        self.base = base
        if head_name not in {'accum', 'income'}:
            raise ValueError(f"head_name must be 'accum' or 'income', got {head_name}")
        self.head_name = head_name

    def forward(self, x):
        la, li = self.base(x)
        out = la if self.head_name == 'accum' else li
        return torch.sigmoid(out).unsqueeze(-1)


def shap_per_head(model, X_train_values, X_test_values, feature_names,
                  head_name, title, n_background=100, n_explain=200,
                  seed=0):
    """
    Compute and plot SHAP values for a single head.

    Parameters
    ----------
    X_train_values, X_test_values : numpy arrays (already scaled)
    feature_names : list of str
    head_name     : 'accum' or 'income'
    """
    model.eval()
    rng = np.random.RandomState(seed)
    bg_idx = rng.choice(len(X_train_values), size=n_background, replace=False)

    bg_tensor   = torch.tensor(X_train_values[bg_idx],  dtype=torch.float32)
    test_tensor = torch.tensor(X_test_values[:n_explain], dtype=torch.float32)

    wrapper = HeadWrapper(model, head_name)
    expl    = shap.DeepExplainer(wrapper, bg_tensor)
    sv      = expl.shap_values(test_tensor, check_additivity=False)

    # Normalise to 2D (n_samples, n_features) across SHAP versions
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    shap.summary_plot(
        sv, X_test_values[:n_explain],
        feature_names=feature_names,
        show=False, plot_type='dot', max_display=15,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Permutation importance
# -----------------------------------------------------------------------------
def permutation_importance_nn(model, X, y_acc, y_inc, feature_names,
                              n_repeats=10, seed=42):
    """
    Per-head permutation importance: how much does PR-AUC drop when we shuffle
    each column? Implemented manually because sklearn's permutation_importance
    assumes a single-output estimator and our model returns a tuple.

    Returns
    -------
    imp_acc, imp_inc : DataFrames sorted by mean PR-AUC drop (descending)
    base_ap_acc, base_ap_inc : unperturbed PR-AUC values
    """
    rng = np.random.RandomState(seed)
    model.eval()

    with torch.no_grad():
        la, li = model(torch.tensor(X.values, dtype=torch.float32))
        base_pa = torch.sigmoid(la).numpy()
        base_pi = torch.sigmoid(li).numpy()
    base_ap_acc = average_precision_score(y_acc, base_pa)
    base_ap_inc = average_precision_score(y_inc, base_pi)

    imp_acc = {f: [] for f in feature_names}
    imp_inc = {f: [] for f in feature_names}

    X_np = X.values.copy()
    for col_idx, feat in enumerate(feature_names):
        original = X_np[:, col_idx].copy()
        for _ in range(n_repeats):
            X_perm = X_np.copy()
            rng.shuffle(X_perm[:, col_idx])
            with torch.no_grad():
                la, li = model(torch.tensor(X_perm, dtype=torch.float32))
                pa_p = torch.sigmoid(la).numpy()
                pi_p = torch.sigmoid(li).numpy()
            imp_acc[feat].append(base_ap_acc - average_precision_score(y_acc, pa_p))
            imp_inc[feat].append(base_ap_inc - average_precision_score(y_inc, pi_p))
        X_np[:, col_idx] = original

    def _summarize(imp_dict):
        return pd.DataFrame({
            'feature':   list(imp_dict.keys()),
            'mean_drop': [np.mean(v) for v in imp_dict.values()],
            'std_drop':  [np.std(v)  for v in imp_dict.values()],
        }).sort_values('mean_drop', ascending=False)

    return _summarize(imp_acc), _summarize(imp_inc), base_ap_acc, base_ap_inc


def plot_permutation_importance(perm_acc, perm_inc):
    """Side-by-side bar charts of permutation importance for both heads."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, df, title, color in [
        (axes[0], perm_acc, 'Accumulation head', 'steelblue'),
        (axes[1], perm_inc, 'Income head',       'tomato'),
    ]:
        df_sorted = df.sort_values('mean_drop', ascending=True)
        ax.barh(df_sorted['feature'], df_sorted['mean_drop'],
                xerr=df_sorted['std_drop'], color=color,
                error_kw={'ecolor': 'black', 'alpha': 0.5})
        ax.axvline(0, color='black', lw=0.5)
        ax.set_title(f'Permutation importance — {title}')
        ax.set_xlabel('PR-AUC drop when feature is shuffled')
    plt.tight_layout()
    plt.show()
