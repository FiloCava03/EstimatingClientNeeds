"""
shap_utils.py
=============
Explainability per il pipeline Decision Tree.

Per la Neural Network usa direttamente NN/explain.py (shap_per_head,
permutation_importance_nn, plot_permutation_importance).

Questo file contiene solo cio' che serve ESCLUSIVAMENTE ai DT:
  - build_explainers()        -- TreeExplainer per ogni (target, feature-set)
  - get_shap_pos()            -- normalizza output SHAP tra versioni diverse
  - get_base_val()            -- estrae expected value scalare
  - permutation_importance_dt() -- drop PR-AUC su sklearn classifier

NOTA: plot_permutation_importance viene da NN/explain.py ed e' gia' abbastanza
generica da coprire entrambi i casi (prende un DataFrame con feature/mean_drop/std_drop).
Importala da li':
    from NN.explain import plot_permutation_importance
"""

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None


# ---------------------------------------------------------------------------
# Compatibility helpers (TreeExplainer cambia formato tra versioni shap)
# ---------------------------------------------------------------------------

def get_shap_pos(shap_values) -> np.ndarray:
    """
    Restituisce SHAP values per la classe POSITIVA come array 2D
    (n_samples, n_features), indipendentemente dalla versione di shap.

    shap < 0.41  : lista [arr_classe0, arr_classe1]
    shap >= 0.41 : array 3D (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values   # gia' 2D


def get_base_val(explainer) -> float:
    """Valore atteso scalare per la classe positiva da un TreeExplainer."""
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1])
    return float(ev)


# ---------------------------------------------------------------------------
# TreeExplainer per tutti i (target x feature-set) dei DT
# ---------------------------------------------------------------------------

def build_explainers(tuned_results: dict, test_splits: dict) -> tuple:
    """
    Crea un shap.TreeExplainer per ogni combinazione (target, feature_set)
    e calcola i SHAP values sul test set corrispondente.

    Parameters
    ----------
    tuned_results : dict  keyed (t_key, f_key), contiene 'model'
    test_splits   : dict  keyed (t_key, f_key), valore = X_test DataFrame

    Returns
    -------
    explainers : dict  {(t_key, f_key): TreeExplainer}
    shap_pos   : dict  {(t_key, f_key): np.ndarray (n_test, n_features)}
    """
    if shap is None:
        raise ImportError("shap non installato. Esegui: pip install shap")

    combos = [
        ('income', 'base'), ('income', 'engineered'),
        ('accum',  'base'), ('accum',  'engineered'),
    ]
    explainers, shap_pos = {}, {}

    for t_key, f_key in combos:
        model  = tuned_results[(t_key, f_key)]['model']
        X_test = test_splits[(t_key, f_key)]

        expl = shap.TreeExplainer(model)
        sv   = expl.shap_values(X_test)

        explainers[(t_key, f_key)] = expl
        shap_pos[(t_key, f_key)]   = get_shap_pos(sv)

    return explainers, shap_pos


# ---------------------------------------------------------------------------
# Permutation importance per sklearn classifier (single-output)
# ---------------------------------------------------------------------------

def permutation_importance_dt(
    model,
    X:             pd.DataFrame,
    y:             pd.Series,
    feature_names: list,
    n_repeats:     int = 10,
    seed:          int = 42
) -> tuple:
    """
    Drop in PR-AUC quando ogni feature viene rimescolata (n_repeats volte).

    Usa un RandomState privato — non tocca il global numpy RNG.

    Returns
    -------
    result  : DataFrame [feature, mean_drop, std_drop], ordinato desc
    base_ap : PR-AUC baseline sul test set non perturbato
    """
    from sklearn.metrics import average_precision_score

    rng       = np.random.RandomState(seed)   # isolato dal global RNG
    base_ap   = average_precision_score(y, model.predict_proba(X)[:, 1])
    X_np      = X.values.copy()
    drops     = {f: [] for f in feature_names}

    for col_idx, feat in enumerate(feature_names):
        original = X_np[:, col_idx].copy()
        for _ in range(n_repeats):
            X_perm = X_np.copy()
            rng.shuffle(X_perm[:, col_idx])
            p = model.predict_proba(
                pd.DataFrame(X_perm, columns=feature_names)
            )[:, 1]
            drops[feat].append(base_ap - average_precision_score(y, p))
        X_np[:, col_idx] = original

    result = pd.DataFrame({
        'feature'  : list(drops.keys()),
        'mean_drop': [np.mean(v) for v in drops.values()],
        'std_drop' : [np.std(v)  for v in drops.values()],
    }).sort_values('mean_drop', ascending=False)

    return result, base_ap
