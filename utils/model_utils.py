"""
model_utils.py
==============
Training, cross-validation e valutazione per il pipeline Decision Tree.

NOTA SUI SPLIT:
  Questa versione NON contiene piu' build_train_test_splits().
  Per il DT si usa direttamente prepare_model_data() di NN/features.py
  che fa un 3-way split (Train/Val/Test) con joint stratification — uguale
  a quello che usa la NN. Cosi' i due notebook lavorano sulla stessa
  partizione dei dati.

  Nel DT notebook:
      from NN.features import prepare_model_data
      X_train, X_val, X_test, y_train, y_val, y_test, features, qt, rs =
          prepare_model_data(needs_engineered)

  Poi per comodita' il DT fonde Train+Val in un unico set per la CV
  (non ha bisogno di Val per early stopping come la NN).

Functions
---------
train_evaluate_dt    -- Stratified K-Fold CV + test-set evaluation
display_results_table -- stampa ASCII compatta dei risultati
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def train_evaluate_dt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model,
    k_folds: int = 5
) -> dict:
    """
    Valuta un classifier sklearn in due fasi:

    (a) Stratified K-Fold CV sul training set — usato per la selezione del modello.
        StratifiedKFold preserva le proporzioni di classe in ogni fold.

    (b) Singola valutazione sul test set — riportata solo una volta per il modello
        finale, come stima unbiased della generalizzazione.

    Parameters
    ----------
    X_train, y_train : feature e label di training
    X_test,  y_test  : feature e label di test (mai visti durante la CV)
    model            : classifier sklearn compatibile (richiede predict_proba)
    k_folds          : numero di fold CV (default 5)

    Returns
    -------
    dict con chiavi:
      cv_metrics    -- {metrica: {'mean': float, 'std': float}}
      test_metrics  -- {metrica: float}
      y_test_pred   -- predizioni hard sul test set
      y_test_proba  -- probabilita' classe positiva sul test set
      feature_names -- lista nomi colonne
      model         -- classifier refitted sull'intero training set
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    cv_scores = {m: [] for m in
                 ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'auc_pr']}

    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx],  y_train.iloc[va_idx]

        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_va)
        y_proba = model.predict_proba(X_va)[:, 1]

        cv_scores['accuracy'].append(accuracy_score(y_va, y_pred))
        cv_scores['precision'].append(precision_score(y_va, y_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_va, y_pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y_va, y_pred, zero_division=0))
        cv_scores['roc_auc'].append(roc_auc_score(y_va, y_proba))
        cv_scores['auc_pr'].append(average_precision_score(y_va, y_proba))

    cv_metrics = {
        m: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        for m, v in cv_scores.items()
    }

    # Refit sull'intero training set — questo e' il modello che usiamo per l'inference
    model.fit(X_train, y_train)
    y_test_pred  = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'accuracy' : accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_test_pred, zero_division=0),
        'f1'       : f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc'  : roc_auc_score(y_test, y_test_proba),
        'auc_pr'   : average_precision_score(y_test, y_test_proba),
    }

    return {
        'cv_metrics'   : cv_metrics,
        'test_metrics' : test_metrics,
        'y_test_pred'  : y_test_pred,
        'y_test_proba' : y_test_proba,
        'feature_names': list(X_train.columns),
        'model'        : model,
    }


def display_results_table(
    results: dict,
    model_name: str,
    feature_label: str,
    target_label: str
) -> None:
    """Stampa una tabella ASCII compatta con metriche CV e test set."""
    title  = f'  {model_name} | {feature_label} | {target_label}  '
    width  = max(49, len(title))
    border = '=' * width

    print()
    print(border)
    print(title)
    print(border)
    print(f'{"Metric":>10}  {"CV (mean ± std)":>18}  {"Test Set":>8}')

    labels = {
        'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
        'f1': 'F1-score', 'roc_auc': 'AUC-ROC', 'auc_pr': 'AUC-PR',
    }
    for key, label in labels.items():
        cv   = results['cv_metrics'][key]
        test = results['test_metrics'][key]
        print(f'{label:>10}  {cv["mean"]:.3f} ± {cv["std"]:.3f}    {test:.3f}')
    print()
