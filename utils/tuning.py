"""
tuning.py
=========
Optuna-based hyperparameter optimisation for sklearn Decision Trees.

The search maximises mean AUC-PR across a lightweight 3-fold stratified CV
on the *training* set only — the held-out test set is never touched during
the search, preserving its role as an unbiased final estimator.

After the search, the best model is passed to train_evaluate_dt (in
model_utils) for a proper 5-fold CV evaluation and test-set report.
"""

import numpy as np
import pandas as pd
import optuna

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# Suppress Optuna's per-trial INFO logs — they clutter the notebook output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_dt(
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    n_trials:     int = 40,
    random_state: int = 42
) -> tuple:
    """
    Search for the best Decision Tree hyperparameters using Optuna (TPE sampler).

    The objective function is mean AUC-PR over a 3-fold stratified CV.
    AUC-PR is preferred over accuracy or ROC-AUC because:
      - It is more informative under class imbalance.
      - It directly measures performance on the positive (need) class.

    Parameters
    ----------
    X_train      : training features
    y_train      : binary training labels (0 = no need, 1 = has need)
    n_trials     : number of Optuna trials (more = better coverage, slower)
    random_state : seed for reproducibility

    Returns
    -------
    best_model : DecisionTreeClassifier initialised with the best params
                 (not yet fitted — fitting happens in train_evaluate_dt)
    study      : the completed optuna.Study object (useful for inspecting
                 the search history or plotting the optimisation curve)
    """

    def objective(trial: optuna.Trial) -> float:
        """
        One Optuna trial: sample parameters, run 3-fold CV, return mean AUC-PR.

        The search space covers the most influential DT hyperparameters:
          max_depth         – controls tree complexity (over/underfitting)
          min_samples_split – minimum samples to consider a split
          min_samples_leaf  – minimum leaf size (acts as regularisation)
          criterion         – impurity measure (gini vs entropy)
          class_weight      – whether to upweight the minority class
        """
        params = {
            'max_depth'        : trial.suggest_int('max_depth', 2, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 80),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 3, 50),
            'criterion'        : trial.suggest_categorical(
                                     'criterion', ['gini', 'entropy']),
            'class_weight'     : trial.suggest_categorical(
                                     'class_weight', [None, 'balanced']),
            'random_state'     : random_state,
        }

        # Use 3 folds (not 5) to keep the inner search fast — we are only
        # looking for a rough optimum here; the final 5-fold CV in
        # train_evaluate_dt gives the accurate performance estimate.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        fold_scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train):
            m = DecisionTreeClassifier(**params)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = m.predict_proba(X_train.iloc[va_idx])[:, 1]
            fold_scores.append(
                average_precision_score(y_train.iloc[va_idx], proba)
            )

        return float(np.mean(fold_scores))

    # TPESampler uses Bayesian inference to guide the search toward
    # promising regions of the hyperparameter space
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Build the best model (unfitted — caller is responsible for fitting)
    best_model = DecisionTreeClassifier(
        **study.best_params,
        random_state=random_state
    )

    return best_model, study
