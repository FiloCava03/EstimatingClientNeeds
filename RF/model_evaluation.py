import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from tabulate import tabulate


def optimize_rf(X_train, y_train, n_trials, random_state):
    """
    Hyperparameter tuning for a Random Forest via Optuna (TPE sampler).
    The search maximizes mean PR-AUC across a 3-fold stratified CV, which is
    a cheaper proxy than the full 5-fold evaluation used downstream.

    The sampler is seeded explicitly, otherwise Optuna explores different
    configurations on every run and the final model is non-reproducible —
    even with scikit-learn's random_state fixed.

    Returns a RandomForestClassifier configured with the best parameters
    found, but NOT yet fitted (fitting happens in train_evaluate_model).
    """

    def objective(trial):
        # Search space kept moderate on purpose: wider ranges slow Optuna
        # down without materially improving the best model on this dataset
        rf_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 80),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            # Targets are imbalanced (especially IncomeInvestment): class_weight
            # rebalances loss per class, a cleaner alternative to over/undersampling
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
        }

        # 3 folds here, not 5, because this CV runs inside every Optuna trial
        # and the cost compounds quickly. The 5-fold CV in train_evaluate_model
        # is the one we report from.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        pr_auc_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**rf_params)
            model.fit(X_f_train, y_f_train)

            # predict_proba[:, 1] = probability of the positive class,
            # which is what PR-AUC (average_precision_score) expects
            preds = model.predict_proba(X_f_val)[:, 1]
            pr_auc_scores.append(average_precision_score(y_f_val, preds))

        return np.mean(pr_auc_scores)

    # Silence Optuna's per-trial INFO logs to keep the notebook readable
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Seeding the TPE sampler is the key reproducibility fix: without it,
    # Optuna picks different trials on each run and best_params drifts
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Return an unfitted model with the best-found hyperparameters.
    # Fitting on the full training set is done by the caller, so the same
    # function can be reused for different training subsets if needed.
    best_model = RandomForestClassifier(
        **study.best_params,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return best_model


def train_evaluate_model(
    X_train, y_train, X_test, y_test, n_splits, random_state, model
):
    """
    Evaluate a model in two stages:
      (a) Stratified K-Fold cross-validation on the training set — used
          to pick the champion between competing feature sets/models.
      (b) A single evaluation on the held-out test set — reported only
          once, for the final model, to estimate real-world performance.

    Returns a dict with cv_metrics (mean/std per metric), test_metrics,
    and the model refitted on the FULL training set (ready for inference).
    """
    # Stratified (not plain KFold) to preserve class proportions in every
    # fold — matters because the positive class is the minority one
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Accumulators for per-fold scores; PR-AUC is the primary metric
    # but the others are tracked for diagnostic purposes
    cv_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "pr_auc": [],
    }

    # --- (a) Cross-validation on the training set ---
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit fresh on each fold's training subset
        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_fold)
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]

        cv_metrics["accuracy"].append(accuracy_score(y_val_fold, y_val_pred))
        cv_metrics["precision"].append(precision_score(y_val_fold, y_val_pred))
        cv_metrics["recall"].append(recall_score(y_val_fold, y_val_pred))
        cv_metrics["f1"].append(f1_score(y_val_fold, y_val_pred))
        cv_metrics["pr_auc"].append(average_precision_score(y_val_fold, y_val_proba))

    # --- (b) Final refit on the FULL training set and test evaluation ---
    # This is the model that gets returned and used downstream (NBA, etc.)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    return {
        "cv_metrics": {
            metric: {"mean": np.mean(scores), "std": np.std(scores)}
            for metric, scores in cv_metrics.items()
        },
        "test_metrics": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
            "pr_auc": average_precision_score(y_test, y_test_proba),
        },
        "model": model,
    }


def display_results_table(results_dict, model_name, feature_type):
    """
    Pretty-print a side-by-side table with CV mean, CV std and test-set
    metrics, for quick visual comparison between candidate models.
    Layout is intentionally the same for every call, so the notebook reads
    like a consistent evaluation report rather than ad-hoc printouts.
    """
    cv_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "PR-AUC"],
        "CV Mean": [
            results_dict["cv_metrics"]["accuracy"]["mean"],
            results_dict["cv_metrics"]["precision"]["mean"],
            results_dict["cv_metrics"]["recall"]["mean"],
            results_dict["cv_metrics"]["f1"]["mean"],
            results_dict["cv_metrics"]["pr_auc"]["mean"],
        ],
        "CV Std": [
            results_dict["cv_metrics"]["accuracy"]["std"],
            results_dict["cv_metrics"]["precision"]["std"],
            results_dict["cv_metrics"]["recall"]["std"],
            results_dict["cv_metrics"]["f1"]["std"],
            results_dict["cv_metrics"]["pr_auc"]["std"],
        ],
        "Test Set": [
            results_dict["test_metrics"]["accuracy"],
            results_dict["test_metrics"]["precision"],
            results_dict["test_metrics"]["recall"],
            results_dict["test_metrics"]["f1"],
            results_dict["test_metrics"]["pr_auc"],
        ],
    }

    # Round to 4 decimals: enough to see differences between models without
    # pretending to have more precision than the CV std actually supports
    df = pd.DataFrame(cv_data).round(4)

    print(f"\n{model_name} - {feature_type}")
    print("=" * 60)
    # showindex=False hides the default 0..4 row index, which is noise here
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
