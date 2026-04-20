"""
plotting.py
===========
All visualisation helpers for the Needs-based recommendation project.

Each function produces one self-contained figure and calls plt.show() so
they can be called anywhere in the notebook without extra boilerplate.

Functions
---------
plot_class_balance          – bar charts of IncomeInvestment / AccumulationInvestment
plot_roc_curves             – ROC curves for base vs engineered features
plot_pr_curves              – Precision-Recall curves
plot_confusion_matrices     – 2×2 grid of confusion matrices
plot_feature_importance     – horizontal bar chart of Gini importances
plot_cv_fold_performance    – per-fold bar chart for F1 / AUC-ROC / Accuracy
plot_baseline_vs_tuned      – grouped bar chart comparing baseline vs Optuna results
plot_shap_bar               – global SHAP mean-|value| bar chart
plot_shap_beeswarm          – global SHAP beeswarm / summary plot
plot_shap_waterfall         – local SHAP waterfall for a single client
plot_permutation_importance – horizontal bar chart of permutation drops
plot_recommendation_freq    – frequency bar chart for first-choice products
plot_suitability_scatter    – client risk vs recommended-product risk scatter
plot_need_probability_hist  – histogram of predicted need probabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    roc_curve, confusion_matrix, precision_recall_curve,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------------------------
# 1. Class balance
# ---------------------------------------------------------------------------

def plot_class_balance(needs_df: pd.DataFrame) -> None:
    """
    Two side-by-side bar charts showing the class distribution of the two
    binary target variables (IncomeInvestment, AccumulationInvestment).
    """
    targets = ['IncomeInvestment', 'AccumulationInvestment']
    labels  = ['Income Investment', 'Accumulation Investment']
    colors  = ['#4c72b0', '#dd8452']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, label, color in zip(axes, targets, labels, colors):
        counts = needs_df[col].value_counts().sort_index()
        total  = counts.sum()

        bars = ax.bar(
            ['No (0)', 'Yes (1)'],
            counts.values,
            color=[color + '88', color],
            edgecolor='white', linewidth=1.5
        )
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f'{count:,}\n({count/total:.1%})',
                ha='center', va='bottom', fontsize=11
            )
        ax.set_title(f'Class Balance — {label}', fontsize=13, pad=10)
        ax.set_ylabel('Count')
        ax.set_ylim(0, total * 0.75)

    plt.tight_layout()
    plt.suptitle('Target Variable Distributions', fontsize=14, y=1.02)
    plt.show()

    # Overlap statistics
    both    = ((needs_df['IncomeInvestment'] == 1) &
               (needs_df['AccumulationInvestment'] == 1)).sum()
    neither = ((needs_df['IncomeInvestment'] == 0) &
               (needs_df['AccumulationInvestment'] == 0)).sum()
    n = len(needs_df)
    print(f'Clients with BOTH needs:    {both:,} ({both/n:.1%})')
    print(f'Clients with NEITHER need:  {neither:,} ({neither/n:.1%})')


# ---------------------------------------------------------------------------
# 2. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    splits:       dict,
    all_results:  dict,
    TARGET_CONFIG: list,
    FEATURE_CONFIG: list
) -> None:
    """
    One ROC-curve figure per target (base vs engineered features).
    Includes the random-classifier diagonal for reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (t_key, t_label, _) in zip(axes, TARGET_CONFIG):
        for f_key, f_label, _ in FEATURE_CONFIG:
            res = all_results[(t_key, f_key)]
            _, X_test, _, y_test = splits[(f_key, t_key)]

            fpr, tpr, _ = roc_curve(y_test, res['y_test_proba'])
            auc_val     = res['test_metrics']['roc_auc']
            ax.plot(fpr, tpr, lw=2, label=f'{f_label}  (AUC = {auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve — {t_label}', fontsize=12, pad=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.suptitle('Decision Tree — ROC Curves (Test Set)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    splits:        dict,
    all_results:   dict,
    TARGET_CONFIG:  list,
    FEATURE_CONFIG: list
) -> None:
    """
    One PR-curve figure per target.  The dashed horizontal line shows the
    random-classifier baseline (= positive class prevalence).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (t_key, t_label, y_full) in zip(axes, TARGET_CONFIG):
        baseline = y_full.mean()   # random classifier AP = class prevalence

        for f_key, f_label, _ in FEATURE_CONFIG:
            res = all_results[(t_key, f_key)]
            _, X_test, _, y_test = splits[(f_key, t_key)]

            prec, rec, _ = precision_recall_curve(y_test, res['y_test_proba'])
            ap = res['test_metrics']['auc_pr']
            ax.plot(rec, prec, lw=2,
                    label=f'{f_label}  (AUC-PR = {ap:.3f})')

        ax.axhline(baseline, color='grey', linestyle='--', lw=1.5,
                   label=f'Random (AP = {baseline:.2f})')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision-Recall Curve — {t_label}', fontsize=12, pad=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.suptitle('Decision Tree — Precision-Recall Curves (Test Set)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    splits:        dict,
    all_results:   dict,
    TARGET_CONFIG:  list,
    FEATURE_CONFIG: list
) -> None:
    """2×2 grid of confusion matrices (target rows × feature-set columns)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, (t_key, t_label, _) in enumerate(TARGET_CONFIG):
        for col, (f_key, f_label, _) in enumerate(FEATURE_CONFIG):
            ax  = axes[row][col]
            res = all_results[(t_key, f_key)]
            _, _, _, y_test = splits[(f_key, t_key)]

            cm   = confusion_matrix(y_test, res['y_test_pred'])
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['No (0)', 'Yes (1)']
            )
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(f'{t_label}\n{f_label}', fontsize=10, pad=8)

    plt.suptitle('Decision Tree — Confusion Matrices (Test Set)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Feature importance (Gini)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    all_results:  dict,
    TARGET_CONFIG: list,
    feature_set:  str = 'base'
) -> None:
    """
    Horizontal bar chart of mean Gini impurity decrease for each feature.
    Uses the feature set specified by `feature_set` ('base' or 'engineered').
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (t_key, t_label, _) in zip(axes, TARGET_CONFIG):
        res        = all_results[(t_key, feature_set)]
        model      = res['model']
        feat_names = res['feature_names']

        importances = pd.Series(
            model.feature_importances_, index=feat_names
        ).sort_values()

        bar_colors = [
            '#4c72b0' if v > importances.median() else '#a0b8d4'
            for v in importances
        ]
        importances.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
        ax.set_xlabel('Mean Impurity Decrease (Gini importance)', fontsize=10)
        ax.set_title(f'Feature Importance — {t_label}', fontsize=12, pad=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        # Value annotations at the end of each bar
        for val, patch in zip(importances.values, ax.patches):
            ax.text(val + 0.001, patch.get_y() + patch.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)

    label_str = 'Base' if feature_set == 'base' else 'Engineered'
    plt.suptitle(
        f'Decision Tree — Feature Importance ({label_str} Features, Test Set)',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. CV fold performance
# ---------------------------------------------------------------------------

def plot_cv_fold_performance(
    splits:        dict,
    TARGET_CONFIG:  list,
    DT_BASELINE:    dict,
    k_folds:        int = 5,
    random_state:   int = 42
) -> None:
    """
    Per-fold bar charts for F1, AUC-ROC, and Accuracy.
    Uses the BASE feature set and the baseline (untuned) DT configuration.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    metrics_to_plot = ['f1', 'roc_auc', 'accuracy']
    metric_labels   = {'f1': 'F1-score', 'roc_auc': 'AUC-ROC',
                       'accuracy': 'Accuracy'}

    kf  = StratifiedKFold_helper(k_folds, random_state)   # see below
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)

    for row, (t_key, t_label, _) in enumerate(TARGET_CONFIG):
        X_train, _, y_train, _ = splits[('base', t_key)]
        fold_scores = {m: [] for m in metrics_to_plot}

        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                             random_state=random_state)

        for tr_idx, va_idx in kf.split(X_train, y_train):
            dt = DecisionTreeClassifier(**DT_BASELINE)
            dt.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            y_pred  = dt.predict(X_train.iloc[va_idx])
            y_proba = dt.predict_proba(X_train.iloc[va_idx])[:, 1]
            y_val   = y_train.iloc[va_idx]

            fold_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            fold_scores['roc_auc'].append(roc_auc_score(y_val, y_proba))
            fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))

        for col, metric in enumerate(metrics_to_plot):
            ax     = axes[row][col]
            scores = fold_scores[metric]
            ax.bar(range(1, k_folds + 1), scores,
                   color='#4c72b0', alpha=0.8, edgecolor='white')
            ax.axhline(np.mean(scores), color='red', lw=1.5, linestyle='--',
                       label=f'Mean = {np.mean(scores):.3f}')
            ax.set_xlabel(f'Fold (k={k_folds})')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f'{t_label}\n{metric_labels[metric]}', fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xticks(range(1, k_folds + 1))
            ax.set_ylim(max(0, min(scores) - 0.05),
                        min(1, max(scores) + 0.05))

    plt.suptitle('Decision Tree — CV Fold Performance (Base Features)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def StratifiedKFold_helper(k, seed):
    """Thin wrapper so the import stays local and the outer scope stays clean."""
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# 7. Baseline vs tuned comparison
# ---------------------------------------------------------------------------

def plot_baseline_vs_tuned(
    all_results:    dict,
    tuned_results:  dict,
    TARGET_CONFIG:  list,
    FEATURE_CONFIG: list
) -> None:
    """
    Grouped bar chart comparing baseline and Optuna-tuned AUC-PR.
    Diamond markers overlay the corresponding F1 scores.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x     = np.arange(len(FEATURE_CONFIG))
    width = 0.35

    for ax, (t_key, t_label, _) in zip(axes, TARGET_CONFIG):
        base_auc = [all_results[(t_key,  f_key)]['test_metrics']['auc_pr']
                    for f_key, _, _ in FEATURE_CONFIG]
        tune_auc = [tuned_results[(t_key, f_key)]['test_metrics']['auc_pr']
                    for f_key, _, _ in FEATURE_CONFIG]
        base_f1  = [all_results[(t_key,  f_key)]['test_metrics']['f1']
                    for f_key, _, _ in FEATURE_CONFIG]
        tune_f1  = [tuned_results[(t_key, f_key)]['test_metrics']['f1']
                    for f_key, _, _ in FEATURE_CONFIG]

        bars1 = ax.bar(x - width/2, base_auc, width,
                       label='Baseline AUC-PR', color='#a0b8d4',
                       edgecolor='white')
        bars2 = ax.bar(x + width/2, tune_auc, width,
                       label='Tuned AUC-PR',    color='#4c72b0',
                       edgecolor='white')
        ax.scatter(x - width/2, base_f1, marker='D', color='#e05c2a',
                   zorder=5, s=60, label='Baseline F1')
        ax.scatter(x + width/2, tune_f1, marker='D', color='#c0392b',
                   zorder=5, s=60, label='Tuned F1')

        ax.set_xticks(x)
        ax.set_xticklabels([f_label for _, f_label, _ in FEATURE_CONFIG])
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.set_title(t_label, fontsize=12, pad=10)
        ax.legend(fontsize=8, loc='lower right')

        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle('Baseline vs Optuna-tuned — AUC-PR and F1 (Test Set)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 8-10. SHAP plots (require shap to be installed)
# ---------------------------------------------------------------------------

def plot_shap_bar(
    shap_pos:  dict,
    test_splits: dict,
    combos:    list
) -> None:
    """
    Global SHAP importance bar chart (mean |SHAP value|) for each combo.
    `combos` is a list of (t_key, f_key, title) tuples.
    """
    for t_key, f_key, title in combos:
        sv_pos = shap_pos[(t_key, f_key)]
        X_te   = test_splits[(t_key, f_key)]

        mean_abs = pd.Series(
            np.abs(sv_pos).mean(axis=0), index=X_te.columns
        ).sort_values(ascending=True)

        bar_colors = [
            '#4c72b0' if v > mean_abs.median() else '#a0b8d4'
            for v in mean_abs
        ]
        fig, ax = plt.subplots(figsize=(10, max(4, len(mean_abs) * 0.45)))
        mean_abs.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'SHAP Global Importance (Bar)\n{title}',
                     fontsize=12, pad=10)
        for patch, val in zip(ax.patches, mean_abs.values):
            ax.text(val + 0.001, patch.get_y() + patch.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.show()


def plot_shap_beeswarm(
    shap_pos:    dict,
    test_splits: dict,
    combos:      list
) -> None:
    """
    Global SHAP beeswarm (summary) plot for each combo.
    Red = high feature value, Blue = low feature value.
    """
    import shap
    for t_key, f_key, title in combos:
        n_feats = test_splits[(t_key, f_key)].shape[1]
        fig, ax = plt.subplots(figsize=(10, max(5, n_feats * 0.5)))
        plt.sca(ax)
        shap.summary_plot(
            shap_pos[(t_key, f_key)],
            test_splits[(t_key, f_key)],
            show=False, plot_size=None, color_bar=True,
        )
        ax.set_title(f'SHAP Beeswarm (Global + Direction)\n{title}',
                     fontsize=12, pad=10)
        plt.tight_layout()
        plt.show()


def plot_shap_waterfall(
    shap_pos:    dict,
    test_splits: dict,
    explainers:  dict,
    combos:      list,
    client_idx:  int = 0
) -> None:
    """
    Local SHAP waterfall plot for a single client across all combos.

    Parameters
    ----------
    client_idx : index of the test-set client to explain
    """
    import shap
    from utils.shap_utils import get_base_val   # local import to avoid circular

    for t_key, f_key, title in combos:
        sv_pos  = shap_pos[(t_key, f_key)]
        X_te    = test_splits[(t_key, f_key)]
        expl    = explainers[(t_key, f_key)]
        n_feats = X_te.shape[1]

        fig, ax = plt.subplots(figsize=(10, max(5, n_feats * 0.45)))
        plt.sca(ax)
        shap.waterfall_plot(
            shap.Explanation(
                values        = sv_pos[client_idx],
                base_values   = get_base_val(expl),
                data          = X_te.iloc[client_idx].values,
                feature_names = X_te.columns.tolist(),
            ),
            max_display = n_feats,
            show        = False,
        )
        ax.set_title(
            f'SHAP Waterfall | {title} | Client #{client_idx}',
            fontsize=12, pad=10
        )
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 11. Permutation importance
# ---------------------------------------------------------------------------

def plot_permutation_importance(
    perm_df:   pd.DataFrame,
    base_ap:   float,
    title:     str
) -> None:
    """
    Horizontal bar chart of permutation importance (mean PR-AUC drop).
    Error bars show ± 1 standard deviation across repeats.
    """
    df = perm_df.sort_values('mean_drop', ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))

    ax.barh(df['feature'], df['mean_drop'], xerr=df['std_drop'],
            color='#4c72b0', edgecolor='white', capsize=4)
    ax.axvline(0, color='grey', lw=1, linestyle='--')
    ax.set_xlabel(f'Mean PR-AUC drop (baseline = {base_ap:.4f})', fontsize=10)
    ax.set_title(f'Permutation Importance\n{title}', fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 12-14. Recommendation visualisations
# ---------------------------------------------------------------------------

def plot_recommendation_freq(recs_df: pd.DataFrame) -> None:
    """Bar chart: how often each product is recommended as first choice."""
    prod_counts = recs_df['Rec_1'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(prod_counts.index.astype(str), prod_counts.values,
           color='#4c72b0', edgecolor='white')
    ax.set_title('Recommended Product Frequency (Rec_1)', fontsize=12, pad=10)
    ax.set_xlabel('Product ID')
    ax.set_ylabel('Number of clients')

    for patch, val in zip(ax.patches, prod_counts.values):
        ax.text(patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 1, str(int(val)),
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_suitability_scatter(
    recs_df:     pd.DataFrame,
    products_df: pd.DataFrame,
    needs_df:    pd.DataFrame
) -> None:
    """
    Scatter plot: client RiskPropensity (x-axis) vs recommended product Risk
    (y-axis).  Points are coloured by product type (income / accumulation).
    The red dashed diagonal is the 'perfect match' reference line.
    """
    from matplotlib.patches import Patch

    rec_valid  = recs_df[recs_df['Rec_1'].notna()]
    client_risk = needs_df.loc[rec_valid.index, 'RiskPropensity'].values
    prod_ids    = rec_valid['Rec_1'].values

    prod_risk = np.array([
        products_df.loc[products_df['IDProduct'] == pid, 'Risk'].values[0]
        for pid in prod_ids
    ])
    prod_type = np.array([
        int(products_df.loc[products_df['IDProduct'] == pid, 'Type'].values[0])
        for pid in prod_ids
    ])

    colors = ['#4c72b0' if t == 0 else '#dd8452' for t in prod_type]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(client_risk, prod_risk, c=colors, alpha=0.5, s=30,
               edgecolors='none')

    max_val = max(client_risk.max(), prod_risk.max()) + 0.05
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, alpha=0.6,
            label='Product risk = Client risk')
    ax.set_xlabel('Client RiskPropensity')
    ax.set_ylabel('Recommended Product Risk (Rec_1)')
    ax.set_title('Suitability Check: Client vs Product Risk', fontsize=12, pad=10)

    legend_handles = [
        Patch(color='#4c72b0', label='Income product (Type=0)'),
        Patch(color='#dd8452', label='Accumulation product (Type=1)'),
    ]
    ax.legend(handles=legend_handles + [ax.get_lines()[0]], fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_need_probability_hist(
    p_inc_all:    np.ndarray,
    p_acc_all:    np.ndarray,
    need_threshold: float = 0.5
) -> None:
    """
    Two overlapping histograms of predicted need probabilities — one per target.
    A vertical dashed line marks the decision threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    data   = [p_inc_all, p_acc_all]
    titles = ['P(Income need)', 'P(Accumulation need)']
    colors = ['#4c72b0', '#dd8452']

    for ax, probs, title, color in zip(axes, data, titles, colors):
        ax.hist(probs, bins=40, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(need_threshold, color='red', lw=1.5, linestyle='--',
                   label=f'Threshold = {need_threshold}')
        ax.set_xlabel('Predicted probability', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(fontsize=9)

    plt.suptitle('Predicted Need Probability Distribution', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
