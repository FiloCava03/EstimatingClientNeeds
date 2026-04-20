"""
Evaluation plots for the multi-task network.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_test_curves(y_acc, p_acc, y_inc, p_inc):
    """
    ROC and Precision-Recall curves for both heads, side by side.

    PR curves matter more than ROC for the imbalanced Income head: ROC-AUC is
    optimistic when the negative class dominates, whereas PR-AUC directly
    reflects precision at every recall level and is what drives outbound-
    marketing cost.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- ROC ---
    fpr_a, tpr_a, _ = roc_curve(y_acc, p_acc)
    fpr_i, tpr_i, _ = roc_curve(y_inc, p_inc)
    axes[0].plot(fpr_a, tpr_a, color='steelblue', lw=2,
                 label=f'Accumulation (AUC = {auc(fpr_a, tpr_a):.3f})')
    axes[0].plot(fpr_i, tpr_i, color='darkorange', lw=2,
                 label=f'Income (AUC = {auc(fpr_i, tpr_i):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('ROC — test set', fontsize=14, pad=15)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    # --- Precision-Recall ---
    prec_a, rec_a, _ = precision_recall_curve(y_acc, p_acc)
    prec_i, rec_i, _ = precision_recall_curve(y_inc, p_inc)
    base_a = sum(y_acc) / len(y_acc)
    base_i = sum(y_inc) / len(y_inc)

    axes[1].plot(rec_a, prec_a, color='steelblue', lw=2,
                 label=f'Accumulation (PR-AUC = {auc(rec_a, prec_a):.3f})')
    axes[1].plot(rec_i, prec_i, color='darkorange', lw=2,
                 label=f'Income (PR-AUC = {auc(rec_i, prec_i):.3f})')
    axes[1].axhline(base_a, color='steelblue',  ls='--', alpha=0.3,
                    label='Accumulation random baseline')
    axes[1].axhline(base_i, color='darkorange', ls='--', alpha=0.3,
                    label='Income random baseline')
    axes[1].set_title('Precision-Recall — test set', fontsize=14, pad=15)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    plt.show()
