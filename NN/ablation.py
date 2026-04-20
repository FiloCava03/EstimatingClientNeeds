"""
Ablation study: is the performance of the multi-task network coming from the
shared representation, the KL consistency term, the noise augmentation, or
none of the above?

We compare four variants against the tuned multi-task model:

    1. Single-Task                      -- no shared trunk, no KL
    2. Multi-Task (no KL, no aug)       -- shared trunk only
    3. Multi-Task (KL, no aug)          -- + joint-distribution KL penalty
    4. Multi-Task (KL + aug)            -- + Gaussian noise on continuous cols

All variants are trained from scratch with the same seed and reported on the
validation set; test remains untouched for final evaluation.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from NN.dataset  import FinancialNeedsDataset
from NN.model    import MultiTaskNeedsMLP, SingleTaskMLP
from NN.train    import train_multitask_model
from NN.calibration import get_raw_probs


@dataclass
class Variant:
    name: str
    multi_task:  bool
    use_kl:      bool
    use_augment: bool


def train_single_head(in_dim, train_loader, val_loader, pos_weight,
                      target_idx, epochs=40, seed=42):
    """
    target_idx: 0 -> Income head, 1 -> Accumulation head.

    No early stopping here -- we want the ablation to reflect the architecture,
    not the stopping criterion. Returns probabilities on the validation set.
    """
    torch.manual_seed(seed)
    m = SingleTaskMLP(in_dim=in_dim)
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        m.train()
        for X_batch, y_inc_b, y_acc_b in train_loader:
            targets = y_inc_b if target_idx == 0 else y_acc_b
            opt.zero_grad()
            logits = m(X_batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight)
            loss.backward()
            opt.step()

    m.eval()
    probs = []
    with torch.no_grad():
        for X_batch, _, _ in val_loader:
            probs.extend(torch.sigmoid(m(X_batch)).numpy())
    return np.array(probs)


def run_variant(v, X_train, y_train, X_val, y_val, val_loader,
                feature_cols, continuous_mask,
                w_a, w_i, joint_prior, seed=42):
    """
    Train one ablation variant and return its validation metrics.
    """
    torch.manual_seed(seed)

    train_ds = FinancialNeedsDataset(
        X_train, y_train,
        augment=v.use_augment, noise_std=0.05,
        continuous_mask=continuous_mask,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    ya = y_val['AccumulationInvestment'].values
    yi = y_val['IncomeInvestment'].values

    if v.multi_task:
        m = MultiTaskNeedsMLP(in_dim=len(feature_cols))
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        lam = 0.15 if v.use_kl else 0.0
        m, _ = train_multitask_model(
            m, train_loader, val_loader, opt,
            w_a, w_i, joint_prior,
            epochs=100, patience=15, lam=lam,
        )
        pa, pi, _, _ = get_raw_probs(m, val_loader)
    else:
        # Two separate single-task models, each the same size as one
        # multi-task branch. Capacity is matched; only structure differs.
        pa = train_single_head(len(feature_cols), train_loader, val_loader,
                               w_a, target_idx=1, seed=seed)
        pi = train_single_head(len(feature_cols), train_loader, val_loader,
                               w_i, target_idx=0, seed=seed)

    return {
        'variant':     v.name,
        'PR-AUC Acc':  average_precision_score(ya, pa),
        'PR-AUC Inc':  average_precision_score(yi, pi),
        'ROC-AUC Acc': roc_auc_score(ya, pa),
        'ROC-AUC Inc': roc_auc_score(yi, pi),
    }
