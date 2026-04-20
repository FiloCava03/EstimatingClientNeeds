"""
Multi-task and single-task MLP architectures for the Needs model.

The multi-task network is the production model. The single-task variant is
kept here (rather than in the notebook) because the ablation study trains
an identical-capacity single-task model and compares it head-to-head against
the multi-task one; keeping both definitions in one place makes the "same
capacity, different structure" claim obvious by inspection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskNeedsMLP(nn.Module):
    """
    Shared trunk + two task-specific heads (Accumulation, Income).

    Rationale for shared-trunk design: the two targets are correlated through
    life-cycle variables (age, wealth, risk). A shared representation lets the
    gradient from each head regularise the other, which is especially useful
    for the minority Income class.
    """

    def __init__(self, in_dim, trunk=(64, 32), head=(16,), p=0.25):
        super().__init__()

        layers, d = [], in_dim
        for h in trunk:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)

        def make_head():
            L, d_h = [], d
            for h in head:
                L += [nn.Linear(d_h, h), nn.GELU(), nn.Dropout(p)]
                d_h = h
            L += [nn.Linear(d_h, 1)]          # logits, not probabilities
            return nn.Sequential(*L)

        self.head_accum  = make_head()
        self.head_income = make_head()

    def forward(self, x):
        z = self.trunk(x)
        return self.head_accum(z).squeeze(-1), self.head_income(z).squeeze(-1)


class SingleTaskMLP(nn.Module):
    """
    Single-output counterpart to one branch of MultiTaskNeedsMLP.

    Has exactly the same capacity as the multi-task trunk + one head, so a
    head-to-head comparison isolates the *structural* difference (shared
    representation + KL prior) rather than a capacity difference.
    """

    def __init__(self, in_dim, trunk=(64, 32), head=(16,), p=0.25):
        super().__init__()

        layers, d = [], in_dim
        for h in trunk:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(p)]
            d = h
        for h in head:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def multi_task_loss(logits_a, logits_i, y_a, y_i, w_a, w_i, joint_prior, lam=0.1):
    """
    Weighted BCE on both heads plus a KL penalty anchoring the batch's implied
    joint distribution (Neither, IncomeOnly, AccumOnly, Both) to the empirical
    joint from the training set.

    The KL term is the reason we chose a multi-task formulation: it regularises
    each head against *population-level* advisor behaviour, which dampens the
    label noise that comes from the revealed-preference labelling scheme.
    """
    bce_a = F.binary_cross_entropy_with_logits(logits_a, y_a, pos_weight=w_a)
    bce_i = F.binary_cross_entropy_with_logits(logits_i, y_i, pos_weight=w_i)

    p_a = torch.sigmoid(logits_a)
    p_i = torch.sigmoid(logits_i)

    # Implied joint distribution of the four segments, over the batch
    implied = torch.stack([
        ((1 - p_a) * (1 - p_i)).mean(),       # Neither
        ((1 - p_a) *      p_i ).mean(),       # Income only
        (     p_a  * (1 - p_i)).mean(),       # Accumulation only
        (     p_a  *      p_i ).mean(),       # Both
    ])

    consistency = F.kl_div((implied + 1e-8).log(), joint_prior,
                           reduction='batchmean')

    return bce_a + bce_i + lam * consistency
