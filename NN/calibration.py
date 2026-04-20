"""
Probability calibration and threshold tuning for the Needs model.

The network outputs raw sigmoid probabilities; these are not well-calibrated
out of the box because training uses pos_weight to fight class imbalance.
Calibration is fit on the VALIDATION split and then applied unchanged to the
TEST split, so the test evaluation remains untouched.

Platt scaling is preferred over isotonic here: the validation set is small
(~800 rows) and isotonic is a step function with up to N-1 knots, which tends
to overfit. Platt is a 2-parameter logistic fit that generalises more reliably.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, brier_score_loss


def get_raw_probs(model, loader):
    """Return (p_acc, p_inc, y_acc, y_inc) as flat numpy arrays."""
    model.eval()
    pa, pi, ya, yi = [], [], [], []
    with torch.no_grad():
        for X_batch, y_inc, y_acc in loader:
            la, li = model(X_batch)
            pa.extend(torch.sigmoid(la).numpy())
            pi.extend(torch.sigmoid(li).numpy())
            ya.extend(y_acc.numpy())
            yi.extend(y_inc.numpy())
    return np.array(pa), np.array(pi), np.array(ya), np.array(yi)


def fit_calibrator(p_val, y_val, name):
    """
    Fit Platt scaling on validation, keep it only if it actually improves Brier
    score. If calibration would hurt a particular head (can happen when the
    head is already well-calibrated), we keep the raw sigmoid output for that
    head and log the decision.

    Returns
    -------
    calibrator : callable  (p: np.ndarray) -> np.ndarray
    lr_object  : fitted sklearn LogisticRegression (serialisable)
    kept       : bool  -- whether calibration was actually applied
    """
    lr = LogisticRegression(C=1.0, solver='liblinear')
    lr.fit(p_val.reshape(-1, 1), y_val)
    calibrated = lambda p: lr.predict_proba(p.reshape(-1, 1))[:, 1]

    brier_raw = brier_score_loss(y_val, p_val)
    brier_cal = brier_score_loss(y_val, calibrated(p_val))

    if brier_cal < brier_raw:
        print(f"  {name}: calibration KEPT    (Brier {brier_raw:.4f} -> {brier_cal:.4f})")
        return calibrated, lr, True

    print(f"  {name}: calibration SKIPPED (Brier {brier_raw:.4f} -> {brier_cal:.4f}, would hurt)")
    return (lambda p: p), lr, False


def best_threshold(y_true, p, grid=None):
    """F1-optimal decision threshold on a (held-out) probability array."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, p > t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])
