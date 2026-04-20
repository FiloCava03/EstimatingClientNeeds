"""
Probability calibration and decision-threshold tuning for the Needs models.

Both functions operate on the validation set held out by ``prepare_model_data``.
The test set is never touched here, so it remains an honest estimate of
real-world performance for the final report.

Design choices:

* **Platt scaling** (2-parameter logistic) instead of isotonic regression:
  isotonic has up to N-1 knots and overfits aggressively on small validation
  sets. Platt's two parameters are far more robust on a few hundred clients.

* **Brier-based gate**: we keep the calibrator only if it actually lowers
  Brier on the validation set. Otherwise we fall back to the identity. This
  defends against fitting a calibrator that hurts more than it helps.

* **F1-optimal threshold**: F1 weighs precision and recall symmetrically.
  For asymmetric business losses (mis-selling cost vs. missed revenue),
  swap ``f1_score`` for ``fbeta_score`` with ``beta > 1`` (favour recall)
  or ``beta < 1`` (favour precision).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score


def fit_platt_if_helpful(p_val, y_val, name=""):
    """
    Fit a Platt-scaling calibrator on validation probabilities and keep it
    only if it lowers Brier score vs the raw probabilities.

    Parameters
    ----------
    p_val : np.ndarray, shape (n,)
        Raw predicted probabilities on the validation set.
    y_val : np.ndarray, shape (n,)
        True labels on the validation set.
    name : str
        Label used only for logging (e.g. "Income", "Accumulation").

    Returns
    -------
    calibrator : callable
        Function ``p_raw -> p_cal``. Identity if calibration was rejected.
    kept : bool
        Whether the Platt calibrator was retained.
    """
    lr = LogisticRegression(C=1.0, solver="liblinear")
    lr.fit(p_val.reshape(-1, 1), y_val)
    p_cal = lr.predict_proba(p_val.reshape(-1, 1))[:, 1]

    brier_raw = brier_score_loss(y_val, p_val)
    brier_cal = brier_score_loss(y_val, p_cal)

    if brier_cal < brier_raw:
        print(f"  {name}: calibration KEPT    (Brier {brier_raw:.4f} -> {brier_cal:.4f})")
        return (lambda p: lr.predict_proba(p.reshape(-1, 1))[:, 1]), True

    print(f"  {name}: calibration SKIPPED (Brier {brier_raw:.4f} -> {brier_cal:.4f}, would hurt)")
    return (lambda p: p), False


def best_threshold_f1(y_true, p, grid=None):
    """
    Pick the F1-maximising decision threshold on a fixed grid.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        True labels.
    p : np.ndarray, shape (n,)
        Predicted probabilities (calibrated).
    grid : np.ndarray, optional
        Threshold grid. Defaults to 181 points in [0.05, 0.95].

    Returns
    -------
    threshold : float
        F1-optimal threshold.
    f1 : float
        F1 score achieved at that threshold.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    scores = np.array([f1_score(y_true, p > t) for t in grid])
    i = int(scores.argmax())
    return float(grid[i]), float(scores[i])
