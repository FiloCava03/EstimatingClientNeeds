def pick_champion(res_base, res_eng, metric="pr_auc"):
    """
    Pick the winning model between two candidates (base vs engineered
    features) using the CV mean of the chosen metric as the tiebreaker.

    Rationale for using CV and NOT the test set: the test set is meant to
    be touched exactly once — at the very end, to report final performance
    of the already-chosen model. Selecting a model based on its test score
    turns that score into a training signal, and the resulting test metric
    is no longer an honest estimate of real-world performance. The CV mean
    is the correct number for model selection because the CV folds are part
    of the training regime and can be "spent" on decisions.

    Ties are broken in favor of the BASE model (Occam's razor): with equal
    CV performance we prefer the simpler, more interpretable feature set.

    Parameters
    ----------
    res_base, res_eng : dict
        Results from train_evaluate_model(). Each must contain
        res["cv_metrics"][metric]["mean"] and res["model"].
    metric : str, default "pr_auc"
        Key in cv_metrics to use as the selection criterion.
        Default is PR-AUC because at least one of the two targets
        (IncomeInvestment, ~38% positives) is imbalanced, and PR-AUC is
        more informative than ROC-AUC under class imbalance.

    Returns
    -------
    tuple of (champion_result, winner_name, cv_winner, cv_loser)
        champion_result : dict
            The full result dict of the winner (model + metrics).
        winner_name : {'base', 'engineered'}
            Which feature set won — needed at inference time to pick the
            right X matrix.
        cv_winner : float
            CV mean of the winner (for logging).
        cv_loser : float
            CV mean of the loser (for logging).
    """
    score_base = res_base["cv_metrics"][metric]["mean"]
    score_eng = res_eng["cv_metrics"][metric]["mean"]

    # Strict > so that in case of a tie we fall back to the simpler
    # base model (Occam's razor — and the Module 2 slides mention it)
    if score_eng > score_base:
        return res_eng, "engineered", score_eng, score_base
    else:
        return res_base, "base", score_base, score_eng
