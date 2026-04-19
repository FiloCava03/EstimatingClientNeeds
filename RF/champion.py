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

    Returns:
        champion_result : the full result dict of the winner (model + metrics)
        winner_name     : 'base' or 'engineered'
        cv_winner       : CV mean of the winner (for logging)
        cv_loser        : CV mean of the loser  (for logging)
    """
    score_base = res_base["cv_metrics"][metric]["mean"]
    score_eng = res_eng["cv_metrics"][metric]["mean"]

    # Strict > so that in case of a tie we fall back to the simpler
    # base model (Occam's razor — and the Module 2 slides mention it)
    if score_eng > score_base:
        return res_eng, "engineered", score_eng, score_base
    else:
        return res_base, "base", score_base, score_eng
