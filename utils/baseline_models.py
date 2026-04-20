"""
baseline_models.py
==================
Baseline classifiers for the Needs prediction task.

Implements the full suite of baselines recommended in the project roadmap:
  - Gaussian Naive Bayes  (GNB)
  - Support Vector Machine (SVM, with probability calibration)
  - K-Nearest Neighbours  (KNN)
  - Random Forest         (RF)
  - Logistic Regression   (LR)

All models are evaluated through train_evaluate_dt (same interface) so
results are directly comparable with the Decision Tree experiments.

Design notes
------------
* SVM does NOT natively output probabilities.  We wrap it in
  CalibratedClassifierCV (Platt scaling) to get predict_proba — this is
  required for AUC-ROC / AUC-PR computation and for the NBA