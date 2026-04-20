"""
utils/
======
Moduli per il pipeline Decision Tree.

Le funzioni condivise con il notebook NN vengono importate direttamente
da NN/ — non sono duplicate qui.

Mappa delle dipendenze
----------------------
Feature engineering
    prepare_features()           <- questo file (wrapper sottile)
      chiama  NN.features.clean_and_engineer()   <- FONTE CANONICA

Split + scaling
    NN.features.prepare_model_data()             <- usa direttamente nel notebook

Training DT
    train_evaluate_dt()
    display_results_table()

Tuning DT
    optimize_dt()                <- Optuna

SHAP / Permutation importance
    DT: build_explainers(), get_shap_pos(), get_base_val(), permutation_importance_dt()
    NN: usa NN.explain direttamente

Recommendation
    compute_score(), generate_recommendations(), analyse_coverage()

Plotting
    Tutte le plot_*() per il DT
    ROC/PR curves: usa NN.evaluate.plot_test_curves() (condiviso)
"""

from .data_loading        import load_data
from .feature_engineering import prepare_features
from .model_utils         import train_evaluate_dt, display_results_table
from .tuning              import optimize_dt
from .shap_utils          import (build_explainers, get_shap_pos,
                                   get_base_val, permutation_importance_dt)
from .recommendation      import (compute_score, generate_recommendations,
                                   analyse_coverage)
from .plotting            import *
