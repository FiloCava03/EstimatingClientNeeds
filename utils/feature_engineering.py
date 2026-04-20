"""
feature_engineering.py
=======================
Feature engineering per il pipeline Decision Tree.

FONTE CANONICA: NN/features.py (scritto dal team NN).
Questo file e' un wrapper sottile: chiama clean_and_engineer() della sua
implementazione e restituisce i due subset di colonne che servono ai DT.

NON duplicare logica di cleaning o feature construction qui — qualsiasi
modifica alle feature deve essere fatta in NN/features.py cosi' entrambi
i notebook restano allineati automaticamente.

Struttura delle dipendenze
--------------------------
  DT notebook  ->  prepare_features()
                     chiama NN.features.clean_and_engineer() internamente
  NN notebook  ->  NN.features.clean_and_engineer() + NN.features.prepare_model_data()
                     direttamente
"""

import numpy as np
import pandas as pd
from NN.features import clean_and_engineer  # noqa: F401 — fonte canonica


# Subset di colonne per i Decision Trees (nessuno scaling necessario,
# gli alberi sono invarianti a trasformazioni monotone).

BASE_COLS = [
    'Age', 'Gender', 'FamilyMembers',
    'FinancialEducation', 'RiskPropensity',
    'Wealth_log', 'Income_log',
]

ENG_COLS = [
    'Gender', 'FinancialEducation',
    'Wealth_log', 'Income_log',
    'YearsToRetire', 'Age_sq',
    'LifeStage_working', 'LifeStage_retired',
    'WealthPerWorkYear_log', 'IncomeWealthRatio_log',
    'Sophistication', 'RiskGap',
    'DependentsPerIncome',
    'Age_x_Wealth_log',   # termine di interazione, utile per split DT
]


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Costruisce le due matrici di feature per il pipeline Decision Tree.

    Usa clean_and_engineer() di NN/features.py (fonte canonica) e aggiunge
    solo il termine di interazione Age x Wealth_log, che e' utile come split
    esplicito per gli alberi ma non necessario per la NN (lo impara il trunk).

    Parameters
    ----------
    df : pd.DataFrame
        Sheet 'Needs' caricato da load_data(), con le colonne target ancora presenti.

    Returns
    -------
    X_base       : pd.DataFrame  -- feature demografiche + log-finanziarie
    X_engineered : pd.DataFrame  -- feature life-cycle + interazioni
    """
    # Pulizia e feature engineering canonici (condivisi con la NN)
    X = clean_and_engineer(df)

    # Unico termine aggiunto qui: interazione Age x Wealth_log.
    # I DT ne hanno bisogno come feature esplicita; la NN lo impara implicitamente
    # attraverso il trunk condiviso.
    X['Age_x_Wealth_log'] = X['Age'] * X['Wealth_log']

    # Seleziona i subset di colonne, ignorando eventuali colonne mancanti
    # (rende visibili refactor futuri invece di silenziare shape mismatch)
    available = set(X.columns)
    X_base       = X[[c for c in BASE_COLS if c in available]].reset_index(drop=True)
    X_engineered = X[[c for c in ENG_COLS  if c in available]].reset_index(drop=True)

    return X_base, X_engineered
