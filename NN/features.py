"""
Data cleaning, feature engineering, and splitting for the Needs model.

Key changes vs. previous version:
    - Strict 3-way split: Train / Validation / Test. No information leakage
      between early stopping / threshold tuning (Val) and final reporting (Test).
    - `prepare_model_data` now returns the fitted QuantileTransformer and
      RobustScaler so the exact same transformation can be applied to new data
      at inference time (and serialized with joblib).
    - Stratification uses the joint (Accumulation, Income) target so all four
      segments (Neither, IncomeOnly, AccumOnly, Both) are proportional across
      all three splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler


# ---------------------------------------------------------------------------
# Cleaning + feature engineering 
# ---------------------------------------------------------------------------
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw data and engineers financial lifecycle features."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Sanity clips
    df['Age'] = df['Age'].clip(18, 100)
    df['FamilyMembers'] = df['FamilyMembers'].clip(1, 8)
    df['Wealth'] = df['Wealth'].clip(lower=0.1)
    df['Income'] = df['Income'].clip(lower=0.1)

    # Winsorize tails on fat-tailed columns
    for col in ['Wealth', 'Income']:
        lo, hi = df[col].quantile([0.005, 0.995])
        df[col] = df[col].clip(lo, hi)

    # ---- Feature engineering ----
    X = df.copy()
    X['Wealth_log'] = np.log1p(X['Wealth'])
    X['Income_log'] = np.log1p(X['Income'])

    # Life-cycle
    X['LifeStage_working'] = ((X['Age'] >= 25) & (X['Age'] < 60)).astype(int)
    X['LifeStage_retired'] = (X['Age'] >= 65).astype(int)
    X['Age_sq']            = X['Age'] ** 2
    X['YearsToRetire']     = (67 - X['Age']).clip(lower=0)

    # Financial behaviour proxies
    X['WealthPerWorkYear']     = X['Wealth'] / (X['Age'] - 18 + 1).clip(lower=1)
    X['WealthPerWorkYear_log'] = np.log1p(X['WealthPerWorkYear'])
    X['IncomeWealthRatio']     = X['Income'] / X['Wealth']
    X['IncomeWealthRatio_log'] = np.log1p(X['IncomeWealthRatio'])
    X['Sophistication']        = X['FinancialEducation'] * X['Wealth_log']

    # Risk coherence
    expected_risk = np.clip(1 - X['Age'] / 100, 0, 1)
    X['RiskGap']  = X['RiskPropensity'] - expected_risk

    # Household burden
    X['DependentsPerIncome'] = X['FamilyMembers'] / (X['Income'] + 1)

    return X


# ---------------------------------------------------------------------------
# Train / Validation / Test split + scaling
# ---------------------------------------------------------------------------
FAT_TAIL_FEATURES = [
    'Wealth', 'Income', 'Wealth_log', 'Income_log',
    'WealthPerWorkYear', 'WealthPerWorkYear_log',
    'IncomeWealthRatio', 'IncomeWealthRatio_log',
]


def prepare_model_data(
    df_engineered: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split engineered data into Train / Val / Test and fit scalers on Train only.

    Parameters
    ----------
    test_size : fraction of the full dataset reserved as hold-out (never seen
                until final reporting).
    val_size  : fraction of the remaining (train+val) pool used for Val.
                So final proportions are roughly 0.64 / 0.16 / 0.20.

    Returns
    -------
    X_train, X_val, X_test : scaled feature frames
    y_train, y_val, y_test : 2-column target frames
    features               : list of feature column names (order-preserving)
    qt                     : fitted QuantileTransformer (fat-tail columns)
    rs                     : fitted RobustScaler          (other numeric columns)
    """
    targets  = ['IncomeInvestment', 'AccumulationInvestment']
    features = [c for c in df_engineered.columns if c not in targets]

    fat_tail_features = [f for f in FAT_TAIL_FEATURES if f in features]
    standard_features = [c for c in features if c not in fat_tail_features]

    # Joint target for stratification: 0=Neither, 1=IncomeOnly, 2=AccumOnly, 3=Both
    y_strat = (df_engineered['AccumulationInvestment'] * 2
               + df_engineered['IncomeInvestment'])

    # ---- First split: (Train+Val) vs Test ----
    X_trval, X_test, y_trval, y_test, strat_trval, _ = train_test_split(
        df_engineered[features],
        df_engineered[targets],
        y_strat,
        test_size=test_size,
        stratify=y_strat,
        random_state=random_state,
    )

    # ---- Second split: Train vs Val (stratified again) ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval,
        y_trval,
        test_size=val_size,
        stratify=strat_trval,
        random_state=random_state,
    )

    # ---- Fit scalers on TRAIN only ----
    qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
    rs = RobustScaler()

    X_train_s, X_val_s, X_test_s = X_train.copy(), X_val.copy(), X_test.copy()

    X_train_s[fat_tail_features] = qt.fit_transform(X_train[fat_tail_features])
    X_val_s[fat_tail_features]   = qt.transform(X_val[fat_tail_features])
    X_test_s[fat_tail_features]  = qt.transform(X_test[fat_tail_features])

    X_train_s[standard_features] = rs.fit_transform(X_train[standard_features])
    X_val_s[standard_features]   = rs.transform(X_val[standard_features])
    X_test_s[standard_features]  = rs.transform(X_test[standard_features])

    return (X_train_s, X_val_s, X_test_s,
            y_train, y_val, y_test,
            features, qt, rs)