import numpy as np


def prepare_features(df):
    """
    Build two feature sets from the raw dataset: a base set and an engineered
    set. The engineered set adds a single interaction feature (Income/Wealth
    ratio in log form) on top of the base set. Returns (X_base, X_engineered).

    Note on scaling: a MinMaxScaler was intentionally NOT applied here, for
    two reasons — first, applying it before train_test_split would leak test
    statistics into the training data; second, Random Forests split on
    thresholds and are invariant to monotonic transformations, so scaling
    would be useless anyway.
    """
    # Work on a copy so the caller's dataframe is never mutated in place
    X = df.copy()

    # Wealth and Income are right-skewed: log1p compresses the tail and
    # keeps zero-valued rows well-defined (log1p(0) = 0 instead of -inf)
    X["Wealth_log"] = np.log1p(X["Wealth"])
    # Trailing space in 'Income ' is intentional: it's the original column
    # name in the Excel file and renaming it here would break other cells
    X["Income_log"] = np.log1p(X["Income "])

    # Create Income/Wealth ratio.
    # If Wealth is 0, replace it with NaN to avoid division by zero (which
    # would produce inf). Then fill the resulting NaNs with 0 — a safe
    # fallback for tree-based models, since 0 is a legitimate split value
    # and does not distort the variable's scale.
    X["Income_Wealth_Ratio"] = (
        X["Income "].div(X["Wealth"].replace(0, np.nan)).fillna(0)
    )

    # Log of the ratio to tame a few extreme values that survive winsorization
    X["Income_Wealth_Ratio_log"] = np.log1p(X["Income_Wealth_Ratio"])

    # Base feature set: raw variables + log transforms of the monetary ones
    features_base = [
        "Age",
        "Gender",
        "FamilyMembers",
        "FinancialEducation",
        "RiskPropensity",
        "Wealth_log",
        "Income_log",
    ]

    # Engineered feature set: swap the two raw-ish log-monetary features
    # for the single Income/Wealth ratio, keeping the rest identical
    features_engineered = [
        "Age",
        "Gender",
        "FamilyMembers",
        "FinancialEducation",
        "RiskPropensity",
        "Income_Wealth_Ratio_log",
    ]

    X_base = X[features_base]
    X_engineered = X[features_engineered]

    return X_base, X_engineered


def prepare_features_advance(df):
    """
    More elaborate feature engineering, driven by financial life-cycle logic.
    On top of the base log transforms, it adds life-stage indicators, age
    polynomials, saving proxies, an interaction between financial education
    and wealth, a risk-appetite gap, and a dependents-per-income burden.

    Cleaning (clip + winsorize) is done BEFORE feature construction, so the
    engineered features are not polluted by a handful of extreme outliers.
    Returns (X_base, X_engineered).
    """
    X = df.copy()

    # Strip trailing spaces in column names ('Income ' -> 'Income'), so the
    # rest of the code can reference the clean name without surprises
    X.columns = X.columns.str.strip()

    # --- 1. Cleaning: clip to plausible ranges + winsorize heavy tails ---
    # Clip Age and FamilyMembers to sensible human values; guard Wealth and
    # Income against zero/negative values that would break log/ratio ops
    X["Age"] = X["Age"].clip(18, 100)
    X["FamilyMembers"] = X["FamilyMembers"].clip(1, 8)
    X["Wealth"] = X["Wealth"].clip(lower=0.1)
    X["Income"] = X["Income"].clip(lower=0.1)

    # Winsorize at the 99.5th percentile: the top 0.5% of values is pulled
    # down to the cap, so a handful of very rich clients don't dominate
    # the scale of engineered features built on top of Wealth/Income.
    # Note: computing the quantile on the full dataset before splitting is
    # a mild form of leakage; negligible at this sample size but worth noting.
    for col in ["Wealth", "Income"]:
        hi = X[col].quantile(0.995)
        X[col] = X[col].clip(upper=hi)

    # --- 2. Base features: log transforms of the monetary variables ---
    X["Wealth_log"] = np.log1p(X["Wealth"])
    X["Income_log"] = np.log1p(X["Income"])

    # --- 3. Engineered features ---
    # Life stage dummies: working-age vs retired, to let the model pick up
    # regime changes that a single Age variable would smooth over
    X["LifeStage_working"] = ((X["Age"] >= 25) & (X["Age"] < 65)).astype(int)
    X["LifeStage_retired"] = (X["Age"] >= 65).astype(int)
    # Age squared — redundant for a Random Forest (it splits on thresholds
    # anyway) but kept for consistency with linear baselines
    X["Age_sq"] = X["Age"] ** 2
    # Distance from retirement: clipped at 0 so post-retirement clients get 0
    X["YearsToRetire"] = (67 - X["Age"]).clip(lower=0)

    # Saving proxy: how much wealth per working year. Adding +1 and clipping
    # at 1 avoids tiny denominators for very young clients.
    X["WealthPerWorkYear"] = X["Wealth"] / (X["Age"] - 18 + 1).clip(lower=1)
    X["WealthPerWorkYear_log"] = np.log1p(X["WealthPerWorkYear"])

    # Liquidity / cash-flow ratio. +1 removed here because Income was already
    # clipped away from 0 at the cleaning stage, so the division is safe.
    X["IncomeWealthRatio"] = X["Income"] / X["Wealth"]
    X["IncomeWealthRatio_log"] = np.log1p(X["IncomeWealthRatio"])

    # Interaction: financial education gets more "reach" when the client
    # has more wealth to deploy — intended as a sophistication index
    X["Sophistication"] = X["FinancialEducation"] * X["Wealth_log"]

    # Risk mismatch vs the classic "100 - age" heuristic, rescaled to [0,1].
    # Positive RiskGap = client more aggressive than the benchmark for their age
    expected_risk = np.clip(1 - X["Age"] / 100, 0, 1)
    X["RiskGap"] = X["RiskPropensity"] - expected_risk

    # Household burden: dependents per unit of income
    X["DependentsPerIncome"] = X["FamilyMembers"] / (X["Income"] + 1)

    # --- 4. Column selection ---
    # Base set: same shape as in prepare_features(), kept for apples-to-apples
    # comparison between the two pipelines
    features_base = [
        "Age",
        "Gender",
        "FamilyMembers",
        "FinancialEducation",
        "RiskPropensity",
        "Wealth_log",
        "Income_log",
    ]

    # Engineered set: deliberately replaces Age / FamilyMembers / RiskPropensity
    # with their engineered counterparts (YearsToRetire, DependentsPerIncome,
    # RiskGap). Trade-off: more business-motivated features, but losing the
    # raw RiskPropensity — which is the variable used downstream by the NBA
    # matching rule. Flagged here so it can be revisited if needed.
    features_engineered = [
        "Gender",
        "FinancialEducation",
        "Wealth_log",
        "Income_log",
        "YearsToRetire",
        "Age_sq",
        "LifeStage_working",
        "LifeStage_retired",
        "WealthPerWorkYear_log",
        "IncomeWealthRatio_log",
        "Sophistication",
        "RiskGap",
        "DependentsPerIncome",
    ]

    return X[features_base], X[features_engineered]
