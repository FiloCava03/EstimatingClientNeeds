"""
Next Best Action (NBA) — match clients to suitable products.

The matching rule mirrors the PoliMI module:

  * Filter products with ``Risk <= client RiskPropensity``
    (MiFID/IDD regulatory suitability).
  * Among the eligible ones, pick the highest-Risk product
    (max expected return within the client's risk budget).

A separate decision is taken for each head (Income, Accumulation), each
with its own per-head probability threshold tuned upstream on the
validation set. A single client may therefore receive up to two
recommendations.
"""

import pandas as pd


def recommend_product(client_risk, product_type_df):
    """
    Among products with Risk <= client_risk (regulatory suitability under
    MiFID/IDD), return the (product_id, product_risk) of the one with the
    HIGHEST risk — i.e. maximize expected return within the client's
    risk budget. Returns (None, None) if no product fits -> catalogue gap.
    """
    compatible = product_type_df[product_type_df["Risk"] <= client_risk]
    if compatible.empty:
        return None, None
    best = compatible.loc[compatible["Risk"].idxmax()]
    return int(best["IDProduct"]), float(best["Risk"])


def build_nba_table(
    client_ids,
    risk_propensity,
    p_inc,
    p_acc,
    thr_inc,
    thr_acc,
    income_products,
    accum_products,
):
    """
    Build the per-client NBA table.

    Each row corresponds to one client and may contain up to two
    recommendations (Income + Accumulation). A recommendation is emitted
    only if (a) the head's calibrated probability is above its tuned
    threshold AND (b) the catalogue contains at least one product
    compatible with the client's RiskPropensity.

    Parameters
    ----------
    client_ids : array-like
        Client identifiers (one per test row).
    risk_propensity : array-like
        Client RiskPropensity values, same length and order as ``client_ids``.
    p_inc, p_acc : array-like
        Calibrated probabilities for Income and Accumulation, same length
        and order as ``client_ids``.
    thr_inc, thr_acc : float
        Per-head decision thresholds tuned on the validation set.
    income_products, accum_products : pd.DataFrame
        Sub-catalogues of the two product families. Must contain
        ``IDProduct`` and ``Risk`` columns.

    Returns
    -------
    pd.DataFrame
        One row per client with columns: ClientID, RiskPropensity,
        P(Income), P(Accumulation), IncomeProductID, IncomeProductRisk,
        AccumulationProductID, AccumulationProductRisk.
    """
    rows = []
    for i, client_id in enumerate(client_ids):
        row = {
            "ClientID": int(client_id),
            "RiskPropensity": float(risk_propensity[i]),
            "P(Income)": float(p_inc[i]),
            "P(Accumulation)": float(p_acc[i]),
        }

        # Income — independent decision with its own tuned threshold
        if p_inc[i] >= thr_inc:
            pid, prisk = recommend_product(row["RiskPropensity"], income_products)
            row["IncomeProductID"], row["IncomeProductRisk"] = pid, prisk
        else:
            row["IncomeProductID"], row["IncomeProductRisk"] = None, None

        # Accumulation — same logic, independent decision
        if p_acc[i] >= thr_acc:
            pid, prisk = recommend_product(row["RiskPropensity"], accum_products)
            row["AccumulationProductID"], row["AccumulationProductRisk"] = pid, prisk
        else:
            row["AccumulationProductID"], row["AccumulationProductRisk"] = None, None

        rows.append(row)

    return pd.DataFrame(rows)


def coverage_summary(nba):
    """
    Operational coverage view: counts of clients by combination of
    recommendations actually emitted. Useful for downstream campaign
    planning ("how many clients get the Both treatment?").
    """
    has_income = nba["IncomeProductID"].notna()
    has_accum = nba["AccumulationProductID"].notna()
    n = len(nba)

    df = pd.DataFrame(
        {
            "Segment": [
                "No recommendation",
                "Only Income",
                "Only Accumulation",
                "Both",
            ],
            "Clients": [
                int((~has_income & ~has_accum).sum()),
                int((has_income & ~has_accum).sum()),
                int((~has_income & has_accum).sum()),
                int((has_income & has_accum).sum()),
            ],
        }
    )
    df["Percentage"] = (df["Clients"] / n * 100).round(1)
    return df


def refined_coverage(nba, thr_inc, thr_acc):
    """
    Diagnostic coverage view: decomposes the "No recommendation" bucket
    into (a) NO NEED PREDICTED — correct silence, the model thinks the
    client doesn't need either product family — and (b) CATALOGUE GAP —
    a need was predicted but no product in the catalogue is compatible
    with the client's risk tolerance.

    The catalogue-gap segment is the actionable one for the business:
    it identifies product-line holes, not model failures.
    """
    has_any = nba["IncomeProductID"].notna() | nba["AccumulationProductID"].notna()
    need_predicted = (nba["P(Income)"] >= thr_inc) | (nba["P(Accumulation)"] >= thr_acc)

    no_need = ~need_predicted & ~has_any
    catalogue_gap = need_predicted & ~has_any
    covered = has_any
    n = len(nba)

    return pd.DataFrame(
        {
            "Segment": [
                "No need predicted (correct silence)",
                "Catalogue gap (need predicted, no compatible product)",
                "At least one recommendation emitted",
            ],
            "Clients": [int(no_need.sum()), int(catalogue_gap.sum()), int(covered.sum())],
            "Percentage": [
                round(no_need.mean() * 100, 1),
                round(catalogue_gap.mean() * 100, 1),
                round(covered.mean() * 100, 1),
            ],
        }
    )
