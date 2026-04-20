"""
recommendation.py
=================
Next Best Action (NBA) engine for the Needs-based investment product
recommendation system.

Pipeline
--------
1. compute_score        – MiFID-compliant composite score for one client-product pair
2. generate_recommendations – top-K product recommendations for every client
3. analyse_coverage     – catalogue-gap decomposition (correct silence vs gap)

Design principles
-----------------
* Regulatory compliance is a HARD gate (ComplianceMask = 0 → score = 0).
  No ML score can override a MiFID suitability constraint.

Reproducibility note
--------------------
Any random values used in the NBA pipeline (e.g. mock Utility scores)
must be generated with np.random.default_rng(seed) — a PRIVATE Generator
fully isolated from the global numpy random state.
Using np.random.seed() + np.random.uniform() is fragile: the number of
internal RNG draws inside sklearn / Optuna helpers can differ between code
versions, silently shifting the global state and changing results even
when the same seed is set.  default_rng() is immune to this.
* Suitability is a SOFT preference: a Gaussian centred on the client's
  RiskPropensity rewards products whose risk matches the client's tolerance
  without penalising modest mismatches too harshly.
* Business utility is a secondary weight so that, among equally suitable
  products, the firm can prefer higher-margin offerings — but only after
  regulatory and need filters are satisfied.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single client-product scoring
# ---------------------------------------------------------------------------

def compute_score(
    p_need:      float,
    client_risk: float,
    product_row: pd.Series,
    owned_set:   set
) -> float:
    """
    Composite MiFID-compliant score for recommending one product to one client.

    Formula:   Score = P(need) × Suitability × Utility × ComplianceMask

    ComplianceMask (hard gate — any breach → score = 0)
    ---------------------------------------------------
    * Risk breach:   product_risk > client_risk + 0.05
                     The 0.05 tolerance avoids rejecting a product that is
                     only marginally above the client's propensity due to
                     rounding in the data.
    * Already owned: product is in owned_set (no re-recommendation).

    Suitability (soft preference)
    ------------------------------
    Gaussian: exp(−3 × (product_risk − client_risk)²)
    Peaks at 1.0 when product_risk == client_risk, decays smoothly as the
    gap widens.  This prevents systematically pushing low-risk clients to
    the cheapest product and high-risk clients to the most aggressive one.

    Parameters
    ----------
    p_need       : float  calibrated P(client needs this product type)
    client_risk  : float  client's RiskPropensity in [0, 1]
    product_row  : pd.Series  one row from products_df (needs 'Risk', 'Type',
                  'IDProduct', 'Utility')
    owned_set    : set    product IDs already held by this client

    Returns
    -------
    float — composite score (0 if any hard constraint is violated)
    """
    product_risk = float(product_row['Risk'])
    product_id   = product_row['IDProduct']
    utility      = float(product_row.get('Utility', 1.0))

    # ── Hard MiFID gate ──────────────────────────────────────────────────
    if product_risk > client_risk + 0.05:
        return 0.0   # risk breach: never recommend this product to this client

    if product_id in owned_set:
        return 0.0   # client already owns it: no value in re-recommending

    # ── Soft suitability score ────────────────────────────────────────────
    # Gaussian centred on the client's own risk tolerance.
    # sigma is implicit in the coefficient 3; larger = narrower peak.
    gap         = product_risk - client_risk
    suitability = float(np.exp(-3.0 * gap ** 2))

    # ── Composite score ───────────────────────────────────────────────────
    return p_need * suitability * utility


# ---------------------------------------------------------------------------
# Batch recommendation generator
# ---------------------------------------------------------------------------

def generate_recommendations(
    needs_df:     pd.DataFrame,
    products_df:  pd.DataFrame,
    p_inc_all:    np.ndarray,
    p_acc_all:    np.ndarray,
    owned_lookup: dict = None,
    top_k:        int  = 3
) -> pd.DataFrame:
    """
    Generate up to top_k MiFID-compliant product recommendations for every client.

    For each client:
      1. For each product, select P(need) based on the product's type:
           Type = 1 (accumulation) → p_acc_all[i]
           Type = 0 (income)       → p_inc_all[i]
      2. Compute the composite score.
      3. Rank products by score descending; keep top_k.
      4. Pad with (None, 0.0) if fewer than top_k products pass the hard gate.

    Parameters
    ----------
    needs_df     : original client DataFrame (used for RiskPropensity and count)
    products_df  : product catalogue with columns IDProduct, Type, Risk, Utility
    p_inc_all    : (n_clients,) array of P(income need) for every client
    p_acc_all    : (n_clients,) array of P(accumulation need) for every client
    owned_lookup : dict {client_index → set(product_ids already owned)}
                   defaults to empty dict (no existing holdings assumed)
    top_k        : number of recommendations to return per client (default 3)

    Returns
    -------
    pd.DataFrame with columns:
        ClientIdx, Rec_1, Score_1, Rec_2, Score_2, ..., Rec_{k}, Score_{k}
    """
    if owned_lookup is None:
        owned_lookup = {}

    client_risks = needs_df['RiskPropensity'].values
    rows = []

    for i in range(len(needs_df)):
        owned  = owned_lookup.get(i, set())
        scores = []

        for _, prod in products_df.iterrows():
            # Route to the correct need probability based on product type
            p_need = p_acc_all[i] if prod['Type'] == 1 else p_inc_all[i]
            s = compute_score(p_need, client_risks[i], prod, owned)
            if s > 0:
                scores.append((int(prod['IDProduct']), round(s, 4)))

        # Sort descending by score and pad to top_k with (None, 0.0)
        scores.sort(key=lambda x: x[1], reverse=True)
        scores += [(None, 0.0)] * (top_k - len(scores))

        row = {'ClientIdx': i}
        for rank in range(top_k):
            row[f'Rec_{rank+1}']   = scores[rank][0]
            row[f'Score_{rank+1}'] = scores[rank][1]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def analyse_coverage(
    needs_df:      pd.DataFrame,
    recs_df:       pd.DataFrame,
    p_inc_all:     np.ndarray,
    p_acc_all:     np.ndarray,
    need_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Decompose clients with no recommendation into two distinct buckets:

    (a) Correct silence — the model predicted no need above the threshold.
        Under MiFID/IDD, NOT recommending when there is no need is compliant
        and correct behaviour.  Do NOT act on this segment.

    (b) Catalogue gap — the model DID predict a need, but no compliant product
        exists in the current catalogue for this client's risk tolerance.
        This is a PRODUCT LINE OPPORTUNITY: the product team should consider
        adding suitable instruments for this risk segment.

    Parameters
    ----------
    needs_df        : client DataFrame
    recs_df         : recommendations DataFrame (from generate_recommendations)
    p_inc_all       : (n_clients,) income-need probabilities
    p_acc_all       : (n_clients,) accumulation-need probabilities
    need_threshold  : probability cut-off to classify a client as 'in need'

    Returns
    -------
    pd.DataFrame summarising counts and percentages for the three segments
    """
    n = len(needs_df)

    # A client is 'in need' if either need exceeds the threshold
    need_predicted = (p_acc_all >= need_threshold) | (p_inc_all >= need_threshold)
    has_any_rec    = recs_df['Rec_1'].notna().values

    # Three mutually exclusive segments
    no_need       = (~need_predicted) & (~has_any_rec)
    catalogue_gap =  need_predicted   & (~has_any_rec)
    covered       =  has_any_rec

    summary = pd.DataFrame({
        'Segment': [
            'No need predicted (correct silence)',
            'Catalogue gap (need predicted, no compliant product)',
            'At least one recommendation emitted',
        ],
        'Clients': [
            int(no_need.sum()),
            int(catalogue_gap.sum()),
            int(covered.sum()),
        ],
        'Pct (%)': [
            round(no_need.mean() * 100, 1),
            round(catalogue_gap.mean() * 100, 1),
            round(covered.mean() * 100, 1),
        ],
    })

    # Per-need-type partial gaps (useful for the product team)
    acc_gap = (p_acc_all >= need_threshold) & (~has_any_rec)
    inc_gap = (p_inc_all >= need_threshold) & (~has_any_rec)

    print('=== REFINED COVERAGE: why are some clients not covered? ===')
    print(summary.to_string(index=False))
    print(f'\nPartial catalogue gaps (per need head):')
    print(f'  Accumulation need predicted but NO compatible product: '
          f'{int(acc_gap.sum())}')
    print(f'  Income        need predicted but NO compatible product: '
          f'{int(inc_gap.sum())}')
    print('\nBusiness insight: catalogue gap clients are actionable — the model '
          'correctly\nidentified a need, but no product in the current catalogue '
          'fits their risk\ntolerance.  This is a product line opportunity, not '
          'a model limitation.')

    return summary
