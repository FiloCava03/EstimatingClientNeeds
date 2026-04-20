"""
recommendation.py
=================
Next Best Action (NBA) engine per il pipeline Decision Tree.

match_best_product() e' condivisa con il notebook NN (stessa logica di
matching MiFID). Tutto il resto usa l'interfaccia sklearn del DT:
predict_proba() su feature non scalate, soglia fissa 0.5.

Pipeline
--------
1. match_best_product()      -- regola MiFID: prodotto piu' rischioso ancora compliance
2. generate_recommendations() -- applica la regola a tutti i clienti con DT proba
3. analyse_coverage()        -- scompone clienti in silence / covered / gap
4. plot_nba_diagnostics()    -- scatter suitability + frequency bar
5. print_top_products()      -- top-k prodotti per head
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Matching rule — condivisa con notebook NN (logica identica)
# ---------------------------------------------------------------------------

def match_best_product(client_risk: float, catalogue: pd.DataFrame):
    """
    Seleziona il prodotto con il risk piu' alto ancora compliance.

    Regola MiFID: product_risk <= client_risk.
    Tra tutti i prodotti ammissibili prende quello che massimizza il risk,
    cioe' il piu' adatto al profilo del cliente senza violare i limiti.

    Questa funzione e' identica a quella usata nel notebook NN — non dipende
    ne' dal tipo di modello ne' dalla pipeline di scaling.

    Parameters
    ----------
    client_risk : float  RiskPropensity del cliente in [0, 1]
    catalogue   : pd.DataFrame  subset di products_df filtrato per Type
                  (solo accum o solo income), colonne IDProduct e Risk

    Returns
    -------
    (product_id, product_risk) : (int, float)
    oppure (None, None) se nessun prodotto e' compliance (catalogue gap)
    """
    eligible = catalogue[catalogue['Risk'] <= client_risk + 1e-9]
    if eligible.empty:
        return None, None
    best = eligible.loc[eligible['Risk'].idxmax()]
    return int(best['IDProduct']), float(best['Risk'])


# ---------------------------------------------------------------------------
# Batch NBA — specifica DT
# ---------------------------------------------------------------------------

def generate_recommendations(
    needs_df:    pd.DataFrame,
    products_df: pd.DataFrame,
    model_acc,                   # sklearn classifier per need Accumulation
    model_inc,                   # sklearn classifier per need Income
    X_features:  pd.DataFrame,   # feature NON scalate (DT invariante a scaling)
    thr_acc:     float = 0.5,
    thr_inc:     float = 0.5,
) -> pd.DataFrame:
    """
    Genera una raccomandazione per ogni cliente usando i modelli DT.

    Differenze rispetto alla NBA del notebook NN:
    - Le probabilita' vengono da predict_proba() sklearn, non da sigmoid(logits)
    - Le feature NON sono scalate (i DT sono invarianti a trasformazioni monotone)
    - Non c'e' calibrazione Platt (il DT e' gia' abbastanza calibrato dopo Optuna)
    - La soglia e' 0.5 di default; puo' essere ottimizzata con best_threshold()
      di NN/calibration.py se si vuole allineare alla pipeline NN

    Parameters
    ----------
    needs_df    : DataFrame originale (serve solo RiskPropensity)
    products_df : catalogo prodotti con colonne IDProduct, Type, Risk
    model_acc   : DT fittato per predire AccumulationInvestment
    model_inc   : DT fittato per predire IncomeInvestment
    X_features  : feature matrix (n_clients, n_features), NON scalata
    thr_acc     : soglia probabilita' per classificare need accum
    thr_inc     : soglia probabilita' per classificare need income

    Returns
    -------
    pd.DataFrame con colonne:
        ClientID, ClientRisk, P_Accum, P_Income,
        NeedAccum, NeedIncome,
        Rec_Accum_ID, Rec_Accum_Risk,
        Rec_Income_ID, Rec_Income_Risk
    """
    # Probabilita' dirette dal DT — nessuno scaling, nessuna calibrazione
    p_acc_all = model_acc.predict_proba(X_features)[:, 1]
    p_inc_all = model_inc.predict_proba(X_features)[:, 1]

    need_acc = p_acc_all >= thr_acc
    need_inc = p_inc_all >= thr_inc

    # Separa catalogo per tipo — fatto una volta sola fuori dal loop
    accum_products  = products_df[products_df['Type'] == 1].reset_index(drop=True)
    income_products = products_df[products_df['Type'] == 0].reset_index(drop=True)

    client_risks = needs_df['RiskPropensity'].values
    rows = []

    for i in range(len(needs_df)):
        cr    = client_risks[i]
        rec_a = match_best_product(cr, accum_products)  if need_acc[i] else (None, None)
        rec_i = match_best_product(cr, income_products) if need_inc[i] else (None, None)

        rows.append({
            'ClientID'        : i + 1,
            'ClientRisk'      : round(float(cr), 4),
            'P_Accum'         : round(float(p_acc_all[i]), 4),
            'P_Income'        : round(float(p_inc_all[i]), 4),
            'NeedAccum'       : bool(need_acc[i]),
            'NeedIncome'      : bool(need_inc[i]),
            'Rec_Accum_ID'    : rec_a[0],
            'Rec_Accum_Risk'  : rec_a[1],
            'Rec_Income_ID'   : rec_i[0],
            'Rec_Income_Risk' : rec_i[1],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage decomposition
# ---------------------------------------------------------------------------

def analyse_coverage(nba: pd.DataFrame) -> pd.DataFrame:
    """
    Scompone i clienti in tre segmenti mutuamente esclusivi:

    1. Correct silence  — modello non ha previsto nessun need (MiFID compliant).
    2. Covered          — almeno una raccomandazione emessa.
    3. Catalogue gap    — need previsto ma nessun prodotto compliance disponibile.
                          Segnale per il product team, non errore del modello.

    Parameters
    ----------
    nba : DataFrame da generate_recommendations()

    Returns
    -------
    pd.DataFrame con Segment / Clients / Pct (%)
    """
    covered_acc = nba['Rec_Accum_ID'].notna()
    covered_inc = nba['Rec_Income_ID'].notna()
    covered_any = covered_acc | covered_inc

    gap_acc = nba['NeedAccum']  & ~covered_acc
    gap_inc = nba['NeedIncome'] & ~covered_inc
    silent  = ~nba['NeedAccum'] & ~nba['NeedIncome']

    n = len(nba)
    summary = pd.DataFrame({
        'Segment': [
            'No need predicted (correct silence)',
            'At least one recommendation emitted',
            'Accumulation gap (need flagged, no compliant product)',
            'Income gap (need flagged, no compliant product)',
        ],
        'Clients': [int(silent.sum()), int(covered_any.sum()),
                    int(gap_acc.sum()),  int(gap_inc.sum())],
        'Pct (%)': [round(silent.mean()*100, 1), round(covered_any.mean()*100, 1),
                    round(gap_acc.mean()*100, 1),  round(gap_inc.mean()*100, 1)],
    })

    print('=== COVERAGE DECOMPOSITION ===')
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_nba_diagnostics(nba: pd.DataFrame, products_df: pd.DataFrame) -> None:
    """
    Suitability scatter (client risk vs product risk) + recommendation frequency.

    Parameters
    ----------
    nba         : DataFrame da generate_recommendations()
    products_df : catalogo prodotti
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Suitability scatter ───────────────────────────────────────────────
    vis = nba[nba['Rec_Accum_ID'].notna() | nba['Rec_Income_ID'].notna()].copy()
    vis['PickedRisk'] = vis['Rec_Accum_Risk'].fillna(vis['Rec_Income_Risk'])
    vis['Head']       = np.where(vis['Rec_Accum_ID'].notna(), 'Accumulation', 'Income')

    sns.scatterplot(ax=axes[0], data=vis, x='ClientRisk', y='PickedRisk',
                    hue='Head', alpha=0.55, s=30,
                    palette={'Accumulation': 'steelblue', 'Income': 'darkorange'})
    max_r = float(max(vis['ClientRisk'].max(), vis['PickedRisk'].max()))
    axes[0].plot([0, max_r], [0, max_r], 'r--', lw=1, alpha=0.6, label='risk = risk')
    axes[0].set_title('Suitability: client risk vs recommended product risk')
    axes[0].set_xlabel('Client risk propensity')
    axes[0].set_ylabel('Recommended product risk')
    axes[0].legend()

    # ── Recommendation frequency ──────────────────────────────────────────
    freq_acc = nba['Rec_Accum_ID'].dropna().astype(int).value_counts().sort_index()
    freq_inc = nba['Rec_Income_ID'].dropna().astype(int).value_counts().sort_index()
    freq = (pd.concat([freq_acc.rename('Accumulation'),
                       freq_inc.rename('Income')], axis=1)
              .fillna(0).astype(int))
    freq.plot(kind='bar', ax=axes[1],
              color=['steelblue', 'darkorange'], width=0.8)
    axes[1].set_title('Recommendation frequency by product')
    axes[1].set_xlabel('Product ID')
    axes[1].set_ylabel('# recommendations')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def print_top_products(nba: pd.DataFrame, products_df: pd.DataFrame, k: int = 3) -> None:
    """Top-k prodotti piu' raccomandati per ciascun head."""
    freq_acc = nba['Rec_Accum_ID'].dropna().astype(int).value_counts()
    freq_inc = nba['Rec_Income_ID'].dropna().astype(int).value_counts()

    for freq, title in [(freq_acc, 'TOP ACCUMULATION'), (freq_inc, 'TOP INCOME')]:
        print(f'\n=== {title} (top {k}) ===')
        if freq.empty:
            print('  (no recommendations emitted)')
            continue
        for pid in freq.nlargest(k).index:
            risk = products_df.loc[products_df['IDProduct'] == pid, 'Risk'].iloc[0]
            print(f'  ProductID {pid:<4} | Risk {risk:.3f} | '
                  f'recommended to {int(freq[pid])} clients')
