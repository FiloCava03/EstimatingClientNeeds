"""
data_loading.py
===============
Utilities for loading and initial preprocessing of the Needs dataset.

Handles:
  - Reading the three Excel sheets (Needs, Products, Metadata)
  - Stripping column-name whitespace
  - Dropping the uninformative ID column
"""

import numpy as np
import pandas as pd


def load_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three sheets from the project Excel file.

    Parameters
    ----------
    file_path : str
        Path to Dataset2_Needs.xls (or .xlsx).

    Returns
    -------
    needs_df : pd.DataFrame
        Client-level data with targets IncomeInvestment and AccumulationInvestment.
    products_df : pd.DataFrame
        Product catalogue (IDProduct, Type, Risk, …).
    metadata_df : pd.DataFrame
        Variable-level metadata.
    """
    # Read all three sheets in a single I/O pass
    needs_df    = pd.read_excel(file_path, sheet_name='Needs')
    products_df = pd.read_excel(file_path, sheet_name='Products')
    metadata_df = pd.read_excel(file_path, sheet_name='Metadata')

    # Strip trailing/leading whitespace that Excel sometimes introduces in headers
    needs_df.columns    = needs_df.columns.str.strip()
    products_df.columns = products_df.columns.str.strip()

    # The ID column carries no predictive signal — drop it immediately
    needs_df = needs_df.drop(columns=['ID'])

    return needs_df, products_df, metadata_df
