import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_variable_summary(df, metadata_df):
    # Create empty lists to store the chosen statistics
    stats_dict = {
        "Variable": [],
        "Description": [],
        "Mean": [],
        "Std": [],
        "Missing": [],
        "Min": [],
        "Max": [],
    }

    # Create a metadata dictionary for easy lookup
    meta_dict = dict(zip(metadata_df["Metadata"], metadata_df["Unnamed: 1"]))

    for col in df.columns:
        stats_dict["Variable"].append(col)
        stats_dict["Description"].append(meta_dict.get(col, "N/A"))

        # Calculate some statistics for each column
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict["Mean"].append(f"{df[col].mean():.2f}")
            stats_dict["Std"].append(f"{df[col].std():.2f}")
            stats_dict["Min"].append(f"{df[col].min():.2f}")
            stats_dict["Max"].append(f"{df[col].max():.2f}")
        else:
            stats_dict["Mean"].append("N/A")
            stats_dict["Std"].append("N/A")
            stats_dict["Min"].append("N/A")
            stats_dict["Max"].append("N/A")

        stats_dict["Missing"].append(df[col].isna().sum())

    return pd.DataFrame(stats_dict)


def plot_histogram(data, title, xlabel, ylabel="Frequency"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(title, pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_target_balance(df, target_cols, figsize=(12, 4)):
    """
    Side-by-side bar chart showing the class balance of each target variable.
    Useful as a first sanity check before modelling: it immediately shows
    whether the problem is balanced, moderately imbalanced, or severely
    imbalanced — information that directly drives the choice of metric
    (accuracy vs PR-AUC) and of class_weight='balanced' in the classifier.
    """
    fig, axes = plt.subplots(1, len(target_cols), figsize=figsize)
    # Normalize to a 1-element iterable when there's only one target, so
    # the loop below works in both cases without special-casing
    if len(target_cols) == 1:
        axes = [axes]

    for ax, target in zip(axes, target_cols):
        counts = df[target].value_counts().sort_index()
        prevalence = df[target].mean()
        # Bar chart + the exact prevalence in the title, so the reader
        # doesn't need to read the y-axis ticks to know the imbalance
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="muted")
        ax.set_title(f"{target}\n(positive rate = {prevalence:.1%})")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df, features, n_cols=3, figsize=(15, 10)):
    """
    Grid of histograms with KDE overlay for a list of numeric features.
    Meant for the EDA step where you want to see all relevant variables
    at once, spotting skewness, multimodality or suspicious spikes.

    Automatically chooses the number of rows from n_cols so the caller
    doesn't have to compute the grid dimensions by hand.
    """
    # Ceiling division: enough rows to fit every feature in the grid
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten for easy indexing even when the grid is 1D or 2D
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(feature)
        axes[i].set_xlabel("")

    # Hide leftover empty subplots when len(features) is not a perfect
    # multiple of n_cols — otherwise empty axes show up as white boxes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(df, features, target, n_cols=3, figsize=(15, 10)):
    """
    Grid of overlaid KDE plots showing how each feature's distribution
    shifts between the two classes of a binary target.
    A large visual gap between the two curves hints at a feature that
    carries signal for the classifier; nearly identical curves hint at
    a feature that will be ignored.

    This is a "which features might matter?" plot — not a formal test.
    """
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, feature in enumerate(features):
        # One KDE per class, filled with transparency so overlap is visible.
        # sort_index() keeps class 0 before class 1 for consistent coloring
        for val in sorted(df[target].unique()):
            subset = df[df[target] == val][feature]
            sns.kdeplot(
                subset, label=f"{target}={val}", ax=axes[i], fill=True, alpha=0.3
            )
        axes[i].set_title(feature)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Feature distributions split by {target}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, columns=None, figsize=(10, 8), annot=True):
    """
    Correlation heatmap of numeric features, centred at 0 so positive and
    negative correlations are visually distinguishable. Defaults to all
    numeric columns when `columns` is None.

    Useful to spot redundancy in engineered features (e.g. Wealth_log vs
    WealthPerWorkYear_log) before feeding them to the model. Note that for
    tree-based models redundancy is not a correctness issue — the RF will
    just pick one and ignore the other — but it can dilute feature
    importance across correlated variables, making the importance plot
    harder to interpret.
    """
    # Default: every numeric column in the dataframe
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[columns].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,  # show the numeric value in each cell
        cmap="coolwarm",
        center=0,  # so 0-correlation is visually neutral (white)
        fmt=".2f",  # two decimals is plenty for visual scanning
        vmin=-1,
        vmax=1,  # fixed scale, for comparability across runs
        square=True,
    )
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()


def plot_missing_values(df, figsize=(10, 5)):
    """
    Horizontal bar chart of missingness per column, showing only the
    columns that actually have at least one missing value. If no column
    has missing values, it prints a short message instead of showing an
    empty chart — this is the common case on already-cleaned datasets.
    """
    # Count NaNs per column, then keep only columns where count > 0
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        print("No missing values in the dataset.")
        return

    plt.figure(figsize=figsize)
    sns.barplot(x=missing.values, y=missing.index, color="tomato")
    plt.title("Missing values per column")
    plt.xlabel("Number of missing values")
    plt.tight_layout()
    plt.show()
