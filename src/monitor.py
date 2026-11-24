"""
Monitoring and Retraining module for MP141.
Computes Population Stability Index (PSI), group-level performance metrics,
and evaluates model promotion criteria.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Compute Population Stability Index (PSI) between reference (expected)
    and target (actual) distributions using 10 quantile-based bins.
    
    Formula: PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
    """
    # Clean nulls
    exp = expected.dropna().values
    act = actual.dropna().values

    if len(exp) == 0 or len(act) == 0:
        return 0.0

    # Determine quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp, percentiles)

    # Ensure strictly increasing edges to prevent duplicate bin errors
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    # Bin frequencies
    exp_counts, _ = np.histogram(exp, bins=bin_edges)
    act_counts, _ = np.histogram(act, bins=bin_edges)

    # Proportions with smoothing epsilon to avoid division by zero
    exp_pct = np.maximum(exp_counts / len(exp), epsilon)
    act_pct = np.maximum(act_counts / len(act), epsilon)

    # Normalize back to sum=1
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    # Calculate PSI
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_val, 4))


def compute_features_psi(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_columns: List[str]
) -> Tuple[Dict[str, float], str]:
    """
    Compute PSI for each numeric feature and return all PSI values
    along with the name of the feature that drifted the most.
    """
    psi_dict: Dict[str, float] = {}

    for col in numeric_columns:
        if col in train_df.columns and col in later_df.columns:
            psi_dict[col] = calculate_psi(train_df[col], later_df[col], num_bins=10)

    most_drifted = max(psi_dict, key=psi_dict.get) if psi_dict else ""
    return psi_dict, most_drifted


def compute_mae_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred: np.ndarray
) -> Tuple[Dict[str, float], str]:
    """
    Compute MAE separately for each group in group_col (e.g., locality)
    and return the worst-served group.
    """
    mae_by_group: Dict[str, float] = {}
    temp_df = df.copy()
    temp_df["_abs_err"] = np.abs(temp_df[y_true_col].values - y_pred)

    for group_name, group_data in temp_df.groupby(group_col):
        mae = float(group_data["_abs_err"].mean())
        mae_by_group[str(group_name)] = round(mae, 2)

    worst_group = max(mae_by_group, key=mae_by_group.get) if mae_by_group else ""
    return mae_by_group, worst_group


def evaluate_promotion(
    mae_v1: float,
    mae_v2: float,
    threshold: float = 0.10
) -> Tuple[str, float]:
    """
    Client's promotion rule: Promote v2 only if its MAE on the check dataset
    is at least 10% lower than v1's.
    
    Returns: ('promote' or 'keep', improvement_percentage)
    """
    if mae_v1 <= 0:
        return "keep", 0.0

    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= threshold else "keep"
    return decision, float(round(improvement, 4))
