"""
Battery State of Health (SoH) Estimation Module

Provides multiple methods for calculating battery health:
1. Capacity-based SoH from OCV tests
2. Dynamic capacity-based SoH
3. Internal resistance-based SoH
4. Combined capacity/resistance averaging

All calculation functions return pd.Series for easy integration with pandas workflows.
"""

import numpy as np
import pandas as pd


# --------------------------
# Capacity-Based SoH Methods
# --------------------------


# Calculates SoH from OCV test data using the ratio of max discharge capacity to nominal capacity
def calculate_soh_cap_ocv(
    df: pd.DataFrame,
    nominal_capacity: float,
    column: str = "Discharge_Capacity(Ah)",
    temp_column: str = "Temperature(°C)",
    ref_temp: float = 22.0,
) -> pd.Series:
    """
    Calculates SoH as the ratio of max discharge capacity to nominal capacity,
    with optional correction for temperature deviations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with battery test data.
    nominal_capacity : float
        Nominal capacity in Ah.
    column : str
        Column name with discharge capacity (default "Discharge_Capacity(Ah)").
    temp_column : str
        Column name with temperature values (default "Temperature(°C)").
    ref_temp : float
        Reference temperature (default 25°C).

    Returns
    -------
    pd.Series
        SoH with temperature correction applied if temperature column is present.

    Raises
    ------
    ValueError
        If nominal capacity is non-positive.
    """
    if nominal_capacity <= 0:
        raise ValueError("nominal_capacity must be positive")

    max_discharge_capacity = df[column].max()
    soh_value = max_discharge_capacity / nominal_capacity

    if temp_column in df.columns:
        avg_temp = df[temp_column].iloc[0] if df[temp_column].nunique() == 1 else df[temp_column].mean()
        delta = avg_temp - ref_temp

        # Asymmetric penalty: higher penalty for overheating
        if delta > 0:
            penalty = max(0.0, 1 - 0.005 * delta)
        elif delta < 0:
            penalty = max(0.0, 1 - 0.0001 * abs(delta))
        else:
            penalty = 1.0

        soh_value *= penalty

    return pd.Series(soh_value, index=df.index, name="SoH_by_Cap")


# Calculates SoH from dynamic test data taking into account the initial charge
def calculate_soh_cap_dyn(
    df: pd.DataFrame,
    nominal_capacity: float,
    step_index_value: int,
    step_index_col: str = "Step_Index",
    charge_capacity_col: str = "Charge_Capacity(Ah)",
    discharge_capacity_col: str = "Discharge_Capacity(Ah)",
    temp_column: str = "Temperature(°C)",
    ref_temp: float = 22.0,
) -> pd.Series:
    """
    Calculates SoH using a multi-step capacity difference approach 
    with optional correction for temperature deviations.

    The calculation performs:
    1. Finds max charge capacity (Q_ch_max)
    2. Finds max charge capacity per step (Q_ch_step_max)
    3. Calculates Q_diff = Q_ch_max - Q_ch_step_max
    4. Final capacity = max discharge capacity (Q_dis_max) - Q_diff
    5. SoH = Final capacity / nominal capacity

    df : pd.DataFrame
        Input dataframe containing battery cycling test data. Must include:
        - Step index column
        - Charge capacity measurements
        - Discharge capacity measurements
    nominal_capacity : float
        Battery's nominal capacity in ampere-hours (Ah).
    step_index_value : int
        Specific step number to use for Q_ch_step_max calculation.
    step_index_col : str, optional
        Column name containing step indices (default: "Step_Index")
    charge_capacity_col : str, optional
        Column name for charge capacity measurements (default: "Charge_Capacity(Ah)")
    discharge_capacity_col : str, optional
        Column name for discharge capacity measurements (default: "Discharge_Capacity(Ah)")
    temp_column : str
        Column name with temperature values (default "Temperature(°C)").
    ref_temp : float
        Reference temperature (default 25°C).

    Returns
    -------
    pd.Series
        SoH with temperature correction applied if temperature column is present.

    Raises
    ------
    ValueError
        If nominal capacity is non-positive
    """
    if nominal_capacity <= 0:
        raise ValueError("Nominal capacity must be positive")

    # Step 1: Get maximum charge capacity
    q_ch_max = df[charge_capacity_col].max()

    # Step 2: Get maximum charge capacity per step
    q_ch_step_max = df.loc[
        df[step_index_col] == step_index_value, charge_capacity_col
    ].max()

    # Step 3: Calculate capacity difference
    q_diff = q_ch_max - q_ch_step_max

    # Step 4: Calculate final available capacity
    q_dis_max = df[discharge_capacity_col].max()
    final_capacity = q_dis_max - q_diff

    # Step 5: Calculate SoH
    soh_value = final_capacity / nominal_capacity

    if temp_column in df.columns:
        avg_temp = df[temp_column].iloc[0] if df[temp_column].nunique() == 1 else df[temp_column].mean()
        delta = avg_temp - ref_temp

        # Asymmetric penalty: higher penalty for overheating
        if delta > 0:
            penalty = max(0.0, 1 - 0.005 * delta)
        elif delta < 0:
            penalty = max(0.0, 1 - 0.0001 * abs(delta))
        else:
            penalty = 1.0

        soh_value *= penalty

    return pd.Series(soh_value, index=df.index, name="SoH_by_Cap")


# ----------------------------
# Resistance-Based SoH Methods
# ----------------------------


def calculate_soh_ir(
    df: pd.DataFrame,
    ir_nominal: float,
    resistance_col: str = "IR(Ohm)",
    alpha: float = 0.5,
) -> pd.Series:
    """
    Estimates battery SoH based on an exponential model of internal resistance increase.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing resistance measurements
    r_nominal : float
        Nominal internal resistance of a new cell (Ohms).
    alpha : float, optional
        Empirical degradation rate coefficient (default: 0.5)
    resistance_col : str, optional
        Column name with resistance data (default: 'IR(Ohm)')

    Returns
    -------
    pd.Series
        Estimated SoH.
    """
    delta_r = df[resistance_col] - ir_nominal
    soh = np.exp(-alpha * delta_r.clip(lower=0))

    return pd.Series(soh, index=df.index, name="SoH_by_IR")


# -------------------
# Combined SoH Methods
# -------------------


def average_soh(
    cap_dict: dict[str, pd.Series],
    ir_dict: dict[str, pd.Series],
    weight_cap: float = 1.4,
    weight_ir: float = 0.6,
) -> dict[str, pd.Series]:
    """
    Calculates the weighted average State of Health (SoH) by combining capacity 
    and internal resistance measurements at each temperature point.

    Parameters:
    -----------
    cap_dict : dict[str, pd.Series]
        Dictionary where keys are temperature points and values are pd.Series
        containing capacity-based SoH measurements.
    ir_dict : dict[str, pd.Series]
        Dictionary where keys are temperature points and values are pd.Series
        containing internal resistance-based SoH measurements.
    weight_cap : float, optional
        Weight for capacity measurements (default: 1.0).
    weight_ir : float, optional
        Weight for internal resistance measurements (default: 0.8).

    Returns:
    --------
    dict[str, pd.Series]
        Dictionary with temperature keys, where values are pd.Series containing
        the weighted average SoH values.

    Raises:
    -------
    ValueError
        If:
        - The length of capacity and internal resistance Series don't match
        - Temperature keys in cap_dict and ir_dict don't match
        - Weights are negative
    """
    # Validate weights
    if weight_cap < 0 or weight_ir < 0:
        raise ValueError("Weights must be non-negative")
    
    # Check if temperature points match
    if set(cap_dict.keys()) != set(ir_dict.keys()):
        raise ValueError("Temperature points in cap_dict and ir_dict don't match")
    
    result = {}
    for temp in cap_dict:
        cap_series = cap_dict[temp]
        ir_series = ir_dict[temp]
        
        # Validate series length
        if len(cap_series) != len(ir_series):
            raise ValueError(f"Length mismatch for temperature {temp}: "
                           f"capacity has {len(cap_series)} points, "
                           f"resistance has {len(ir_series)} points")
        
        # Calculate weighted average
        total_weight = weight_cap + weight_ir
        avg_soh = (weight_cap * cap_series + weight_ir * ir_series) / total_weight
        avg_soh.name = "SoH"
        result[temp] = avg_soh
    
    return result