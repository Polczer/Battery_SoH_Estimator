"""
Feature Extraction Functions Module

Contains specialized functions for extracting physical battery characteristics from test data.
Handles both OCV (Open Circuit Voltage) and dynamic test profiles with appropriate methods.
"""

import numpy as np
import pandas as pd
from scipy.signal import medfilt


# --------------------------
# Main Feature Extraction
# --------------------------


def extract_features(
    df: pd.DataFrame, processing_functions: dict[str, callable]
) -> pd.DataFrame:
    """
    Applies feature extraction functions to input DataFrame, handling both Series and DataFrame outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input test data (OCV or Dynamic).
    processing_functions : dict[str, callable]
        Dictionary where each key is a desired feature name, and each value is a function.
        that takes a DataFrame and returns a Series with that feature.

    Returns
    -------
    pd.DataFrame
        Combined features with:
        - Single column for Series returns
        - Multiple columns for DataFrame returns
        - All columns properly named
        - Any feature that fails will be skipped with an error.
    """
    # Initialize containers for different return types
    series_features = {}
    dataframe_features = []

    for name, func in processing_functions.items():
        try:

            result = func(df)  # Pass kwargs

            if isinstance(result, pd.Series):
                # For Series: use the provided name as column name
                result.name = name
                series_features[name] = result

            elif isinstance(result, pd.DataFrame):
                # For DataFrames: preserve original column names
                dataframe_features.append(result)

            else:
                # Convert other types to Series
                series_features[name] = pd.Series(result, name=name)

        except Exception as e:
            print(f"Feature extraction error '{name}': {e}")

    # Combine all results
    final_df = pd.concat([pd.DataFrame(series_features)] + dataframe_features, axis=1)

    return final_df


# --------------------------
# Resistance Calculations
# --------------------------


# Calculates internal resistance for ocv tests using pulse detection
def get_internal_resistance_ocv(
    df: pd.DataFrame,
    current_col: str,
    voltage_col: str,
    temp_col: str,
    base_temp_ocv: float,
    current_threshold_ocv: float,
    window: int,
    smoothing_window_ocv: int,
) -> pd.Series:
    """
    Calculates internal resistance (IR) from pulse edges in OCV tests and applies:
    - Temperature-based correction (asymmetric model)
    - Median smoothing filter

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with current, voltage, and optionally temperature.
    current_col : str
        Column name for current data.
    voltage_col : str
        Column name for voltage data.
    temp_col : str
        Column name for temperature.
    base_temp_ocv : float
        Reference temperature in °C.
    current_threshold_ocv : float
        Minimum pulse amplitude for edge detection.
    window : int
        Number of points before/after edge to calculate ΔV/ΔI.
    smoothing_window_ocv : int
        Rolling window for median filtering (set to 1 to disable filtering).

    Returns
    -------
    pd.Series
        Time-aligned internal resistance series (Ohms), smoothed and temperature-corrected.
    """
    voltage = df[voltage_col].values
    current = df[current_col].values
    ir_series = pd.Series(index=df.index, dtype=float)
    last_ir = None

    for i in range(1, len(current)):
        # Detect rising or falling edge of current pulse
        if (abs(current[i - 1]) < current_threshold_ocv <= abs(current[i])) or (
            abs(current[i - 1]) >= current_threshold_ocv > abs(current[i])
        ):
            if i - window >= 0 and i + window < len(current):
                delta_v = voltage[i + window] - voltage[i - window]
                delta_i = current[i + window] - current[i - window]
                if abs(delta_i) > 1e-5:
                    last_ir = abs(delta_v / delta_i)
        ir_series.iloc[i] = last_ir

    # Apply temperature correction
    if temp_col in df.columns:
        if df[temp_col].nunique() == 1:
            avg_temp = df[temp_col].iloc[0]
        else:
            avg_temp = df[temp_col].mean()

        delta = avg_temp - base_temp_ocv

        # Asymmetric correction model
        if delta > 0:
            correction_factor = 1 + 0.025 * delta
        elif delta < 0:
            correction_factor = 1 + 0.0001 * abs(delta)
        else:
            correction_factor = 1.0

        ir_series *= correction_factor

    # Apply median smoothing filter
    if smoothing_window_ocv > 1:
        if smoothing_window_ocv % 2 == 0:
            smoothing_window_ocv += 1  # Ensure odd window size
        ir_series = ir_series.rolling(
            window=smoothing_window_ocv, center=True, min_periods=1
        ).median()

    return ir_series.ffill()


# Calculates internal resistance for dynamic tests using differential analysis
def get_internal_resistance_dyn(
    df: pd.DataFrame,
    current_col: str,
    voltage_col: str,
    temp_col: str,
    base_temp_dyn: float,
    current_threshold_dyn: float,
    smoothing_window_dyn: int,
) -> pd.Series:
    """
    Computes internal resistance during dynamic load tests using local dV/dI analysis,
    with optional correction for temperature effects.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series measurement data.
    current_col : str
        Column name for current values.
    voltage_col : str
        Column name for voltage values.
    temp_col : str
        Column name for temperature.
    base_temp : float
        Reference temperature for correction.
    current_threshold : float
        Threshold for detecting significant current change.
    smoothing_window : int
        Rolling window size for smoothing (median).

    Returns
    -------
    pd.Series
        Instantaneous IR values (Ohms), forward-filled and smoothed.
    """
    current = df[current_col].values
    voltage = df[voltage_col].values
    IR_series = pd.Series(index=df.index, dtype=float)

    delta_I = np.diff(current)
    delta_V = np.diff(voltage)

    # Calculate IR only at significant current change points
    for i in range(1, len(delta_I)):
        if abs(delta_I[i]) >= current_threshold_dyn:
            R = delta_V[i] / delta_I[i]
            IR_series.iloc[i + 1] = abs(R)

    # Optional median smoothing
    if smoothing_window_dyn > 1:
        if smoothing_window_dyn % 2 == 0:
            smoothing_window_dyn += 1  # Ensure odd window for centered smoothing
        IR_series = IR_series.rolling(
            window=smoothing_window_dyn, min_periods=1, center=True
        ).median()

    # Temperature correction
    if temp_col in df.columns:
        # Assume constant temperature per test
        avg_temp = (
            df[temp_col].iloc[0] if df[temp_col].nunique() == 1 else df[temp_col].mean()
        )
        delta = avg_temp - base_temp_dyn

        # Asymmetric penalty: more sensitivity to high temps
        if delta > 0:
            correction_factor = 1 + 0.01 * delta  # increase resistance at high temp
        elif delta < 0:
            correction_factor = 1 + 0.015 * abs(delta)  # moderate increase at low temp
        else:
            correction_factor = 1.0

        IR_series *= correction_factor

    return IR_series.ffill()


# --------------------------
# State of Charge
# --------------------------


# Calculates SoC by cumulative capacity
def get_soc_ocv(
    df: pd.DataFrame,
    initial_soc: float,  # Initial SOC (1.0 = fully charged for ocv tests)
) -> pd.Series:
    """
    Calculates SOC from cumulative Charge/Discharge Capacity data.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'Charge_Capacity(Ah)' and 'Discharge_Capacity(Ah)' columns.
    initial_soc : float
        Initial SOC value before the test (0-1 range).

    Returns:
    --------
    pd.Series
        SOC values (ranging from 0 to 1).
    """
    # Calculate changes in charge/discharge capacities
    delta_charge = df["Charge_Capacity(Ah)"].diff().fillna(0)
    delta_discharge = df["Discharge_Capacity(Ah)"].diff().fillna(0)

    # Use maximum discharge capacity as nominal capacity
    nominal_capacity = df["Discharge_Capacity(Ah)"].max()

    # SOC calculation: initial + (charge - discharge) / nominal capacity
    soc = initial_soc + (delta_charge - delta_discharge).cumsum() / nominal_capacity

    # Upper limit only
    soc = np.minimum(soc, 1)

    return soc


# Сalculates SoC by voltage approximation with filtering
def get_soc_dyn(
    df: pd.DataFrame,
    voltage_col: str,
    v_min: float,  # For each Dynamic test
    v_max: float,  # For each Dynamic test
    smoothing_kernel_dyn: int,  # Selected according to plots
) -> pd.Series:
    """
    Estimate SoC using only voltage, assuming a linear relationship between voltage and SoC.
    Applies median filtering to suppress outliers and noise.

    Parameters:
        df (pd.DataFrame): Input dataframe with voltage data.
        voltage_col (str): Name of the voltage column.
        v_min (float): Voltage corresponding to 0% SoC.
        v_max (float): Voltage corresponding to 100% SoC.
        smoothing_kernel (int): Kernel size for median filtering (must be odd).

    Returns:
        pd.Series
            SOC values (ranging from 0 to 1).
    """
    # Normalize voltage to SoC range [0, 1]
    soc = (df[voltage_col] - v_min) / (v_max - v_min)
    soc = soc.clip(0, 1)

    # Ensure kernel size is odd and > 1
    if smoothing_kernel_dyn < 3:
        smoothing_kernel_dyn = 3
    if smoothing_kernel_dyn % 2 == 0:
        smoothing_kernel_dyn += 1

    # Apply median filter to smooth spikes/outliers
    soc_smoothed = medfilt(soc, kernel_size=smoothing_kernel_dyn)

    return pd.Series(soc_smoothed, index=df.index)


# --------------------------
# Temperature Features
# --------------------------


def add_temp_features(
    df: pd.DataFrame,
    temp_col: str = "Temperature(°C)",
    feature: str = None
) -> pd.DataFrame | pd.Series:
    """
    Generates temperature-related features for SoH estimation.
    Returns either a single Series (if feature specified) or full DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a temperature column.
    temp_col : str, optional
        Name of the temperature column. Default is 'Temperature(°C)'.
    feature : str, optional
        Specific feature to return as Series. Options:
        - 'weight_0': Proximity weight to 0°C
        - 'weight_25': Proximity weight to 25°C
        - 'weight_45': Proximity weight to 45°C
        - 'nearest_temp': Categorical reference temperature
        - 'is_extreme_low': Extreme cold flag
        - 'is_extreme_high': Extreme heat flag
        If None, returns all features as DataFrame.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Temperature features. Single Series if `feature` specified, else DataFrame.

    Raises
    ------
    ValueError
        If specified feature doesn't exist.
    """
    df = df.copy()

    # Handle missing values
    if df[temp_col].isnull().any():
        print(f"Warning: {df[temp_col].isnull().sum()} NaN values filled with 25°C.")
        df[temp_col] = df[temp_col].fillna(25)

    t = df[temp_col]
    eps = 1e-6

    # Calculate all features
    features = {
        # Proximity weights
        "weight_0": 1 / (np.abs(t - 0) + eps),
        "weight_25": 1 / (np.abs(t - 25) + eps),
        "weight_45": 1 / (np.abs(t - 45) + eps),

        # Categorical
        "nearest_temp": np.select(
            [(t <= 12.5), (t > 12.5) & (t <= 35)],
            [0, 25],
            default=45
        ),

        # Extreme flags
        "is_extreme_low": (t < 0).astype(int),
        "is_extreme_high": (t > 45).astype(int)
    }

    # Normalize weights
    total_weight = features["weight_0"] + features["weight_25"] + features["weight_45"]
    for w in ["weight_0", "weight_25", "weight_45"]:
        features[w] /= total_weight

    # Return requested output format
    if feature is not None:
        if feature not in features:
            raise ValueError(
                f"Invalid feature '{feature}'. Available: {list(features.keys())}"
            )
        return features[feature]

    return pd.DataFrame(features, index=df.index)