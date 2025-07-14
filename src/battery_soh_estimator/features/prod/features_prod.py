"""
This module defines the main feature set used for production-ready 
State-of-Health (SoH) estimation in lithium-ion batteries.

It includes physically interpretable features such as internal resistance, 
state-of-charge (SoC), and their dynamic behaviors, extracted from real-time 
voltage, current, and temperature telemetry.

These features are intended for use in machine learning models integrated 
into battery management systems (BMS).
"""

from battery_soh_estimator.features.core import common_features
from battery_soh_estimator.features.core import features_func

# Dictionary of core features for production SoH estimation.
prod_features = {

    # Unpack base features
    **common_features.base_features,

    # IR for dynamic current profiles
    "IR(Ohm)": lambda df: features_func.get_internal_resistance_dyn(
        df,
        current_col="Current(A)",
        voltage_col="Voltage(V)",
        temp_col="Temperature(°C)",
        base_temp_dyn=25.0,
        current_threshold_dyn=10,
        smoothing_window_dyn=50
    ),

    # Dynamic SOC estimation
    "SoC": lambda df: features_func.get_soc_dyn(
        df,
        voltage_col="Voltage(V)",
        v_min=3,
        v_max=4.2,
        smoothing_kernel_dyn=50
    ),

    # SOC variability over 20 samples
    "SoC_variance": lambda df: (
        features_func.get_soc_dyn(
            df,
            voltage_col="Voltage(V)",
            v_min=3,
            v_max=4.2,
            smoothing_kernel_dyn=200
        )
        .rolling(20)
        .var()
    ),

    # Absolute rate of IR change
    "IR_change_rate": lambda df: (
        features_func.get_internal_resistance_dyn(
            df,
            current_col="Current(A)",
            voltage_col="Voltage(V)",
            temp_col="Temperature(°C)",
            base_temp_dyn=25.0,
            current_threshold_dyn=10,
            smoothing_window_dyn=120
        )
        .diff()
        .abs()
        .rolling(5)
        .mean()
    ),
}