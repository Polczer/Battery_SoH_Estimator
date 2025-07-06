"""
dyn_features.py

This module defines feature extraction logic for dynamic battery test profiles.

Dynamic features are robust to changing load profiles and support fast, approximate
assessment of battery health and performance.
"""

from battery_soh_estimator.features.core import features_func

# Dictionary of feature extraction functions for dynamic battery tests
# Includes runtime-compatible features and dynamic-specific estimates
dyn_features = {

    # IR for dynamic current profiles
    "IR(Ohm)": lambda df: features_func.get_internal_resistance_dyn(
        df,
        current_col="Current(A)",
        voltage_col="Voltage(V)",
        temp_col="Temperature(°C)",
        base_temp_dyn=25.0,
        current_threshold_dyn=0.4,
        smoothing_window_dyn=5000
    ),

    # Dynamic SOC estimation
    "SoC": lambda df: features_func.get_soc_dyn(
        df,
        voltage_col="Voltage(V)",
        v_min=2.5,
        v_max=4.2,
        smoothing_kernel_dyn=500
    ),

    # SOC variability over 20 samples
    "SoC_variance": lambda df: (
        features_func.get_soc_dyn(
            df,
            voltage_col="Voltage(V)",
            v_min=2.5,
            v_max=4.2,
            smoothing_kernel_dyn=500
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
            current_threshold_dyn=0.4,
            smoothing_window_dyn=5000
        )
        .diff()
        .abs()
        .rolling(5)
        .mean()
    ),
}
