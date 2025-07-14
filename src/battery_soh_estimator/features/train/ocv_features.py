"""
ocv_features.py

This module defines feature extraction logic specific to OCV (Open Circuit Voltage) tests.
These features are used primarily during model training and battery characterization
under low current or rest conditions.

OCV-based features often require clean, low-noise segments for accurate estimation,
and are typically not used during real-time deployment.
"""

from battery_soh_estimator.features.core import features_func

# Dictionary of feature extraction functions for OCV (Open Circuit Voltage) tests
# Keys represent feature names, values are functions that transform input DataFrame
ocv_features = {

    # Internal resistance from OCV pulses
    "IR(Ohm)": lambda df: features_func.get_internal_resistance_ocv(
        df,
        current_col="Current(A)",
        voltage_col="Voltage(V)",
        temp_col="Temperature(°C)",
        base_temp_ocv=30.0,
        current_threshold_ocv=0.01,
        window=1,
        smoothing_window_ocv=30000
    ),

    # State of Charge estimation
    "SoC": lambda df: features_func.get_soc_ocv(
        df,
        initial_soc=1.0
    ),

    # SOC variability over 20 samples
    "SoC_variance": lambda df: (
        features_func.get_soc_ocv(df, initial_soc=1.0)
        .rolling(20)
        .var()
    ),

    # Absolute rate of IR change
    "IR_change_rate": lambda df: (
        features_func.get_internal_resistance_ocv(
            df,
            current_col="Current(A)",
            voltage_col="Voltage(V)",
            temp_col="Temperature(°C)",
            base_temp_ocv=30.0,
            current_threshold_ocv=0.01,
            window=1,
            smoothing_window_ocv=30000
        )
        .diff()
        .abs()
        .rolling(5)
        .mean()
    )
}