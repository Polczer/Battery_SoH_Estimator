"""
common_features.py

This module defines basic and reusable feature transformations that are shared
between both OCV (Open Circuit Voltage) and dynamic test scenarios.

These features are general-purpose, based on direct voltage and current measurements,
and are commonly applicable in both offline training and online inference stages.

Each entry in the `BASE_FEATURES` dictionary maps a feature name to a function
that accepts a pandas DataFrame and returns a transformed Series.
"""

from .features_func import add_temp_features

base_features = {
    # Transfer columns as features
    "Current(A)": lambda df: df["Current(A)"].abs(),  # Absolute current values

    # New features
    "dV/dt(V/s)": lambda df: (
        df["Voltage(V)"]
        .diff()
        .rolling(5)
        .mean()
    ),

    "dI/dt(A/s)": lambda df: (
        df["Current(A)"]
        .abs()
        .diff()
        .rolling(5)
        .mean()  # Smoothed current change rate
    ),

    "power_avg(W)": lambda df: (
        df["Voltage(V)"] * df["Current(A)"].abs()
        .rolling(10)
        .mean()  # 10-point moving average of power
    ),

    "Voltage_Current_cov": lambda df: (
        df["Voltage(V)"]
        .rolling(20)
        .cov(df["Current(A)"])  # Rolling covariance between V and I
    ),

    "energy_wh": lambda df: (
        (df["Voltage(V)"] * df["Current(A)"].abs() * df["Step_Time(s)"].diff().fillna(0))
        .cumsum() / 3600  # Cumulative energy in Watt-hours
    ),

    "is_discharging": lambda df: (df["Current(A)"] > 0).astype(int),  # Discharge flag (1=discharge)

    # Temperature features
    "weight_0": lambda df: add_temp_features(df, feature="weight_0"),
    "weight_25": lambda df: add_temp_features(df, feature="weight_25"),
    "weight_45": lambda df: add_temp_features(df, feature="weight_45"),
    "nearest_temp": lambda df: add_temp_features(df, feature="nearest_temp"),
}