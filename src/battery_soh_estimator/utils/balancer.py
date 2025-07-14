import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def balance_data(
    df,
    n_synthetic=6000,
    target_col='SoH',
    low_thresh=0.92,
    high_thresh=0.95,
    diversity_strength=0.5
):
    """
    Generate synthetic data with maximum diversity
    
    Parameters:
        df: Source DataFrame
        n_synthetic: Number of synthetic samples
        target_col: Target variable
        low_thresh, high_thresh: Target range boundaries
        diversity_strength: Diversity effect strength (0-1)
    """
    # Extract numeric features and target range
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target variable '{target_col}' not found in numeric columns")

    feature_cols = [col for col in numeric_cols if col != target_col]
    target_band = df[(df[target_col] > low_thresh) & (df[target_col] <= high_thresh)].copy()

    if len(target_band) < 10:
        raise ValueError("Insufficient source data in target range")

    # Normalization
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(target_band[feature_cols]),
        columns=feature_cols,
        index=target_band.index
    )

    # Anomaly-aware weighting
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(scaled_df)
    target_band['anomaly_score'] = anomaly_labels
    weights = np.where(anomaly_labels == -1, 1 + diversity_strength, 1 - 0.5 * diversity_strength)

    # Nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=5)
    nbrs.fit(scaled_df)

    synthetic_data = []

    for _ in range(n_synthetic):
        idx = np.random.choice(target_band.index, p=weights / weights.sum())
        base_sample_df = target_band.loc[[idx], feature_cols]
        base_scaled = pd.DataFrame(
            scaler.transform(base_sample_df),
            columns=feature_cols
        )

        # Find neighbors
        distances, indices = nbrs.kneighbors(base_scaled)

        distance_weights = distances[0] ** diversity_strength
        if distance_weights.sum() == 0:
            distance_weights = np.ones_like(distance_weights)

        neighbor_idx = np.random.choice(indices[0], p=distance_weights / distance_weights.sum())
        neighbor_sample = target_band.loc[scaled_df.index[neighbor_idx], feature_cols].values

        # Hybridization
        base_sample = base_sample_df.values.flatten()
        alpha = np.random.beta(1, 1 + diversity_strength * 3)
        synthetic_sample = (1 - alpha) * base_sample + alpha * neighbor_sample

        # Add noise
        noise_std = target_band[feature_cols].std().values * (0.1 + 0.2 * diversity_strength)
        synthetic_sample += np.random.normal(0, noise_std)

        # Create new row
        new_row = dict(zip(feature_cols, synthetic_sample))

        # Generate SoH based on neighbors
        soh_neighbors = target_band.loc[scaled_df.index[indices[0]], target_col].values
        new_soh = np.random.choice(soh_neighbors) * np.random.normal(1.0, 0.01)
        new_row[target_col] = np.clip(new_soh, low_thresh, high_thresh)

        synthetic_data.append(new_row)

    df_synthetic = pd.DataFrame(synthetic_data)

    return pd.concat([df, df_synthetic], ignore_index=True)