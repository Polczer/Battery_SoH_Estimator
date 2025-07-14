"""
Data Loading Module

Provides functionality to:
1. Load and concatenate .xlsx files from a directory
2. Cache loaded data for performance
3. Handle custom file-to-key mappings

Features:
- Automatic sheet concatenation
- Persistent caching
- Custom key mapping
"""

import os

import joblib
import pandas as pd


def load_data_to_dict(
    dir_path: str,
    cache_dir: str = os.path.join("..", "data", "cache"),
    cache_name: str = "loaded_files_cache.pkl",
    **file_keys: str,
) -> dict[str, pd.DataFrame]:
    """
    Loads .xlsx files from the specified directory, concatenates all sheets
    into a single DataFrame per file, and caches results to avoid reloading.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing .xlsx files.
    cache_dir : str
        Directory where the cache file will be stored (default: ../data/cache).
    cache_name : str
        Cache file name.
    **file_keys : str
        Mapping of custom keys to filenames (e.g., train="train_data.xlsx").

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys as specified and values as DataFrames.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    ValueError
        If the directory is empty or no valid .xlsx files were loaded.

    Example:
    -------
    >>> data = load_data_to_dict("data/", train="train_data.xlsx")
    >>> df = data["train"]
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # Load cache if it exists
    if os.path.exists(cache_path):
        loaded_files_cache = joblib.load(cache_path)
    else:
        loaded_files_cache = {}

    data = {}

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' not found!")

    xlsx_files = [f for f in os.listdir(dir_path) if f.endswith(".xlsx")]
    if not xlsx_files:
        raise ValueError(f"No .xlsx files found in '{dir_path}'!")

    for key, filename in file_keys.items():
        if filename in loaded_files_cache:
            print(f"[Skip] '{filename}' already loaded as '{key}'")
            data[key] = loaded_files_cache[filename]
            continue

        full_path = os.path.join(dir_path, filename)
        try:
            xls = pd.ExcelFile(full_path)
            dfs = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
            combined_df = pd.concat(dfs, ignore_index=True)
            data[key] = combined_df
            loaded_files_cache[filename] = combined_df
            print(f"Loaded and combined '{filename}' as '{key}'")
        except Exception as e:
            print(f"Error loading file '{filename}': {str(e)}")

    joblib.dump(loaded_files_cache, cache_path)  # Save updated cache

    if not data:
        raise ValueError("No valid .xlsx files were loaded!")

    return data