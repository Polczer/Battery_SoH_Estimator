"""
DataFrame Processing Utilities Module

Provides helper functions for:
1. Batch processing of DataFrame dictionaries
2. Batch processing of DataFrame nested dictionaries
3. Dictionary merging operations
4. Temperature column addition
5. Processing resting periods in ocv data

All functions maintain original DataFrames and handle errors gracefully.
"""

import pandas as pd

# --------------------------
# Dictionary Processing
# --------------------------


def process_dataframes(
    data_dict: dict, processing_func: callable, **kwargs
) -> dict[str, any]:
    """
    Applies a processing function to each DataFrame in a dictionary and returns the results.

    Parameters
    ----------
    data_dict : dict
        A dictionary where keys are DataFrame names and values
        are DataFrames to be processed.
    processing_func : callable
        A function that takes a DataFrame (and optional keyword arguments) and performs
        operations.
    **kwargs : optional
        Additional keyword arguments passed directly to processing_func.

    Returns
    -------
    dict
        A dictionary with the same keys as `data_dict`, where each value is the result of
        applying processing_func.
        DataFrames that fail during processing will be replaced with None
    """
    results = {}

    for key, df in data_dict.items():
        try:
            results[key] = processing_func(df, **kwargs)
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            results[key] = None

    return results


# --------------------------
# Nested Dictionary Processing
# --------------------------


def process_nested_dicts(
    dict1: dict[str, dict[str, any]],
    dict2: dict[str, dict[str, any]],
    func: callable,
    **kwargs,
) -> dict[str, dict[str, any]]:
    """
    Applies a function to matching top-level keys of two nested dictionaries.

    Parameters
    ----------
    dict1 : dict[str, dict[str, any]]
        First nested dictionary: {key1: {subkey: value}}.
    dict2 : dict[str, dict[str, any]]
        Second nested dictionary: {key1: {subkey: value}}.
    func : callable[dict[str, any], dict[str, any]]
        Function to apply to matching sub-dictionaries.
    **kwargs : optional
        Additional arguments to pass to the function.

    Returns
    -------
    dict[str, dict[str, Any]]
        New nested dictionary with results: {key1: func(dict1[key1], dict2[key1])}.
    """
    result = {}

    for key in dict1:
        if key in dict2:
            try:
                merged = func(dict1[key], dict2[key], **kwargs)
                if merged:
                    result[key] = merged
            except Exception as e:
                print(f"Skipping {key} due to error: {e}")
        else:
            print(f"Key '{key}' not found in second dictionary")

    if not result:
        print("[Warning] No entries were successfully processed")

    return result


# --------------------------
# Dictionary Merging
# --------------------------


def merge_dicts(
    dict1: dict[str, pd.DataFrame], dict2: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Merges two dictionaries containing DataFrames by their common keys with basic length validation.

    Parameters:
        dict1 (dict[str, pd.DataFrame]): First dictionary of DataFrames.
        dict2 (dict[str, pd.DataFrame]): Second dictionary of DataFrames.

    Returns:
        dict[str, pd.DataFrame]: New dictionary with merged DataFrames for each common key.

    Notes:
        - Only keys present in both dictionaries will be included in the result.
        - DataFrames are concatenated horizontally (column-wise).
        - Indexes are reset to avoid alignment issues.
    """
    merged_dict = {}

    for key in dict1:
        if key in dict2:
            try:
                df1 = dict1[key].reset_index(drop=True)
                df2 = dict2[key].reset_index(drop=True)

                if len(df1) != len(df2):
                    raise ValueError(f"Length mismatch: {len(df1)} != {len(df2)}")

                merged_dict[key] = pd.concat([df1, df2], axis=1)
                
            except ValueError as e:
                print(f"Skipping '{key}': {e}")
            except Exception as e:
                print(f"Error merging '{key}': {str(e)}")
        else:
            print(f"Key '{key}' missing in second dictionary")

    return merged_dict


# --------------------------
# Column Manipulation
# --------------------------


def add_temp_column(data_dict: dict[str, pd.DataFrame]):
    """
    Adds a 'Temperature(°C)' column to each DataFrame in the input dictionary.
    The temperature value is extracted from the key string, e.g. 'temp_25' -> 25.

    Parameters
    ----------
    dfs_dict : dict
        Dictionary with keys like 'temp_0', 'temp_25', 'temp_45' and values as pd.DataFrames.

    Returns
    -------
    dict
        New dictionary with updated DataFrames including the 'Temperature(°C)' column.
    """
    updated_dict = {}

    for key, df in data_dict.items():

        # Extract temperature from key (supports formats like "temp_25", "test45", "25C")
        temp_value = None
        try:
            
            # Take digit in key
            temp_value = int("".join(filter(str.isdigit, key)))
        except ValueError:
            temp_value = None

        # Add column to DataFrame copy
        df_copy = df.copy()
        df_copy["Temperature(°C)"] = temp_value
        updated_dict[key] = df_copy

    return updated_dict