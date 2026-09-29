"""
data_loader.py — Load and inspect phishing detection datasets.

This module handles:
    1. Loading CSV files from disk or user-provided paths.
    2. Inspecting the schema: column names, types, missing values, duplicates.
    3. Verifying the target column exists and reporting label distribution.

Usage:
    from src.data_loader import load_primary_dataset, inspect_dataset
    df = load_primary_dataset()
    inspect_dataset(df)
"""

import os
import pandas as pd
import numpy as np

# Import project-level settings
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset file not found at: {filepath}\n"
            f"Please download the dataset and place it at this location.\n"
            f"See README.md for download instructions."
        )

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse CSV file: {filepath}\nError: {e}")

    if df.shape[0] == 0:
        raise ValueError(f"The dataset has zero rows: {filepath}")

    return df


def load_primary_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the primary Hannousse & Yahiouche (2021) phishing dataset.

    The dataset is expected to have:
        - ~11,430 rows
        - 87 features + 1 target column named "status"
        - Target values: "legitimate" and "phishing"

    Parameters
    ----------
    filepath : str, optional
        Override path. Defaults to config.PRIMARY_DATASET_PATH.

    Returns
    -------
    pd.DataFrame
        The loaded primary dataset.
    """
    path = filepath or config.PRIMARY_DATASET_PATH
    print(f"[data_loader] Loading primary dataset from: {path}")
    df = load_csv(path)
    print(f"[data_loader] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


def inspect_dataset(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Inspect a dataset and print a summary report.

    This function checks:
        - Number of rows and columns
        - Column names and data types
        - Missing values per column
        - Duplicate rows
        - Target column label distribution (if provided)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    target_col : str, optional
        Name of the target/label column. If None, uses config default.

    Returns
    -------
    dict
        A dictionary containing inspection results.
    """
    target_col = target_col or config.PRIMARY_TARGET_COLUMN

    info = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column_found": target_col in df.columns,
    }

    print("\n" + "=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Shape:           {info['n_rows']} rows × {info['n_cols']} columns")
    print(f"Duplicate rows:  {info['duplicate_rows']}")
    print(f"Total missing:   {info['total_missing']}")

    # Show columns with missing values
    missing = {k: v for k, v in info["missing_values"].items() if v > 0}
    if missing:
        print("\nColumns with missing values:")
        for col, count in sorted(missing.items(), key=lambda x: -x[1]):
            pct = 100 * count / info["n_rows"]
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("\nNo missing values found.")

    # Target column analysis
    if info["target_column_found"]:
        labels = df[target_col].value_counts()
        info["label_distribution"] = labels.to_dict()
        print(f"\nTarget column '{target_col}' — label distribution:")
        for label, count in labels.items():
            pct = 100 * count / info["n_rows"]
            print(f"  {label}: {count} ({pct:.1f}%)")
    else:
        print(f"\n⚠ Target column '{target_col}' NOT found in dataset.")
        print(f"  Available columns: {info['columns'][:10]}...")
        info["label_distribution"] = {}

    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    print(f"\nData types:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    print("=" * 60)
    return info


def verify_primary_dataset(df: pd.DataFrame) -> dict:
    """
    Verify the primary dataset matches the expected schema.

    Checks that the target column exists and has expected label values.
    Does NOT assert exact row/column counts — those are verified against
    the actual data, not hardcoded expectations.

    Returns
    -------
    dict
        Verification results with warnings if any issues found.
    """
    result = {"valid": True, "warnings": []}

    # Check target column
    if config.PRIMARY_TARGET_COLUMN not in df.columns:
        result["valid"] = False
        result["warnings"].append(
            f"Target column '{config.PRIMARY_TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        return result

    # Check label values
    unique_labels = set(df[config.PRIMARY_TARGET_COLUMN].unique())
    expected_labels = set(config.PRIMARY_LABEL_MAPPING.keys())

    if not expected_labels.issubset(unique_labels):
        result["warnings"].append(
            f"Expected labels {expected_labels}, found {unique_labels}. "
            f"The label encoding in config.py may need updating."
        )

    unexpected = unique_labels - expected_labels
    if unexpected:
        result["warnings"].append(
            f"Unexpected label values found: {unexpected}"
        )

    # Report row count (information, not assertion)
    print(f"[verify] Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"[verify] ⚠ {w}")
    else:
        print("[verify] ✓ Dataset schema looks correct.")

    return result
