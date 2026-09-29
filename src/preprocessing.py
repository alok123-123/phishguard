"""
preprocessing.py — Data cleaning, transformation, and splitting.

This module handles:
    1. Encoding the target column from string labels to integers.
    2. Identifying numeric vs. categorical features.
    3. Handling missing values (imputation fitted on training data only).
    4. Scaling features when needed (e.g., for SVM).
    5. Splitting data into train / validation / test sets.
    6. Saving and loading the preprocessing pipeline for reproducibility.
    7. Detecting and removing duplicate rows.

IMPORTANT: All learned transformations (imputer means, scaler parameters)
are fitted on the TRAINING set only, then applied to validation and test sets.
This prevents data leakage.

Usage:
    from src.preprocessing import preprocess_dataset, load_pipeline
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def encode_target(df: pd.DataFrame, target_col: str = None,
                  label_mapping: dict = None) -> tuple:
    """
    Encode the target column from string labels to integers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with the target column.
    target_col : str
        Name of the target column.
    label_mapping : dict
        Mapping of string labels to integers, e.g. {"legitimate": 0, "phishing": 1}.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series, LabelEncoder)
        - DataFrame of features (target column removed).
        - Series of encoded target values.
        - Fitted LabelEncoder for inverse transformations.
    """
    target_col = target_col or config.PRIMARY_TARGET_COLUMN
    label_mapping = label_mapping or config.PRIMARY_LABEL_MAPPING

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate features and target
    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Create and fit a LabelEncoder
    le = LabelEncoder()

    # If we have a specific mapping, use it; otherwise fit from data
    if label_mapping:
        # Sort labels by their integer value to ensure consistent encoding
        sorted_labels = sorted(label_mapping.keys(), key=lambda k: label_mapping[k])
        le.fit(sorted_labels)
        y = pd.Series(le.transform(y_raw), index=y_raw.index, name=target_col)
    else:
        y = pd.Series(le.fit_transform(y_raw), index=y_raw.index, name=target_col)

    print(f"[preprocessing] Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"[preprocessing] Class distribution after encoding:")
    for val, count in y.value_counts().sort_index().items():
        label_name = le.inverse_transform([val])[0]
        print(f"  {val} ({label_name}): {count}")

    return X, y, le


def identify_feature_types(X: pd.DataFrame) -> tuple:
    """
    Separate numeric and categorical column names.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    tuple of (list, list)
        Lists of numeric and categorical column names.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"[preprocessing] Numeric features:     {len(numeric_cols)}")
    print(f"[preprocessing] Categorical features:  {len(categorical_cols)}")

    if categorical_cols:
        print(f"[preprocessing] Categorical columns: {categorical_cols}")

    return numeric_cols, categorical_cols


def remove_duplicates(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Remove exact duplicate rows from the dataset.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Deduplicated features and targets.
    """
    combined = X.copy()
    combined["__target__"] = y
    n_before = len(combined)
    combined = combined.drop_duplicates()
    n_after = len(combined)

    y_clean = combined["__target__"]
    X_clean = combined.drop(columns=["__target__"])

    removed = n_before - n_after
    if removed > 0:
        print(f"[preprocessing] Removed {removed} duplicate rows ({100*removed/n_before:.1f}%).")
    else:
        print("[preprocessing] No duplicate rows found.")

    return X_clean, y_clean


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = None,
               val_size: float = None,
               random_seed: int = None) -> dict:
    """
    Split data into train, validation, and test sets with stratification.

    The test set is held out completely and never used during training or tuning.
    The validation set is used for model selection / hyperparameter tuning.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
        Fraction for the test set.
    val_size : float
        Fraction of the *remaining* data for validation.
    random_seed : int

    Returns
    -------
    dict
        Keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    test_size = test_size or config.TEST_SIZE
    val_size = val_size or config.VALIDATION_SIZE
    random_seed = random_seed or config.RANDOM_SEED

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    # Second split: separate validation set from training set
    # val_size is relative to the full dataset, so adjust for the remaining data
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_seed,
        stratify=y_temp
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    print(f"\n[preprocessing] Data split (seed={random_seed}):")
    print(f"  Training:   {X_train.shape[0]} samples ({100*X_train.shape[0]/len(X):.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({100*X_val.shape[0]/len(X):.1f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({100*X_test.shape[0]/len(X):.1f}%)")

    return splits


def build_preprocessing_pipeline(numeric_cols: list,
                                  categorical_cols: list,
                                  scale: bool = False) -> ColumnTransformer:
    """
    Build a scikit-learn preprocessing pipeline.

    For numeric columns: impute missing values with median, optionally scale.
    For categorical columns: impute with most frequent value.
    (In the primary dataset, all features are numeric, so the categorical
     branch is included for robustness but may not be exercised.)

    Parameters
    ----------
    numeric_cols : list
        Names of numeric feature columns.
    categorical_cols : list
        Names of categorical feature columns.
    scale : bool
        Whether to apply StandardScaler to numeric features (needed for SVM).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """
    # Numeric pipeline
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # Build the column transformer
    transformers = [("num", numeric_pipeline, numeric_cols)]

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Drop any unexpected columns
    )

    return preprocessor


def preprocess_dataset(df: pd.DataFrame,
                       target_col: str = None,
                       label_mapping: dict = None,
                       scale: bool = False) -> dict:
    """
    Full preprocessing pipeline: encode → deduplicate → split → fit/transform.

    This is the main entry point for preprocessing. It returns everything
    needed for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_col : str
        Target column name.
    label_mapping : dict
        Label encoding mapping.
    scale : bool
        Whether to scale features (set True for SVM).

    Returns
    -------
    dict
        Contains:
        - X_train, X_val, X_test (numpy arrays, preprocessed)
        - y_train, y_val, y_test (numpy arrays)
        - feature_names (list of str)
        - preprocessor (fitted ColumnTransformer)
        - label_encoder (fitted LabelEncoder)
        - numeric_cols, categorical_cols
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Encode target
    X, y, le = encode_target(df, target_col, label_mapping)

    # Step 2: Remove non-feature columns (like 'url' if present)
    # The primary dataset may have a 'url' column that is NOT a model feature
    non_feature_cols = []
    for col in X.columns:
        if col.lower() in ("url", "id", "index"):
            non_feature_cols.append(col)
    if non_feature_cols:
        print(f"[preprocessing] Dropping non-feature columns: {non_feature_cols}")
        X = X.drop(columns=non_feature_cols)

    # Step 3: Remove duplicates
    X, y = remove_duplicates(X, y)

    # Step 4: Identify feature types
    numeric_cols, categorical_cols = identify_feature_types(X)
    feature_names = numeric_cols + categorical_cols

    # Step 5: Split data
    splits = split_data(X, y)

    # Step 6: Build and fit preprocessing pipeline (on training data only!)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols, scale=scale)
    X_train_processed = preprocessor.fit_transform(splits["X_train"])
    X_val_processed = preprocessor.transform(splits["X_val"])
    X_test_processed = preprocessor.transform(splits["X_test"])

    print(f"[preprocessing] Preprocessed feature shape: {X_train_processed.shape}")
    print("=" * 60)

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": splits["y_train"].values,
        "y_val": splits["y_val"].values,
        "y_test": splits["y_test"].values,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def save_pipeline(preprocessor, label_encoder, feature_names: list,
                  metadata: dict = None):
    """
    Save the fitted preprocessing pipeline, label encoder, feature schema,
    and metadata to disk.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    label_encoder : LabelEncoder
        Fitted label encoder.
    feature_names : list
        Ordered list of feature names the model expects.
    metadata : dict, optional
        Additional metadata (training date, dataset info, etc.)
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Save preprocessor
    joblib.dump(preprocessor, config.SAVED_PIPELINE_PATH)
    print(f"[preprocessing] Saved pipeline to {config.SAVED_PIPELINE_PATH}")

    # Save label encoder
    joblib.dump(label_encoder, config.SAVED_LABEL_ENCODER_PATH)
    print(f"[preprocessing] Saved label encoder to {config.SAVED_LABEL_ENCODER_PATH}")

    # Save feature schema as JSON
    schema = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    with open(config.SAVED_FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[preprocessing] Saved feature schema to {config.SAVED_FEATURE_SCHEMA_PATH}")

    # Save metadata
    if metadata:
        with open(config.SAVED_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[preprocessing] Saved metadata to {config.SAVED_METADATA_PATH}")


def load_pipeline() -> tuple:
    """
    Load the saved preprocessing pipeline, label encoder, and feature schema.

    Returns
    -------
    tuple of (ColumnTransformer, LabelEncoder, list)
        Fitted preprocessor, label encoder, and ordered feature names.
    """
    preprocessor = joblib.load(config.SAVED_PIPELINE_PATH)
    label_encoder = joblib.load(config.SAVED_LABEL_ENCODER_PATH)

    with open(config.SAVED_FEATURE_SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    feature_names = schema["feature_names"]

    return preprocessor, label_encoder, feature_names
