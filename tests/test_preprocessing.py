"""
test_preprocessing.py — Tests for data preprocessing pipeline.

Tests cover:
    - Target encoding
    - Feature type identification
    - Duplicate removal
    - Data splitting (stratified, no leakage)
    - Pipeline building and fitting
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import (
    encode_target,
    identify_feature_types,
    remove_duplicates,
    split_data,
    build_preprocessing_pipeline,
)


@pytest.fixture
def sample_dataset():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randint(0, 10, n),
        "feature_c": np.random.uniform(0, 1, n),
        "status": np.random.choice(["legitimate", "phishing"], n),
    })
    return df


@pytest.fixture
def dataset_with_missing():
    """Create a dataset with missing values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "status": np.random.choice(["legitimate", "phishing"], n),
    })
    # Add some missing values
    df.loc[0:4, "feat1"] = np.nan
    df.loc[10:14, "feat2"] = np.nan
    return df


class TestEncodeTarget:
    """Tests for target column encoding."""

    def test_encodes_correctly(self, sample_dataset):
        X, y, le = encode_target(sample_dataset, "status",
                                  {"legitimate": 0, "phishing": 1})
        assert "status" not in X.columns
        assert set(y.unique()).issubset({0, 1})
        assert len(le.classes_) == 2

    def test_missing_target_raises(self, sample_dataset):
        with pytest.raises(KeyError, match="not found"):
            encode_target(sample_dataset, "nonexistent_column")

    def test_preserves_row_count(self, sample_dataset):
        X, y, le = encode_target(sample_dataset, "status",
                                  {"legitimate": 0, "phishing": 1})
        assert len(X) == len(sample_dataset)
        assert len(y) == len(sample_dataset)

    def test_label_encoder_inverse(self, sample_dataset):
        X, y, le = encode_target(sample_dataset, "status",
                                  {"legitimate": 0, "phishing": 1})
        decoded = le.inverse_transform(y.unique())
        assert set(decoded).issubset({"legitimate", "phishing"})


class TestIdentifyFeatureTypes:
    """Tests for feature type identification."""

    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        numeric, categorical = identify_feature_types(df)
        assert len(numeric) == 2
        assert len(categorical) == 0

    def test_mixed_types(self):
        df = pd.DataFrame({
            "num": [1, 2],
            "cat": ["a", "b"],
        })
        numeric, categorical = identify_feature_types(df)
        assert "num" in numeric
        assert "cat" in categorical


class TestRemoveDuplicates:
    """Tests for duplicate row removal."""

    def test_removes_duplicates(self):
        X = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        y = pd.Series([0, 0, 1])
        X_clean, y_clean = remove_duplicates(X, y)
        assert len(X_clean) == 2
        assert len(y_clean) == 2

    def test_no_duplicates(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        X_clean, y_clean = remove_duplicates(X, y)
        assert len(X_clean) == 3


class TestSplitData:
    """Tests for data splitting."""

    def test_split_sizes(self, sample_dataset):
        X, y, _ = encode_target(sample_dataset, "status",
                                 {"legitimate": 0, "phishing": 1})
        splits = split_data(X, y, test_size=0.15, val_size=0.15)

        total = (splits["X_train"].shape[0] + splits["X_val"].shape[0] +
                 splits["X_test"].shape[0])
        assert total == len(X)

    def test_no_sample_overlap(self, sample_dataset):
        """Train, val, and test sets must not share any indices."""
        X, y, _ = encode_target(sample_dataset, "status",
                                 {"legitimate": 0, "phishing": 1})
        splits = split_data(X, y)

        train_idx = set(splits["X_train"].index)
        val_idx = set(splits["X_val"].index)
        test_idx = set(splits["X_test"].index)

        assert train_idx.isdisjoint(val_idx), "Train and val overlap!"
        assert train_idx.isdisjoint(test_idx), "Train and test overlap!"
        assert val_idx.isdisjoint(test_idx), "Val and test overlap!"

    def test_stratification(self, sample_dataset):
        """Check that class distribution is roughly preserved in splits."""
        X, y, _ = encode_target(sample_dataset, "status",
                                 {"legitimate": 0, "phishing": 1})
        splits = split_data(X, y)

        full_ratio = y.mean()
        train_ratio = splits["y_train"].mean()

        # Allow up to 10% deviation due to small sample size
        assert abs(full_ratio - train_ratio) < 0.10


class TestPreprocessingPipeline:
    """Tests for the preprocessing pipeline."""

    def test_pipeline_fits_and_transforms(self, sample_dataset):
        X, y, _ = encode_target(sample_dataset, "status",
                                 {"legitimate": 0, "phishing": 1})
        numeric, categorical = identify_feature_types(X)
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == len(numeric) + len(categorical)

    def test_pipeline_handles_missing_values(self, dataset_with_missing):
        X, y, _ = encode_target(dataset_with_missing, "status",
                                 {"legitimate": 0, "phishing": 1})
        numeric, categorical = identify_feature_types(X)
        pipeline = build_preprocessing_pipeline(numeric, categorical)

        X_transformed = pipeline.fit_transform(X)
        # Should have no NaN after imputation
        assert not np.any(np.isnan(X_transformed))

    def test_scaled_pipeline(self, sample_dataset):
        X, y, _ = encode_target(sample_dataset, "status",
                                 {"legitimate": 0, "phishing": 1})
        numeric, categorical = identify_feature_types(X)
        pipeline = build_preprocessing_pipeline(numeric, categorical, scale=True)

        X_transformed = pipeline.fit_transform(X)

        # Scaled data should have mean ≈ 0 and std ≈ 1
        means = np.abs(X_transformed.mean(axis=0))
        assert np.all(means < 0.5), "Scaled features should have mean near 0"
