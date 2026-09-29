"""
test_prediction.py — Tests for the prediction pipeline.

Tests cover:
    - PhishingPredictor initialization
    - Feature alignment with model schema
    - Prediction output structure
    - Error handling for invalid inputs
    - Warning messages in predictions
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.predict import PhishingPredictor
from src.feature_extraction import extract_url_features, get_feature_names


class TestPhishingPredictor:
    """Tests for the PhishingPredictor class."""

    def test_predictor_init(self):
        """Predictor should initialize without crashing."""
        predictor = PhishingPredictor()
        # May or may not have a loaded model depending on training state
        assert isinstance(predictor.is_loaded, bool)

    def test_model_info(self):
        """get_model_info should return a dict."""
        predictor = PhishingPredictor()
        info = predictor.get_model_info()
        assert isinstance(info, dict)
        assert "loaded" in info

    def test_predict_invalid_url(self):
        """Predicting an invalid URL should return an error, not crash."""
        predictor = PhishingPredictor()
        result = predictor.predict("")
        assert result["error"] is not None

    def test_predict_returns_dict(self):
        """Predict should always return a dict with expected keys."""
        predictor = PhishingPredictor()
        result = predictor.predict("https://example.com")
        expected_keys = [
            "url", "prediction", "prediction_code", "confidence",
            "phishing_probability", "features", "feature_df",
            "preprocessed_input", "warning", "error",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_with_model_not_loaded(self):
        """If model isn't loaded, predict should return an error message."""
        predictor = PhishingPredictor()
        if not predictor.is_loaded:
            result = predictor.predict("https://example.com")
            assert result["error"] is not None
            assert "train" in result["error"].lower() or "model" in result["error"].lower()


class TestFeatureAlignment:
    """Tests for feature alignment with model schema."""

    def test_features_match_schema(self):
        """Extracted features should contain all schema features."""
        schema_names = get_feature_names()
        features = extract_url_features("https://example.com")

        for name in schema_names:
            assert name in features, f"Schema feature '{name}' missing from extraction"

    def test_feature_types_are_numeric(self):
        """All extracted features should be numeric (int or float)."""
        features = extract_url_features("https://example.com")
        for name, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"Feature '{name}' has non-numeric type: {type(value)}"

    def test_binary_features_are_0_or_1(self):
        """Binary features should only have values 0 or 1."""
        features = extract_url_features("https://example.com")
        binary_features = [
            "has_ip_address", "has_https", "has_http", "has_at_sign",
            "has_double_slash_redirect", "is_shortened", "has_port",
            "has_suspicious_tld",
        ]
        for name in binary_features:
            assert features[name] in (0, 1), \
                f"Binary feature '{name}' has value {features[name]}"


class TestPredictionOutput:
    """Tests for prediction output format."""

    def test_prediction_has_warning(self):
        """Every successful prediction should include a warning disclaimer."""
        predictor = PhishingPredictor()
        if predictor.is_loaded:
            result = predictor.predict("https://example.com")
            if result["error"] is None:
                assert result["warning"] is not None
                assert "guarantee" in result["warning"].lower() or \
                       "estimate" in result["warning"].lower()

    def test_prediction_label_is_string(self):
        """Prediction label should be a human-readable string."""
        predictor = PhishingPredictor()
        if predictor.is_loaded:
            result = predictor.predict("https://example.com")
            if result["prediction"] is not None:
                assert isinstance(result["prediction"], str)
                assert result["prediction"] in ("legitimate", "phishing")

    def test_confidence_in_range(self):
        """Confidence should be between 0 and 1."""
        predictor = PhishingPredictor()
        if predictor.is_loaded:
            result = predictor.predict("https://example.com")
            if result["confidence"] is not None:
                assert 0 <= result["confidence"] <= 1
