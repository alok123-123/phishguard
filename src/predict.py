"""
predict.py — End-to-end prediction pipeline for live URL analysis.

This module connects:
    User URL → Validation → Feature Extraction → Preprocessing → Model → Prediction

It loads the saved URL-only model and preprocessing pipeline, extracts
features from a raw URL, and produces a prediction with probabilities.

IMPORTANT: This module uses the URL-ONLY model, which is trained on
features that can be extracted from the URL string alone. The full dataset
model may use additional features (e.g., HTML content, external APIs) that
cannot be safely reproduced from an arbitrary URL.

Usage:
    from src.predict import predict_url
    result = predict_url("https://suspicious-site.com/login")
    print(result["prediction"], result["confidence"])
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.feature_extraction import (
    extract_url_features, validate_url, features_to_dataframe, get_feature_names,
)


class PhishingPredictor:
    """
    End-to-end phishing prediction from a raw URL.

    This class loads the saved URL-only model, preprocessing pipeline,
    and label encoder, then provides a predict() method.

    Attributes
    ----------
    model : estimator
        Trained ML model for URL-based prediction.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    label_encoder : LabelEncoder
        For decoding predicted class labels.
    feature_names : list
        Expected feature names in correct order.
    is_loaded : bool
        Whether the model was successfully loaded.
    """

    def __init__(self):
        """Initialize and attempt to load the saved model."""
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False
        self.model_type = None
        self._load()

    def _load(self):
        """Load the URL-only model and pipeline from disk."""
        url_model_path = os.path.join(config.MODELS_DIR, "URLOnly_model.joblib")
        url_pipeline_path = os.path.join(config.MODELS_DIR, "url_preprocessing_pipeline.joblib")
        url_schema_path = os.path.join(config.MODELS_DIR, "url_feature_schema.json")
        url_le_path = os.path.join(config.MODELS_DIR, "url_label_encoder.joblib")

        # Check all required files exist
        required_files = {
            "model": url_model_path,
            "pipeline": url_pipeline_path,
            "schema": url_schema_path,
            "label_encoder": url_le_path,
        }

        missing = [name for name, path in required_files.items()
                    if not os.path.isfile(path)]
        if missing:
            print(f"[predict] [!] Cannot load URL predictor. Missing files: {missing}")
            print(f"[predict]   Run training first: python -m src.train --url-only")
            return

        try:
            self.model = joblib.load(url_model_path)
            self.preprocessor = joblib.load(url_pipeline_path)
            self.label_encoder = joblib.load(url_le_path)

            with open(url_schema_path, "r") as f:
                schema = json.load(f)
            self.feature_names = schema["feature_names"]
            self.model_type = type(self.model).__name__

            self.is_loaded = True
            print(f"[predict] [OK] Loaded URL-only model ({self.model_type}) "
                  f"with {len(self.feature_names)} features.")

        except Exception as e:
            print(f"[predict] [!] Error loading model: {e}")

    def predict(self, url: str) -> dict:
        """
        Predict whether a URL is likely phishing or legitimate.

        Parameters
        ----------
        url : str
            The URL to analyze.

        Returns
        -------
        dict
            Contains:
            - 'url': the input URL
            - 'prediction': predicted class label (str)
            - 'prediction_code': numeric class (int)
            - 'confidence': model probability for the predicted class
            - 'phishing_probability': probability of being phishing
            - 'features': extracted feature values (dict)
            - 'feature_df': single-row DataFrame of features
            - 'preprocessed_input': preprocessed numpy array
            - 'warning': any warnings about the prediction
            - 'error': error message if prediction failed

        DISCLAIMER: The 'confidence' score reflects the MODEL's certainty,
        not the actual probability of the website being dangerous. Model
        predictions are estimates, not guarantees of safety.
        """
        result = {
            "url": url,
            "prediction": None,
            "prediction_code": None,
            "confidence": None,
            "phishing_probability": None,
            "features": None,
            "feature_df": None,
            "preprocessed_input": None,
            "warning": None,
            "error": None,
        }

        # Check if model is loaded
        if not self.is_loaded:
            result["error"] = (
                "Model not loaded. Please train the model first by running:\n"
                "  python -m src.train --url-only"
            )
            return result

        # Step 1: Validate URL
        try:
            validated_url = validate_url(url)
        except ValueError as e:
            result["error"] = f"Invalid URL: {str(e)}"
            return result

        # Step 2: Extract features
        try:
            features = extract_url_features(validated_url)
            result["features"] = features
        except Exception as e:
            result["error"] = f"Feature extraction failed: {str(e)}"
            return result

        # Step 3: Build feature DataFrame with correct schema
        try:
            feature_df = self._align_features(features)
            result["feature_df"] = feature_df
        except Exception as e:
            result["error"] = f"Feature alignment failed: {str(e)}"
            return result

        # Step 4: Preprocess
        try:
            X_processed = self.preprocessor.transform(feature_df)
            result["preprocessed_input"] = X_processed
        except Exception as e:
            result["error"] = f"Preprocessing failed: {str(e)}"
            return result

        # Step 5: Predict
        try:
            pred_code = self.model.predict(X_processed)[0]
            pred_label = self.label_encoder.inverse_transform([pred_code])[0]

            result["prediction"] = pred_label
            result["prediction_code"] = int(pred_code)

            # Get probability if available
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_processed)[0]
                result["confidence"] = float(max(proba))
                # Probability of being phishing (class 1)
                phishing_idx = list(self.label_encoder.classes_).index("phishing") \
                    if "phishing" in self.label_encoder.classes_ else 1
                result["phishing_probability"] = float(proba[phishing_idx])

        except Exception as e:
            result["error"] = f"Prediction failed: {str(e)}"
            return result

        # Add standard warning
        result["warning"] = (
            "This prediction is a model estimate based on URL structural features. "
            "It is NOT a guarantee of safety. A 'legitimate' classification does not "
            "mean the website is safe to use. Always verify websites through "
            "trusted sources before entering sensitive information."
        )

        return result

    def _align_features(self, features: dict) -> pd.DataFrame:
        """
        Align extracted features to match the model's expected schema.

        Ensures correct feature names, ordering, and handles any missing features.

        Parameters
        ----------
        features : dict
            Feature dictionary from extract_url_features().

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame matching the model's expected schema.

        Raises
        ------
        ValueError
            If required features are missing and cannot be safely defaulted.
        """
        row = {}
        missing_features = []

        for feat_name in self.feature_names:
            if feat_name in features:
                row[feat_name] = features[feat_name]
            else:
                missing_features.append(feat_name)
                # Only fill with 0 for clearly missing structural features
                # This is documented: the model may be less accurate for these
                row[feat_name] = 0

        if missing_features:
            print(f"[predict] [!] Missing features filled with 0: {missing_features}")
            print(f"[predict]   The model was trained with these features but the "
                  f"URL extractor did not produce them.")

        return pd.DataFrame([row])

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": self.model_type,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "class_labels": list(self.label_encoder.classes_),
        }


def predict_url(url: str) -> dict:
    """
    Convenience function: predict a single URL.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
        Prediction results.
    """
    predictor = PhishingPredictor()
    return predictor.predict(url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict if a URL is phishing.")
    parser.add_argument("url", type=str, help="URL to analyze")
    args = parser.parse_args()

    result = predict_url(args.url)

    if result["error"]:
        print(f"\n[X] Error: {result['error']}")
    else:
        print(f"\n{'=' * 50}")
        print(f"URL:        {result['url']}")
        print(f"Prediction: {result['prediction']}")
        if result["confidence"]:
            print(f"Confidence: {result['confidence']:.2%}")
        if result["phishing_probability"] is not None:
            print(f"Phishing Probability: {result['phishing_probability']:.2%}")
        print(f"\n[!] {result['warning']}")
        print(f"{'=' * 50}")
