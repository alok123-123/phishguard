"""
train.py — Train and save machine learning models for phishing detection.

This module:
    1. Loads the preprocessed data.
    2. Trains four candidate models: Random Forest, XGBoost, LightGBM, SVM.
    3. Evaluates each on the validation set for model selection.
    4. Saves trained models, the preprocessing pipeline, and metadata.
    5. Optionally trains a "URL-only" model on the subset of features that
       can be extracted from a URL string (for live prediction).

Usage (command line):
    python -m src.train                          # Train on primary dataset
    python -m src.train --dataset path/to/data.csv
    python -m src.train --url-only               # Train URL-feature-only model

Usage (in code):
    from src.train import train_all_models
    results = train_all_models(data)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_primary_dataset, inspect_dataset, verify_primary_dataset
from src.preprocessing import (
    preprocess_dataset, save_pipeline, encode_target,
    identify_feature_types, remove_duplicates, split_data,
    build_preprocessing_pipeline,
)
from src.feature_extraction import get_feature_names, compare_with_dataset_features


def _get_model(model_name: str, random_seed: int = None):
    """
    Create an untrained model instance by name.

    Parameters
    ----------
    model_name : str
        One of: "RandomForest", "XGBoost", "LightGBM", "SVM"
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    estimator
        A scikit-learn compatible model.
    """
    seed = random_seed or config.RANDOM_SEED

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )

    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("[train] ⚠ XGBoost not installed. Skipping.")
            return None
        return XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )

    elif model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            print("[train] ⚠ LightGBM not installed. Skipping.")
            return None
        return LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            class_weight="balanced",
            verbose=-1,  # Suppress LightGBM output
            n_jobs=-1,
        )

    elif model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,   # Enables predict_proba (slower but needed)
            random_state=seed,
            class_weight="balanced",
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_single_model(model_name: str, X_train, y_train, X_val, y_val,
                        random_seed: int = None) -> dict:
    """
    Train a single model and evaluate on validation data.

    Parameters
    ----------
    model_name : str
    X_train, y_train : training data
    X_val, y_val : validation data
    random_seed : int

    Returns
    -------
    dict
        Contains 'model', 'val_accuracy', 'val_f1', 'train_time_seconds'.
        Returns None for 'model' if the library is unavailable.
    """
    print(f"\n{'─' * 40}")
    print(f"Training: {model_name}")
    print(f"{'─' * 40}")

    model = _get_model(model_name, random_seed)
    if model is None:
        return {"model": None, "model_name": model_name, "skipped": True}

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Validation predictions
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="weighted")

    print(f"  Training time:      {train_time:.2f} seconds")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation F1:       {val_f1:.4f}")

    return {
        "model": model,
        "model_name": model_name,
        "val_accuracy": round(val_acc, 4),
        "val_f1": round(val_f1, 4),
        "train_time_seconds": round(train_time, 2),
        "skipped": False,
    }


def train_all_models(data: dict) -> dict:
    """
    Train all candidate models and return results.

    Parameters
    ----------
    data : dict
        Output from preprocess_dataset() containing X_train, y_train, etc.

    Returns
    -------
    dict
        model_name → training result dict
    """
    results = {}

    for model_name in config.MODEL_NAMES:
        result = train_single_model(
            model_name,
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        results[model_name] = result

    # Summary table
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON (Validation Set)")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Accuracy':<12} {'F1':<12} {'Time (s)':<10} {'Status'}")
    print(f"{'─' * 60}")
    for name, r in results.items():
        if r.get("skipped"):
            print(f"{name:<15} {'—':<12} {'—':<12} {'—':<10} SKIPPED")
        else:
            print(f"{name:<15} {r['val_accuracy']:<12.4f} {r['val_f1']:<12.4f} "
                  f"{r['train_time_seconds']:<10.2f} OK")
    print(f"{'=' * 60}")

    return results


def save_models(results: dict, data: dict, dataset_path: str = None):
    """
    Save all trained models, the preprocessing pipeline, and metadata.

    Parameters
    ----------
    results : dict
        Output from train_all_models().
    data : dict
        Output from preprocess_dataset().
    dataset_path : str
        Path to the dataset used for training.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Save each model
    for model_name, result in results.items():
        if result.get("skipped") or result.get("model") is None:
            continue
        model_path = config.SAVED_MODEL_TEMPLATE.format(model_name=model_name)
        joblib.dump(result["model"], model_path)
        print(f"[train] Saved {model_name} → {model_path}")

    # Save preprocessing pipeline and schema
    metadata = {
        "training_date": datetime.now().isoformat(),
        "dataset_path": dataset_path or config.PRIMARY_DATASET_PATH,
        "random_seed": config.RANDOM_SEED,
        "n_train_samples": int(data["X_train"].shape[0]),
        "n_val_samples": int(data["X_val"].shape[0]),
        "n_test_samples": int(data["X_test"].shape[0]),
        "n_features": len(data["feature_names"]),
        "feature_names": data["feature_names"],
        "target_column": config.PRIMARY_TARGET_COLUMN,
        "label_mapping": config.PRIMARY_LABEL_MAPPING,
        "models_trained": {
            name: {
                "val_accuracy": r.get("val_accuracy"),
                "val_f1": r.get("val_f1"),
                "train_time_seconds": r.get("train_time_seconds"),
                "skipped": r.get("skipped", False),
            }
            for name, r in results.items()
        },
    }

    save_pipeline(
        data["preprocessor"],
        data["label_encoder"],
        data["feature_names"],
        metadata=metadata,
    )


def train_url_only_model(df: pd.DataFrame) -> dict:
    """
    Train a separate model using ONLY features extractable from a URL.

    The primary dataset has 87 features, but only some overlap with what
    we can compute from a URL string. This function:
        1. Identifies shared features between the dataset and URL extractor.
        2. Trains a Random Forest on the shared subset.
        3. Saves it as the "URL-only" model for live predictions.

    Parameters
    ----------
    df : pd.DataFrame
        The primary dataset.

    Returns
    -------
    dict
        Training results and feature mapping report.
    """
    print("\n" + "=" * 60)
    print("TRAINING URL-ONLY MODEL")
    print("=" * 60)

    # Encode target
    X, y, le = encode_target(df)

    # Drop non-feature columns
    non_feature_cols = [c for c in X.columns if c.lower() in ("url", "id", "index")]
    if non_feature_cols:
        print(f"[train] Dropping non-feature columns: {non_feature_cols}")
        X = X.drop(columns=non_feature_cols)

    # Compare features
    url_feature_names = get_feature_names()
    report = compare_with_dataset_features(list(X.columns))

    shared_features = report["shared"]
    if len(shared_features) == 0:
        print("\n⚠ WARNING: No features are shared between the URL extractor and the dataset.")
        print("  The URL-only model CANNOT be trained with this dataset.")
        print("  The dataset uses different feature names/definitions than the URL extractor.")
        print("  For live URL prediction, the model will use URL-extracted features directly.")
        print("  Training a standalone URL-only model from URL-extracted features...")

        # Train on URL-extracted features instead
        return _train_standalone_url_model(df, le)

    # Use only shared features
    X_shared = X[shared_features]
    print(f"\n[train] Training URL-only model with {len(shared_features)} shared features:")
    for f in shared_features:
        print(f"  - {f}")

    # Remove duplicates
    X_shared, y = remove_duplicates(X_shared, y)

    # Split
    splits = split_data(X_shared, y)

    # Build pipeline (no scaling for RF)
    numeric_cols = X_shared.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_shared.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols, scale=False)

    X_train = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=config.RANDOM_SEED,
        n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, splits["y_train"].values)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(splits["y_val"], val_preds)
    val_f1 = f1_score(splits["y_val"], val_preds, average="weighted")
    print(f"\n[train] URL-only model validation accuracy: {val_acc:.4f}")
    print(f"[train] URL-only model validation F1:       {val_f1:.4f}")

    # Save
    url_model_path = os.path.join(config.MODELS_DIR, "URLOnly_model.joblib")
    url_pipeline_path = os.path.join(config.MODELS_DIR, "url_preprocessing_pipeline.joblib")
    url_schema_path = os.path.join(config.MODELS_DIR, "url_feature_schema.json")

    joblib.dump(model, url_model_path)
    joblib.dump(preprocessor, url_pipeline_path)

    schema = {"feature_names": shared_features, "n_features": len(shared_features)}
    with open(url_schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Save label encoder for URL model
    url_le_path = os.path.join(config.MODELS_DIR, "url_label_encoder.joblib")
    joblib.dump(le, url_le_path)

    print(f"[train] Saved URL-only model → {url_model_path}")
    print("=" * 60)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": shared_features,
        "label_encoder": le,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "feature_report": report,
    }


def _train_standalone_url_model(df: pd.DataFrame, le) -> dict:
    """
    Train a URL-only model by extracting URL features from the dataset's URL column.

    If the dataset has a 'url' column, we extract features from each URL
    and train a model on those extracted features. This ensures the model
    uses the exact same feature set as the live predictor.
    """
    from src.feature_extraction import extract_url_features

    if "url" not in df.columns:
        print("⚠ No 'url' column found in the dataset.")
        print("  Cannot train a standalone URL-only model without URLs.")
        print("  The live URL analyzer will still work but without a pre-trained model.")
        return {"model": None, "skipped": True, "reason": "No URL column in dataset"}

    print("[train] Extracting URL features from dataset URLs...")
    target_col = config.PRIMARY_TARGET_COLUMN

    feature_rows = []
    labels = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            feats = extract_url_features(row["url"])
            feature_rows.append(feats)
            labels.append(row[target_col])
        except (ValueError, Exception):
            skipped += 1
            continue

    print(f"[train] Extracted features from {len(feature_rows)} URLs ({skipped} skipped).")

    if len(feature_rows) < 100:
        print("⚠ Too few valid URLs to train a model.")
        return {"model": None, "skipped": True, "reason": "Too few valid URLs"}

    X_url = pd.DataFrame(feature_rows)
    y_url = pd.Series(labels)

    # Encode labels
    y_encoded = pd.Series(le.transform(y_url), name=target_col)

    # Remove duplicates
    X_url, y_encoded = remove_duplicates(X_url, y_encoded)

    # Split
    splits = split_data(X_url, y_encoded)

    # Build pipeline
    numeric_cols = X_url.columns.tolist()
    preprocessor = build_preprocessing_pipeline(numeric_cols, [], scale=False)

    X_train = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])

    # Train
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=config.RANDOM_SEED,
        n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, splits["y_train"].values)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(splits["y_val"], val_preds)
    val_f1 = f1_score(splits["y_val"], val_preds, average="weighted")

    print(f"\n[train] Standalone URL-only model validation accuracy: {val_acc:.4f}")
    print(f"[train] Standalone URL-only model validation F1:       {val_f1:.4f}")

    # Save
    url_model_path = os.path.join(config.MODELS_DIR, "URLOnly_model.joblib")
    url_pipeline_path = os.path.join(config.MODELS_DIR, "url_preprocessing_pipeline.joblib")
    url_schema_path = os.path.join(config.MODELS_DIR, "url_feature_schema.json")
    url_le_path = os.path.join(config.MODELS_DIR, "url_label_encoder.joblib")

    joblib.dump(model, url_model_path)
    joblib.dump(preprocessor, url_pipeline_path)
    joblib.dump(le, url_le_path)

    feature_names = list(X_url.columns)
    with open(url_schema_path, "w") as f:
        json.dump({"feature_names": feature_names, "n_features": len(feature_names)}, f, indent=2)

    print(f"[train] Saved standalone URL-only model → {url_model_path}")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "label_encoder": le,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    }


# ─── Command-line interface ───
def main():
    parser = argparse.ArgumentParser(
        description="Train phishing detection models."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to the dataset CSV file."
    )
    parser.add_argument(
        "--url-only", action="store_true",
        help="Also train a URL-feature-only model for live predictions."
    )
    args = parser.parse_args()

    # Load dataset
    df = load_primary_dataset(args.dataset)
    inspect_dataset(df)
    verify_primary_dataset(df)

    # Preprocess
    data = preprocess_dataset(df)

    # Train all models
    results = train_all_models(data)

    # Save everything
    save_models(results, data, args.dataset)

    # Optionally train URL-only model
    if args.url_only:
        train_url_only_model(df)

    print("\n✓ Training complete!")
    print(f"  Models saved to: {config.MODELS_DIR}")
    print(f"  Run evaluation:  python -m src.evaluate")
    print(f"  Launch app:      streamlit run app.py")


if __name__ == "__main__":
    main()
