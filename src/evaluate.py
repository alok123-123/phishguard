"""
evaluate.py — Evaluate trained models and generate performance reports.

This module:
    1. Loads trained models and the test set.
    2. Computes accuracy, precision, recall, F1, ROC-AUC.
    3. Generates confusion matrices and classification reports.
    4. Produces comparison charts across all models.
    5. Saves results and visualizations to the reports/ directory.

IMPORTANT: All metrics are computed on the HELD-OUT TEST SET that was
never used during training or hyperparameter selection.

Usage:
    python -m src.evaluate
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def evaluate_model(model, X_test, y_test, model_name: str,
                   label_encoder=None) -> dict:
    """
    Evaluate a single model on the test set.

    Parameters
    ----------
    model : estimator
        Trained scikit-learn compatible model.
    X_test : array-like
        Test features (preprocessed).
    y_test : array-like
        True test labels (encoded).
    model_name : str
        Name of the model (for display and saving).
    label_encoder : LabelEncoder, optional
        For decoding label names.

    Returns
    -------
    dict
        All computed metrics, predictions, and probabilities.
    """
    print(f"\n{'─' * 40}")
    print(f"Evaluating: {model_name}")
    print(f"{'─' * 40}")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report (text)
    if label_encoder is not None:
        target_names = list(label_encoder.classes_)
    else:
        target_names = [str(c) for c in sorted(np.unique(y_test))]
    cls_report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    result = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "y_pred": y_pred,
        "prec_per_class": prec_per_class.tolist(),
        "rec_per_class": rec_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }

    # ROC-AUC (only for binary classification with probability support)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                result["roc_auc"] = round(roc_auc, 4)
                result["y_proba"] = y_proba[:, 1]
                print(f"  ROC-AUC:   {roc_auc:.4f}")

                # Average precision (PR-AUC)
                ap = average_precision_score(y_test, y_proba[:, 1])
                result["average_precision"] = round(ap, 4)
                print(f"  Avg Prec:  {ap:.4f}")
        except Exception as e:
            print(f"  ⚠ Could not compute ROC-AUC: {e}")

    print(f"\n  Classification Report:\n{cls_report}")

    return result


def plot_confusion_matrix(cm, model_name: str, target_names: list,
                          save_path: str = None):
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.
    model_name : str
    target_names : list
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix → {save_path}")

    plt.close(fig)
    return fig


def plot_roc_curves(all_results: dict, y_test, target_names: list,
                    save_path: str = None):
    """
    Plot ROC curves for all models that support probability predictions.

    Parameters
    ----------
    all_results : dict
        model_name → evaluation result dict.
    y_test : array-like
    target_names : list
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in all_results.items():
        if "y_proba" in result:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
            auc = result.get("roc_auc", 0)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved ROC curves → {save_path}")

    plt.close(fig)
    return fig


def plot_model_comparison(all_results: dict, save_path: str = None):
    """
    Bar chart comparing all models on key metrics.

    Parameters
    ----------
    all_results : dict
    save_path : str, optional
    """
    models = []
    metrics_data = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    for name, result in all_results.items():
        models.append(name)
        metrics_data["Accuracy"].append(result["accuracy"])
        metrics_data["Precision"].append(result["precision"])
        metrics_data["Recall"].append(result["recall"])
        metrics_data["F1"].append(result["f1_score"])

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved model comparison → {save_path}")

    plt.close(fig)
    return fig


def plot_precision_recall_curve(all_results: dict, y_test,
                                 save_path: str = None):
    """
    Plot precision-recall curves for all models with probability support.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in all_results.items():
        if "y_proba" in result:
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_test, result["y_proba"]
            )
            ap = result.get("average_precision", 0)
            ax.plot(recall_vals, precision_vals,
                    label=f"{name} (AP = {ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved PR curves → {save_path}")

    plt.close(fig)
    return fig


def evaluate_all_models(data: dict) -> dict:
    """
    Load all saved models and evaluate them on the test set.

    Parameters
    ----------
    data : dict
        Output from preprocess_dataset() containing X_test, y_test, etc.

    Returns
    -------
    dict
        model_name → evaluation results.
    """
    # Load label encoder
    label_encoder = joblib.load(config.SAVED_LABEL_ENCODER_PATH)
    target_names = list(label_encoder.classes_)

    all_results = {}

    for model_name in config.MODEL_NAMES:
        model_path = config.SAVED_MODEL_TEMPLATE.format(model_name=model_name)
        if not os.path.isfile(model_path):
            print(f"[evaluate] ⚠ Model not found: {model_path} — skipping.")
            continue

        model = joblib.load(model_path)
        result = evaluate_model(
            model, data["X_test"], data["y_test"],
            model_name, label_encoder
        )
        all_results[model_name] = result

        # Save confusion matrix plot
        cm_path = os.path.join(config.REPORTS_DIR, f"confusion_matrix_{model_name}.png")
        plot_confusion_matrix(
            np.array(result["confusion_matrix"]),
            model_name, target_names, cm_path
        )

    # Save comparison plots
    if all_results:
        plot_roc_curves(
            all_results, data["y_test"], target_names,
            os.path.join(config.REPORTS_DIR, "roc_curves.png")
        )
        plot_model_comparison(
            all_results,
            os.path.join(config.REPORTS_DIR, "model_comparison.png")
        )
        plot_precision_recall_curve(
            all_results, data["y_test"],
            os.path.join(config.REPORTS_DIR, "precision_recall_curves.png")
        )

    # Save metrics JSON
    metrics_json = {}
    for name, result in all_results.items():
        metrics_json[name] = {
            k: v for k, v in result.items()
            if k not in ("y_pred", "y_proba", "classification_report")
        }
    metrics_path = os.path.join(config.REPORTS_DIR, "evaluation_metrics.json")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n[evaluate] Saved metrics → {metrics_path}")

    # Save classification reports
    for name, result in all_results.items():
        report_path = os.path.join(config.REPORTS_DIR, f"classification_report_{name}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report: {name}\n")
            f.write("=" * 50 + "\n")
            f.write(result["classification_report"])
        print(f"[evaluate] Saved classification report → {report_path}")

    return all_results


def main():
    """Run evaluation from command line."""
    import argparse
    from src.data_loader import load_primary_dataset, load_csv
    from src.preprocessing import preprocess_dataset

    parser = argparse.ArgumentParser(description="Evaluate phishing detection models.")
    parser.add_argument("model_file", nargs="?", default=None, help="Optional path to a specific model file (.joblib or .pkl)")
    parser.add_argument("--model", type=str, default=None, help="Path to a specific model file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV for evaluation")
    args = parser.parse_args()

    model_path = args.model or args.model_file

    if model_path:
        if not os.path.isfile(model_path):
            print(f"[evaluate] Error: Model file not found: {model_path}")
            return

        print(f"[evaluate] Loading custom model from: {model_path}")
        model = joblib.load(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Determine dataset
        dataset_path = args.dataset
        if not dataset_path:
            # Check default primary dataset first
            if os.path.isfile(config.PRIMARY_DATASET_PATH):
                dataset_path = config.PRIMARY_DATASET_PATH
            # If model features match PhiUSIIL and PhiUSIIL dataset exists, prefer that
            phiusiil_default = r"C:\Users\Alok\Downloads\PhiUSIIL_Phishing_URL_Dataset.csv"
            if hasattr(model, "feature_names_in_") and os.path.isfile(phiusiil_default):
                sample_cols = list(pd.read_csv(phiusiil_default, nrows=2).columns)
                if all(f in sample_cols for f in model.feature_names_in_[:5]):
                    dataset_path = phiusiil_default

        print(f"[evaluate] Using dataset for evaluation: {dataset_path}")
        df = load_csv(dataset_path)

        # Check target column
        target_col = None
        for candidate in ["status", "label", "Result", "target", "CLASS"]:
            if candidate in df.columns:
                target_col = candidate
                break

        if not target_col:
            print(f"[evaluate] Error: Could not find target column in dataset.")
            return

        # Prepare features
        if hasattr(model, "feature_names_in_"):
            feature_cols = [f for f in model.feature_names_in_ if f in df.columns]
            X = df[feature_cols]
        else:
            X = df.drop(columns=[target_col])
            # Drop url if present
            if "url" in X.columns:
                X = X.drop(columns=["url"])
            if "URL" in X.columns:
                X = X.drop(columns=["URL"])

        y_raw = df[target_col]
        # Encode y if string
        if y_raw.dtype == object:
            mapping = {"legitimate": 0, "phishing": 1, "benign": 0, "malicious": 1}
            y = y_raw.map(mapping).fillna(0).astype(int)
        else:
            y = y_raw.astype(int)

        # Evaluate on test split (15%)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
        )

        res = evaluate_model(model, X_test, y_test, model_name)
        auc = res.get("roc_auc", "N/A")
        if isinstance(auc, float):
            auc = f"{auc:.4f}"

        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY: {model_name}")
        print(f"{'=' * 60}")
        print(f"Test Samples: {len(y_test)}")
        print(f"Accuracy:     {res['accuracy']:.4f}")
        print(f"Precision:    {res['precision']:.4f}")
        print(f"Recall:       {res['recall']:.4f}")
        print(f"F1-Score:     {res['f1_score']:.4f}")
        print(f"ROC-AUC:      {auc}")
        print(f"{'=' * 60}")
        print(f"\nConfusion matrix saved to: {config.REPORTS_DIR}/confusion_matrix_{model_name}.png")
        return

    # Load and preprocess data (same splits as training due to same seed)
    df = load_primary_dataset()
    data = preprocess_dataset(df)

    # Evaluate
    results = evaluate_all_models(data)

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY (Test Set)")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'AUC':<10}")
    print(f"{'─' * 65}")
    for name, r in results.items():
        auc = r.get("roc_auc", "N/A")
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        print(f"{name:<15} {r['accuracy']:<10.4f} {r['precision']:<10.4f} "
              f"{r['recall']:<10.4f} {r['f1_score']:<10.4f} {auc:<10}")
    print(f"{'=' * 60}")

    print(f"\n✓ Reports saved to: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
