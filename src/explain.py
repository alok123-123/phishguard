"""
explain.py — SHAP and LIME explanations for model predictions.

This module:
    1. Computes SHAP global feature importance (summary plot).
    2. Computes SHAP local explanation for a single prediction.
    3. Computes LIME explanation for a single prediction.
    4. Handles errors gracefully so the app doesn't crash.

IMPORTANT NOTES:
    - Feature contributions explain MODEL BEHAVIOR, not proof of malicious intent.
    - SHAP values show what the model relied on, not what is objectively suspicious.
    - Some model types may have limited SHAP support (e.g., SVM uses KernelSHAP
      which is slow; tree-based models use TreeSHAP which is fast).

Usage:
    from src.explain import explain_shap_global, explain_shap_local, explain_lime
"""

import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Suppress verbose SHAP/LIME output
warnings.filterwarnings("ignore", category=UserWarning)


def explain_shap_global(model, X_data, feature_names: list,
                        model_name: str = "Model",
                        max_samples: int = 500,
                        save_path: str = None) -> dict:
    """
    Compute SHAP global feature importance and generate a summary plot.

    For tree-based models (Random Forest, XGBoost, LightGBM), uses TreeSHAP.
    For other models, attempts KernelSHAP with a smaller background sample.

    Parameters
    ----------
    model : estimator
        Trained model.
    X_data : array-like
        Feature data to explain (typically the test set or a sample of it).
    feature_names : list
        Feature names matching X_data columns.
    model_name : str
        Name of the model (for display).
    max_samples : int
        Maximum number of samples to explain (for performance).
    save_path : str, optional
        Path to save the summary plot.

    Returns
    -------
    dict
        Contains 'shap_values', 'expected_value', 'feature_importance', 'figure'.
        Returns {'error': str} if explanation fails.
    """
    try:
        import shap
    except ImportError:
        return {"error": "SHAP library not installed. Install with: pip install shap"}

    try:
        # Limit samples for performance
        if hasattr(X_data, 'shape') and X_data.shape[0] > max_samples:
            indices = np.random.RandomState(config.RANDOM_SEED).choice(
                X_data.shape[0], max_samples, replace=False
            )
            X_sample = X_data[indices] if isinstance(X_data, np.ndarray) else X_data.iloc[indices]
        else:
            X_sample = X_data

        # Choose the right explainer based on model type
        model_type = type(model).__name__
        tree_models = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
                       "GradientBoostingClassifier", "DecisionTreeClassifier"}

        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # KernelSHAP for non-tree models (slower, use small background)
            bg_size = min(50, X_sample.shape[0])
            background = shap.sample(X_sample, bg_size,
                                     random_state=config.RANDOM_SEED)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample[:min(100, len(X_sample))])

        # Handle different SHAP value formats
        # For binary classification, shap_values may be a list of two arrays
        if isinstance(shap_values, list):
            # Use the positive class (phishing = class 1)
            sv_for_plot = shap_values[1]
        else:
            sv_for_plot = shap_values

        # Compute mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(sv_for_plot).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(mean_abs_shap)],
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False)

        # Generate summary plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a DataFrame for the summary plot
        if isinstance(X_sample, np.ndarray):
            X_df = pd.DataFrame(X_sample, columns=feature_names[:X_sample.shape[1]])
        else:
            X_df = pd.DataFrame(X_sample, columns=feature_names)

        # Use bar plot for global importance (more reliable than beeswarm)
        top_n = min(20, len(feature_names))
        top_features = importance_df.head(top_n)

        ax.barh(range(top_n), top_features["importance"].values[::-1],
                color="#2196F3", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[explain] Saved SHAP global plot → {save_path}")

        plt.close(fig)

        return {
            "shap_values": sv_for_plot,
            "expected_value": getattr(explainer, "expected_value", None),
            "feature_importance": importance_df,
            "figure": fig,
            "X_sample": X_sample,
        }

    except Exception as e:
        error_msg = f"SHAP global explanation failed for {model_name}: {str(e)}"
        print(f"[explain] ⚠ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}


def explain_shap_local(model, single_input, feature_names: list,
                       background_data=None,
                       model_name: str = "Model") -> dict:
    """
    Compute SHAP explanation for a single prediction.

    Parameters
    ----------
    model : estimator
        Trained model.
    single_input : array-like
        Single sample to explain (1D or 2D with one row).
    feature_names : list
        Feature names.
    background_data : array-like, optional
        Background data for KernelSHAP (needed for non-tree models).
    model_name : str

    Returns
    -------
    dict
        Contains 'shap_values', 'base_value', 'contributions', 'figure'.
    """
    try:
        import shap
    except ImportError:
        return {"error": "SHAP library not installed."}

    try:
        # Ensure 2D input
        if isinstance(single_input, pd.DataFrame):
            X_input = single_input.values
        elif isinstance(single_input, np.ndarray) and single_input.ndim == 1:
            X_input = single_input.reshape(1, -1)
        else:
            X_input = np.array(single_input).reshape(1, -1)

        model_type = type(model).__name__
        tree_models = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
                       "GradientBoostingClassifier"}

        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
        else:
            if background_data is None:
                return {"error": "Background data required for non-tree model SHAP explanation."}
            bg = shap.sample(background_data, min(50, len(background_data)),
                             random_state=config.RANDOM_SEED)
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        shap_values = explainer.shap_values(X_input)

        # For binary classification
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Positive class, first (only) sample
            base_value = explainer.expected_value[1] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value

        # Build contributions table
        contributions = pd.DataFrame({
            "feature": feature_names[:len(sv)],
            "value": X_input[0][:len(sv)],
            "shap_value": sv,
            "abs_shap": np.abs(sv),
        }).sort_values("abs_shap", ascending=False)

        # Waterfall-style bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(contributions))
        top = contributions.head(top_n)

        colors = ["#E53935" if v > 0 else "#43A047" for v in top["shap_value"]]
        ax.barh(range(top_n), top["shap_value"].values[::-1], color=colors[::-1])
        labels = [f"{row['feature']} = {row['value']:.2f}"
                  for _, row in top.iterrows()]
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels[::-1], fontsize=9)
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
        ax.set_title(f"SHAP Local Explanation — {model_name}", fontsize=13)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.close(fig)

        return {
            "shap_values": sv,
            "base_value": float(base_value) if base_value is not None else None,
            "contributions": contributions,
            "figure": fig,
        }

    except Exception as e:
        error_msg = f"SHAP local explanation failed: {str(e)}"
        print(f"[explain] ⚠ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}


def explain_lime(model, single_input, feature_names: list,
                 training_data=None, model_name: str = "Model",
                 num_features: int = 15) -> dict:
    """
    Generate a LIME explanation for a single prediction.

    LIME creates a local surrogate model around the prediction point
    and shows which features contributed most to the decision.

    Parameters
    ----------
    model : estimator
    single_input : array-like
    feature_names : list
    training_data : array-like
        Training data for LIME's background distribution.
    model_name : str
    num_features : int
        Number of top features to show.

    Returns
    -------
    dict
        Contains 'explanation', 'feature_weights', 'figure'.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return {"error": "LIME library not installed. Install with: pip install lime"}

    try:
        # Ensure proper input shape
        if isinstance(single_input, pd.DataFrame):
            X_input = single_input.values.flatten()
        elif isinstance(single_input, np.ndarray):
            X_input = single_input.flatten()
        else:
            X_input = np.array(single_input).flatten()

        if training_data is None:
            return {"error": "Training data required for LIME explanation."}

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=["Legitimate", "Phishing"],
            mode="classification",
            random_state=config.RANDOM_SEED,
        )

        # Generate explanation
        # Use predict_proba if available for better explanations
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            # Fallback for models without predict_proba
            def predict_fn(X):
                preds = model.predict(X)
                return np.column_stack([1 - preds, preds])

        explanation = explainer.explain_instance(
            X_input,
            predict_fn,
            num_features=num_features,
            num_samples=1000,
        )

        # Extract feature weights
        feature_weights = explanation.as_list()
        weights_df = pd.DataFrame(feature_weights, columns=["feature_rule", "weight"])
        weights_df["abs_weight"] = weights_df["weight"].abs()
        weights_df = weights_df.sort_values("abs_weight", ascending=False)

        # Create a visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        n_show = min(num_features, len(weights_df))
        display_df = weights_df.head(n_show)

        colors = ["#E53935" if w > 0 else "#43A047" for w in display_df["weight"]]
        ax.barh(range(n_show), display_df["weight"].values[::-1],
                color=colors[::-1])
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(display_df["feature_rule"].values[::-1], fontsize=9)
        ax.set_xlabel("Feature Weight", fontsize=11)
        ax.set_title(f"LIME Explanation — {model_name}", fontsize=13)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.close(fig)

        # Get prediction probabilities from the explanation
        pred_proba = explanation.predict_proba

        return {
            "explanation": explanation,
            "feature_weights": weights_df,
            "figure": fig,
            "predict_proba": pred_proba,
        }

    except Exception as e:
        error_msg = f"LIME explanation failed: {str(e)}"
        print(f"[explain] ⚠ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}


def generate_plain_english_explanation(contributions: pd.DataFrame,
                                       prediction_label: str,
                                       top_n: int = 5) -> str:
    """
    Generate a plain-English explanation of what drove the prediction.

    Parameters
    ----------
    contributions : pd.DataFrame
        DataFrame with 'feature', 'value', 'shap_value' (or 'weight') columns.
    prediction_label : str
        The predicted class label (e.g., "phishing" or "legitimate").
    top_n : int
        Number of top features to include.

    Returns
    -------
    str
        Human-readable explanation paragraph.
    """
    disclaimer = (
        "⚠️ Note: These are explanations of the model's reasoning, "
        "not proof of whether the website is actually malicious or safe. "
        "The model's prediction is an estimate based on learned patterns."
    )

    if contributions is None or len(contributions) == 0:
        return f"The model predicted this URL as **{prediction_label}**.\n\n{disclaimer}"

    top_feats = contributions.head(top_n)

    parts = [f"The model predicted this URL as **{prediction_label}**. "
             f"Here are the top factors that influenced this prediction:\n"]

    for i, (_, row) in enumerate(top_feats.iterrows(), 1):
        feature = row.get("feature", row.get("feature_rule", "unknown"))
        value = row.get("value", "")
        shap_val = row.get("shap_value", row.get("weight", 0))

        direction = "toward phishing" if shap_val > 0 else "toward legitimate"
        strength = abs(shap_val)

        if strength > 0.1:
            strength_word = "strongly"
        elif strength > 0.01:
            strength_word = "moderately"
        else:
            strength_word = "slightly"

        if value != "":
            parts.append(f"{i}. **{feature}** (value: {value:.2f}) "
                         f"pushed the prediction {strength_word} {direction}.")
        else:
            parts.append(f"{i}. **{feature}** "
                         f"pushed the prediction {strength_word} {direction}.")

    parts.append(f"\n{disclaimer}")
    return "\n".join(parts)
