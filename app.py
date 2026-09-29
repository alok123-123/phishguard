"""
app.py — PhishGuard: Explainable Phishing Website Detection

This is the main Streamlit application file. It creates a polished
cybersecurity dashboard with the following pages:

    1. Home Dashboard     — Project overview and how it works
    2. URL Analyzer       — Analyze a URL for phishing indicators
    3. Model Performance  — Evaluation metrics and charts
    4. Explainability     — SHAP and LIME explanations
    5. Dataset Explorer   — Explore the training dataset
    6. About Project      — Methodology, limitations, references

Launch with:
    streamlit run app.py

Author: PhishGuard Project
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import warnings
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title=f"{config.APP_TITLE} — {config.APP_SUBTITLE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
# CUSTOM CSS — Modern cybersecurity aesthetic
# ═══════════════════════════════════════════════════════════
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #121e36 50%, #0d1a30 100%);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] span {
        color: #c8d6e5 !important;
    }

    /* ─── Metric cards ─── */
    .metric-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
        border: 1px solid #2d3b4e;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #60a5fa;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ─── Result cards ─── */
    .result-safe {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }
    .result-danger {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
    }

    /* ─── Section headers ─── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #3b82f6;
        display: inline-block;
    }

    /* ─── Warning banner ─── */
    .warning-banner {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #fef3c7;
        font-size: 0.9rem;
    }

    /* ─── Feature table ─── */
    .feature-table {
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 32px 0;
        font-size: 0.8rem;
        border-top: 1px solid #1e293b;
        margin-top: 48px;
    }

    /* ─── Hide Streamlit default elements ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_predictor():
    """Load the prediction model (cached across reruns)."""
    try:
        from src.predict import PhishingPredictor
        predictor = PhishingPredictor()
        return predictor
    except Exception as e:
        st.error(f"Failed to load prediction model: {e}")
        return None


@st.cache_data
def load_evaluation_metrics():
    """Load saved evaluation metrics."""
    metrics_path = os.path.join(config.REPORTS_DIR, "evaluation_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_metadata():
    """Load saved model metadata."""
    if os.path.isfile(config.SAVED_METADATA_PATH):
        with open(config.SAVED_METADATA_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_cached():
    """Load the primary dataset (cached)."""
    try:
        from src.data_loader import load_primary_dataset
        return load_primary_dataset()
    except Exception:
        return None


def render_metric_card(value, label, color="#60a5fa"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
def render_sidebar():
    """Render the sidebar with navigation and project info."""
    with st.sidebar:
        st.markdown(f"# {config.APP_ICON} {config.APP_TITLE}")
        st.markdown(f"*{config.APP_SUBTITLE}*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠 Home Dashboard",
                "🔍 URL Analyzer",
                "📊 Model Performance",
                "🧠 Explainability",
                "📁 Dataset Explorer",
                "ℹ️ About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model status indicator
        predictor = load_predictor()
        if predictor and predictor.is_loaded:
            st.success("✓ Model Loaded", icon="🟢")
            info = predictor.get_model_info()
            st.caption(f"Type: {info['model_type']}")
            st.caption(f"Features: {info['n_features']}")
        else:
            st.error("✗ Model Not Loaded", icon="🔴")
            st.caption("Run training first")

        st.markdown("---")
        st.caption("PhishGuard v1.0")
        st.caption("Educational Project")

        return page


# ═══════════════════════════════════════════════════════════
# PAGE: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════
def page_home():
    """Render the home dashboard page."""
    st.markdown(f"# {config.APP_ICON} {config.APP_TITLE}")
    st.markdown(f"### {config.APP_SUBTITLE}")
    st.markdown("---")

    # Warning banner
    st.markdown("""
    <div class="warning-banner">
        ⚠️ <strong>Important Disclaimer:</strong> This tool provides model-based estimates,
        NOT guarantees of website safety. A "legitimate" classification does not mean a
        website is safe. Always verify through trusted sources before entering sensitive information.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Overview metrics
    metadata = load_model_metadata()
    metrics = load_evaluation_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_features = metadata.get("n_features", "—") if metadata else "—"
        render_metric_card(n_features, "Model Features", "#60a5fa")
    with col2:
        n_train = metadata.get("n_train_samples", "—") if metadata else "—"
        render_metric_card(n_train, "Training Samples", "#34d399")
    with col3:
        n_models = sum(1 for m in (metadata or {}).get("models_trained", {}).values()
                       if not m.get("skipped", True)) if metadata else "—"
        render_metric_card(n_models, "Models Trained", "#f59e0b")
    with col4:
        if metrics:
            best_f1 = max(m.get("f1_score", 0) for m in metrics.values())
            render_metric_card(f"{best_f1:.1%}", "Best F1 Score", "#a78bfa")
        else:
            render_metric_card("—", "Best F1 Score", "#a78bfa")

    st.markdown("")
    st.markdown("---")

    # How it works
    st.markdown("## How It Works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣ Input URL")
        st.markdown("Enter a website URL you want to analyze.")

    with col2:
        st.markdown("### 2️⃣ Extract Features")
        st.markdown(
            "The system extracts structural features from the URL "
            "(length, special characters, domain patterns, etc.)."
        )

    with col3:
        st.markdown("### 3️⃣ ML Prediction")
        st.markdown(
            "A trained Random Forest model analyzes the features "
            "and predicts whether the URL appears legitimate or phishing."
        )

    with col4:
        st.markdown("### 4️⃣ Explain Results")
        st.markdown(
            "SHAP and LIME explain which features influenced the "
            "prediction, helping you understand the model's reasoning."
        )

    st.markdown("---")

    # Dataset & Model info
    st.markdown("## Training Data")
    st.info(
        "**Primary Dataset:** Hannousse & Yahiouche (2021) — "
        "Web Page Phishing Detection Dataset\n\n"
        "- Source: Mendeley Data (CC BY 4.0 license)\n"
        "- Reported: ~11,430 URLs with 87+ features\n"
        "- Target: Binary classification (legitimate vs. phishing)\n\n"
        "The URL-only model uses structural features extractable from the URL text. "
        "Features requiring website visits (HTML content, page rank, etc.) are not "
        "used in live prediction to ensure safety."
    )


# ═══════════════════════════════════════════════════════════
# PAGE: URL ANALYZER
# ═══════════════════════════════════════════════════════════
def page_url_analyzer():
    """Render the URL analysis page."""
    st.markdown("# 🔍 URL Analyzer")
    st.markdown("Enter a URL to analyze it for potential phishing indicators.")
    st.markdown("---")

    # Warning
    st.markdown("""
    <div class="warning-banner">
        ⚠️ This tool analyzes URL structure only. It does NOT visit the website,
        download content, or execute any code. Predictions are model estimates,
        not security guarantees.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Load predictor
    predictor = load_predictor()
    if not predictor or not predictor.is_loaded:
        st.error(
            "⚠️ **Model not loaded.** Please train the model first:\n\n"
            "```bash\n"
            "python -m src.train --url-only\n"
            "```"
        )
        return

    # URL input
    url_input = st.text_input(
        "🌐 Enter URL to analyze",
        placeholder="https://example.com/login?user=test",
        help="Enter the full URL including http:// or https://",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze URL", type="primary",
                                    use_container_width=True)

    if analyze_clicked and url_input:
        with st.spinner("Analyzing URL..."):
            result = predictor.predict(url_input)

        if result["error"]:
            st.error(f"❌ **Error:** {result['error']}")
            return

        # ─── Prediction Result ───
        prediction = result["prediction"]
        confidence = result.get("confidence")
        phishing_prob = result.get("phishing_probability")

        if prediction == "phishing":
            st.markdown(f"""
            <div class="result-danger">
                <h2 style="color: #fca5a5; margin: 0;">⚠️ Potential Phishing Detected</h2>
                <p style="color: #fecaca; font-size: 1.1rem; margin: 8px 0 0 0;">
                    The model classifies this URL as <strong>likely phishing</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2 style="color: #6ee7b7; margin: 0;">✓ Appears Legitimate</h2>
                <p style="color: #a7f3d0; font-size: 1.1rem; margin: 8px 0 0 0;">
                    The model classifies this URL as <strong>likely legitimate</strong>.
                    This does NOT guarantee the website is safe.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction.upper())
        with col2:
            if confidence is not None:
                st.metric("Model Confidence", f"{confidence:.1%}")
            else:
                st.metric("Model Confidence", "N/A")
        with col3:
            if phishing_prob is not None:
                st.metric("Phishing Probability", f"{phishing_prob:.1%}")
            else:
                st.metric("Phishing Probability", "N/A")

        st.warning(result["warning"])

        # ─── Extracted Features ───
        st.markdown("### 📋 Extracted URL Features")
        if result["features"]:
            features_df = pd.DataFrame(
                list(result["features"].items()),
                columns=["Feature", "Value"]
            )
            st.dataframe(features_df, use_container_width=True, hide_index=True)

        # ─── Explanations ───
        st.markdown("### 🧠 Prediction Explanations")

        # SHAP local explanation
        with st.expander("SHAP Explanation", expanded=True):
            try:
                from src.explain import explain_shap_local, generate_plain_english_explanation
                shap_result = explain_shap_local(
                    predictor.model,
                    result["preprocessed_input"],
                    predictor.feature_names,
                    model_name="URL-Only Model",
                )
                if "error" in shap_result:
                    st.warning(f"SHAP explanation unavailable: {shap_result['error']}")
                else:
                    if shap_result.get("figure"):
                        st.pyplot(shap_result["figure"])
                    if shap_result.get("contributions") is not None:
                        explanation_text = generate_plain_english_explanation(
                            shap_result["contributions"], prediction
                        )
                        st.markdown(explanation_text)
            except Exception as e:
                st.warning(f"SHAP explanation error: {str(e)}")

        # LIME explanation
        with st.expander("LIME Explanation", expanded=False):
            try:
                from src.explain import explain_lime

                # We need training data for LIME — try to load it
                try:
                    from src.data_loader import load_primary_dataset
                    from src.preprocessing import preprocess_dataset
                    df = load_dataset_cached()
                    if df is not None:
                        data = preprocess_dataset(df)
                        training_data = data["X_train"]
                    else:
                        training_data = None
                except Exception:
                    training_data = None

                if training_data is not None:
                    lime_result = explain_lime(
                        predictor.model,
                        result["preprocessed_input"],
                        predictor.feature_names,
                        training_data=training_data,
                        model_name="URL-Only Model",
                    )
                    if "error" in lime_result:
                        st.warning(f"LIME unavailable: {lime_result['error']}")
                    else:
                        if lime_result.get("figure"):
                            st.pyplot(lime_result["figure"])
                        if lime_result.get("feature_weights") is not None:
                            st.markdown("**Feature contributions:**")
                            st.dataframe(
                                lime_result["feature_weights"][["feature_rule", "weight"]],
                                use_container_width=True, hide_index=True,
                            )
                else:
                    st.info(
                        "LIME requires training data. Please ensure the dataset "
                        "is available in the data/ directory."
                    )
            except Exception as e:
                st.warning(f"LIME explanation error: {str(e)}")

    elif analyze_clicked and not url_input:
        st.warning("Please enter a URL to analyze.")


# ═══════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
def page_model_performance():
    """Render model performance evaluation page."""
    st.markdown("# 📊 Model Performance")
    st.markdown("Evaluation metrics computed on the **held-out test set** "
                "(never seen during training or tuning).")
    st.markdown("---")

    metrics = load_evaluation_metrics()
    metadata = load_model_metadata()

    if not metrics:
        st.warning(
            "⚠️ No evaluation results found. Run evaluation first:\n\n"
            "```bash\n"
            "python -m src.evaluate\n"
            "```"
        )
        return

    # ─── Summary Cards ───
    st.markdown("### Model Comparison")
    cols = st.columns(len(metrics))
    for i, (model_name, m) in enumerate(metrics.items()):
        with cols[i]:
            render_metric_card(f"{m['f1_score']:.1%}", f"{model_name} F1",
                               "#60a5fa" if i == 0 else "#a78bfa")

    # ─── Detailed Metrics Table ───
    st.markdown("### Detailed Metrics")
    metrics_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1 Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 'N/A')}",
        }
        for name, m in metrics.items()
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ─── Confusion Matrices ───
    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(min(len(metrics), 4))
    for i, (model_name, m) in enumerate(metrics.items()):
        with cm_cols[i % 4]:
            cm_path = os.path.join(config.REPORTS_DIR,
                                   f"confusion_matrix_{model_name}.png")
            if os.path.isfile(cm_path):
                st.image(cm_path, caption=model_name, use_container_width=True)
            elif "confusion_matrix" in m:
                fig, ax = plt.subplots(figsize=(5, 4))
                cm = np.array(m["confusion_matrix"])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Legit", "Phish"],
                            yticklabels=["Legit", "Phish"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(model_name)
                st.pyplot(fig)
                plt.close(fig)

    # ─── Comparison Charts ───
    st.markdown("### Comparison Charts")
    col1, col2 = st.columns(2)

    with col1:
        roc_path = os.path.join(config.REPORTS_DIR, "roc_curves.png")
        if os.path.isfile(roc_path):
            st.image(roc_path, caption="ROC Curves", use_container_width=True)
        else:
            st.info("ROC curves not yet generated. Run evaluation.")

    with col2:
        comp_path = os.path.join(config.REPORTS_DIR, "model_comparison.png")
        if os.path.isfile(comp_path):
            st.image(comp_path, caption="Model Comparison", use_container_width=True)
        else:
            st.info("Comparison chart not yet generated. Run evaluation.")

    # PR curves
    pr_path = os.path.join(config.REPORTS_DIR, "precision_recall_curves.png")
    if os.path.isfile(pr_path):
        st.markdown("### Precision-Recall Curves")
        st.image(pr_path, caption="Precision-Recall Curves",
                 use_container_width=True)

    # ─── Classification Reports ───
    st.markdown("### Classification Reports")
    for model_name in metrics:
        report_path = os.path.join(config.REPORTS_DIR,
                                   f"classification_report_{model_name}.txt")
        if os.path.isfile(report_path):
            with st.expander(f"📋 {model_name} Classification Report"):
                with open(report_path, "r") as f:
                    st.text(f.read())

    # ─── Evaluation methodology note ───
    st.markdown("### Evaluation Methodology")
    if metadata:
        st.info(
            f"**Dataset split:**\n"
            f"- Training: {metadata.get('n_train_samples', '?')} samples\n"
            f"- Validation: {metadata.get('n_val_samples', '?')} samples\n"
            f"- Test: {metadata.get('n_test_samples', '?')} samples\n"
            f"- Random seed: {metadata.get('random_seed', '?')}\n\n"
            f"All metrics shown above are from the **test set** evaluation. "
            f"The test set was held out during training and hyperparameter selection."
        )


# ═══════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
def page_explainability():
    """Render the explainability page with SHAP and LIME."""
    st.markdown("# 🧠 Explainability")
    st.markdown(
        "Understanding **why** the model makes its predictions using "
        "SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable "
        "Model-agnostic Explanations)."
    )
    st.markdown("---")

    st.markdown("""
    <div class="warning-banner">
        ⚠️ Feature contributions explain the <strong>model's reasoning</strong>, not
        whether a feature is objectively suspicious. A feature pushing the prediction
        toward "phishing" means the model learned that pattern from training data,
        not that it is proof of malicious intent.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # SHAP Global
    st.markdown("### SHAP Global Feature Importance")
    st.markdown(
        "Shows which features the model relies on most across all predictions."
    )

    shap_path = os.path.join(config.REPORTS_DIR, "shap_global_importance.png")
    if os.path.isfile(shap_path):
        st.image(shap_path, caption="SHAP Global Feature Importance",
                 use_container_width=True)
    else:
        if st.button("Generate SHAP Global Importance", type="primary"):
            with st.spinner("Computing SHAP values (this may take a minute)..."):
                try:
                    from src.explain import explain_shap_global
                    import joblib

                    # Load the best model (try RandomForest first)
                    model_path = config.SAVED_MODEL_TEMPLATE.format(
                        model_name="RandomForest"
                    )
                    if os.path.isfile(model_path):
                        model = joblib.load(model_path)
                    else:
                        st.warning("No saved model found.")
                        return

                    # Load preprocessed data
                    df = load_dataset_cached()
                    if df is None:
                        st.warning("Dataset not available.")
                        return

                    from src.preprocessing import preprocess_dataset
                    data = preprocess_dataset(df)

                    result = explain_shap_global(
                        model, data["X_test"],
                        data["feature_names"],
                        model_name="RandomForest",
                        max_samples=300,
                        save_path=shap_path,
                    )

                    if "error" in result:
                        st.warning(f"SHAP error: {result['error']}")
                    else:
                        if result.get("figure"):
                            st.pyplot(result["figure"])
                        if result.get("feature_importance") is not None:
                            st.markdown("**Top 20 Features by Mean |SHAP Value|:**")
                            st.dataframe(
                                result["feature_importance"].head(20),
                                use_container_width=True, hide_index=True,
                            )
                        st.success("SHAP global importance saved. Reload to see it.")

                except Exception as e:
                    st.error(f"SHAP computation failed: {str(e)}")
                    st.text(traceback.format_exc())

    # SHAP explanation
    st.markdown("---")
    st.markdown("### What do SHAP values mean?")
    st.markdown("""
    - **SHAP values** measure how much each feature contributed to a specific prediction.
    - A **positive SHAP value** pushes the prediction toward the "phishing" class.
    - A **negative SHAP value** pushes the prediction toward the "legitimate" class.
    - The **magnitude** indicates how strong the contribution is.
    - SHAP is based on cooperative game theory (Shapley values) and provides
      theoretically sound feature attributions.
    """)

    st.markdown("### What does LIME show?")
    st.markdown("""
    - **LIME** creates a simple, interpretable model around a single prediction.
    - It perturbs the input features and observes how predictions change.
    - The resulting feature weights show local, per-prediction importance.
    - LIME explanations are approximate — they explain a local neighborhood,
      not the entire model.
    """)


# ═══════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════
def page_dataset_explorer():
    """Render the dataset exploration page."""
    st.markdown("# 📁 Dataset Explorer")
    st.markdown("Explore the primary training dataset.")
    st.markdown("---")

    df = load_dataset_cached()
    if df is None:
        st.warning(
            "⚠️ Dataset not found. Please download the dataset and place it at:\n\n"
            f"```\n{config.PRIMARY_DATASET_PATH}\n```\n\n"
            "See the README for download instructions."
        )

        # Allow CSV upload
        uploaded = st.file_uploader("Or upload a CSV file:", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded uploaded file: {uploaded.name}")
        else:
            return

    # ─── Overview ───
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(f"{df.shape[0]:,}", "Rows")
    with col2:
        render_metric_card(df.shape[1], "Columns")
    with col3:
        render_metric_card(int(df.isnull().sum().sum()), "Missing Values")
    with col4:
        render_metric_card(int(df.duplicated().sum()), "Duplicate Rows")

    # ─── Label Distribution ───
    if config.PRIMARY_TARGET_COLUMN in df.columns:
        st.markdown("### Label Distribution")
        label_counts = df[config.PRIMARY_TARGET_COLUMN].value_counts()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                label_counts.reset_index().rename(
                    columns={"index": "Label", config.PRIMARY_TARGET_COLUMN: "Count"}
                ),
                use_container_width=True, hide_index=True,
            )
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#43A047", "#E53935"]
            label_counts.plot(kind="bar", color=colors[:len(label_counts)], ax=ax)
            ax.set_title("Class Distribution", fontsize=14)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_xlabel("")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ─── Data Types ───
    st.markdown("### Feature Data Types")
    dtype_counts = df.dtypes.astype(str).value_counts()
    dtype_df = pd.DataFrame({
        "Data Type": dtype_counts.index.astype(str),
        "Count": dtype_counts.values,
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ─── Missing Values ───
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        st.markdown("### Columns with Missing Values")
        missing_df = pd.DataFrame({
            "Column": missing_cols.index,
            "Missing Count": missing_cols.values,
            "Missing %": (100 * missing_cols.values / len(df)).round(2),
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("✓ No missing values in the dataset.")

    # ─── Sample Data ───
    st.markdown("### Sample Data (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # ─── Feature Statistics ───
    st.markdown("### Feature Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(
            df[numeric_cols].describe().round(3).T,
            use_container_width=True,
        )

    # ─── Column List ───
    with st.expander("📝 All Column Names"):
        for i, col in enumerate(df.columns):
            st.text(f"{i+1}. {col} ({df[col].dtype})")


# ═══════════════════════════════════════════════════════════
# PAGE: ABOUT PROJECT
# ═══════════════════════════════════════════════════════════
def page_about():
    """Render the About page with project details."""
    st.markdown("# ℹ️ About PhishGuard")
    st.markdown("---")

    st.markdown("## Problem Statement")
    st.markdown("""
    Phishing websites impersonate legitimate sites to steal sensitive information
    like passwords, credit card numbers, and personal data. Manual detection is
    slow and error-prone. Machine learning can help automate detection, but models
    must be **transparent** and **explainable** — users need to understand *why*
    a URL is flagged, not just *whether* it is.
    """)

    st.markdown("## Objectives")
    st.markdown("""
    1. Build an ML system that classifies URLs as legitimate or phishing.
    2. Use structural URL features that can be extracted without visiting the website.
    3. Compare multiple ML algorithms (Random Forest, XGBoost, LightGBM, SVM).
    4. Provide explainable predictions using SHAP and LIME.
    5. Deploy as an interactive web application.
    """)

    st.markdown("## Methodology")
    st.markdown("""
    1. **Data Collection**: Primary dataset from Hannousse & Yahiouche (2021).
    2. **Preprocessing**: Missing value imputation, feature encoding, train/val/test split.
    3. **Feature Extraction**: 28 structural URL features extracted from URL text.
    4. **Model Training**: Four candidate models trained and compared.
    5. **Evaluation**: Accuracy, precision, recall, F1, ROC-AUC on held-out test set.
    6. **Explainability**: SHAP for global/local feature importance; LIME for local explanations.
    7. **Deployment**: Streamlit web application.
    """)

    st.markdown("## Dataset Sources")
    st.markdown("""
    | Dataset | Source | Description |
    |---------|--------|-------------|
    | Primary (Hannousse & Yahiouche, 2021) | [Mendeley Data](https://data.mendeley.com/datasets/c2gw7fy2j4/3) | ~11,430 URLs, 87 features, CC BY 4.0 |
    | UCI Phishing Websites | [UCI ML Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) | 11,055 instances, 30 features |
    | Tan (2018) | Research dataset | 10,000 URLs, 48 Selenium-extracted features |

    **Note:** The live URL analyzer uses only URL-structural features. Dataset features
    requiring HTML content or external services are used only in offline model training.
    """)

    st.markdown("## Algorithms Used")
    st.markdown("""
    - **Random Forest**: Ensemble of decision trees; robust, handles non-linear relationships.
    - **XGBoost**: Gradient boosted trees; high performance on tabular data.
    - **LightGBM**: Optimized gradient boosting; fast training on large datasets.
    - **SVM (RBF kernel)**: Finds optimal decision boundary in high-dimensional space.
    """)

    st.markdown("## Limitations")
    st.warning("""
    1. **URL-only features**: The live analyzer uses only URL structural features,
       not HTML content, JavaScript behavior, or external reputation services.
    2. **No real-time threat intelligence**: The model does not check live blacklists.
    3. **Training data bias**: The model's accuracy depends on the training dataset;
       new phishing techniques may not be represented.
    4. **Not a security product**: This is an educational project, not a replacement
       for browser security, antivirus software, or enterprise security tools.
    5. **Model confidence ≠ actual risk**: A high confidence score reflects model
       certainty, not the actual probability of a website being dangerous.
    """)

    st.markdown("## Future Enhancements")
    st.markdown("""
    - Integration with URL reputation APIs (e.g., VirusTotal, Google Safe Browsing).
    - Safe HTML content analysis in a sandboxed environment.
    - Real-time model retraining with new phishing examples.
    - Browser extension for on-the-fly URL checking.
    - Support for multi-class classification (e.g., phishing, malware, spam, legitimate).
    """)

    st.markdown("## References")
    st.markdown("""
    1. Hannousse, A., & Yahiouche, S. (2021). Web page phishing detection.
       *Mendeley Data*, V3. DOI: 10.17632/c2gw7fy2j4.3
    2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
       model predictions. *NeurIPS*.
    3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?"
       Explaining the predictions of any classifier. *KDD*.
    4. UCI Machine Learning Repository — Phishing Websites Dataset.
    """)


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════
def main():
    load_css()
    page = render_sidebar()

    if "Home" in page:
        page_home()
    elif "URL Analyzer" in page:
        page_url_analyzer()
    elif "Model Performance" in page:
        page_model_performance()
    elif "Explainability" in page:
        page_explainability()
    elif "Dataset Explorer" in page:
        page_dataset_explorer()
    elif "About" in page:
        page_about()

    # Footer
    st.markdown("""
    <div class="footer">
        PhishGuard — Explainable Phishing Website Detection<br>
        Educational Project | Model predictions are estimates, not guarantees
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
