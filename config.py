"""
config.py — Central configuration for the PhishGuard project.

This file stores all paths, constants, random seeds, and model settings
used across the project. Changing a value here changes it everywhere.
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

# ─── Project root directory (auto-detected) ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Directory paths ───
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# ─── Random seed for reproducibility ───
# Using the same seed everywhere ensures you get the same results each time.
RANDOM_SEED = 42

# ─── Dataset configuration ───
# Primary dataset: Hannousse & Yahiouche (2021) Web Page Phishing Detection
# Source: https://data.mendeley.com/datasets/c2gw7fy2j4/3
# License: CC BY 4.0
# The dataset has ~11,430 rows and 87 features + 1 label column.
# The target column is named "status" with values: "legitimate" and "phishing".
PRIMARY_DATASET_FILENAME = "dataset_phishing.csv"
PRIMARY_DATASET_PATH = os.path.join(DATA_DIR, PRIMARY_DATASET_FILENAME)
PRIMARY_TARGET_COLUMN = "status"  # Documented label column name
PRIMARY_LABEL_MAPPING = {"legitimate": 0, "phishing": 1}

# ─── Model configuration ───
# We train these four candidate models and compare them.
MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM", "SVM"]
DEFAULT_MODEL = "RandomForest"  # Baseline model

# Train/validation/test split ratios
TEST_SIZE = 0.15       # 15% held out for final testing — never touched during training
VALIDATION_SIZE = 0.15  # 15% of the remaining 85% used for validation/tuning

# Cross-validation folds (used during hyperparameter tuning)
CV_FOLDS = 5

# ─── Saved model and pipeline file names ───
SAVED_MODEL_TEMPLATE = os.path.join(MODELS_DIR, "{model_name}_model.joblib")
SAVED_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
SAVED_FEATURE_SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")
SAVED_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
SAVED_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# ─── Feature extraction settings ───
# Maximum URL length we accept (security measure against absurdly long inputs)
MAX_URL_LENGTH = 2048

# Known URL shortening domains (documented, commonly referenced list)
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
    "bl.ink", "lnkd.in", "rb.gy",
}

# ─── Streamlit display settings ───
APP_TITLE = "PhishGuard"
APP_SUBTITLE = "Explainable Phishing Website Detection"
APP_ICON = "🛡️"

# ─── Ensure required directories exist ───
for _dir in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
