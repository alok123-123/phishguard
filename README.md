# 🛡️ PhishGuard — Explainable Phishing Website Detection

> Using Behavioral and Structural Features with Machine Learning and Explainable AI

---

## ⚠️ Disclaimer

This is an **educational project**. Model predictions are **estimates**, not guarantees of website safety. A "legitimate" classification does not mean a website is safe. Always verify websites through trusted sources before entering sensitive information.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Setup Instructions](#setup-instructions)
6. [Dataset Acquisition](#dataset-acquisition)
7. [Training the Model](#training-the-model)
8. [Running Evaluation](#running-evaluation)
9. [Launching the Application](#launching-the-application)
10. [Running Tests](#running-tests)
11. [Google Colab Instructions](#google-colab-instructions)
12. [Deployment](#deployment)
13. [Git & GitHub Instructions](#git--github-instructions)
14. [Troubleshooting](#troubleshooting)
15. [Limitations & Known Issues](#limitations--known-issues)
16. [References](#references)

---

## Project Overview

**PhishGuard** analyzes a website URL and estimates whether it is likely to be legitimate or phishing. The system:

- ✅ Accepts a URL from the user
- ✅ Extracts 28 structural features from the URL text (without visiting the website)
- ✅ Passes features through a trained machine learning model
- ✅ Displays the predicted class and model confidence
- ✅ Explains the prediction using **SHAP** and **LIME**
- ✅ Shows feature importance and evaluation metrics
- ✅ Clearly communicates that predictions are estimates

---

## Features

| Feature | Description |
|---------|-------------|
| URL Feature Extraction | 28 structural features from URL text (no website visits) |
| 4 ML Models | Random Forest, XGBoost, LightGBM, SVM |
| SHAP Explanations | Global and local feature importance |
| LIME Explanations | Individual prediction explanations |
| Streamlit Dashboard | 6-page cybersecurity-themed web app |
| Automated Tests | 40+ tests for feature extraction, preprocessing, prediction |
| Reproducible Pipeline | Saved models, pipelines, and metadata |

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP, LIME |
| Visualization | matplotlib, seaborn |
| Web Application | Streamlit |
| Testing | pytest |
| Version Control | Git, GitHub |

---

## Project Structure

```
phishguard/
├── app.py                  ← Streamlit web application (main entry point)
├── config.py               ← Central configuration (paths, constants, settings)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Files to exclude from Git
│
├── src/                    ← Core source modules
│   ├── __init__.py
│   ├── data_loader.py      ← Load and inspect datasets
│   ├── preprocessing.py    ← Data cleaning, splitting, pipeline
│   ├── feature_extraction.py ← Extract URL features (28 structural features)
│   ├── train.py            ← Train ML models (RF, XGBoost, LightGBM, SVM)
│   ├── evaluate.py         ← Evaluate models, generate metrics/charts
│   ├── explain.py          ← SHAP and LIME explanations
│   └── predict.py          ← End-to-end URL prediction pipeline
│
├── notebooks/              ← Jupyter notebooks for experimentation
│   └── phishing_detection.ipynb
│
├── models/                 ← Saved trained models and pipelines (git-ignored)
│   └── .gitkeep
│
├── reports/                ← Generated evaluation reports and charts (git-ignored)
│   └── .gitkeep
│
├── data/                   ← Dataset files (git-ignored — too large for Git)
│   └── .gitkeep
│
└── tests/                  ← Automated tests
    ├── __init__.py
    ├── test_feature_extraction.py
    ├── test_preprocessing.py
    └── test_prediction.py
```

---

## Setup Instructions

### Prerequisites

- **Python 3.9 or newer** — [Download here](https://www.python.org/downloads/)
- **Git** (optional, for version control) — [Download here](https://git-scm.com/downloads)

### Step-by-Step Setup (Windows)

1. **Open Command Prompt or PowerShell**

2. **Navigate to the project folder:**
   ```bash
   cd "C:\Users\Alok\Downloads\phising detection"
   ```

3. **Create a virtual environment** (keeps packages isolated):
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   # PowerShell:
   .\venv\Scripts\Activate.ps1

   # Command Prompt:
   venv\Scripts\activate.bat
   ```
   You should see `(venv)` at the start of your prompt.

5. **Install all dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. **Download the dataset** (see next section).

7. **Train the model** (see "Training the Model" section).

8. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

---

## Dataset Acquisition

### Primary Dataset: Hannousse & Yahiouche (2021)

This is the main dataset used for training.

| Property | Value |
|----------|-------|
| **Name** | Web Page Phishing Detection Dataset |
| **Authors** | Hannousse, A. & Yahiouche, S. |
| **Year** | 2021 |
| **Source** | [Mendeley Data](https://data.mendeley.com/datasets/c2gw7fy2j4/3) |
| **License** | CC BY 4.0 |
| **Reported size** | ~11,430 URLs, 87 features |
| **Target column** | `status` (values: `legitimate`, `phishing`) |

**Download instructions:**

1. Visit: https://data.mendeley.com/datasets/c2gw7fy2j4/3
2. Click the **Download** button.
3. Extract the ZIP file.
4. Find the CSV file (e.g., `dataset_phishing.csv`).
5. Copy it to:
   ```
   phishguard/data/dataset_phishing.csv
   ```

> **Note:** If the filename is different, update `PRIMARY_DATASET_FILENAME` in `config.py`.

### Additional Datasets (Optional)

| Dataset | Source | Notes |
|---------|--------|-------|
| UCI Phishing Websites | [UCI Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) | Different feature schema; for cross-dataset evaluation |
| Tan (2018) | Research dataset | 48 Selenium features; requires separate feature mapping |

**Important:** These datasets have different column names and feature definitions. Do NOT combine them with the primary dataset without explicit feature mapping.

---

## Training the Model

**Run from the project root directory** (where `app.py` is):

### Train all models (primary dataset features):
```bash
python -m src.train
```

### Train all models + URL-only model (for live predictions):
```bash
python -m src.train --url-only
```

### Train with a custom dataset path:
```bash
python -m src.train --dataset "path/to/your/dataset.csv" --url-only
```

**What happens during training:**
1. Loads and inspects the dataset
2. Encodes the target labels
3. Removes duplicates
4. Splits into train (70%) / validation (15%) / test (15%)
5. Fits preprocessing pipeline on training data only
6. Trains 4 models: Random Forest, XGBoost, LightGBM, SVM
7. Reports validation accuracy and F1 for each model
8. Saves models, pipelines, and metadata to `models/`
9. (With `--url-only`) Trains a separate URL-feature-only model

**Output files saved to `models/`:**
- `RandomForest_model.joblib`, `XGBoost_model.joblib`, etc.
- `preprocessing_pipeline.joblib`
- `label_encoder.joblib`
- `feature_schema.json`
- `model_metadata.json`
- `URLOnly_model.joblib` (if `--url-only` was used)

---

## Running Evaluation

```bash
python -m src.evaluate
```

This computes metrics on the **held-out test set** and saves:
- Confusion matrices (PNG images)
- ROC curves
- Precision-recall curves
- Model comparison bar chart
- `evaluation_metrics.json`
- Classification reports (text files)

All saved to the `reports/` directory.

---

## Launching the Application

```bash
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501` with:

| Page | Description |
|------|-------------|
| 🏠 Home Dashboard | Project overview, how it works, model info |
| 🔍 URL Analyzer | Enter a URL → get prediction + SHAP/LIME explanation |
| 📊 Model Performance | Test set metrics, confusion matrices, ROC curves |
| 🧠 Explainability | SHAP global importance, SHAP/LIME methodology |
| 📁 Dataset Explorer | Browse dataset, label distribution, missing values |
| ℹ️ About Project | Methodology, limitations, references |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

This runs all automated tests and shows pass/fail status.

Tests include:
- URL validation (valid, invalid, edge cases)
- Feature extraction (28 features, correct types, binary values)
- Preprocessing (encoding, splitting, pipeline, missing values)
- Prediction pipeline (output format, error handling, warnings)

---

## Google Colab Instructions

1. Upload the project files to Google Drive or clone from GitHub.
2. In a Colab notebook, mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install dependencies:
   ```python
   !pip install -r /content/drive/MyDrive/phishguard/requirements.txt
   ```
4. Change to the project directory:
   ```python
   import os
   os.chdir('/content/drive/MyDrive/phishguard')
   ```
5. Upload the dataset to `data/dataset_phishing.csv`.
6. Train models:
   ```python
   !python -m src.train --url-only
   ```
7. Evaluate:
   ```python
   !python -m src.evaluate
   ```
8. To run Streamlit in Colab, use [localtunnel](https://theboroithagain.medium.com/):
   ```python
   !streamlit run app.py &>/content/logs.txt &
   !npx localtunnel --port 8501
   ```

---

## Deployment

### Streamlit Community Cloud (Free)

1. Push your project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your GitHub repo, branch, and set `app.py` as the main file.
5. Click **Deploy**.

**Important:** You need to include the trained model files (`models/` directory) in the repository for deployment, or use a cloud storage service. Since model files can be large, consider using [Git LFS](https://git-lfs.github.com/) or training on deployment.

---

## Git & GitHub Instructions

### Initialize Git (first time only):
```bash
cd "C:\Users\Alok\Downloads\phising detection"
git init
git add .
git commit -m "Initial commit: PhishGuard project"
```

### Create a GitHub repository:
1. Go to [github.com/new](https://github.com/new)
2. Name it `phishguard`
3. Leave it public (or private)
4. Do NOT initialize with README (you already have one)
5. Click **Create repository**

### Push to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/phishguard.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Subsequent commits:
```bash
git add .
git commit -m "Your message describing the changes"
git push
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install xgboost` |
| `ModuleNotFoundError: No module named 'lightgbm'` | Run `pip install lightgbm` |
| `FileNotFoundError: Dataset file not found` | Download the dataset and place it in `data/dataset_phishing.csv` |
| `Model not loaded` in Streamlit | Run `python -m src.train --url-only` first |
| PowerShell execution policy error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| Port 8501 already in use | Add `--server.port 8502` to the streamlit command |
| Tests fail | Ensure you're in the project root directory |
| SHAP/LIME errors | These are optional; the app continues without them |

### Checking Python version:
```bash
python --version
```
Must be 3.9 or higher.

### Checking installed packages:
```bash
python -m pip list
```

---

## Limitations & Known Issues

1. **URL-only features:** The live analyzer extracts structural features from the URL text only. It does NOT visit the website or analyze HTML/JavaScript content.

2. **No real-time threat intelligence:** The model does not query live blacklists, DNS records, or WHOIS databases.

3. **Training data bias:** Model performance depends on the training dataset. Novel phishing techniques may not be detected.

4. **Probability calibration:** Model probabilities may not be perfectly calibrated. The "confidence" score reflects model certainty, not actual risk probability.

5. **Feature gap:** The primary dataset has 87 features; the URL extractor can produce 28. The URL-only model is trained on a reduced feature set.

6. **Single dataset:** Cross-dataset generalization evaluation is not implemented by default because the secondary datasets have incompatible feature schemas.

7. **Not a security product:** This project is for educational purposes only.

---

## References

1. Hannousse, A., & Yahiouche, S. (2021). Web page phishing detection. *Mendeley Data*, V3. DOI: 10.17632/c2gw7fy2j4.3

2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of KDD*.

4. UCI Machine Learning Repository. Phishing Websites Dataset. https://archive.ics.uci.edu/dataset/327/phishing+websites

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*.

---

## License

This project is for educational purposes. The primary dataset is licensed under **CC BY 4.0** (Hannousse & Yahiouche, 2021). Please cite the original authors if you use their dataset.

---

*Built as a college project for demonstrating explainable machine learning in cybersecurity.*
