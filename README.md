# 🩺 End-to-End Disease Prediction System

A production-ready Machine Learning and Streamlit web application designed for multi-disease prediction (Heart Disease, Diabetes, Parkinson's) using advanced classification algorithms, data preprocessing pipelines, and interactive diagnostic visualizations.

---

## 📌 Features

- **Multi-Disease Diagnostic Suite**:
  - Heart Disease Prediction
  - Diabetes Risk Assessment
  - Parkinson's Disease Detection
- **Robust ML Pipeline**: Automated data loading, feature scaling, missing value imputation, hyperparameter tuning, and cross-validation.
- **Model Ensembling**: Trains and evaluates Logistic Regression, Random Forest, Support Vector Machines (SVM), Gradient Boosting, XGBoost, and LightGBM models.
- **Interactive Web App**: Built with Streamlit for real-time risk predictions, visual feature impact analysis, and PDF report generation.
- **Production Ready**: Modular software design, full type hinting, PEP8 compliance, unit tests, and comprehensive logging.

---

## 📂 Repository Structure

```
Disease_Prediction/
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation & usage guide
├── .gitignore                    # Git ignore file
├── app.py                        # Streamlit web application entrypoint
├── src/                          # Modular Python source package
│   ├── __init__.py
│   ├── config.py                 # Paths, hyperparameter grids & configurations
│   ├── utils.py                  # Helper functions, logging & metric export
│   ├── data_loader.py            # Dataset loading & dynamic synthetic fallback generator
│   ├── preprocessing.py          # Data cleaning, scaling & encoding pipelines
│   ├── visualization.py          # Seaborn/Matplotlib EDA & evaluation plotting
│   ├── train.py                  # Model training & cross-validation runner
│   ├── evaluate.py               # Model evaluation & metric calculation
│   └── predict.py                # Inference pipeline for real-time predictions
├── dataset/                      # CSV dataset files
├── models/                       # Trained model artifacts (.joblib)
├── notebooks/                    # Jupyter notebooks for exploratory analysis
└── outputs/                      # Saved charts, evaluation plots & metrics
```

---

## 🛠️ Quick Start & Installation

### 1. Clone or Open Workspace in VS Code
```bash
git clone https://github.com/your-username/Disease_Prediction.git
cd Disease_Prediction
```

### 2. Create and Activate Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### Train Models & Generate Pipelines
To run data preprocessing, train all classification models, and save evaluation metrics & artifacts:
```bash
python -m src.train
```

### Launch Web Dashboard
To run the interactive Streamlit application:
```bash
streamlit run app.py
```

---

## 🧪 Testing & Quality Assurance
Run unit tests across data preprocessing and inference pipelines:
```bash
pytest tests/
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
