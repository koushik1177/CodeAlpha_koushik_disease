"""
Central Configuration Module for Disease Prediction System.

This module defines directory paths, disease-specific feature schemas,
hyperparameter search grids, and default random seeds for model reproducibility.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# ==========================================
# BASE PATH CONFIGURATIONS
# ==========================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure directories exist
for path in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# ==========================================
# DISEASE DATASET CONFIGURATIONS
# ==========================================
DISEASE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "heart": {
        "name": "Heart Disease",
        "file_name": "heart.csv",
        "target_col": "target",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "feature_names": {
            "age": "Age (years)",
            "sex": "Sex (1 = Male, 0 = Female)",
            "cp": "Chest Pain Type (0-3)",
            "trestbps": "Resting Blood Pressure (mm Hg)",
            "chol": "Serum Cholestoral (mg/dl)",
            "fbs": "Fasting Blood Sugar > 120 mg/dl (1/0)",
            "restecg": "Resting Electrocardiographic Results (0-2)",
            "thalach": "Maximum Heart Rate Achieved",
            "exang": "Exercise Induced Angina (1/0)",
            "oldpeak": "ST Depression Induced by Exercise",
            "slope": "Slope of Peak Exercise ST Segment (0-2)",
            "ca": "Number of Major Vessels (0-4)",
            "thal": "Thalassemia (1 = normal, 2 = fixed, 3 = reversable)"
        }
    },
    "diabetes": {
        "name": "Diabetes Risk",
        "file_name": "diabetes.csv",
        "target_col": "Outcome",
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "feature_names": {
            "Pregnancies": "Number of Pregnancies",
            "Glucose": "Glucose Level (mg/dL)",
            "BloodPressure": "Blood Pressure (mm Hg)",
            "SkinThickness": "Skin Thickness (mm)",
            "Insulin": "Insulin Level (iu/mL)",
            "BMI": "Body Mass Index (weight in kg/(height in m)^2)",
            "DiabetesPedigreeFunction": "Diabetes Pedigree Function Score",
            "Age": "Age (years)"
        }
    },
    "kidney": {
        "name": "Chronic Kidney Disease",
        "file_name": "kidney.csv",
        "target_col": "classification",
        "features": [
            "age", "bp", "sg", "al", "su", "bgr", "bu",
            "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm"
        ],
        "feature_names": {
            "age": "Age (years)",
            "bp": "Blood Pressure (mm Hg)",
            "sg": "Specific Gravity (1.005 - 1.025)",
            "al": "Albumin (0-5)",
            "su": "Sugar Level (0-5)",
            "bgr": "Blood Glucose Random (mg/dL)",
            "bu": "Blood Urea (mg/dL)",
            "sc": "Serum Creatinine (mg/dL)",
            "sod": "Sodium (mEq/L)",
            "pot": "Potassium (mEq/L)",
            "hemo": "Hemoglobin (g/dL)",
            "pcv": "Packed Cell Volume (%)",
            "wbcc": "White Blood Cell Count (cells/cumm)",
            "rbcc": "Red Blood Cell Count (millions/cumm)",
            "htn": "Hypertension (1 = Yes, 0 = No)",
            "dm": "Diabetes Mellitus (1 = Yes, 0 = No)"
        }
    }
}

# ==========================================
# HYPERPARAMETER GRIDS FOR ML MODELS
# ==========================================
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "GradientBoosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5]
    }
}
