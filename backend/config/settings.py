from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DIRECTORIES
# ============================================================

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"


# ============================================================
# DATABASE
# ============================================================

DATABASE_PATH = (
    BASE_DIR
    / "database"
    / "disease_prediction.db"
)


# ============================================================
# MACHINE LEARNING SETTINGS
# ============================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# MODEL PARAMETERS
# ============================================================

PARAM_GRIDS = {

    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0]
    },

    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    },

    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"]
    },

    "GradientBoosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5]
    }
}


# ============================================================
# DISEASE CONFIGURATION
# ============================================================

DISEASE_CONFIGS = {

    "heart": {

        "name": "Heart Disease",

        "file_name": "heart.csv",

        "target_col": "target",

        "features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    },

    "diabetes": {

        "name": "Diabetes",

        "file_name": "diabetes.csv",

        "target_col": "Outcome",

        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    },

    "kidney": {

        "name": "Chronic Kidney Disease",

        "file_name": "kidney.csv",

        "target_col": "target",

        "features": [
            "age",
            "bp",
            "sg",
            "al",
            "su",
            "bgr",
            "bu",
            "sc",
            "sod",
            "pot",
            "hemo",
            "pcv",
            "wbcc",
            "rbcc",
            "htn",
            "dm"
        ]
    }
}