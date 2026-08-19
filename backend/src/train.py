"""
Training Module for Disease Prediction System.

This module orchestrates data loading, preprocessing, model hyperparameter
tuning using GridSearchCV, model evaluation, and artifact persistence
across all disease domains.
"""

from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


from config.settings import (
    DISEASE_CONFIGS,
    PARAM_GRIDS,
    RANDOM_STATE
)

from backend.src.data_loader import load_disease_data
from backend.src.preprocessing import prepare_dataset
from backend.src.evaluate import evaluate_classification_model
from backend.src.utils import (
    logger,
    save_artifact,
    save_json
)


# ============================================================
# CANDIDATE MODELS
# ============================================================

def get_candidate_models() -> Dict[
    str,
    Tuple[Any, Dict[str, Any]]
]:
    """
    Constructs candidate classification algorithms and
    their hyperparameter search grids.
    """

    models = {

        "LogisticRegression": (

            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),

            PARAM_GRIDS[
                "LogisticRegression"
            ]
        ),


        "RandomForest": (

            RandomForestClassifier(
                random_state=RANDOM_STATE
            ),

            PARAM_GRIDS[
                "RandomForest"
            ]
        ),


        "SVM": (

            SVC(
                probability=True,
                random_state=RANDOM_STATE
            ),

            PARAM_GRIDS[
                "SVM"
            ]
        ),


        "GradientBoosting": (

            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

            PARAM_GRIDS[
                "GradientBoosting"
            ]
        )
    }


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if HAS_XGBOOST:

        models["XGBoost"] = (

            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            ),

            {
                "n_estimators": [
                    50,
                    100
                ],

                "learning_rate": [
                    0.01,
                    0.1
                ]
            }
        )


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    if HAS_LIGHTGBM:

        models["LightGBM"] = (

            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1
            ),

            {
                "n_estimators": [
                    50,
                    100
                ],

                "learning_rate": [
                    0.01,
                    0.1
                ]
            }
        )


    return models


# ============================================================
# TRAIN ONE DISEASE MODEL
# ============================================================

def train_and_evaluate_disease_model(
    disease_type: str
) -> Dict[str, Any]:
    """
    Runs the complete training and evaluation pipeline
    for one disease domain.
    """

    if disease_type not in DISEASE_CONFIGS:

        raise KeyError(
            f"Invalid disease type '{disease_type}'. "
            f"Valid diseases: "
            f"{list(DISEASE_CONFIGS.keys())}"
        )


    config = DISEASE_CONFIGS[
        disease_type
    ]

    disease_name = config[
        "name"
    ]


    logger.info(
        f"=== Starting Training Pipeline for: "
        f"{disease_name} ==="
    )


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    X, y = load_disease_data(
        disease_type
    )


    # ========================================================
    # 2. PREPROCESS DATA
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_dataset(
        X,
        y,
        disease_type
    )


    # ========================================================
    # 3. GET MODELS
    # ========================================================

    candidate_models = (
        get_candidate_models()
    )


    best_overall_model = None

    best_overall_score = -1.0

    best_model_name = ""

    all_metrics = []


    # ========================================================
    # 4. TRAIN AND TUNE MODELS
    # ========================================================

    for (
        model_name,
        (
            estimator,
            param_grid
        )
    ) in candidate_models.items():

        logger.info(
            f"Tuning {model_name} "
            f"for {disease_name}..."
        )


        grid_search = GridSearchCV(

            estimator=estimator,

            param_grid=param_grid,

            cv=5,

            scoring="f1",

            n_jobs=-1
        )


        grid_search.fit(
            X_train,
            y_train
        )


        best_estimator = (
            grid_search.best_estimator_
        )


        metrics = (
            evaluate_classification_model(
                best_estimator,
                X_test,
                y_test,
                disease_name,
                model_name
            )
        )


        all_metrics.append(
            metrics
        )


        # ----------------------------------------------------
        # Select best model based on F1 score
        # ----------------------------------------------------

        if (
            metrics["f1_score"]
            > best_overall_score
        ):

            best_overall_score = (
                metrics["f1_score"]
            )

            best_overall_model = (
                best_estimator
            )

            best_model_name = (
                model_name
            )


    # ========================================================
    # 5. CHECK BEST MODEL
    # ========================================================

    if best_overall_model is None:

        raise RuntimeError(
            f"No successful model was trained "
            f"for {disease_name}."
        )


    logger.info(
        f"Best model for {disease_name}: "
        f"{best_model_name} "
        f"(F1 Score: "
        f"{best_overall_score:.4f})"
    )


    # ========================================================
    # 6. SAVE BEST MODEL
    # ========================================================

    artifact_filename = (
        f"model_{disease_type}.joblib"
    )


    save_artifact(

        best_overall_model,

        filename=artifact_filename,

        subfolder="models"
    )


    # ========================================================
    # 7. SAVE METRICS
    # ========================================================

    metrics_summary = {

        "disease_type":
            disease_type,

        "disease_name":
            disease_name,

        "best_model":
            best_model_name,

        "best_f1_score":
            best_overall_score,

        "all_candidate_metrics":
            all_metrics
    }


    save_json(

        metrics_summary,

        f"metrics_{disease_type}.json"
    )


    logger.info(
        f"=== Completed Training Pipeline "
        f"for: {disease_name} ==="
    )


    return metrics_summary


# ============================================================
# TRAIN ALL DISEASE MODELS
# ============================================================

def run_full_training_pipeline() -> Dict[str, Any]:
    """
    Executes training pipelines across all configured
    disease datasets.
    """

    results = {}


    for disease_type in (
        DISEASE_CONFIGS.keys()
    ):

        results[disease_type] = (
            train_and_evaluate_disease_model(
                disease_type
            )
        )


    logger.info(
        "=== All Disease Training Pipelines "
        "Completed Successfully ==="
    )


    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_full_training_pipeline()