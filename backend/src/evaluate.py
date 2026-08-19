"""
Evaluation Module for Disease Prediction System.

This module computes classification metrics:
Accuracy, Precision, Recall, F1-Score, and ROC-AUC,
and generates evaluation charts.
"""

from typing import Dict, Any

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from backend.src.utils import logger
from backend.src.visualization import (
    plot_confusion_matrix,
    plot_roc_curve
)


def evaluate_classification_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    disease_name: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluates a trained classification model.

    Args:
        model: Fitted scikit-learn classifier.
        X_test: Scaled test features.
        y_test: True test labels.
        disease_name: Disease name.
        model_name: Model algorithm name.

    Returns:
        Dictionary containing classification metrics.
    """

    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(X_test)


    # ========================================================
    # PROBABILITY / DECISION SCORES
    # ========================================================

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(model, "decision_function"):

        y_prob = model.decision_function(
            X_test
        )

    else:

        y_prob = y_pred


    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    accuracy = float(
        accuracy_score(
            y_test,
            y_pred
        )
    )


    precision = float(
        precision_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    recall = float(
        recall_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    f1 = float(
        f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    try:

        roc_auc = float(
            roc_auc_score(
                y_test,
                y_prob
            )
        )

    except Exception:

        roc_auc = 0.5


    # ========================================================
    # LOG RESULTS
    # ========================================================

    logger.info(
        f"[{disease_name}] {model_name} -> "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, "
        f"AUC: {roc_auc:.4f}"
    )


    # ========================================================
    # GENERATE CONFUSION MATRIX
    # ========================================================

    try:

        plot_confusion_matrix(
            y_test,
            y_pred,
            disease_name,
            model_name,
            save=True
        )

    except Exception as e:

        logger.warning(
            f"Could not generate confusion matrix "
            f"for {model_name}: {e}"
        )


    # ========================================================
    # GENERATE ROC CURVE
    # ========================================================

    try:

        plot_roc_curve(
            y_test,
            y_prob,
            disease_name,
            model_name,
            save=True
        )

    except Exception as e:

        logger.warning(
            f"Could not generate ROC curve "
            f"for {model_name}: {e}"
        )


    # ========================================================
    # RETURN METRICS
    # ========================================================

    return {

        "model_name": model_name,

        "disease_name": disease_name,

        "accuracy": round(
            accuracy,
            4
        ),

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1_score": round(
            f1,
            4
        ),

        "roc_auc": round(
            roc_auc,
            4
        )
    }