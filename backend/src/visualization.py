"""
Visualization Module for Disease Prediction System.

This module generates and saves diagnostic charts including confusion matrices,
ROC curves, feature importances, and multi-model metric comparison charts.
"""

from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

from config.settings import OUTPUTS_DIR
from backend.src.utils import logger


# ============================================================
# VISUALIZATION SETUP
# ============================================================

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sns.set_theme(
    style="whitegrid"
)

plt.rcParams.update(
    {
        "font.size": 11,
        "figure.autolayout": True
    }
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disease_name: str,
    model_name: str,
    save: bool = True
) -> plt.Figure:
    """
    Plots and optionally saves a confusion matrix heatmap.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Negative (0)",
            "Positive (1)"
        ],
        yticklabels=[
            "Negative (0)",
            "Positive (1)"
        ],
        ax=ax
    )

    ax.set_title(
        f"Confusion Matrix - "
        f"{disease_name}\n"
        f"({model_name})",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Label",
        fontweight="bold"
    )


    if save:

        filename = (
            f"cm_"
            f"{disease_name.lower().replace(' ', '_')}_"
            f"{model_name.lower()}.png"
        )

        filepath = (
            OUTPUTS_DIR /
            filename
        )

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            f"Saved confusion matrix plot to: "
            f"{filepath}"
        )


    plt.close(fig)

    return fig


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_name: str,
    model_name: str,
    save: bool = True
) -> plt.Figure:
    """
    Plots and optionally saves a ROC curve.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.plot(
        fpr,
        tpr,
        color="#2563EB",
        lw=2,
        label=(
            f"ROC Curve "
            f"(AUC = {roc_auc:.3f})"
        )
    )


    ax.plot(
        [0, 1],
        [0, 1],
        color="#9CA3AF",
        lw=1.5,
        linestyle="--",
        label="Random Classifier"
    )


    ax.set_xlim(
        [0.0, 1.0]
    )

    ax.set_ylim(
        [0.0, 1.05]
    )


    ax.set_xlabel(
        "False Positive Rate "
        "(1 - Specificity)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Positive Rate "
        "(Sensitivity)",
        fontweight="bold"
    )


    ax.set_title(
        f"ROC Curve - "
        f"{disease_name}\n"
        f"({model_name})",
        fontsize=13,
        fontweight="bold"
    )


    ax.legend(
        loc="lower right"
    )


    if save:

        filename = (
            f"roc_"
            f"{disease_name.lower().replace(' ', '_')}_"
            f"{model_name.lower()}.png"
        )

        filepath = (
            OUTPUTS_DIR /
            filename
        )

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            f"Saved ROC curve plot to: "
            f"{filepath}"
        )


    plt.close(fig)

    return fig


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    disease_name: str,
    top_n: int = 10,
    save: bool = True
) -> plt.Figure:
    """
    Generates a horizontal bar chart displaying
    the top feature importances.
    """

    indices = (
        np.argsort(importances)[::-1][:top_n]
    )

    top_features = [
        feature_names[i]
        for i in indices
    ]

    top_importances = (
        importances[indices]
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    y_pos = np.arange(
        len(top_features)
    )


    ax.barh(
        y_pos,
        top_importances,
        align="center",
        color="#059669",
        alpha=0.85
    )


    ax.set_yticks(
        y_pos
    )

    ax.set_yticklabels(
        top_features
    )

    ax.invert_yaxis()


    ax.set_xlabel(
        "Relative Feature Importance Score",
        fontweight="bold"
    )


    ax.set_title(
        f"Top {top_n} Key Feature Predictors - "
        f"{disease_name}",
        fontsize=13,
        fontweight="bold"
    )


    if save:

        filename = (
            f"feature_importance_"
            f"{disease_name.lower().replace(' ', '_')}.png"
        )

        filepath = (
            OUTPUTS_DIR /
            filename
        )

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            f"Saved feature importance plot to: "
            f"{filepath}"
        )


    plt.close(fig)

    return fig


# ============================================================
# MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    metrics_df: Any,
    disease_name: str,
    save: bool = True
) -> plt.Figure:
    """
    Plots a multi-metric comparison bar chart
    comparing algorithm performances.
    """

    df_melted = metrics_df.melt(
        id_vars=["Model"],
        var_name="Metric",
        value_name="Score"
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="Set2",
        ax=ax
    )


    ax.set_ylim(
        [0.0, 1.1]
    )


    ax.set_title(
        f"Model Performance Comparison - "
        f"{disease_name}",
        fontsize=13,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Algorithm",
        fontweight="bold"
    )


    ax.set_ylabel(
        "Metric Score (0.0 - 1.0)",
        fontweight="bold"
    )


    ax.legend(
        loc="lower right"
    )


    if save:

        filename = (
            f"model_comparison_"
            f"{disease_name.lower().replace(' ', '_')}.png"
        )

        filepath = (
            OUTPUTS_DIR /
            filename
        )

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            f"Saved model comparison plot to: "
            f"{filepath}"
        )


    plt.close(fig)

    return fig