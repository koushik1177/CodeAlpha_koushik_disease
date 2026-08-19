"""
Prediction Module for Disease Prediction System.

This module provides the DiseasePredictor class for loading fitted artifacts
and executing single-instance predictions with risk probabilities.
"""

from typing import Dict, Any

import pandas as pd

from config.settings import DISEASE_CONFIGS
from backend.src.preprocessing import DataPreprocessor
from backend.src.utils import logger, load_artifact


class DiseasePredictor:
    """
    Inference pipeline manager for loading disease models
    and serving predictions.
    """

    def __init__(self, disease_type: str):
        """
        Initializes the predictor for a given disease domain.

        Args:
            disease_type: Key identifying the disease.
        """

        # Validate disease type
        if disease_type not in DISEASE_CONFIGS:
            raise KeyError(
                f"Invalid disease type '{disease_type}'. "
                f"Available diseases: {list(DISEASE_CONFIGS.keys())}"
            )

        self.disease_type = disease_type

        self.config = DISEASE_CONFIGS[disease_type]

        # Load fitted preprocessor
        self.preprocessor = DataPreprocessor.load(
            disease_type
        )

        # Load trained model
        model_filename = (
            f"model_{disease_type}.joblib"
        )

        self.model = load_artifact(
            model_filename,
            subfolder="models"
        )

        logger.info(
            f"Loaded predictor pipeline for disease: "
            f"'{self.config['name']}'"
        )

    def predict_single(
        self,
        input_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes disease risk prediction for a single patient.

        Args:
            input_dict:
                Feature values keyed by feature names.

        Returns:
            Dictionary containing prediction,
            probability, risk percentage and status.
        """

        # ----------------------------------------------------
        # Check feature configuration
        # ----------------------------------------------------

        if "features" not in self.config:
            raise KeyError(
                f"Feature configuration missing for "
                f"disease '{self.disease_type}'. "
                f"Add 'features' to DISEASE_CONFIGS in "
                f"config/settings.py."
            )

        expected_features = self.config["features"]

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_dict
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features for "
                f"{self.disease_type}: "
                f"{missing_features}"
            )

        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        df_input = pd.DataFrame(
            [input_dict]
        )

        df_input = df_input[
            expected_features
        ]

        # ----------------------------------------------------
        # Preprocess input
        # ----------------------------------------------------

        scaled_input = (
            self.preprocessor.transform(
                df_input
            )
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = int(
            self.model.predict(
                scaled_input
            )[0]
        )

        # ----------------------------------------------------
        # Calculate probability
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(
                    scaled_input
                )
            )

            probability = float(
                probabilities[0][1]
            )

        else:

            probability = float(
                prediction
            )

        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        risk_category = (
            "High Risk"
            if prediction == 1
            else "Low Risk / Normal"
        )

        status = (
            "Positive / Elevated Risk"
            if prediction == 1
            else "Negative / Low Risk"
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "disease": self.config["name"],

            "prediction": prediction,

            "probability": round(
                probability,
                4
            ),

            "risk_percentage": round(
                probability * 100,
                2
            ),

            "risk_category": risk_category,

            "status": status
        }