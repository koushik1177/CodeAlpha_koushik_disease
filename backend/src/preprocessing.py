"""
Preprocessing Module for Disease Prediction System.

This module provides data cleaning, missing value imputation,
standard feature scaling, and train-test splitting workflows.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config.settings import TEST_SIZE, RANDOM_STATE
from backend.src.utils import (
    logger,
    save_artifact,
    load_artifact
)


class DataPreprocessor:
    """
    Encapsulates feature imputation and scaling pipelines
    for a disease domain.
    """

    def __init__(self, disease_type: str):
        self.disease_type = disease_type

        self.imputer = SimpleImputer(
            strategy="median"
        )

        self.scaler = StandardScaler()

        self.is_fitted = False

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Fits the imputer and scaler and returns
        scaled data.
        """

        logger.info(
            f"Fitting preprocessor pipeline "
            f"for '{self.disease_type}'..."
        )

        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(
            X_imputed
        )

        self.is_fitted = True

        return X_scaled

    def transform(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Transforms input data using the fitted
        preprocessing pipeline.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before "
                "calling transform()."
            )

        X_imputed = self.imputer.transform(X)

        X_scaled = self.scaler.transform(
            X_imputed
        )

        return X_scaled

    def save(self) -> str:
        """
        Saves the fitted preprocessor.
        """

        filename = (
            f"preprocessor_"
            f"{self.disease_type}.joblib"
        )

        save_artifact(
            self,
            filename=filename,
            subfolder="models"
        )

        return filename

    @classmethod
    def load(
        cls,
        disease_type: str
    ) -> "DataPreprocessor":
        """
        Loads a saved preprocessor.
        """

        filename = (
            f"preprocessor_"
            f"{disease_type}.joblib"
        )

        instance = load_artifact(
            filename=filename,
            subfolder="models"
        )

        return instance


def prepare_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    disease_type: str
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DataPreprocessor
]:
    """
    Prepares the dataset by performing:

    1. Train-test split
    2. Missing-value imputation
    3. Feature scaling
    4. Saving the fitted preprocessor
    """

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(
        f"Split '{disease_type}' data into "
        f"Train shape: {X_train.shape}, "
        f"Test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Create preprocessor
    # --------------------------------------------------------

    preprocessor = DataPreprocessor(
        disease_type=disease_type
    )

    # --------------------------------------------------------
    # Fit on training data
    # --------------------------------------------------------

    X_train_scaled = (
        preprocessor.fit_transform(X_train)
    )

    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    X_test_scaled = (
        preprocessor.transform(X_test)
    )

    # --------------------------------------------------------
    # Save fitted preprocessor
    # --------------------------------------------------------

    preprocessor.save()

    # --------------------------------------------------------
    # Return processed data
    # --------------------------------------------------------

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.values,
        y_test.values,
        preprocessor
    )
