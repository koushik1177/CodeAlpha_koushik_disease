"""
Data Loader Module for Disease Prediction System.

This module handles loading CSV datasets from disk. If a dataset is missing,
it automatically generates a synthetic dataset for training and testing.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import (
    DISEASE_CONFIGS,
    DATASET_DIR,
    RANDOM_STATE
)

from backend.src.utils import logger


def generate_synthetic_data(
    disease_type: str,
    num_samples: int = 300
) -> pd.DataFrame:
    """
    Generates a synthetic dataset for a specified disease domain.
    """

    np.random.seed(RANDOM_STATE)

    if disease_type not in DISEASE_CONFIGS:
        raise KeyError(
            f"Invalid disease type '{disease_type}'. "
            f"Valid keys: {list(DISEASE_CONFIGS.keys())}"
        )

    config = DISEASE_CONFIGS[disease_type]

    target_col = config["target_col"]


    # ========================================================
    # HEART DISEASE
    # ========================================================

    if disease_type == "heart":

        data = {

            "age": np.random.randint(
                29, 78, num_samples
            ),

            "sex": np.random.choice(
                [0, 1],
                num_samples
            ),

            "cp": np.random.choice(
                [0, 1, 2, 3],
                num_samples
            ),

            "trestbps": np.random.randint(
                94, 200,
                num_samples
            ),

            "chol": np.random.randint(
                126, 564,
                num_samples
            ),

            "fbs": np.random.choice(
                [0, 1],
                num_samples,
                p=[0.85, 0.15]
            ),

            "restecg": np.random.choice(
                [0, 1, 2],
                num_samples
            ),

            "thalach": np.random.randint(
                71, 202,
                num_samples
            ),

            "exang": np.random.choice(
                [0, 1],
                num_samples
            ),

            "oldpeak": np.round(
                np.random.uniform(
                    0.0,
                    6.2,
                    num_samples
                ),
                1
            ),

            "slope": np.random.choice(
                [0, 1, 2],
                num_samples
            ),

            "ca": np.random.choice(
                [0, 1, 2, 3, 4],
                num_samples
            ),

            "thal": np.random.choice(
                [1, 2, 3],
                num_samples
            ),

            target_col: np.random.choice(
                [0, 1],
                num_samples,
                p=[0.46, 0.54]
            )
        }


    # ========================================================
    # DIABETES
    # ========================================================

    elif disease_type == "diabetes":

        data = {

            "Pregnancies": np.random.randint(
                0,
                17,
                num_samples
            ),

            "Glucose": np.random.randint(
                44,
                199,
                num_samples
            ),

            "BloodPressure": np.random.randint(
                24,
                122,
                num_samples
            ),

            "SkinThickness": np.random.randint(
                7,
                99,
                num_samples
            ),

            "Insulin": np.random.randint(
                14,
                846,
                num_samples
            ),

            "BMI": np.round(
                np.random.uniform(
                    18.2,
                    67.1,
                    num_samples
                ),
                1
            ),

            "DiabetesPedigreeFunction": np.round(
                np.random.uniform(
                    0.08,
                    2.42,
                    num_samples
                ),
                3
            ),

            "Age": np.random.randint(
                21,
                81,
                num_samples
            ),

            target_col: np.random.choice(
                [0, 1],
                num_samples,
                p=[0.65, 0.35]
            )
        }


    # ========================================================
    # KIDNEY DISEASE
    # ========================================================

    elif disease_type == "kidney":

        data = {

            "age": np.random.randint(
                12,
                85,
                num_samples
            ),

            "bp": np.random.choice(
                [60, 70, 80, 90, 100, 110, 120],
                num_samples
            ),

            "sg": np.random.choice(
                [
                    1.005,
                    1.010,
                    1.015,
                    1.020,
                    1.025
                ],
                num_samples
            ),

            "al": np.random.choice(
                [0, 1, 2, 3, 4, 5],
                num_samples
            ),

            "su": np.random.choice(
                [0, 1, 2, 3, 4, 5],
                num_samples
            ),

            "bgr": np.random.randint(
                70,
                490,
                num_samples
            ),

            "bu": np.random.randint(
                10,
                390,
                num_samples
            ),

            "sc": np.round(
                np.random.uniform(
                    0.4,
                    15.0,
                    num_samples
                ),
                1
            ),

            "sod": np.random.randint(
                110,
                160,
                num_samples
            ),

            "pot": np.round(
                np.random.uniform(
                    2.5,
                    7.5,
                    num_samples
                ),
                1
            ),

            "hemo": np.round(
                np.random.uniform(
                    3.1,
                    17.8,
                    num_samples
                ),
                1
            ),

            "pcv": np.random.randint(
                16,
                54,
                num_samples
            ),

            "wbcc": np.random.randint(
                2200,
                26400,
                num_samples
            ),

            "rbcc": np.round(
                np.random.uniform(
                    2.1,
                    8.0,
                    num_samples
                ),
                1
            ),

            "htn": np.random.choice(
                [0, 1],
                num_samples,
                p=[0.6, 0.4]
            ),

            "dm": np.random.choice(
                [0, 1],
                num_samples,
                p=[0.65, 0.35]
            ),

            target_col: np.random.choice(
                [0, 1],
                num_samples,
                p=[0.37, 0.63]
            )
        }


    else:

        raise ValueError(
            f"Unknown disease_type: {disease_type}"
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(data)

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        DATASET_DIR /
        config["file_name"]
    )

    df.to_csv(
        save_path,
        index=False
    )

    logger.info(
        f"Generated synthetic dataset for "
        f"'{disease_type}' at: {save_path}"
    )

    return df


def load_disease_data(
    disease_type: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset for a specific disease domain
    and separates features and target.
    """

    if disease_type not in DISEASE_CONFIGS:

        raise KeyError(
            f"Invalid disease key '{disease_type}'. "
            f"Valid keys: "
            f"{list(DISEASE_CONFIGS.keys())}"
        )


    config = DISEASE_CONFIGS[disease_type]

    file_path = (
        DATASET_DIR /
        config["file_name"]
    )


    # ========================================================
    # LOAD OR GENERATE DATASET
    # ========================================================

    if not file_path.exists():

        logger.warning(
            f"File {file_path} not found. "
            f"Generating synthetic dataset..."
        )

        df = generate_synthetic_data(
            disease_type
        )

    else:

        df = pd.read_csv(
            file_path
        )

        logger.info(
            f"Loaded existing dataset from: "
            f"{file_path} "
            f"(Shape: {df.shape})"
        )


    # ========================================================
    # TARGET COLUMN
    # ========================================================

    target_col = config["target_col"]

    if target_col not in df.columns:

        raise KeyError(
            f"Target column '{target_col}' "
            f"not found in dataset "
            f"{file_path}"
        )


    # ========================================================
    # FEATURES AND TARGET
    # ========================================================

    X = df.drop(
        columns=[target_col]
    )

    y = df[target_col]


    return X, y