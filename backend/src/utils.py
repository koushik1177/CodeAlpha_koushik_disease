import json
import logging
from typing import Any, Dict

import joblib

from config.settings import (
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR
)


# ============================================================
# LOGGER
# ============================================================

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def setup_logger(
    name: str = "disease_prediction"
) -> logging.Logger:

    logger = logging.getLogger(name)

    logger.setLevel(
        logging.INFO
    )

    if not logger.handlers:

        console_handler = (
            logging.StreamHandler()
        )

        console_handler.setLevel(
            logging.INFO
        )

        console_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - "
            "%(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler.setFormatter(
            console_formatter
        )

        logger.addHandler(
            console_handler
        )


        log_file = (
            LOGS_DIR / "app.log"
        )

        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8"
        )

        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        )

        file_handler.setFormatter(
            file_formatter
        )

        logger.addHandler(
            file_handler
        )

    return logger


logger = setup_logger()


# ============================================================
# SAVE ARTIFACT
# ============================================================

def save_artifact(
    obj: Any,
    filename: str,
    subfolder: str = "models"
):

    target_dir = (
        MODELS_DIR
        if subfolder == "models"
        else OUTPUTS_DIR
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        target_dir / filename
    )

    joblib.dump(
        obj,
        file_path
    )

    logger.info(
        f"Successfully saved artifact: "
        f"{file_path}"
    )

    return file_path


# ============================================================
# LOAD ARTIFACT
# ============================================================

def load_artifact(
    filename: str,
    subfolder: str = "models"
):

    target_dir = (
        MODELS_DIR
        if subfolder == "models"
        else OUTPUTS_DIR
    )

    file_path = (
        target_dir / filename
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Artifact not found: {file_path}"
        )

    obj = joblib.load(
        file_path
    )

    logger.info(
        f"Successfully loaded artifact: "
        f"{file_path}"
    )

    return obj


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data: Dict[str, Any],
    filename: str
):

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        OUTPUTS_DIR / filename
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    logger.info(
        f"Saved JSON metrics to: "
        f"{file_path}"
    )

    return file_path