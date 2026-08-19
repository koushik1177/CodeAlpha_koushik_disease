"""
Unit Test Suite for Disease Prediction System.
"""

import unittest
import numpy as np
import pandas as pd
from src.config import DISEASE_CONFIGS
from src.data_loader import load_disease_data
from src.preprocessing import prepare_dataset
from src.predict import DiseasePredictor


class TestDiseasePredictionPipeline(unittest.TestCase):

    def test_disease_configs_exist(self):
        self.assertIn("heart", DISEASE_CONFIGS)
        self.assertIn("diabetes", DISEASE_CONFIGS)
        self.assertIn("kidney", DISEASE_CONFIGS)

    def test_data_loader(self):
        for disease in ["heart", "diabetes", "kidney"]:
            X, y = load_disease_data(disease)
            self.assertGreater(X.shape[0], 0)
            self.assertEqual(X.shape[0], y.shape[0])

    def test_preprocessing(self):
        X, y = load_disease_data("heart")
        X_train, X_test, y_train, y_test, preprocessor = prepare_dataset(X, y, "heart")
        self.assertEqual(X_train.shape[1], X_test.shape[1])

    def test_inference_pipeline(self):
        heart_sample = {
            "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
            "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0,
            "slope": 2, "ca": 0, "thal": 2
        }
        predictor = DiseasePredictor("heart")
        res = predictor.predict_single(heart_sample)
        self.assertIn("prediction", res)
        self.assertIn("probability", res)
        self.assertIn("risk_category", res)


if __name__ == "__main__":
    unittest.main()
