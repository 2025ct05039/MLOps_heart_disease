import pytest
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import HeartDiseasePreprocessor

@pytest.fixture
def sample_data():
""" Create sample_data for testing """
    data = {
        "age": [40,50,60],
        "sex": [1, 0, 1],
        "cp": [2,3,1],
        "trestbps": [140, 130, 120],
        "chol": [250, 200, 180],
        "fbs": [0, 1, 0],
        "restecg": [0, 1, 2],
        "thalach": [150, 140, 160],
        "exang": [0, 1, 0],
        "oldpeak": [1.5, 2.0, 0.5],
        "slope": [2, 1, 3],
        "ca": [0,1,2],
        "thal": [3,7,6 ],
    }
    return pd.DataFrame(data)

def test_preprocessor_fit(sample_data):
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(sample_data)

    assert preprocessor.is_fitted == True
    assert preprocessor.feature_names == sample_data.columns.tolist()

def test_preprocessor_transform(sample_data):
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    assert transformed.shape == sample_data.shape
    assert np.allclose(transformed.mean(), 0, atol=1e-10)  # Mean should be close to 0 after scaling
    assert np.allclose(transformed.std(), 1, atol=1e-10)   # Std should be close to 1 after scaling        

def test_preprocessor_fit_transform(sample_data):
    preprocessor = HeartDiseasePreprocessor()
    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape == sample_data.shape
    assert preprocessor.is_fitted == True
