import pytest
import pandas as pd
import numpy as np

from src.training.train_models import ModelTrainer

@pytest.fixture
def trainer():
    """ Create a ModelTrainer instance for testing """
    data_path = "data/raw/heart.csv"
    return ModelTrainer(data_path)

def test_data_preparation(trainer):
    X_train, X_test, y_train, y_test = trainer.prepare_data()

    # Check that the shapes of the train and test sets are correct
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

def test_logistic_regression_training(trainer):
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    model, metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)

    # Check that results for logistic regression are stored
    assert "test_accuracy" in metrics
    assert metrics["test_accuracy"] > 0.5

def test_random_forest_training(trainer):
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    model, metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)

    # Check that results for random forest are stored
    assert "test_roc_auc" in metrics
    assert metrics["test_roc_auc"] > 0.5
