import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle
import json

class HeartDiseasePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X):
        """ Fit the processor to the data. """
        X_cleaned = X.fillna(X.mean())
        self.scaler.fit(X_cleaned)
        self.feature_names = X_cleaned.columns.tolist()
        self.is_fitted = True
        return self

    def transform(self, X):
        # Transform the data using the fitted processor.
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        X_cleaned = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X_cleaned)

        return pd.DataFrame(X_scaled, columns=self.feature_names)

    def fit_transform(self, X):
        """ Fit and transform the data using the processor. """

        return self.fit(X).transform(X)

    def save(self, path):
        """ Save processor to a file. """
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler,'feature_names': self.feature_names, 'is_fitted': self.is_fitted}, f)

    def load(self, path):
        """ Load processor from a file. """

        with open(path, 'rb') as f:
            data  = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_fitted = data['is_fitted']

        return self

        