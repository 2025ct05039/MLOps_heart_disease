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
        self.imputation_values = None
        self.is_fitted = False

    @staticmethod
    def _clean_numeric_data(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        cleaned = X.replace("?", np.nan)
        numeric = cleaned.apply(pd.to_numeric, errors="coerce").astype(float)
        invalid = numeric.isna() & cleaned.notna()
        if invalid.any().any():
            invalid_columns = invalid.columns[invalid.any()].tolist()
            raise ValueError(
                f"Non-numeric values found in feature columns: {invalid_columns}"
            )
        return numeric

    def fit(self, X):
        """ Fit the processor to the data. """
        X_cleaned = self._clean_numeric_data(X)
        self.imputation_values = X_cleaned.mean()
        all_missing = self.imputation_values[self.imputation_values.isna()].index.tolist()
        if all_missing:
            raise ValueError(
                f"Cannot impute features with no numeric values: {all_missing}"
            )
        X_cleaned = X_cleaned.fillna(self.imputation_values)
        self.scaler.fit(X_cleaned)
        self.feature_names = X_cleaned.columns.tolist()
        self.is_fitted = True
        return self

    def transform(self, X):
        # Transform the data using the fitted processor.
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        missing_columns = [name for name in self.feature_names if name not in X.columns]
        unexpected_columns = [name for name in X.columns if name not in self.feature_names]
        if missing_columns or unexpected_columns:
            raise ValueError(
                "Input feature columns do not match fitted data "
                f"(missing: {missing_columns}, unexpected: {unexpected_columns})"
            )

        X_cleaned = self._clean_numeric_data(X.loc[:, self.feature_names])
        X_cleaned = X_cleaned.fillna(self.imputation_values)
        X_scaled = self.scaler.transform(X_cleaned)

        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def fit_transform(self, X):
        """ Fit and transform the data using the processor. """

        return self.fit(X).transform(X)

    def save(self, path):
        """ Save processor to a file. """
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'imputation_values': self.imputation_values,
                'is_fitted': self.is_fitted
            }, f)

    def load(self, path):
        """ Load processor from a file. """

        with open(path, 'rb') as f:
            data  = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.imputation_values = data.get('imputation_values')
            if self.imputation_values is None and self.feature_names is not None:
                self.imputation_values = pd.Series(
                    self.scaler.mean_, index=self.feature_names
                )
            self.is_fitted = data['is_fitted']

        return self

        