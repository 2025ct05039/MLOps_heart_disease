import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle
import json

class HeartDiseasePreprocessor:
    def __init__(self):
        self.config_path = config_path
        self.config = self.load_config()
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate features and target
        X = df.drop(columns=self.config['target_column'])
        y = df[self.config['target_column']]

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X[self.config['numerical_features']])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.config['numerical_features'])

        # Encode categorical features
        X_encoded = self.encoder.fit_transform(X[self.config['categorical_features']])
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(self.config['categorical_features']))

        # Combine scaled and encoded features
        X_preprocessed = pd.concat([X_scaled_df, X_encoded_df], axis=1)

        return X_preprocessed, y

    def save_preprocessor(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'encoder': self.encoder}, f)

    def load_preprocessor(self, path: str):
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
            self.scaler = preprocessor['scaler']
            self.encoder = preprocessor['encoder']