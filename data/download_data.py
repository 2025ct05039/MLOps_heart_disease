# data/download_data.py

import pandas as pd

path = "combined_heart_dataset.csv"

df = pd.read_csv(path, header=None, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]   ) 

df.to_csv("data/raw/heart.csv", index=False)
