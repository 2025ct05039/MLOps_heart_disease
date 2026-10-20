# data/download_data.py

import pandas as pd

path = "combined_heart_dataset.csv"

df = pd.read_csv(path, header=0, names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]   ) 

df.to_csv("data/raw/heart.csv", index=False)

print(f"Dataset shape: {df.shape}")
print(f"First few rows: {df.head()}")
print(f"Last few rows: {df.tail()}")
print(f"data types: {df.dtypes}")
print(f"Missing values: {df.isnull().sum()}")