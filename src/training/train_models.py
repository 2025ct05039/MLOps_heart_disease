import pandas as pd
import numpy as np  
from pathlib import Path
import pickle
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessor import HeartDiseasePreprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2

class ModelTrainer:
    """ train and evaluate models for heart disease prediction. """
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(data_path)
        self.preprocessor = HeartDiseasePreprocessor()
        self.models = {
        }
        self.results = {}

    def prepare_data(self):
        """Prepare the data for training and testing."""
        #Seperate features and target variable
        X = self.df.drop('target', axis=1)
        y = (self.df['target'] > 0).astype(int)  # Convert target to binary (0 or 1)
        
        #Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        #Preprocess the data
        self.preprocessor.fit(X_train)
        X_train_scaled = self.preprocessor.transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train a logistic regression model."""

        print("\n" + "="*20 + " Training Logistic Regression Model " + "="*20 + "\n")

        with mlflow.start_run(run_name="Logistic Regression"):
            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
            
            # Log parameters
            mlflow.log_param('model_type', 'Logistic Regression')
            mlflow.log_param('max_iter', 1000)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("C", 1.0)
            mlflow.log_param("solver", 'lbfgs')

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)

            #Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "logistic_regression_model")
            self.models['logistic_regression'] = model
            self.results['logistic_regression'] = metrics

            print("\n" + "="*20 + " Logistic Regression Model Training Complete " + "="*20 + "\n")

            return model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train a random forest classifier model."""

        print("\n" + "="*20 + " Training Random Forest Model " + "="*20 + "\n")

        with mlflow.start_run(run_name="Random Forest"):
            model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=5, n_jobs=-1, random_state=42)

            # Log parameters
            mlflow.log_param('model_type', 'Random Forest')
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 10)
            mlflow.log_param('min_samples_split', 5)    
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_jobs", -1)
            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)

            #Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            self.models['random_forest'] = model
            self.results['random_forest'] = metrics

            print("\n" + "="*20 + " Random Forest Model Training Complete " + "="*20 + "\n")

            return model, metrics

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate the model and return metrics."""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba)
        }

        print("\n" + "="*20 + " Model Evaluation Metrics " + "="*20 + "\n")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return metrics

    def save_best_model(self, model_name, save_path):
        """Save the best model and preprocessor to disk."""

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model = self.models[model_name]

        #Save model 
        model_path = save_path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        #Save preprocessor
        preprocessor_path = save_path / "preprocessor.pkl"
        self.preprocessor.save(preprocessor_path)

        #Save results
        results_path = save_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nModel saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        print(f"Results saved to {results_path}")

    def main(self):
        """Main function to run the training and evaluation pipeline."""
        #Set up MLflow experiment
        mlflow.set_experiment("Heart Disease Prediction")

        # Prepare the data
        trainer = ModelTrainer("data/raw/heart.csv")
        X_train, X_test, y_train, y_test = trainer.prepare_data()

        print(f"\n Data prepared : ") 
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f" Target distribution train): {np.bincount(y_train)}")

        #Train models
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)

        #Compare and save the best model
        print ("\n" + "="*20 + " Comparing Models " + "="*20 + "\n")

        for name , metrics in trainer.results.items():
            print(f"\n{name.upper()}:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
        #Save the best model
        best_model_name = max(trainer.results, key=lambda x: trainer.results[x]['test_f1'])
        print(f"\nBest model based on test F1 score: {best_model_name}")
        trainer.save_best_model(best_model_name, "src/models")

        print("\n" + "="*20 + " Training and Evaluation Pipeline Complete " + "="*20 + "\n")
        print(f"Best model saved to src/models/{best_model_name}.pkl")

if __name__ == "__main__":
    #Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)

    #Run the training and evaluation pipeline
    trainer = ModelTrainer("data/raw/heart.csv")
    trainer.main()
    
