# Heart Disease Prediction

A machine-learning project that prepares heart disease data, trains and evaluates classification models, tracks experiments with MLflow, and serves predictions through a Flask API. The API can also be packaged with Docker and deployed to Kubernetes.

> This project is for educational and research use only. It is not a medical device and must not be used to diagnose or treat any condition.

## Contents

- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Set up the Python environment](#set-up-the-python-environment)
- [Prepare data and train models](#prepare-data-and-train-models)
- [Run the API](#run-the-api)
- [Run tests](#run-tests)
- [Build and run with Docker](#build-and-run-with-docker)
- [Deploy to Kubernetes with Minikube](#deploy-to-kubernetes-with-minikube)
- [API reference](#api-reference)

## Project structure

```text
data/
  raw/heart.csv                 Prepared input data
  processed/                    Generated data visualizations
deployment/
  docker/                       Dockerfile and container dependencies
  kubernetes/                   Deployment and Service manifests
notebooks/                      Exploratory data analysis
src/
  api/app.py                    Flask prediction API
  models/                       Model, preprocessor, and evaluation results
  preprocessing/preprocessor.py Data cleaning, imputation, and scaling
  training/train_models.py      Training, evaluation, and MLflow logging
tests/                          API, model, and preprocessing tests
```

## Requirements

- Python 3.11 or later
- Docker, if building/running a container
- Minikube and `kubectl`, if deploying to Kubernetes

The root `requirements.txt` contains the Python dependencies for development, training, and testing. The container uses the smaller dependency list in `deployment/docker/requirements.txt`.

## Set up the Python environment

Run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, allow scripts for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Prepare data and train models

The repository includes `combined_heart_dataset.csv`. To create or refresh `data/raw/heart.csv` from it:

```powershell
python data\download_data.py
```

Then run the training pipeline:

```powershell
python -m src.training.train_models
```

The pipeline splits the data into training and test sets, fits the preprocessor on the training data, trains logistic regression and random forest classifiers, evaluates them, and logs runs to MLflow. It selects the model with the highest test F1 score and saves the model, preprocessor, and metrics under `src/models/`.

The API currently loads `src/models/random_forest.pkl` and `src/models/preprocessor.pkl`. If training selects a different best model, ensure the selected model is saved or copied to the filename the API expects before starting the API.

## Run the API

Start the Flask development server from the repository root:

```powershell
python -m src.api.app
```

The API listens on `http://127.0.0.1:5000`.

Check the health endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/health
```

Send a prediction request:

```powershell
$body = @{
    age = 50
    sex = 1
    cp = 2
    trestbps = 140
    chol = 250
    fbs = 0
    restecg = 1
    thalach = 150
    exang = 0
    oldpeak = 1.5
    slope = 2
    ca = 0
    thal = 3
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:5000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Run tests

Run the test suite from the repository root:

```powershell
pytest
```

## Build and run with Docker

Build the image using the repository root as the build context:

```powershell
docker build -t heart-disease-api:latest -f deployment/docker/Dockerfile .
```

Run the API container and map port 5000:

```powershell
docker run --rm -p 5000:5000 heart-disease-api:latest
```

The API is then available at `http://127.0.0.1:5000`.

## Deploy to Kubernetes with Minikube

Start Minikube and build the image directly into Minikube's image store:

```powershell
minikube start
minikube image build -t heart-disease-api:latest -f deployment/docker/Dockerfile .
```

Apply the Deployment and Service manifests:

```powershell
kubectl apply -f deployment/kubernetes/
kubectl get deployments,pods,services
```

Wait for the Deployment to become ready:

```powershell
kubectl rollout status deployment/heart-disease-api
```

Forward the Service to a local port:

```powershell
kubectl port-forward service/heart-disease-api-service 8080:80
```

Keep that command running, then check the API in another terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/health
```

To remove the deployed resources:

```powershell
kubectl delete -f deployment/kubernetes/
```

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns API health, model-load status, and a timestamp. |
| `GET` | `/info` | Returns API name, version, model, and endpoint information. |
| `POST` | `/predict` | Accepts the 13 feature values as JSON and returns a prediction and confidence. |

The `/predict` request must use `Content-Type: application/json` and include these numeric fields:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal
```

A successful response includes `prediction` (0 or 1), `confidence`, `prediction:label`, and a timestamp. The target is treated as binary during training: values greater than zero map to 1.
