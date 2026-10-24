import pytest

@pytest.fixture
def client():
    from src.api.app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json["status"] == "healthy"
    assert "model_loaded" in response.json
    assert "timestamp" in response.json


def test_predict_endpoint(client):
    # Sample input data for prediction
    test_data = {
        "age": 50,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    response = client.post('/predict', json=test_data)

    assert response.status_code == 200
    assert 'prediction' in response.json
    assert 'confidence' in response.json

    