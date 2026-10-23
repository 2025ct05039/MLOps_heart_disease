from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import pandas as pd
import pickle
from pathlib import Path

import logging
from datetime import datetime

#Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest.pkl"
PREPROCESSOR_PATH = Path(__file__).parent.parent / "models" / "preprocessor.pkl"

#Load model and preprocessor

def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        from src.preprocessing.preprocessor import HeartDiseasePreprocessor
        preprocessor = HeartDiseasePreprocessor()
        preprocessor.load(str(PREPROCESSOR_PATH))

        logger.info(" Model and preprocessor loaded successfully")
        return model,preprocessor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None
    
model, preprocessor = load_model_and_preprocessor()

#Request schema

class PredictionRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.route('/health',methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({
        'status' : 'healthy',
        'model_loaded' : model is not None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/predict',methods=['POST'])
def predict():
    """Make prediction endpoint"""
    try:
        #Parse Request
        data = request.get_json()
    
        #Validate
        request_data = PredictionRequest(**data)

        #Create Dataframe
        input_df = pd.DataFrame([request_data.dict()])

        #Preprocess
        input_processed = preprocessor.transform(input_df)

        #Predict
        prediction = model.predict(input_processed)[0]
        confidence = float(model.predict.proba(input_processed)[0][int(prediction)])

        #Log
        logger.info(f"Prediction made: {prediction}, confidence: {confidence:.4f}")

        response = {
            'prediction':int(prediction),
            'confidence': float(confidence),
            'prediction:label': 'Heart Disease Present' if prediction ==1 else 'No Heart Disease',
            'timestamp' : datetime.utcnow().isoformat()
        }

        return jsonify(response), 200

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': 'Invalid request', 'details': e.errors()}), 400
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('\info', methods=['GET'])

def info():
    """API Information endpoint"""
    return jsonify({
        'name': 'Heart Disease Prediction API',
        'version': '1.1.0',
        'model': 'Random Forest Classifier',
        'endpoints': {
            '/health': 'GET- Health check',
            '/predict': 'POST- Make prediction',
            'info': 'GET - API informaition'
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)

