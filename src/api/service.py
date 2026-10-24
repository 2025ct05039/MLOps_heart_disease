import pickle
from pathlib import path
import pandas as pd

class PredictionService:
    """Service for making predictions """

    def __init__(self,model_path,preprocessor_path):
        self.model_path = Path(model_path)
        import sys

        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        self.load()
    
    def _load(self):
        """Load model and preprocessor """
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        from src.preprocessing.preprocessor import HeartDiseasePreprocessor
        self.preprocessor = HeartDiseasePreprocessor()
        self.preprocessor.load(str(self.preprocessor_path))

    def predict(self,data_dict):
        """Make prediction from dictionary"""
        df = pd.DataFrame([data_dict])
        df_processed = self.preprocessor.transform(df)

        prediction = self.model.predict(df_processed)[0]
        confidence = self.model.predict.proba(df.processed)[0][int(prediction)]

        return {
            'predicton' : int(prediction),
            'confidence' : int(confidence)
        }

