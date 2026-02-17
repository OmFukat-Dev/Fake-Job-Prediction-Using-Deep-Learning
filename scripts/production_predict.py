"""
Production-ready job prediction with saved tokenizer
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class JobFraudDetector:
    def __init__(self, model_path='../models/lstm_text_model.h5', 
                 tokenizer_path='../models/tokenizer.pkl', threshold=0.3):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.threshold = threshold
        
    def preprocess_text(self, text_data):
        """Preprocess text data for prediction"""
        combined_text = f"{text_data.get('title', '')} {text_data.get('description', '')} {text_data.get('requirements', '')} {text_data.get('company_profile', '')}"
        
        # Tokenize and pad using the saved tokenizer
        sequences = self.tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        return padded
    
    def preprocess_numeric(self, numeric_data):
        """Preprocess numeric features"""
        features = [
            numeric_data.get('telecommuting', 0),
            numeric_data.get('has_company_logo', 0), 
            numeric_data.get('has_questions', 0)
        ]
        return np.array([features])
    
    def predict(self, job_data):
        """Predict if a job is fraudulent"""
        try:
            # Preprocess features
            text_features = self.preprocess_text(job_data)
            numeric_features = self.preprocess_numeric(job_data)
            
            # Predict
            prediction_prob = self.model.predict([text_features, numeric_features], verbose=0)[0][0]
            is_fake = prediction_prob > self.threshold
            
            # Generate risk assessment
            if prediction_prob > 0.6:
                risk_level = "HIGH"
                action = "BLOCK"
            elif prediction_prob > 0.3:
                risk_level = "MEDIUM" 
                action = "REVIEW"
            else:
                risk_level = "LOW"
                action = "ALLOW"
            
            return {
                'success': True,
                'is_fake': bool(is_fake),
                'confidence': float(prediction_prob),
                'risk_level': risk_level,
                'recommended_action': action,
                'threshold_used': self.threshold
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = JobFraudDetector(threshold=0.3)
    
    # Test examples
    test_jobs = [
        {
            'title': 'Work From Home Data Entry',
            'description': 'Earn $5000 per month working from home. No experience needed! Quick money!',
            'requirements': 'None, just a computer and internet',
            'company_profile': '',
            'telecommuting': 1,
            'has_company_logo': 0,
            'has_questions': 0
        },
        {
            'title': 'Senior Software Engineer at Google',
            'description': 'Join our team to build innovative solutions. Competitive salary and benefits.',
            'requirements': '5+ years experience, Python, Cloud computing',
            'company_profile': 'Google is a leading technology company...',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 1
        }
    ]
    
    for i, job in enumerate(test_jobs, 1):
        print(f"\n🔍 Analyzing Job #{i}: {job['title']}")
        result = detector.predict(job)
        
        if result['success']:
            print(f"   Predicted Fake: {result['is_fake']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Action: {result['recommended_action']}")
        else:
            print(f"   Error: {result['error']}")