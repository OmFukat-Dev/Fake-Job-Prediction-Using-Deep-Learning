"""
Script to predict if a job posting is fake
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_job(title, description, requirements, company_profile, 
                telecommuting=0, has_company_logo=0, has_questions=0):
    """
    Predict if a job posting is fake
    """
    # Load model
    model = load_model('../models/lstm_text_model.h5')
    
    # Prepare text
    combined_text = f"{title} {description} {requirements} {company_profile}"
    
    # Tokenize (you should save/load your tokenizer properly)
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    # For now, we'll use a simple approach - in production, save the fitted tokenizer
    tokenizer.fit_on_texts([combined_text])
    
    text_seq = tokenizer.texts_to_sequences([combined_text])
    text_padded = pad_sequences(text_seq, maxlen=200, padding='post', truncating='post')
    
    # Prepare numeric features
    numeric_features = np.array([[telecommuting, has_company_logo, has_questions]])
    
    # Predict
    prediction_prob = model.predict([text_padded, numeric_features])[0][0]
    is_fake = prediction_prob > 0.4  # Using optimal threshold
    
    return {
        'is_fake': bool(is_fake),
        'confidence': float(prediction_prob),
        'risk_level': 'HIGH' if prediction_prob > 0.7 else 'MEDIUM' if prediction_prob > 0.4 else 'LOW'
    }

if __name__ == "__main__":
    # Example usage
    result = predict_job(
        title="Software Engineer",
        description="We are looking for a skilled software engineer...",
        requirements="5+ years experience, Python, Django...",
        company_profile="Established tech company since 2010...",
        telecommuting=1,
        has_company_logo=1,
        has_questions=1
    )
    
    print("🔍 Job Prediction Result:")
    print(f"Fake: {result['is_fake']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Risk Level: {result['risk_level']}")