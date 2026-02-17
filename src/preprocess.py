import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .config import CONFIG

# Download NLTK data (run once)
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("⚠️  NLTK downloads may require internet connection")

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(data):
    """Preprocess the entire dataset"""
    print("🔄 Preprocessing data...")
    
    # Handle missing values
    data = data.fillna('')
    
    # Clean text columns
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].apply(clean_text)
    
    print("✅ Data preprocessing complete!")
    return data