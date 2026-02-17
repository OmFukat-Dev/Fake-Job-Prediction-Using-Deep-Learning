import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import CONFIG

def load_raw_data():
    """Load the raw dataset from CSV"""
    try:
        data = pd.read_csv(CONFIG['data']['raw_path'])  # Fixed: CONFIG not CONTEG
        print(f"✅ Data loaded successfully! Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print("❌ Dataset file not found. Please download it from Kaggle.")
        print(f"Expected file: {CONFIG['data']['raw_path']}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def split_data(data, target_column='fraudulent'):
    """Split data into train, validation, and test sets"""
    if data is None:
        return None, None, None
        
    # Split into train+val and test
    train_val_data, test_data = train_test_split(
        data, 
        test_size=CONFIG['data']['test_size'], 
        random_state=CONFIG['data']['random_state'], 
        stratify=data[target_column]
    )
    
    # Split train_val into train and validation
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=CONFIG['data']['val_size']/(1-CONFIG['data']['test_size']), 
        random_state=CONFIG['data']['random_state'], 
        stratify=train_val_data[target_column]
    )
    
    print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def save_split_data(train_data, val_data, test_data):
    """Save the split datasets to processed folder"""
    if train_data is not None:
        train_data.to_csv(CONFIG['data']['processed_path'] + 'train_data.csv', index=False)
        val_data.to_csv(CONFIG['data']['processed_path'] + 'val_data.csv', index=False)
        test_data.to_csv(CONFIG['data']['processed_path'] + 'test_data.csv', index=False)
        print("✅ Split data saved successfully!")