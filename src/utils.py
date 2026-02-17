import pickle
import json
from tensorflow.keras.models import save_model, load_model

def save_model_file(model, filepath):
    """Save model to file"""
    if filepath.endswith('.h5'):
        model.save(filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    print(f"✅ Model saved to {filepath}")

def load_model_file(filepath):
    """Load model from file"""
    if filepath.endswith('.h5'):
        return load_model(filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def save_metrics(metrics, filepath):
    """Save evaluation metrics to file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {filepath}")