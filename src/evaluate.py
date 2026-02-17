from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, predictions_binary)
    precision = precision_score(y_test, predictions_binary)
    recall = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    cm = confusion_matrix(y_test, predictions_binary)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print(f"📊 Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return metrics