from tensorflow.keras.callbacks import EarlyStopping
from .config import CONFIG

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['model']['batch_size'],
        epochs=CONFIG['model']['epochs'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history, model