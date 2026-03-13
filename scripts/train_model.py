import os
import sys
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import CONFIG, setup_directories
from src.model import create_combined_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model_file, save_metrics


def build_text_features(df, tokenizer, max_len):
    combined = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['company_profile'].fillna('')
    )
    seq = tokenizer.texts_to_sequences(combined)
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')


def build_numeric_features(df):
    cols = ['telecommuting', 'has_company_logo', 'has_questions']
    return df[cols].fillna(0).astype(np.float32).values


def main():
    print("Starting model training...")
    setup_directories()

    train_path = os.path.join('data', 'processed', 'train_data.csv')
    val_path = os.path.join('data', 'processed', 'val_data.csv')
    test_path = os.path.join('data', 'processed', 'test_data.csv')

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing processed data: {p}. Run scripts/run_data_cleaning.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(
        train_df['title'].fillna('') + ' ' +
        train_df['description'].fillna('') + ' ' +
        train_df['requirements'].fillna('') + ' ' +
        train_df['company_profile'].fillna('')
    )
    tokenizer_path = os.path.join('models', 'tokenizer.pkl')
    joblib.dump(tokenizer, tokenizer_path)
    print(f"Tokenizer saved: {tokenizer_path}")

    max_len = CONFIG['model']['text_seq_length']

    X_train_text = build_text_features(train_df, tokenizer, max_len)
    X_val_text = build_text_features(val_df, tokenizer, max_len)
    X_test_text = build_text_features(test_df, tokenizer, max_len)

    X_train_num = build_numeric_features(train_df)
    X_val_num = build_numeric_features(val_df)
    X_test_num = build_numeric_features(test_df)

    y_train = train_df['fraudulent'].astype(np.float32).values
    y_val = val_df['fraudulent'].astype(np.float32).values
    y_test = test_df['fraudulent'].astype(np.float32).values

    vocab_size = min(10000, len(tokenizer.word_index) + 1)
    metadata_dim = X_train_num.shape[1]

    model = create_combined_model(vocab_size, metadata_dim)

    _, model = train_model(
        model,
        [X_train_text, X_train_num],
        y_train,
        [X_val_text, X_val_num],
        y_val
    )

    model_path = os.path.join('models', 'lstm_text_model.h5')
    save_model_file(model, model_path)

    metrics = evaluate_model(model, [X_test_text, X_test_num], y_test)
    metrics_path = os.path.join('models', 'model_performance.txt')
    save_metrics(metrics, metrics_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
