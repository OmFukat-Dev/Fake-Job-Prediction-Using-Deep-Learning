# Fake Job Detection using Deep Learning

A deep learning project to detect fraudulent job postings using LSTM networks and metadata features.

## Project Structure

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Place the dataset at:
   `data/raw/fake_job_postings.csv`

## Train the Model
1. Clean and split the data:
   `python scripts/run_data_cleaning.py`
2. Train and save the model + tokenizer:
   `python scripts/train_model.py`

## Run the App
`streamlit run app.py`

## Outputs
- Model: `models/lstm_text_model.h5`
- Tokenizer: `models/tokenizer.pkl`
- Metrics: `models/model_performance.txt`
