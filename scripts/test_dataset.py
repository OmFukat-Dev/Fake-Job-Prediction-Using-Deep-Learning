import pandas as pd
import os

def test_dataset():
    print("Testing dataset...")
    
    # Check if file exists
    file_path = "data/raw/fake_job_postings.csv"
    if not os.path.exists(file_path):
        print("❌ Dataset file not found!")
        print("Please download from: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction")
        print("And save as: data/raw/fake_job_postings.csv")
        return False
    
    # Try to load the data
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Fraudulent jobs: {data['fraudulent'].sum()} out of {len(data)}")
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    test_dataset()