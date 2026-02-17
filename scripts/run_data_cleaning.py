import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_raw_data, split_data, save_split_data
from src.preprocess import preprocess_data
from src.config import setup_directories

def main():
    print("🚀 Starting data cleaning pipeline...")
    
    # Setup directories
    setup_directories()
    
    # Load data
    data = load_raw_data()
    if data is None:
        return
    
    # Preprocess data
    cleaned_data = preprocess_data(data)
    
    # Split data
    train_data, val_data, test_data = split_data(cleaned_data)
    
    # Save split data
    save_split_data(train_data, val_data, test_data)
    
    print("✅ Data cleaning pipeline completed successfully!")

if __name__ == "__main__":
    main()