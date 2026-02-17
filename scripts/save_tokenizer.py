"""
Save the fitted tokenizer for consistent predictions
"""

import pandas as pd
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import sys

# Add the parent directory to Python path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to main project folder
sys.path.append(project_root)

print("💾 Saving fitted tokenizer...")

# Get the correct path
data_path = os.path.join(project_root, 'data', 'processed', 'train_data.csv')
print(f"Looking for data at: {data_path}")

# Check if file exists first
if not os.path.exists(data_path):
    print("❌ Processed data not found! Running data cleaning first...")
    
    # Try to run data cleaning
    try:
        import subprocess
        print("🔄 Running data cleaning script...")
        
        # Use the correct Python executable from venv
        python_exe = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
        cleaning_script = os.path.join(project_root, 'scripts', 'run_data_cleaning.py')
        
        result = subprocess.run([python_exe, cleaning_script], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Data cleaning failed: {result.stderr}")
            print("Please run: python scripts/run_data_cleaning.py manually first")
            input("Press Enter to exit...")
            sys.exit(1)
        else:
            print("✅ Data cleaning completed successfully!")
            
    except Exception as e:
        print(f"❌ Data cleaning failed: {e}")
        print("Please run: python scripts/run_data_cleaning.py manually first")
        input("Press Enter to exit...")
        sys.exit(1)

try:
    # Load training data
    train_data = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully! Shape: {train_data.shape}")

    # Combine text features
    text_columns = ['description', 'requirements', 'company_profile']
    train_data['combined_text'] = train_data[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data['combined_text'])

    # Save tokenizer
    tokenizer_path = os.path.join(project_root, 'models', 'tokenizer.pkl')
    joblib.dump(tokenizer, tokenizer_path)
    print(f"✅ Tokenizer saved to: {tokenizer_path}")

    # Test loading
    loaded_tokenizer = joblib.load(tokenizer_path)
    print(f"✅ Tokenizer loaded successfully! Vocabulary size: {len(loaded_tokenizer.word_index)}")
    print("🎉 Tokenizer creation complete!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease make sure you have:")
    print("1. Downloaded the dataset to data/raw/fake_job_postings.csv")
    print("2. Run: python scripts/run_data_cleaning.py")
    input("Press Enter to exit...")
    sys.exit(1)