import yaml
import os

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("❌ config.yaml not found. Using default settings.")
        return {}

def setup_directories():
    """Create all necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/embeddings',
        'models',
        'notebooks',
        'src',
        'scripts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ All directories created successfully!")

# Global config variable
CONFIG = load_config()