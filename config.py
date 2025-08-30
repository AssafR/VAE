"""
Configuration file for VAE training
"""
import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from YAML file"""
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        print("Warning: config.yaml not found. Using default values.")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")
        print("Using default values.")
        return get_default_config()

def get_default_config():
    """Return default configuration if YAML loading fails"""
    return {
        'huggingface': {
            'username': 'AssafR2',
            'repo_name': 'vae-celeba',
            'repo_type': 'model',
            'token': None
        },
        'training': {
            'batch_size': 128,
            'z_size': 512,
            'epochs': 40,
            'learning_rate': 0.0005,
            'checkpoint_interval': 5,
            'sample_interval': 60
        },
        'model': {
            'layer_count': 5,
            'im_size': 128
        },
        'paths': {
            'checkpoint_dir': 'checkpoints',
            'results_dir': 'results',
            'data_dir': '.'
        }
    }

# Load configuration
config = load_config()

# Hugging Face Hub Configuration
HF_USERNAME = config['huggingface']['username']
REPO_NAME = config['huggingface']['repo_name']
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
REPO_TYPE = config['huggingface']['repo_type']
HF_TOKEN = config['huggingface']['token']

# Training Configuration
BATCH_SIZE = config['training']['batch_size']
Z_SIZE = config['training']['z_size']
TRAIN_EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']
CHECKPOINT_INTERVAL = config['training']['checkpoint_interval']
SAMPLE_INTERVAL = config['training']['sample_interval']

# Model Configuration
LAYER_COUNT = config['model']['layer_count']
IM_SIZE = config['model']['im_size']

# Paths
CHECKPOINT_DIR = Path(config['paths']['checkpoint_dir'])
RESULTS_DIR = Path(config['paths']['results_dir'])
DATA_DIR = Path(config['paths']['data_dir'])

# Ensure directories exist
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "rec").mkdir(exist_ok=True)
(RESULTS_DIR / "gen").mkdir(exist_ok=True)

def print_config_status():
    """Print configuration status - only call this when needed"""
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        print("Warning: Hugging Face token not set in config.yaml")
        print("Edit config.yaml and set your token in the 'huggingface.token' field")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Checkpointing to Hub will fail until token is set.")
    else:
        print(f"✓ Hugging Face token loaded for repository: {REPO_ID}")

# Only print status if this file is run directly
if __name__ == "__main__":
    print_config_status()
