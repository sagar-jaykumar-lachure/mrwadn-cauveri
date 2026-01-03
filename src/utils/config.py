"""
Utility functions for the flood prediction system
"""
import yaml
import os
import logging
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "logs", log_file: str = "flood_prediction.log"):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to store log files
        log_file: Name of the log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories based on configuration
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for path_key, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
