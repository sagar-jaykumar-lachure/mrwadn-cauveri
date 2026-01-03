"""
Prediction script for flood events
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from src.utils import load_config, setup_logging
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer
from src.models import EnsembleFloodModel
import logging

logger = logging.getLogger(__name__)


def predict_flood(config_path: str = "config.yaml", 
                 data_path: str = None,
                 model_path: str = None):
    """
    Make flood predictions on new data
    
    Args:
        config_path: Path to configuration file
        data_path: Path to input data
        model_path: Path to trained model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    setup_logging(log_dir, 'flood_prediction_inference.log')
    
    logger.info("Starting flood prediction")
    
    # Default paths
    if model_path is None:
        model_dir = config.get('paths', {}).get('model_dir', 'models')
        model_path = os.path.join(model_dir, 'ensemble_model.pkl')
    
    if data_path is None:
        data_dir = config.get('paths', {}).get('data_dir', 'data')
        data_path = os.path.join(data_dir, 'test_data.csv')
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    ensemble = EnsembleFloodModel(config)
    ensemble.load(model_path)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature engineering
    feature_engineer = FloodFeatureEngineer(config)
    df = feature_engineer.engineer_features(df)
    
    # Get feature names
    feature_cols = feature_engineer.get_feature_names(df)
    
    # Prepare features
    data_processor = FloodDataProcessor(config)
    X = df[feature_cols].values
    
    # Normalize features (using saved scaler from model)
    # Note: In production, the scaler should be saved and loaded
    # For now, we'll use the features as-is
    
    # Make predictions
    logger.info("Making predictions")
    results = ensemble.predict(X)
    
    # Add predictions to dataframe
    df['flood_prediction'] = results['predictions']
    df['flood_probability'] = results['probabilities']
    df['prediction_confidence'] = results['confidence']
    
    # Identify high-confidence predictions
    confidence_threshold = config.get('prediction', {}).get('confidence_threshold', 0.8)
    df['high_confidence'] = df['prediction_confidence'] >= confidence_threshold
    
    # Save results
    output_path = data_path.replace('.csv', '_predictions.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Print summary
    total = len(df)
    flood_count = df['flood_prediction'].sum()
    high_conf_count = df['high_confidence'].sum()
    
    logger.info(f"\nPrediction Summary:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Predicted floods: {flood_count} ({flood_count/total*100:.2f}%)")
    logger.info(f"Predicted no flood: {total - flood_count} ({(total-flood_count)/total*100:.2f}%)")
    logger.info(f"High confidence predictions: {high_conf_count} ({high_conf_count/total*100:.2f}%)")
    
    # Show some examples
    logger.info(f"\nSample predictions:")
    sample_df = df[['timestamp', 'flood_prediction', 'flood_probability', 'prediction_confidence']].head(10) if 'timestamp' in df.columns else df[['flood_prediction', 'flood_probability', 'prediction_confidence']].head(10)
    logger.info(f"\n{sample_df}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flood events')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to input data CSV file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    
    args = parser.parse_args()
    
    predict_flood(args.config, args.data, args.model)
