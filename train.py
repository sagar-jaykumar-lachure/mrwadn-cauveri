"""
Main training pipeline for flood prediction models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from src.utils import load_config, setup_logging, create_directories
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer
from src.models import (
    RandomForestFloodModel,
    XGBoostFloodModel,
    LightGBMFloodModel,
    LSTMFloodModel,
    GRUFloodModel,
    EnsembleFloodModel
)
from src.evaluation import FloodModelEvaluator
import logging

logger = logging.getLogger(__name__)


def train_models(config_path: str = "config.yaml", data_path: str = None):
    """
    Train flood prediction models
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    setup_logging(log_dir)
    
    # Create directories
    create_directories(config)
    
    logger.info("Starting flood prediction model training")
    
    # Check if data path is provided
    if data_path is None:
        data_path = os.path.join(config.get('paths', {}).get('data_dir', 'data'), 'flood_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please provide a valid data file or create sample data")
        return
    
    # Initialize components
    data_processor = FloodDataProcessor(config)
    feature_engineer = FloodFeatureEngineer(config)
    evaluator = FloodModelEvaluator(config)
    
    # Load and process data
    logger.info(f"Loading data from {data_path}")
    df = data_processor.load_data(data_path)
    
    # Handle missing values
    df = data_processor.handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineer.engineer_features(df)
    
    # Get feature names
    feature_cols = feature_engineer.get_feature_names(df)
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Split data
    train_df, val_df, test_df = data_processor.split_data(df, 'flood_event')
    
    # Normalize features
    X_train, X_val, X_test = data_processor.normalize_features(train_df, val_df, test_df, feature_cols)
    
    # Get targets
    y_train = train_df['flood_event'].values
    y_val = val_df['flood_event'].values
    y_test = test_df['flood_event'].values
    
    # Get ensemble model types from config
    ensemble_models = config.get('model', {}).get('ensemble_models', ['random_forest', 'xgboost'])
    
    # Initialize ensemble
    ensemble = EnsembleFloodModel(config)
    
    # Train models based on configuration
    model_dir = config.get('paths', {}).get('model_dir', 'models')
    
    for model_type in ensemble_models:
        logger.info(f"Training {model_type} model")
        
        if model_type == 'random_forest':
            model = RandomForestFloodModel(config)
            model.train(X_train, y_train)
            model.save(os.path.join(model_dir, 'random_forest_model.pkl'))
            ensemble.add_model('random_forest', model, weight=1.0)
            
            # Evaluate
            y_pred = (model.predict(X_test) >= 0.5).astype(int)
            y_pred_proba = model.predict(X_test)
            evaluator.evaluate_model(y_test, y_pred, y_pred_proba, 
                                   'random_forest', 'evaluation')
        
        elif model_type == 'xgboost':
            model = XGBoostFloodModel(config)
            model.train(X_train, y_train, X_val, y_val)
            model.save(os.path.join(model_dir, 'xgboost_model.pkl'))
            ensemble.add_model('xgboost', model, weight=1.2)
            
            # Evaluate
            y_pred = (model.predict(X_test) >= 0.5).astype(int)
            y_pred_proba = model.predict(X_test)
            evaluator.evaluate_model(y_test, y_pred, y_pred_proba,
                                   'xgboost', 'evaluation')
        
        elif model_type == 'lightgbm':
            model = LightGBMFloodModel(config)
            model.train(X_train, y_train, X_val, y_val)
            model.save(os.path.join(model_dir, 'lightgbm_model.pkl'))
            ensemble.add_model('lightgbm', model, weight=1.1)
            
            # Evaluate
            y_pred = (model.predict(X_test) >= 0.5).astype(int)
            y_pred_proba = model.predict(X_test)
            evaluator.evaluate_model(y_test, y_pred, y_pred_proba,
                                   'lightgbm', 'evaluation')
        
        elif model_type == 'lstm':
            # Prepare sequences for LSTM
            sequence_length = config.get('model', {}).get('lstm', {}).get('sequence_length', 30)
            input_shape = (sequence_length, X_train.shape[1])
            
            model = LSTMFloodModel(config, input_shape)
            model.train(X_train, y_train, X_val, y_val,
                       os.path.join(model_dir, 'lstm_model.h5'))
            
            # For ensemble, we need to handle sequence prediction differently
            logger.info("LSTM model trained but not added to ensemble (requires sequence handling)")
        
        elif model_type == 'gru':
            # Prepare sequences for GRU
            sequence_length = config.get('model', {}).get('lstm', {}).get('sequence_length', 30)
            input_shape = (sequence_length, X_train.shape[1])
            
            model = GRUFloodModel(config, input_shape)
            model.train(X_train, y_train, X_val, y_val,
                       os.path.join(model_dir, 'gru_model.h5'))
            
            logger.info("GRU model trained but not added to ensemble (requires sequence handling)")
    
    # Save ensemble
    ensemble.save(os.path.join(model_dir, 'ensemble_model.pkl'))
    
    # Evaluate ensemble
    logger.info("Evaluating ensemble model")
    ensemble_results = ensemble.predict(X_test)
    evaluator.evaluate_model(
        y_test,
        ensemble_results['predictions'],
        ensemble_results['probabilities'],
        'ensemble',
        'evaluation'
    )
    
    logger.info("Training complete!")
    logger.info(f"Models saved to {model_dir}")
    logger.info(f"Evaluation results saved to evaluation/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flood prediction models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data CSV file')
    
    args = parser.parse_args()
    
    train_models(args.config, args.data)
