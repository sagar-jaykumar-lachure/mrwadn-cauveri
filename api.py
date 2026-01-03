"""
Flask API for flood prediction service
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from src.utils import load_config, setup_logging
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer
from src.models import EnsembleFloodModel
import logging

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and config
model = None
config = None
feature_engineer = None
data_processor = None


def initialize_model(config_path='config.yaml', model_path='models/ensemble_model.pkl'):
    """Initialize the model and configuration"""
    global model, config, feature_engineer, data_processor
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    setup_logging(log_dir, 'flood_api.log')
    
    logger.info("Initializing flood prediction API")
    
    # Load model
    model = EnsembleFloodModel(config)
    if os.path.exists(model_path):
        model.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Initialize feature engineer and data processor
    feature_engineer = FloodFeatureEngineer(config)
    data_processor = FloodDataProcessor(config)
    
    logger.info("API initialization complete")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict flood event
    
    Expected JSON format:
    {
        "timestamp": "2023-01-01 12:00:00",
        "rainfall": 10.5,
        "temperature": 28.0,
        "humidity": 75.0,
        "wind_speed": 12.0,
        "pressure": 1013.0,
        "water_level": 3.5,
        "flow_rate": 175.0,
        "upstream_level": 3.8
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Ensure timestamp is present
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        # Feature engineering
        df = feature_engineer.engineer_features(df)
        
        # Get feature names
        feature_cols = feature_engineer.get_feature_names(df)
        
        # Prepare features
        X = df[feature_cols].values
        
        # Make predictions
        results = model.predict(X)
        
        # Prepare response
        response = []
        for i in range(len(df)):
            response.append({
                'timestamp': str(df['timestamp'].iloc[i]) if 'timestamp' in df.columns else None,
                'prediction': int(results['predictions'][i]),
                'probability': float(results['probabilities'][i]),
                'confidence': float(results['confidence'][i]),
                'risk_level': get_risk_level(results['probabilities'][i]),
                'message': get_prediction_message(
                    results['predictions'][i],
                    results['probabilities'][i],
                    results['confidence'][i]
                )
            })
        
        return jsonify({
            'success': True,
            'predictions': response if len(response) > 1 else response[0]
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction from CSV file
    
    Expected: CSV file in request files
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Feature engineering
        df = feature_engineer.engineer_features(df)
        
        # Get feature names
        feature_cols = feature_engineer.get_feature_names(df)
        
        # Prepare features
        X = df[feature_cols].values
        
        # Make predictions
        results = model.predict(X)
        
        # Add predictions to dataframe
        df['flood_prediction'] = results['predictions']
        df['flood_probability'] = results['probabilities']
        df['prediction_confidence'] = results['confidence']
        df['risk_level'] = [get_risk_level(p) for p in results['probabilities']]
        
        # Convert to JSON
        response = df.to_dict('records')
        
        return jsonify({
            'success': True,
            'total_samples': len(df),
            'flood_predictions': int(results['predictions'].sum()),
            'predictions': response
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return 'low'
    elif probability < 0.6:
        return 'moderate'
    elif probability < 0.8:
        return 'high'
    else:
        return 'critical'


def get_prediction_message(prediction, probability, confidence):
    """Generate human-readable prediction message"""
    risk_level = get_risk_level(probability)
    
    if prediction == 1:
        base_message = f"Flood event predicted with {probability*100:.1f}% probability."
    else:
        base_message = f"No flood event predicted. Risk level: {risk_level}."
    
    confidence_message = f" Confidence: {confidence*100:.1f}%."
    
    if confidence < 0.6:
        warning = " (Low confidence - consider additional monitoring)"
    elif confidence < 0.8:
        warning = " (Moderate confidence)"
    else:
        warning = " (High confidence)"
    
    return base_message + confidence_message + warning


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Ensemble (Random Forest + XGBoost)',
        'features': {
            'weather': config.get('features', {}).get('weather', []),
            'river': config.get('features', {}).get('river', []),
            'engineered': ['temporal', 'lag', 'rolling', 'interaction']
        },
        'thresholds': {
            'flood_threshold': config.get('prediction', {}).get('flood_threshold', 0.5),
            'confidence_threshold': config.get('prediction', {}).get('confidence_threshold', 0.8)
        }
    })


if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False)
