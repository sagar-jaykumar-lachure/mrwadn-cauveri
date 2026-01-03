"""
Ensemble model for flood prediction combining multiple models
"""
import numpy as np
from typing import Dict, Any, List
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class EnsembleFloodModel:
    """
    Ensemble model that combines predictions from multiple models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.prediction_config = config.get('prediction', {})
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Name of the model
            model: Model instance
            weight: Weight for the model in ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def predict_voting(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using majority voting
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            # Convert probabilities to binary predictions
            pred_binary = (pred >= self.prediction_config.get('flood_threshold', 0.5)).astype(int)
            predictions.append(pred_binary)
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = np.round(np.mean(predictions, axis=0)).astype(int)
        
        return final_predictions
    
    def predict_weighted_average(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted average of probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        weighted_predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights[name]
            weighted_predictions.append(pred * weight)
        
        # Weighted average
        final_predictions = np.sum(weighted_predictions, axis=0) / total_weight
        
        return final_predictions
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions using the configured ensemble strategy
        
        Args:
            X: Input features
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        strategy = self.prediction_config.get('ensemble_strategy', 'weighted_average')
        
        # Get individual model predictions
        individual_predictions = {}
        for name, model in self.models.items():
            individual_predictions[name] = model.predict(X)
        
        # Ensemble prediction
        if strategy == 'voting':
            ensemble_pred = self.predict_voting(X)
            probabilities = None
        elif strategy == 'weighted_average':
            probabilities = self.predict_weighted_average(X)
            threshold = self.prediction_config.get('flood_threshold', 0.5)
            ensemble_pred = (probabilities >= threshold).astype(int)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using weighted_average")
            probabilities = self.predict_weighted_average(X)
            threshold = self.prediction_config.get('flood_threshold', 0.5)
            ensemble_pred = (probabilities >= threshold).astype(int)
        
        # Calculate confidence based on agreement between models
        predictions_array = np.array(list(individual_predictions.values()))
        
        # For probability-based models, calculate confidence as agreement level
        if probabilities is not None:
            # Confidence is based on how far the probability is from 0.5
            confidence = np.abs(probabilities - 0.5) * 2
        else:
            # For voting, confidence is based on unanimity
            confidence = np.mean(predictions_array == ensemble_pred, axis=0)
        
        return {
            'predictions': ensemble_pred,
            'probabilities': probabilities if probabilities is not None else ensemble_pred,
            'confidence': confidence,
            'individual_predictions': individual_predictions
        }
    
    def save(self, model_path: str):
        """
        Save the ensemble model
        
        Args:
            model_path: Path to save the ensemble model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'config': self.config
        }
        
        joblib.dump(ensemble_data, model_path)
        logger.info(f"Ensemble model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a saved ensemble model
        
        Args:
            model_path: Path to the saved ensemble model
        """
        ensemble_data = joblib.load(model_path)
        
        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.config = ensemble_data['config']
        
        logger.info(f"Ensemble model loaded from {model_path}")
