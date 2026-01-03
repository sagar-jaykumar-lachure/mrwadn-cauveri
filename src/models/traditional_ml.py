"""
Traditional ML models for flood prediction
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class RandomForestFloodModel:
    """
    Random Forest model for flood prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Random Forest model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rf_config = config.get('model', {}).get('random_forest', {})
        self.model = None
        
    def build_model(self) -> RandomForestClassifier:
        """
        Build Random Forest model
        
        Returns:
            RandomForestClassifier instance
        """
        logger.info("Building Random Forest model")
        
        n_estimators = self.rf_config.get('n_estimators', 200)
        max_depth = self.rf_config.get('max_depth', 20)
        min_samples_split = self.rf_config.get('min_samples_split', 5)
        min_samples_leaf = self.rf_config.get('min_samples_leaf', 2)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Training Random Forest model")
        
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        
        # Get training metrics
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_precision': precision_score(y_train, y_pred),
            'train_recall': recall_score(y_train, y_pred),
            'train_f1': f1_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }
        
        logger.info(f"Random Forest training complete - Train AUC: {metrics['train_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, model_path: str):
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")


class XGBoostFloodModel:
    """
    XGBoost model for flood prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.xgb_config = config.get('model', {}).get('xgboost', {})
        self.model = None
        
    def build_model(self) -> xgb.XGBClassifier:
        """
        Build XGBoost model
        
        Returns:
            XGBClassifier instance
        """
        logger.info("Building XGBoost model")
        
        n_estimators = self.xgb_config.get('n_estimators', 200)
        max_depth = self.xgb_config.get('max_depth', 10)
        learning_rate = self.xgb_config.get('learning_rate', 0.05)
        subsample = self.xgb_config.get('subsample', 0.8)
        colsample_bytree = self.xgb_config.get('colsample_bytree', 0.8)
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Training XGBoost model")
        
        if self.model is None:
            self.build_model()
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Get training metrics
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_precision': precision_score(y_train, y_pred),
            'train_recall': recall_score(y_train, y_pred),
            'train_f1': f1_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }
        
        logger.info(f"XGBoost training complete - Train AUC: {metrics['train_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, model_path: str):
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")


class LightGBMFloodModel:
    """
    LightGBM model for flood prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lgb_config = config.get('model', {}).get('lightgbm', {})
        self.model = None
        
    def build_model(self) -> lgb.LGBMClassifier:
        """
        Build LightGBM model
        
        Returns:
            LGBMClassifier instance
        """
        logger.info("Building LightGBM model")
        
        n_estimators = self.lgb_config.get('n_estimators', 200)
        max_depth = self.lgb_config.get('max_depth', 10)
        learning_rate = self.lgb_config.get('learning_rate', 0.05)
        num_leaves = self.lgb_config.get('num_leaves', 31)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Training LightGBM model")
        
        if self.model is None:
            self.build_model()
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='auc'
        )
        
        # Get training metrics
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_precision': precision_score(y_train, y_pred),
            'train_recall': recall_score(y_train, y_pred),
            'train_f1': f1_score(y_train, y_pred),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }
        
        logger.info(f"LightGBM training complete - Train AUC: {metrics['train_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, model_path: str):
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
