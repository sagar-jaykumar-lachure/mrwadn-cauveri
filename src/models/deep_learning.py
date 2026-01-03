"""
Deep Learning models for flood prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Dict, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class LSTMFloodModel:
    """
    LSTM-based model for flood prediction
    """
    
    def __init__(self, config: Dict[str, Any], input_shape: Tuple[int, int]):
        """
        Initialize LSTM model
        
        Args:
            config: Configuration dictionary
            input_shape: Shape of input data (sequence_length, num_features)
        """
        self.config = config
        self.lstm_config = config.get('model', {}).get('lstm', {})
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model")
        
        units = self.lstm_config.get('units', [128, 64, 32])
        dropout = self.lstm_config.get('dropout', 0.3)
        learning_rate = self.lstm_config.get('learning_rate', 0.001)
        
        model = models.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(layers.Dropout(dropout))
        
        # Middle LSTM layers
        for unit in units[1:-1]:
            model.add(layers.LSTM(unit, return_sequences=True))
            model.add(layers.Dropout(dropout))
        
        # Last LSTM layer
        model.add(layers.LSTM(units[-1], return_sequences=False))
        model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(16, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"LSTM model built with {model.count_params()} parameters")
        
        return model
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None, 
                         sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            X: Input features
            y: Target labels (optional)
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if sequence_length is None:
            sequence_length = self.lstm_config.get('sequence_length', 30)
        
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            if y is not None:
                y_seq.append(y[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_path: str = 'models/lstm_model.h5') -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_path: Path to save the trained model
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Training LSTM model")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # Training parameters
        batch_size = self.lstm_config.get('batch_size', 32)
        epochs = self.lstm_config.get('epochs', 100)
        patience = self.config.get('training', {}).get('early_stopping_patience', 15)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add ModelCheckpoint if path provided
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callback_list.append(
                callbacks.ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("LSTM model training complete")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_seq, _ = self.prepare_sequences(X)
        predictions = self.model.predict(X_seq)
        
        return predictions
    
    def save(self, model_path: str):
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")


class GRUFloodModel(LSTMFloodModel):
    """
    GRU-based model for flood prediction (inherits from LSTM)
    """
    
    def build_model(self) -> keras.Model:
        """
        Build GRU model architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building GRU model")
        
        units = self.lstm_config.get('units', [128, 64, 32])
        dropout = self.lstm_config.get('dropout', 0.3)
        learning_rate = self.lstm_config.get('learning_rate', 0.001)
        
        model = models.Sequential()
        
        # First GRU layer
        model.add(layers.GRU(units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(layers.Dropout(dropout))
        
        # Middle GRU layers
        for unit in units[1:-1]:
            model.add(layers.GRU(unit, return_sequences=True))
            model.add(layers.Dropout(dropout))
        
        # Last GRU layer
        model.add(layers.GRU(units[-1], return_sequences=False))
        model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(16, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        logger.info(f"GRU model built with {model.count_params()} parameters")
        
        return model
