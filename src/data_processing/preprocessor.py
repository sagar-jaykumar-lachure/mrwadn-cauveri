"""
Data preprocessing module for flood prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FloodDataProcessor:
    """
    Preprocesses flood-related data for model training and prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.scaler = self._get_scaler()
        self.feature_columns = None
        
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration"""
        norm_method = self.data_config.get('normalization', 'minmax')
        
        if norm_method == 'minmax':
            return MinMaxScaler()
        elif norm_method == 'standard':
            return StandardScaler()
        elif norm_method == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"Unknown normalization method: {norm_method}. Using MinMaxScaler.")
            return MinMaxScaler()
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            data_path: Path to the data file
            
        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert timestamp to datetime if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        method = self.data_config.get('missing_values', 'interpolate')
        
        logger.info(f"Handling missing values using method: {method}")
        
        if method == 'forward_fill':
            df = df.ffill()
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        
        # Fill any remaining NaN values with forward fill
        df = df.ffill().bfill()
        
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'flood_event') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_split = self.data_config.get('train_split', 0.7)
        val_split = self.data_config.get('val_split', 0.15)
        
        n = len(df)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using the specified scaler
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Tuple of normalized (train_features, val_features, test_features)
        """
        self.feature_columns = feature_cols
        
        # Fit scaler on training data only
        X_train = self.scaler.fit_transform(train_df[feature_cols])
        X_val = self.scaler.transform(val_df[feature_cols])
        X_test = self.scaler.transform(test_df[feature_cols])
        
        logger.info(f"Features normalized using {self.scaler.__class__.__name__}")
        
        return X_train, X_val, X_test
    
    def process_data(self, data_path: str, feature_cols: list, target_col: str = 'flood_event') -> Dict[str, Any]:
        """
        Complete data processing pipeline
        
        Args:
            data_path: Path to the data file
            feature_cols: List of feature column names
            target_col: Name of the target column
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Load data
        df = self.load_data(data_path)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df, target_col)
        
        # Normalize features
        X_train, X_val, X_test = self.normalize_features(train_df, val_df, test_df, feature_cols)
        
        # Extract targets
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': feature_cols,
            'scaler': self.scaler
        }
