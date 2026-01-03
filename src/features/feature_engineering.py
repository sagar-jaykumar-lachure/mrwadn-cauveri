"""
Feature engineering module for flood prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FloodFeatureEngineer:
    """
    Creates features for flood prediction models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('features', {})
        
    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create temporal features from timestamp
        
        Args:
            df: Input DataFrame with timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with added temporal features
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found. Skipping temporal features.")
            return df
        
        logger.info("Creating temporal features")
        
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract temporal features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        df['day_of_month'] = df[timestamp_col].dt.day
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        
        # Season (0: Winter, 1: Spring, 2: Summer, 3: Fall)
        df['season'] = (df['month'] % 12 + 3) // 3
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create lag features for specified columns
        
        Args:
            df: Input DataFrame
            columns: List of column names to create lag features for
            
        Returns:
            DataFrame with added lag features
        """
        lag_periods = self.feature_config.get('lag_periods', [1, 2, 3, 6, 12, 24])
        
        logger.info(f"Creating lag features for {len(columns)} columns with periods: {lag_periods}")
        
        for col in columns:
            if col in df.columns:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df: Input DataFrame
            columns: List of column names to create rolling features for
            
        Returns:
            DataFrame with added rolling features
        """
        rolling_windows = self.feature_config.get('rolling_windows', [3, 6, 12, 24])
        
        logger.info(f"Creating rolling features for {len(columns)} columns with windows: {rolling_windows}")
        
        for col in columns:
            if col in df.columns:
                for window in rolling_windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between related variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added interaction features
        """
        logger.info("Creating interaction features")
        
        # Water level and rainfall interaction
        if 'water_level' in df.columns and 'rainfall' in df.columns:
            df['water_rainfall_interaction'] = df['water_level'] * df['rainfall']
        
        # Flow rate and water level interaction
        if 'flow_rate' in df.columns and 'water_level' in df.columns:
            df['flow_water_interaction'] = df['flow_rate'] * df['water_level']
        
        # Rainfall intensity (rainfall with wind speed)
        if 'rainfall' in df.columns and 'wind_speed' in df.columns:
            df['rainfall_intensity'] = df['rainfall'] * df['wind_speed']
        
        return df
    
    def create_rate_of_change_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create rate of change features
        
        Args:
            df: Input DataFrame
            columns: List of column names to create rate of change features for
            
        Returns:
            DataFrame with added rate of change features
        """
        logger.info(f"Creating rate of change features for {len(columns)} columns")
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change()
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, base_columns: List[str] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input DataFrame
            base_columns: List of base column names to create features from
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Determine base columns for feature engineering
        if base_columns is None:
            base_columns = []
            for category in ['weather', 'river']:
                base_columns.extend(self.feature_config.get(category, []))
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, base_columns)
        
        # Create rolling features
        df = self.create_rolling_features(df, base_columns)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create rate of change features
        df = self.create_rate_of_change_features(df, base_columns)
        
        # Fill NaN values created by feature engineering
        df = df.fillna(method='bfill').fillna(0)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature column names (excluding target and timestamp)
        
        Args:
            df: Input DataFrame
            exclude_cols: List of columns to exclude
            
        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'flood_event', 'date', 'time']
        
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return feature_names
