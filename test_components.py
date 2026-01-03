"""
Quick test script to verify components work
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import load_config, setup_logging
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer

print("=" * 60)
print("Testing Flood Prediction System Components")
print("=" * 60)

# Test 1: Load configuration
print("\n1. Testing configuration loading...")
try:
    config = load_config('config.yaml')
    print("   ✓ Configuration loaded successfully")
    print(f"   - Model types: {config.get('model', {}).get('ensemble_models', [])}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Setup logging
print("\n2. Testing logging setup...")
try:
    setup_logging('logs', 'test.log')
    print("   ✓ Logging configured successfully")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Load sample data
print("\n3. Testing data loading...")
try:
    data_processor = FloodDataProcessor(config)
    df = data_processor.load_data('data/sample_data.csv')
    print(f"   ✓ Data loaded: {len(df)} samples")
    print(f"   - Features: {df.columns.tolist()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Handle missing values
print("\n4. Testing missing value handling...")
try:
    df = data_processor.handle_missing_values(df)
    print(f"   ✓ Missing values handled")
    print(f"   - Remaining NaN: {df.isnull().sum().sum()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Feature engineering
print("\n5. Testing feature engineering...")
try:
    feature_engineer = FloodFeatureEngineer(config)
    df_engineered = feature_engineer.engineer_features(df)
    print(f"   ✓ Features engineered")
    print(f"   - Original features: {len(df.columns)}")
    print(f"   - Engineered features: {len(df_engineered.columns)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Get feature names
print("\n6. Testing feature name extraction...")
try:
    feature_cols = feature_engineer.get_feature_names(df_engineered)
    print(f"   ✓ Feature names extracted")
    print(f"   - Total features for modeling: {len(feature_cols)}")
    print(f"   - Sample features: {feature_cols[:5]}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 7: Data splitting and normalization
print("\n7. Testing data splitting and normalization...")
try:
    # Split the engineered data
    train_df, val_df, test_df = data_processor.split_data(df_engineered, 'flood_event')
    
    # Normalize features
    X_train, X_val, X_test = data_processor.normalize_features(train_df, val_df, test_df, feature_cols)
    
    # Get targets
    y_train = train_df['flood_event'].values
    y_val = val_df['flood_event'].values
    y_test = test_df['flood_event'].values
    
    print(f"   ✓ Data processed successfully")
    print(f"   - Train samples: {len(X_train)}")
    print(f"   - Val samples: {len(X_val)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Feature dimension: {X_train.shape[1]}")
    print(f"   - Train flood events: {y_train.sum()}/{len(y_train)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All component tests passed! ✓")
print("=" * 60)
print("\nSystem is ready for training.")
print("Run: python train.py --data data/sample_data.csv")
