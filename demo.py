"""
Quick demo with a simple model to verify the system works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, setup_logging
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer
from src.models import RandomForestFloodModel, XGBoostFloodModel
from src.evaluation import FloodModelEvaluator

print("=" * 60)
print("Quick Demo: Training Flood Prediction Models")
print("=" * 60)

# Load configuration
config = load_config('config.yaml')
setup_logging('logs', 'demo.log')

# Initialize components
data_processor = FloodDataProcessor(config)
feature_engineer = FloodFeatureEngineer(config)
evaluator = FloodModelEvaluator(config)

# Load data
print("\n1. Loading data...")
df = data_processor.load_data('data/sample_data.csv')
df = data_processor.handle_missing_values(df)
print(f"   Loaded {len(df)} samples")

# Feature engineering
print("\n2. Engineering features...")
df = feature_engineer.engineer_features(df)
feature_cols = feature_engineer.get_feature_names(df)
print(f"   Created {len(feature_cols)} features")

# Split and normalize
print("\n3. Preparing data...")
train_df, val_df, test_df = data_processor.split_data(df, 'flood_event')
X_train, X_val, X_test = data_processor.normalize_features(train_df, val_df, test_df, feature_cols)
y_train = train_df['flood_event'].values
y_val = val_df['flood_event'].values
y_test = test_df['flood_event'].values
print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train Random Forest
print("\n4. Training Random Forest model...")
rf_model = RandomForestFloodModel(config)
rf_model.train(X_train, y_train)
print("   ✓ Random Forest trained")

# Evaluate Random Forest
y_pred_rf = (rf_model.predict(X_test) >= 0.5).astype(int)
y_pred_proba_rf = rf_model.predict(X_test)
metrics_rf = evaluator.calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf)

print("\n   Random Forest Results:")
print(f"   - Accuracy: {metrics_rf['accuracy']:.4f}")
print(f"   - Precision: {metrics_rf['precision']:.4f}")
print(f"   - Recall: {metrics_rf['recall']:.4f}")
print(f"   - F1 Score: {metrics_rf['f1_score']:.4f}")
print(f"   - AUC-ROC: {metrics_rf['auc_roc']:.4f}")

# Train XGBoost
print("\n5. Training XGBoost model...")
xgb_model = XGBoostFloodModel(config)
xgb_model.train(X_train, y_train, X_val, y_val)
print("   ✓ XGBoost trained")

# Evaluate XGBoost
y_pred_xgb = (xgb_model.predict(X_test) >= 0.5).astype(int)
y_pred_proba_xgb = xgb_model.predict(X_test)
metrics_xgb = evaluator.calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb)

print("\n   XGBoost Results:")
print(f"   - Accuracy: {metrics_xgb['accuracy']:.4f}")
print(f"   - Precision: {metrics_xgb['precision']:.4f}")
print(f"   - Recall: {metrics_xgb['recall']:.4f}")
print(f"   - F1 Score: {metrics_xgb['f1_score']:.4f}")
print(f"   - AUC-ROC: {metrics_xgb['auc_roc']:.4f}")

print("\n" + "=" * 60)
print("Demo Complete! Models trained successfully.")
print("=" * 60)
print("\nThe system is working correctly and ready for full use.")
print("Run: python train.py --data data/sample_data.csv")
