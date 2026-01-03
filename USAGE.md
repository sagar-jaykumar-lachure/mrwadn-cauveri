# Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training Models](#training-models)
5. [Making Predictions](#making-predictions)
6. [Using the API](#using-the-api)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python generate_sample_data.py

# 3. Test components
python test_components.py

# 4. Run quick demo
python demo.py

# 5. Train full system
python train.py --data data/sample_data.csv

# 6. Make predictions
python predict.py --data data/test_data.csv
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Basic Installation

```bash
# Clone repository
git clone https://github.com/sagar-jaykumar-lachure/mrwadn-cauveri.git
cd mrwadn-cauveri

# Install core dependencies
pip install -r requirements.txt
```

### Optional: Deep Learning Support

If you want to use LSTM/GRU models:

```bash
pip install tensorflow==2.13.0 keras==2.13.1
```

### Verify Installation

```bash
python test_components.py
```

You should see:
```
============================================================
All component tests passed! ✓
============================================================
```

## Data Preparation

### Data Format

Your data should be a CSV file with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| timestamp | datetime | Observation time | 2023-01-01 12:00:00 |
| rainfall | float | Rainfall in mm/hour | 15.5 |
| temperature | float | Temperature in °C | 28.5 |
| humidity | float | Humidity percentage | 75.0 |
| wind_speed | float | Wind speed in km/h | 12.0 |
| pressure | float | Atmospheric pressure in hPa | 1013.0 |
| water_level | float | River water level in meters | 3.5 |
| flow_rate | float | River flow rate in m³/s | 175.0 |
| upstream_level | float | Upstream water level in meters | 3.8 |
| flood_event | int | 0=No flood, 1=Flood | 0 or 1 |

### Generate Sample Data

```bash
python generate_sample_data.py
```

This creates `data/sample_data.csv` with 5,000 samples.

### Using Your Own Data

1. **Prepare CSV**: Ensure your CSV has the required columns
2. **Handle Missing Values**: The system can handle some missing values, but try to minimize them
3. **Time Order**: Sort data by timestamp (ascending)
4. **Check Data**: View first few rows to verify format

```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.head())
print(df.info())
```

## Training Models

### Basic Training

```bash
python train.py --data data/sample_data.csv
```

### Training Options

```bash
# Custom configuration
python train.py --config custom_config.yaml --data data/flood_data.csv

# Training will create:
# - models/random_forest_model.pkl
# - models/xgboost_model.pkl
# - models/ensemble_model.pkl
# - evaluation/*.png (visualization plots)
# - logs/flood_prediction.log
```

### Training Process

The training script will:
1. Load and validate data
2. Engineer 200+ features
3. Split into train/val/test sets (70%/15%/15%)
4. Train each model in the ensemble
5. Evaluate on test set
6. Save models and visualizations

### Expected Output

```
Starting flood prediction model training
Loading data from data/sample_data.csv
Total features: 213
Training random_forest model
Random Forest training complete - Train AUC: 1.0000
Training xgboost model
XGBoost training complete - Train AUC: 1.0000
Training complete!
Models saved to models/
```

### Training Time

- Small dataset (< 10K samples): 2-5 minutes
- Medium dataset (10K-100K samples): 10-30 minutes
- Large dataset (> 100K samples): 30+ minutes

## Making Predictions

### Single File Prediction

```bash
python predict.py --data data/test_data.csv
```

### Custom Model Path

```bash
python predict.py \
  --data data/test_data.csv \
  --model models/ensemble_model.pkl \
  --config config.yaml
```

### Output

The script creates `data/test_data_predictions.csv` with additional columns:
- `flood_prediction`: 0 (no flood) or 1 (flood)
- `flood_probability`: Probability score (0-1)
- `prediction_confidence`: Confidence level (0-1)
- `high_confidence`: True if confidence > threshold

### Example Output

```
Prediction Summary:
Total samples: 750
Predicted floods: 665 (88.67%)
Predicted no flood: 85 (11.33%)
High confidence predictions: 700 (93.33%)

Sample predictions:
             timestamp  flood_prediction  flood_probability  prediction_confidence
0  2020-01-01 00:00:00                 0           0.123456               0.876543
1  2020-01-01 01:00:00                 1           0.987654               0.987654
```

### Programmatic Usage

```python
from src.models import EnsembleFloodModel
from src.utils import load_config
import pandas as pd

# Load model
config = load_config('config.yaml')
model = EnsembleFloodModel(config)
model.load('models/ensemble_model.pkl')

# Load and prepare data
df = pd.read_csv('new_data.csv')
# ... feature engineering ...

# Predict
results = model.predict(X)
predictions = results['predictions']
probabilities = results['probabilities']
confidence = results['confidence']
```

## Using the API

### Start API Server

```bash
python api.py
```

Server starts on `http://localhost:5000`

### API Endpoints

#### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2023-01-01 12:00:00",
    "rainfall": 15.5,
    "temperature": 28.5,
    "humidity": 75.0,
    "wind_speed": 12.0,
    "pressure": 1013.0,
    "water_level": 3.5,
    "flow_rate": 175.0,
    "upstream_level": 3.8
  }'
```

Response:
```json
{
  "success": true,
  "predictions": {
    "timestamp": "2023-01-01 12:00:00",
    "prediction": 0,
    "probability": 0.234,
    "confidence": 0.766,
    "risk_level": "low",
    "message": "No flood event predicted. Risk level: low. Confidence: 76.6%. (Moderate confidence)"
  }
}
```

#### Batch Prediction

```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "file=@data/test_data.csv"
```

#### Model Info

```bash
curl http://localhost:5000/info
```

## Configuration

### Edit config.yaml

```yaml
# Select models for ensemble
model:
  ensemble_models:
    - random_forest
    - xgboost
    # - lstm  # Requires TensorFlow

# Adjust model parameters
  random_forest:
    n_estimators: 200
    max_depth: 20

# Configure features
features:
  lag_periods: [1, 2, 3, 6, 12, 24]
  rolling_windows: [3, 6, 12, 24]

# Set thresholds
prediction:
  flood_threshold: 0.5  # Classify as flood if probability > 0.5
  confidence_threshold: 0.8  # High confidence if > 0.8
```

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| flood_threshold | Probability cutoff for flood classification | 0.5 | 0.0-1.0 |
| confidence_threshold | Minimum confidence for high-confidence predictions | 0.8 | 0.0-1.0 |
| n_estimators | Number of trees in Random Forest | 200 | 10-1000 |
| learning_rate | XGBoost learning rate | 0.05 | 0.001-0.3 |

### Tuning for Your Data

**High Precision (few false alarms)**:
- Increase `flood_threshold` to 0.6-0.7
- Use XGBoost with `max_depth: 15`

**High Recall (catch all floods)**:
- Decrease `flood_threshold` to 0.3-0.4
- Use ensemble with all models

**Fast Predictions**:
- Use only Random Forest
- Reduce `n_estimators` to 100

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem**: Missing Python packages

**Solution**:
```bash
pip install -r requirements.txt
```

#### 2. Data Format Error

**Problem**: CSV doesn't have required columns

**Solution**: Check your CSV has all required columns:
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.columns.tolist())
```

Required: timestamp, rainfall, temperature, humidity, wind_speed, pressure, water_level, flow_rate, upstream_level, flood_event

#### 3. Low Accuracy

**Problem**: Model performs poorly

**Solutions**:
- Ensure sufficient training data (> 1000 samples)
- Check data quality (no excessive missing values)
- Verify flood events are realistic
- Try different models in ensemble
- Adjust flood_threshold

#### 4. TensorFlow Warnings

**Problem**: Deep learning warnings/errors

**Solution**: Deep learning is optional. Remove 'lstm' and 'gru' from `ensemble_models` in config.yaml

#### 5. Out of Memory

**Problem**: Training crashes

**Solutions**:
- Reduce dataset size
- Use fewer lag/rolling features
- Train models individually
- Use LightGBM instead of XGBoost

### Getting Help

1. Check logs in `logs/` directory
2. Run component tests: `python test_components.py`
3. Try demo with sample data: `python demo.py`
4. Review configuration: `config.yaml`
5. Check ARCHITECTURE.md for system design

### Performance Tips

**Speed up training**:
- Reduce `n_estimators`
- Use fewer lag periods
- Sample large datasets

**Improve accuracy**:
- More training data
- Add domain-specific features
- Tune hyperparameters
- Use cross-validation

**Reduce memory usage**:
- Fewer rolling windows
- Load data in chunks
- Use float32 instead of float64

## Best Practices

1. **Always split data chronologically** - Don't shuffle time series data
2. **Validate predictions** - Check against known flood events
3. **Monitor performance** - Track accuracy over time
4. **Regular retraining** - Update models monthly
5. **Keep logs** - Enable detailed logging for debugging
6. **Backup models** - Save trained models with version numbers
7. **Test changes** - Use validation set before production deployment

## Examples

### Complete Workflow

```bash
# 1. Prepare data
python generate_sample_data.py

# 2. Verify system
python test_components.py

# 3. Quick test
python demo.py

# 4. Train models
python train.py --data data/sample_data.csv

# 5. Make predictions
python predict.py --data data/sample_data.csv

# 6. Start API
python api.py
```

### Python Script Example

```python
#!/usr/bin/env python3
"""Complete flood prediction workflow"""

from src.utils import load_config, setup_logging
from src.data_processing import FloodDataProcessor
from src.features import FloodFeatureEngineer
from src.models import EnsembleFloodModel
from src.evaluation import FloodModelEvaluator

# Setup
config = load_config('config.yaml')
setup_logging('logs', 'workflow.log')

# Process data
processor = FloodDataProcessor(config)
engineer = FloodFeatureEngineer(config)

df = processor.load_data('data/flood_data.csv')
df = processor.handle_missing_values(df)
df = engineer.engineer_features(df)

feature_cols = engineer.get_feature_names(df)
train_df, val_df, test_df = processor.split_data(df)
X_train, X_val, X_test = processor.normalize_features(
    train_df, val_df, test_df, feature_cols
)
y_train = train_df['flood_event'].values
y_test = test_df['flood_event'].values

# Train model
from src.models import XGBoostFloodModel
model = XGBoostFloodModel(config)
model.train(X_train, y_train, X_val, val_df['flood_event'].values)
model.save('models/my_model.pkl')

# Evaluate
evaluator = FloodModelEvaluator(config)
y_pred = (model.predict(X_test) >= 0.5).astype(int)
y_pred_proba = model.predict(X_test)
evaluator.evaluate_model(y_test, y_pred, y_pred_proba, 'my_model')

print("✓ Training complete!")
```
