"""
Models package
"""
from .traditional_ml import RandomForestFloodModel, XGBoostFloodModel, LightGBMFloodModel
from .ensemble import EnsembleFloodModel

# Try to import deep learning models (optional - requires TensorFlow)
try:
    from .deep_learning import LSTMFloodModel, GRUFloodModel
    __all__ = [
        'LSTMFloodModel',
        'GRUFloodModel',
        'RandomForestFloodModel',
        'XGBoostFloodModel',
        'LightGBMFloodModel',
        'EnsembleFloodModel'
    ]
except ImportError:
    # TensorFlow not available, skip deep learning models
    __all__ = [
        'RandomForestFloodModel',
        'XGBoostFloodModel',
        'LightGBMFloodModel',
        'EnsembleFloodModel'
    ]
