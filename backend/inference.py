import os
import sys
import json
import joblib
import numpy as np
import torch
from functools import lru_cache

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Ensure we can import from data directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

from feature_config import FEATURE_COLS, NORMALIZATION_STRATEGY, MODEL_DIR, SCALER_PKL
from collect_data import (
    geocode_location, fetch_osm_features, fetch_google_places,
    fetch_network_metrics, engineer_features
)

sys.path.insert(0, MODEL_DIR)
from model_def import MUISNN

# Initialize globals
_scalers = None
_model = None
_xgb_model = None
_ensemble_weight_ann = None

def load_models():
    """Load the scalers, PyTorch ANN, and XGBoost models into memory."""
    global _scalers, _model, _xgb_model, _ensemble_weight_ann
    if _scalers is None:
        _scalers = joblib.load(SCALER_PKL)
        
    if _model is None or _xgb_model is None or _ensemble_weight_ann is None:
        # Load hyperparams and ensemble weight
        history_path = os.path.join(MODEL_DIR, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {}
            
        _ensemble_weight_ann = history.get('ensemble_weight_ann', 0.5)
        
        # 1. Load PyTorch ANN model
        if _model is None and 'best_params' in history:
            best_params = history['best_params']
            n_layers = best_params['n_layers']
            hidden_dims = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
            
            # Instantiate model architecture
            _model = MUISNN(
                input_dim=len(FEATURE_COLS),
                hidden_dims=hidden_dims,
                dropout_rate=best_params['dropout_rate'],
                activation_fn=best_params['activation_fn']
            )
            
            # Load weights
            model_path = os.path.join(MODEL_DIR, 'ann_model.pth')
            if os.path.exists(model_path):
                _model.load_state_dict(torch.load(model_path, map_location='cpu'))
                _model.eval()
                
        # 2. Load XGBoost model
        if _xgb_model is None:
            xgb_model_path = os.path.join(MODEL_DIR, 'xgb_model.json')
            if xgb is not None and os.path.exists(xgb_model_path):
                try:
                    _xgb_model = xgb.XGBRegressor()
                    _xgb_model.load_model(xgb_model_path)
                except Exception as e:
                    print(f"Error loading XGBoost model: {e}")
                    _xgb_model = None
            else:
                _xgb_model = None

@lru_cache(maxsize=1000)
def predict_for_location(neighbourhood: str, city: str, country: str):
    """
    Given a location name, fetches live data, engineers features, 
    and predicts the MUIS score.
    """
    load_models()
    
    # 1. Geocode
    lat, lon = geocode_location(neighbourhood, city, country)
    if lat is None or lon is None:
        raise ValueError(f"Could not geocode location: {neighbourhood}, {city}, {country}")
        
    # 2. Fetch Data
    osm_elements = fetch_osm_features(lat, lon)
    google_places = fetch_google_places(lat, lon)
    network_metrics = fetch_network_metrics(lat, lon)
    
    # 3. Engineer Features
    features_dict = engineer_features(osm_elements, google_places, network_metrics, lat, lon)
    
    # 4. Normalize Features
    X_raw = np.zeros((1, len(FEATURE_COLS)), dtype=np.float64)
    for i, col in enumerate(FEATURE_COLS):
        X_raw[0, i] = features_dict.get(col, 0.0)
        
    X_norm = np.zeros_like(X_raw)
    for i, col in enumerate(FEATURE_COLS):
        strategy, scaler_obj = _scalers[col]
        val = X_raw[0, i]
        
        if strategy == 'log_robust':
            val_log = np.log1p(val)
            X_norm[0, i] = scaler_obj.transform([[val_log]])[0][0]
        elif strategy == 'robust':
            X_norm[0, i] = scaler_obj.transform([[val]])[0][0]
        else:
            X_norm[0, i] = val
            
    # 5. Predict Score
    # PyTorch ANN Prediction
    y_pred_ann = 5.5
    if _model is not None:
        try:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_norm)
                y_pred_ann_tensor = _model(X_tensor)
                if hasattr(y_pred_ann_tensor, 'item'):
                    y_pred_ann = float(y_pred_ann_tensor.item())
                else:
                    y_pred_ann = float(y_pred_ann_tensor)
        except Exception as e:
            y_pred_ann = 5.5

    # XGBoost Prediction
    y_pred_xgb = 5.5
    if _xgb_model is not None:
        try:
            y_pred_xgb = float(_xgb_model.predict(X_norm)[0])
        except Exception as e:
            y_pred_xgb = 5.5

    # Ensemble Prediction
    if _ensemble_weight_ann is not None:
        score = _ensemble_weight_ann * y_pred_ann + (1.0 - _ensemble_weight_ann) * y_pred_xgb
    else:
        score = y_pred_ann
    
    # Clamp to [0, 10] range
    score = max(0.0, min(10.0, float(score)))
    
    return {
        "location": {"lat": lat, "lon": lon},
        "score": round(score, 2),
        "features": features_dict
    }
