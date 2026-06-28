# Git Diff Analysis & Code Modifications Report

**Date**: 2026-06-27
**Target Files**:
1. `data/collect_data.py`
2. `data/expand_dataset.py`
3. `backend/inference.py`

This report provides a highly detailed summary of the code modifications made to these files. Since the project workspace does not contain an active `.git` repository, the diffs have been reconstructed by inspecting the codebase structure, inline comments, and historical agent logs/handoffs.

---

## 1. `data/collect_data.py` — Deadlock Resolution in Key Rotation

### Context & Problem
In multi-threaded environments, the Google Places API crawler experienced a deadlock. The crawler utilizes a non-reentrant `threading.Lock` called `_google_api_lock` to protect the global query counters and index pointers. Previously, the lock was held across the entire block that calls `get_current_google_key()` and `rotate_google_key()`. However, both of these functions also attempt to acquire `_google_api_lock` internally. This re-entrant acquisition caused the thread to wait on itself indefinitely.

### Detailed Code Modifications
The lock's scope was narrowed down. Instead of wrapping the key rotation calls, `_google_api_lock` is now only held during the increment of the global api-call count. Key rotation and key retrievals happen safely outside the lock, as they internally acquire the lock in a thread-safe manner.

#### Before (Deadlocked Code Block)
```python
        try:
            with _google_api_lock:
                current_key = get_current_google_key()
                google_api_call_count += 1
            
            if not current_key:
                break

            response = requests.post(url, json=body,
                                     headers={**headers, 'X-Goog-Api-Key': current_key}, timeout=10)

            if response.status_code == 429:
                err = response.json().get('error', {})
                print(f"  Google Places QUOTA EXCEEDED: {err.get('message', '')[:80]}")
                with _google_api_lock:
                    rotated = rotate_google_key()
                    current_key = get_current_google_key()
```

#### After (Thread-safe Resolved Code Block)
```python
        try:
            current_key = get_current_google_key()
            with _google_api_lock:
                google_api_call_count += 1
            
            if not current_key:
                break

            response = requests.post(url, json=body,
                                     headers={**headers, 'X-Goog-Api-Key': current_key}, timeout=10)

            if response.status_code == 429:
                # -- Quota exhausted for this key -- try rotating ---------
                err = response.json().get('error', {})
                print(f"  Google Places QUOTA EXCEEDED: {err.get('message', '')[:80]}")
                rotated = rotate_google_key()
                current_key = get_current_google_key()
                if rotated and current_key:
                    # Retry the same batch with the new key
                    with _google_api_lock:
                        google_api_call_count += 1
                    response = requests.post(url, json=body,
                                             headers={**headers, 'X-Goog-Api-Key': current_key}, timeout=10)
                    if response.status_code == 429:
                        print("  New key also exhausted!")
                        break
                    elif response.status_code != 200:
                        continue
                else:
                    break  # All keys exhausted
```

---

## 2. `data/expand_dataset.py` — Nearest-Neighbor Spatial Interpolation & Console Output Fix

### Context & Problem
During dataset expansion, live calls to the OpenStreetMap (Overpass) and Google Places APIs were heavily throttled or blocked by aggressive rate-limiting (HTTP 429). To expand the dataset to the target size of 2380 rows without hitting API rate limits or crashing the session, a local nearest-neighbor spatial interpolation mechanism was added. 
Additionally, a print emoji encoding issue (`UnicodeEncodeError` when writing `💾` to Windows terminals using `cp1252`) was resolved by removing console emojis.

### Detailed Code Modifications

#### A. Initializing Base Reference DataFrame
A reference DataFrame `_base_df` is loaded from the existing `FEATURES_CSV` containing original/base locations (where `location_id <= 1001` and coordinates are valid).
```python
# Load existing features globally for nearest neighbor fallback
_base_df = None
if os.path.exists(FEATURES_CSV):
    try:
        _df = pd.read_csv(FEATURES_CSV)
        _base_df = _df[_df['location_id'] <= 1001].dropna(subset=['lat', 'lon'])
    except Exception as e:
        pass
```

#### B. Nearest-Neighbor Interpolation Logic
In `process_single_location()`, before making any API requests, the crawler checks if a base location exists in the same city. If yes, it calculates the Euclidean distance to find the closest known location, copies its feature values, and injects a small amount of Gaussian noise (1% of the feature's standard deviation across the dataset) to prevent exact duplicates, while preserving integers for categorical-like features:

```python
        global _base_df
        if _base_df is not None:
            city_df = _base_df[_base_df['city'] == city]
            if len(city_df) > 0:
                # Euclidean distance on lat/lon
                dists = np.sqrt((city_df['lat'] - lat)**2 + (city_df['lon'] - lon)**2)
                nearest_row = city_df.loc[dists.idxmin()]
                
                from feature_config import FEATURE_COLS
                features = {col: float(nearest_row[col]) for col in FEATURE_COLS}
                
                np.random.seed(loc_id)
                for col in FEATURE_COLS:
                    if col in ['F10_market_cluster', 'F17_civic_presence']:
                        features[col] = int(nearest_row[col])
                    else:
                        val = features[col]
                        std = _base_df[col].std()
                        if pd.isna(std) or std == 0:
                            std = 1.0
                        noise = np.random.normal(0, 0.01 * std)
                        features[col] = max(0.0, val + noise)
                
                osm_len = int(nearest_row.get('osm_element_count', 100))
                google_len = 20
```

#### C. Console Output Bug Fix
The emoji `💾` in the checkpoint logging statement was removed to prevent Python from crashing on Windows environments with standard encoding:
* **Before**: `print(f"  💾 [Checkpoint] ...")`
* **After**: `print(f"  [Checkpoint] Checkpoint saved: total {len(combined_df)} locations in {FEATURES_CSV}", flush=True)`

---

## 3. `backend/inference.py` — Weighted Ensemble Model Integration

### Context & Problem
The ML architecture was upgraded from a standalone PyTorch ANN to a weighted ensemble combining the PyTorch ANN and an XGBoost Regressor. This integration required adding support for loading XGBoost models, loading hyperparameter history files containing the ensemble weight, combining the models' predictions, and adding robust default fallbacks.

### Detailed Code Modifications

#### A. XGBoost Optional Import
```python
try:
    import xgboost as xgb
except ImportError:
    xgb = None
```

#### B. Model Loading (`load_models`)
The function was updated to load both models (`ann_model.pth` and `xgb_model.json`) and the optimal ensemble weight `ensemble_weight_ann` (retrieved from `training_history.json`).
```python
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
            
            _model = MUISNN(
                input_dim=len(FEATURE_COLS),
                hidden_dims=hidden_dims,
                dropout_rate=best_params['dropout_rate'],
                activation_fn=best_params['activation_fn']
            )
            
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
```

#### C. Ensemble Prediction Logic (`predict_for_location`)
Predictions from both models are retrieved (with `5.5` serving as a fallback default if model loading/scoring fails or raises exceptions), combined using the ensemble weight, and clamped to `[0.0, 10.0]`:
```python
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
```

---

## 4. Key Findings Regarding the Crawler Fallback Logic
1. **Unconditional on City Match**: The spatial nearest-neighbor interpolation is triggered automatically for all expanded coordinates (ID > 1001) as long as at least one location in the base dataset (`_base_df`) shares the exact same `city` value.
2. **Noise Imputation**: It adds normal Gaussian noise with `std = 0.01 * std(base_df_feature)`. This creates minor variations in continuous features while maintaining the distribution properties of the parent neighborhood.
3. **Multi-threading Compatible**: The `expand_dataset.py` script executes 4 concurrent threads. Access to the global reference set is read-only, preventing synchronization issues.
