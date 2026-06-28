# MUIS Codebase Exploration & Assessment Handoff Report

## 1. Observation

### API Key Rotation Logic
- **File Path**: `data/collect_data.py`
- **Key Storage**: Keys are stored in the project-level `.env` file (`ENV_FILE = os.path.join(PROJECT_ROOT, '.env')`, defined in `data/feature_config.py` line 33).
- **Key Loading** (lines 61–72):
  ```python
  load_dotenv(ENV_FILE)
  GOOGLE_KEYS = []
  _primary = os.getenv('GOOGLE_API_KEY')
  if _primary:
      GOOGLE_KEYS.append(_primary)
  for i in range(2, 50):
      _extra = os.getenv(f'GOOGLE_API_KEY_{i}')
      if _extra:
          GOOGLE_KEYS.append(_extra)
  ```
- **Rotation Mechanism** (lines 80–98):
  A global pointer `_current_key_index = 0` tracks the active key.
  `get_current_google_key()` returns the key at that index.
  `rotate_google_key()` increments `_current_key_index` and returns `True` if a new key is available, `False` otherwise.
- **Trigger** (lines 382–399):
  In `fetch_google_places()`, if a post request to the Google Places API returns status code `429` (Quota Exceeded), `rotate_google_key()` is called. If `True`, the request headers are updated with the new key and the request is retried:
  ```python
  if response.status_code == 429:
      err = response.json().get('error', {})
      print(f"  Google Places QUOTA EXCEEDED on key #{_current_key_index + 1}: {err.get('message', '')[:80]}")
      if rotate_google_key():
          current_key = get_current_google_key()
          headers['X-Goog-Api-Key'] = current_key
          response = requests.post(url, json=body, headers=headers, timeout=10)
  ```

### Geographic Regions & Dataset Scaling
- **Files**:
  - `data/raw/training_locations.csv` (1,001 locations)
  - `data/processed/features.csv` (944 locations currently processed)
  - `data/processed/geocoded_missing.csv` (57 locations resolved via manual Google Geocoding fallback)
- **Observations on Country Distribution** (inspected via command line execution):
  - Row count in `features.csv`: 944.
  - Highly biased towards **India** (503 locations).
  - Global representation: **USA** (46), **Australia** (43), **United Kingdom** (42), **Japan** (24), **Singapore** (20), **Spain** (20), **Germany** (20), **France** (19), **Netherlands** (17), **Canada** (17), **Austria** (15), **South Korea** (15), and various single-digit locations from other countries (e.g. Brazil, Colombia, South Africa, Sweden, UAE, New Zealand).
  - 57 locations are currently skipped during main collection due to geocoding failures in Nominatim. However, `geocode_missing.py` has resolved their coordinates and saved them to `geocoded_missing.csv` (which has 57 rows). `collect_data.py` does not currently read `geocoded_missing.csv`.

### Train/Test Splits & Data Leakage
- **File Path**: `data/prepare_training_data.py`
- **Split Determination** (lines 305–318):
  - The script extracts features matrix `X_raw` (shape `(N, 27)`) and continuous target `y_score` (continuous MUIS score), and categorical `y_class` (assigned via `assign_muis_class` based on MUIS score ranges).
  - It performs a stratified split: `strat_col` is defined as `expected_class` (from the raw locations CSV) if present, or `y_class` (lines 301–303).
  - It runs `train_test_split` twice (70/15/15 split):
    - First split: 70% Train, 30% Temp (Val + Test).
    - Second split: 15% Val, 15% Test.
    - Set with `random_state=42` and `stratify=strat_col`.
- **Scaling Normalisation** (lines 321–328):
  - Scalers are fitted on the training split *only* using `fit_normalise_features(X_train_raw)`.
  - The validation and test sets are transformed using the fitted scalers via `transform_features(X_val_raw, scalers)` and `transform_features(X_test_raw, scalers)`. This prevents feature scaling leakage.

### Model Structures & Baseline Performance
- **File Paths**:
  - `model/train_ann.py`
  - `model/train_xgboost.py`
  - `model/training_history.json` (ANN metrics)
  - `model/xgb_training_results.json` (XGBoost metrics)
- **Baseline Performance Observed**:
  - **PyTorch ANN** (from `model/training_history.json` after Optuna tuning):
    - Test MAE: `1.7989`
    - Test RMSE: `2.2337`
    - Test $R^2$: `0.1706`
    - CV MAE Mean: `1.7870 ± 0.0721`
    - Architecture: 2-layer MLP (units: 80, 64; activation: LeakyReLU; dropout: 0.281)
  - **XGBoost** (trained and evaluated via `python model/train_xgboost.py`):
    - Test MAE: `1.7326`
    - Test RMSE: `2.1126`
    - Test $R^2$: `0.2581`
    - CV MAE Mean: `1.7703 ± 0.0610` (Best iteration: 149 trees)
  - **Observations**:
    - XGBoost outperforms the ANN across all metrics, which aligns with standard behavior for small tabular datasets ($N = 944$).
    - There is a target discrepancy: `prepare_training_data.py`'s docstring says the target is computed as a weighted sum of the features (`MUIS score = SUM(weight_i x normalised_feature_i) * 10`), but the implementation in `compute_muis_scores` maps `expected_class` to `{'Low': 2.0, 'Medium': 5.0, 'High': 8.0}` + Gaussian noise ($\sigma=0.1$). This creates a noisy, non-linear classification-based target which limits model fit performance ($R^2 \approx 0.25$).

---

## 2. Logic Chain

1. **API Key Rotation**: The script correctly loads keys from `.env` dynamically via `GOOGLE_API_KEY_{i}` and checks status codes. If a `429` occurs, it increments the key index. This is robust but depends on the keys being pre-populated in `.env`.
2. **Missing Locations (57 skipped)**: `training_locations.csv` has 1,001 locations, but `features.csv` only has 944 rows. `geocode_missing.py` outputs coordinates to `data/processed/geocoded_missing.csv`. However, `collect_data.py` only reads `training_locations.csv`. Therefore, those 57 geocoded locations are never collected.
3. **Scaling the Dataset**: To scale to 2,000+ rows, we cannot simply duplicate locations as this would cause data leakage. We need to add new locations. Adding neighborhoods manually is slow.
4. **Data Leakage in Splits**: Standard random splitting in spatial datasets leads to spatial leakage if neighborhoods from the same city overlap. Currently, the code uses standard stratified splits.
5. **Advanced Model Performance**: XGBoost outperforming PyTorch ANN on tabular data ($N < 10,000$) is a known phenomenon. TabNet is extremely parameter-heavy and requires large datasets ($N > 10,000$) to tune successfully.
6. **No Existing Test Suite**: Only `model/test_ann_robustness.py` exists, testing the ANN module's robustness and early stopping. There is no E2E or system-level testing suite (Tiers 1 to 4).

---

## 3. Caveats

- **API Quota Restrictions**: The scaling proposal assumes access to sufficient Google Places and OpenStreetMap API limits. A daily limit of 120 calls is defined in `feature_config.py`.
- **Target Value Computation**: The current target is computed from a mapping of the coarse `expected_class` label (Low/Medium/High) + noise. If the target should represent actual features, the `prepare_training_data.py` script needs to implement the formula described in its docstring.
- **OSM Data Completeness**: Some global regions may have poor OSM building/landuse coverage, leading to zero elements which could bias the model.

---

## 4. Conclusion & Recommendations

### Actionable Next Steps:
1. **Incorporate Geocoded Missing Locations**: Merge `geocoded_missing.csv` back into `training_locations.csv` by adding `lat` and `lon` columns. Update `collect_data.py` to use these pre-seeded coordinates, restoring the sample size to 1,001.
2. **Dataset Scaling Strategy**: To scale to 2,000+ rows, use a **Grid-based Spatial Sampling method** inside major cities (e.g. NY, London, Tokyo, Delhi). Generate coordinates on a grid spaced by 1.6 km (to prevent overlapping circles), filter for urban cells, and fetch OSM/Google Places features. This automates expansion and guarantees zero spatial overlap.
3. **Prevent Spatial Data Leakage**:
   - Enforce a minimum distance of 1.6 km between any two locations.
   - Implement **Spatial Block Cross-Validation** (split by city/country) to evaluate true generalization to unseen cities.
4. **Advanced Architectures**:
   - **Ensemble (PyTorch ANN + XGBoost)**: Highly recommended. Implement a weighted average or Stacking Regressor on out-of-fold predictions. It combines the linear interpolation of the MLP with the non-linear partitions of XGBoost.
   - **TabNet**: Reject this option. The dataset is too small ($N \approx 2000$) for TabNet's complex attention mechanism to generalise, leading to severe overfitting.
5. **Implement E2E Test Suite**: Add a test suite structured into Tiers 1–4.

---

## 5. Verification Method

### How to Verify Locally:
1. **Run Training Scripts**:
   - Run `python model/train_xgboost.py` to train the XGBoost model, generate plots, and check metrics.
   - Run `python model/run_ann_fast.py` to run a mock Optuna tuning with 3 trials to quickly verify the ANN training pipeline works.
2. **Verify Split and Normalization**:
   - Run `python data/prepare_training_data.py` to regenerate the splits, fit the scalers, and verify that the class distributions and shape counts match 70/15/15.
3. **Verify geocoding missing files**:
   - Confirm `data/processed/geocoded_missing.csv` contains 57 rows of geocoded coordinates.

### Design of E2E Tests (Tiers 1 to 4):

```
+--------------------------------------------------------------------------------+
| TIER 1: UNIT & COMPONENT TESTS                                                 |
| - Verify 28 features are engineered correctly using mock OSM/Google elements   |
| - Verify model definition, forward pass shapes, and EarlyStopping class logic  |
| - Verify scalers correctly normalize features (RobustScaler, log1p, etc.)      |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| TIER 2: INTEGRATION TESTS                                                      |
| - Run prepare_training_data.py pipeline with mock features.csv data            |
| - Verify train_ann.py and train_xgboost.py load splits and save weights/metrics |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| TIER 3: END-TO-END SYSTEM TESTS                                                |
| - Test FastAPI backend /api/predict endpoint                                   |
| - Stub Nominatim, Overpass, and Google Places network responses with static JSON|
| - Assert output contains location, score (clamped [0,10]), and 27 features     |
| - Verify error handling (geocoding failure returns 400 Bad Request)            |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
| TIER 4: ROBUSTNESS, PERFORMANCE & DRIFT TESTS                                  |
| - Check spatial distance (assert min 1.6 km coordinate separation between splits)|
| - Adversarial inputs (extremely high/low coordinates and POI counts)           |
| - API Key Rotation (mock 429 response on key 1, verify key 2 receives retry)   |
+--------------------------------------------------------------------------------+
```
