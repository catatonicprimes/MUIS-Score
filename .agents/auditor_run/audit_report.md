## Forensic Audit Report

**Work Product**: MUIS Project Codebase (specifically `data/expand_dataset.py`, `data/collect_data.py`, `backend/inference.py`, and `data/processed/features.csv`)
**Profile**: General Project
**Verdict**: INTEGRITY VIOLATION

### Phase Results
1. **Hardcoded Bypass and Facade Check in `data/expand_dataset.py`**: **FAIL**
   - *Details*: The script contains a fallback bypass logic (lines 46–70) that checks if a baseline dataset exists. If the city of the new location exists in the baseline, the script copies the features of the nearest spatial neighbor in that city and adds a small amount of random Gaussian noise (`np.random.normal(0, 0.01 * std)`), completely bypassing the actual web crawling/API query logic for OSM, Google Places, and network metrics.
2. **Hardcoded Bypass Check in `data/collect_data.py`**: **PASS**
   - *Details*: The raw crawler helper functions correctly define the query procedures.
3. **Hardcoded Bypass Check in `backend/inference.py`**: **PASS**
   - *Details*: The inference module correctly uses the defined API crawler and model evaluation logic. It contains fallback return values (`5.5`) in case of exceptions/unloaded models, which are appropriate defensive defaults.
4. **Behavioral Verification (Test Suite Execution)**: **PASS**
   - *Details*: The test suite `tests/test_e2e.py` executed successfully with all 313 tests passing. (Note: PyTorch is mocked in this test to avoid Windows DLL loading errors).
5. **Dataset Integrity Verification**: **FAIL**
   - *Details*: Analysis of the generated `data/processed/features.csv` revealed that out of 1,436 expanded locations (where `location_id > 1001`), 1,426 locations (99.30%) match the features of a baseline location in the same city with a small Gaussian noise perturbation (<= 0.05 * standard deviation). Only 10 locations (representing the first few rows from New York) contain actual crawled feature data, confirming that the vast majority of the expanded dataset is fabricated synthetic data.
6. **Acceptability Criteria check**: **FAIL**
   - *Details*: The final model performance saved in `model/training_history.json` lists a Test MAE of `2.0316`, which fails to improve upon the simple ANN baseline of `1.79` specified in the project requirements.

---

### Evidence

#### 1. Code Bypass in `data/expand_dataset.py` (Lines 46–70)
```python
        global _base_df
        if _base_df is not None:
            city_df = _base_df[_base_df['city'] == city]
            if len(city_df) > 0:
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

#### 2. Dataset Fabricated Rows Verification
Executing the audit analysis script on `data/processed/features.csv`:
```
Total rows in features.csv: 2380
Baseline rows (location_id <= 1001): 944
Expanded rows (location_id > 1001): 1436

Checking for synthetic data copying...
Total expanded rows checked: 1436
Expanded rows matching baseline features with <= 0.05*std noise: 1426 (99.30%)
```
This confirms that 1,426 out of 1,436 expanded locations have fabricated feature vectors derived from nearest-neighbor cloning instead of genuine API data acquisition.

#### 3. Model Performance in `model/training_history.json`
```json
{
  "test_mae": 2.031686544418335,
  "test_rmse": 2.4192135334014893,
  "test_r2": 0.019658198992734577,
  ...
}
```
The MAE is 2.03, which is worse than the baseline of 1.79.
