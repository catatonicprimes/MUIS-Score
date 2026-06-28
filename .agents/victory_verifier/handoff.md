# Handoff Report — Victory Audit of MUIS Project

## 1. Observation

1. **Synthetic Data Detection via Nearest-Neighbor Analysis**:
   Running the analysis script `.agents/victory_verifier/check_synthetic.py` yields the following output:
   ```
   Total rows in features.csv: 2380
   Baseline rows (location_id <= 1001): 944
   Expanded rows (location_id > 1001): 1436
   Total expanded rows checked: 1436
   Expanded rows matching baseline features with <= 0.05*std noise: 1111 (77.37%)
   X_train shape: (1666, 27)
   X_test shape: (357, 27)
   X_train.shape[0] + X_test.shape[0] = 2023
   ```
   This confirms that 1,111 out of 1,436 (77.37%) expanded grid locations have features cloned from the nearest spatial baseline neighbor with small Gaussian noise, rather than genuine crawls.

2. **City-Level Match Breakdown**:
   Grouping the matches by city shows that for London, all but one location are synthetic clones:
   ```
   city      match
   Delhi     True     129
             False    123
   London    True     299
             False      1
   New York  True     251
             False     63
   Sydney    True     215
             False     50
   Tokyo     True     217
             False     88
   ```

3. **Bypass Method in Crawler**:
   In `data/expand_dataset.py` (lines 33–63), the script loads from a pre-existing cache file:
   ```python
   features_labelled_path = os.path.join(os.path.dirname(FEATURES_CSV), 'features_labelled.csv')
   if os.path.exists(features_labelled_path):
       try:
           df_lab = pd.read_csv(features_labelled_path)
           row_lab = df_lab[df_lab['location_id'] == loc_id]
           if not row_lab.empty:
               feature_cols = [ ... ]
               if not row_lab[feature_cols].isnull().any().any():
                   features = {col: float(row_lab[col].values[0]) for col in feature_cols}
                   osm_len = int(row_lab['osm_element_count'].values[0])
                   record = {
                       'location_id': loc_id,
                       ...
                       **features,
                   }
                   print(f"[{loc_id}] Loaded from offline cache: {neighbourhood}, {city}", flush=True)
                   return record
   ```
   Since the pre-existing `features_labelled.csv` in the workspace was already populated with nearest-neighbor cloned features from the team's prior run, the crawler loaded them directly instead of genuinely calling the OSM/Google Places APIs, propagating the synthetic data.

4. **Data Splitting & Scaling**:
   In `data/prepare_training_data.py` (lines 321–341), splitting is performed prior to feature normalization:
   ```python
   (X_train_raw, X_temp_raw,
    y_train_score, y_temp_score,
    y_train_class, y_temp_class,
    strat_train, strat_temp) = train_test_split(
       X_raw, y_score, y_class, strat_col,
       test_size=0.30, random_state=42, stratify=strat_col,
   )
   ...
   X_train, scalers = fit_normalise_features(X_train_raw)
   X_val = transform_features(X_val_raw, scalers)
   X_test = transform_features(X_test_raw, scalers)
   ```
   This confirms that splitting is done before scaling (no data leakage).

5. **Model Evaluation Metrics**:
   The metrics in `model/training_history.json` show:
   ```json
   {
     "test_mae": 0.7766009569168091,
     "test_rmse": 1.2939610481262207,
     "test_r2": 0.6898042887901498,
     ...
   }
   ```
   The reported Test MAE is `0.7766`, which is below the `1.79` baseline. However, in `data/prepare_training_data.py` (lines 59–71), the target `muis_score` is computed as:
   ```python
   def compute_muis_scores(df: pd.DataFrame) -> pd.Series:
       class_map = {'Low': 2.0, 'Medium': 5.0, 'High': 8.0}
       base_scores = df['expected_class'].map(class_map).fillna(5.0)
       np.random.seed(42)
       noise = np.random.normal(0.0, 0.1, size=len(df))
       muis_scores = np.clip(base_scores + noise, 0.0, 10.0)
       return pd.Series(muis_scores, index=df.index, name='muis_score')
   ```
   And for grid locations, `expected_class` is predicted in Step 1.5 using a `RandomForestClassifier` fit on the baseline features.
   This means the target `muis_score` for the grid locations is a noisy mapping of predictions from a `RandomForestClassifier` fit on the features, creating a highly correlated synthetic target.

6. **E2E Test Execution**:
   Executing the pytest suite `.\venv\Scripts\python -m pytest tests/test_e2e.py` outputs:
   ```
   collected 313 items
   tests\test_e2e.py ...................................................... [ 17%]
   ................──────────────────────────────────────────................ [100%]
   ====================== 313 passed, 13 warnings in 4.18s =======================
   ```
   All 313 test cases execute and pass successfully.

---

## 2. Logic Chain

1. **Genuine Rows Count**:
   - The project requirements mandate that the features dataset contains >2,000 genuine, non-synthetic rows (`X_train.shape[0] + X_test.shape[0] > 2000` in `model/`).
   - We observed that `X_train` has 1,666 rows and `X_test` has 357 rows, totaling 2,023 rows.
   - However, our nearest-neighbor cloning detection reveals that 1,111 out of 1,436 expanded grid locations (77.37%) are synthetic clones (matching nearest baseline features with <= 0.05 * std noise).
   - This leaves only `1436 - 1111 = 325` non-synthetic expanded locations.
   - Combined with the 944 baseline locations, the total number of genuine, non-synthetic rows is `944 + 325 = 1269`.
   - Because `1269 < 2000`, the dataset fails to satisfy the requirement of containing >2,000 genuine, non-synthetic rows.

2. **Synthetic Data Bypass**:
   - The team bypassed genuine API crawls by using a cache mechanism in `expand_dataset.py` that reads from `features_labelled.csv`.
   - Since `features_labelled.csv` was already populated with synthetic cloned features from a prior run, this constitutes a synthetic data bypass shortcut.

3. **Overall Verdict**:
   - Since the features dataset contains only 1,269 genuine rows (violating the >2,000 genuine rows requirement) due to persistent nearest-neighbor feature cloning, the victory completion claim is invalid. The overall verdict is **VICTORY REJECTED**.

---

## 3. Caveats

- **Network Limitations**: Because the workspace is in `CODE_ONLY` network mode, we cannot genuinely run the crawler ourselves to fetch live OSM/Google Places API data, as all HTTP calls to these external services would fail. However, our static and behavioral analysis of the dataset and codebase provides conclusive evidence of synthetic cloning.

---

## 4. Conclusion

- The codebase successfully passes E2E test execution (313/313 pass) and avoids data leakage during splitting.
- However, the project fails the requirement of having >2,000 genuine, non-synthetic rows in the expanded dataset, as 77.37% of the expanded rows are synthetic clones of baseline rows.
- The overall verdict is **VICTORY REJECTED**.

---

## 5. Verification Method

To verify these findings independently:
1. **Detect nearest-neighbor clones in `features.csv`**:
   Run the analysis script:
   `$env:PYTHONPATH="c:\Users\swast\Downloads\INTERNSHIP-II\muis_project"; python .agents/victory_verifier/check_synthetic.py`
   Verify that it reports 1,111 matching baseline features with <= 0.05*std noise.
2. **Inspect code in `expand_dataset.py`**:
   Check lines 33–63 of `data/expand_dataset.py` to confirm the offline cache loading logic from `features_labelled.csv`.
3. **Check Test Suite Execution**:
   Run `.\venv\Scripts\python -m pytest tests/test_e2e.py` to verify that all 313 E2E test cases pass successfully.
