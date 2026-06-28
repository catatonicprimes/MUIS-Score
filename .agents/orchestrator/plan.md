# Plan — MUIS Dataset Scaling & Advanced Model Architecture

## Objective
Scale the MUIS dataset by fetching additional geographic regions (to >= 2,000 unique rows) and implement a high-performing PyTorch ANN + XGBoost Ensemble tuned via Optuna to beat the 1.79 Test MAE baseline.

## Steps

### Step 1: Exploration and Baseline Verification
- Verify the current features.csv row count and structure.
- Run the existing E2E tests to ensure they are passing.
- Inspect the logic inside `data/expand_dataset.py` and `data/prepare_training_data.py`.

### Step 2: Dataset Expansion
- Execute `data/expand_dataset.py` to crawl additional geographic regions.
- Ensure that the final features.csv contains enough rows to yield > 2,000 rows across train and test sets (`X_train.shape[0] + X_test.shape[0] > 2000`).
- Verify key rotation, timeout handling, and ensure no corrupt/empty coordinates.
- Clean up any temporary or cached garbage files.

### Step 3: Data Preparation
- Run `data/prepare_training_data.py` to construct normalized splits (`X_train`, `X_val`, `X_test`, `y_train_score`, etc.).
- Ensure feature scaling parameters (scalers) are fit ONLY on the training split, then applied to the validation and test splits (no data leakage).

### Step 4: Advanced Model Implementation & Tuning
- Implement `model/train_ensemble.py` which:
  - Trains a PyTorch ANN (fully connected layers with LeakyReLU/ReLU, Dropout, and weight decay/L2 regularization).
  - Trains an XGBoost Regressor.
  - Uses Optuna to tune hyper-parameters of both components (e.g. layers, units, learning rate, dropout, max_depth, learning_rate, subsample, etc.) based on validation MAE.
  - Combines the tuned ANN and XGBoost into a Weighted Ensemble.
  - Logs results (Test MAE, RMSE, R2) to `model/training_history.json`.
- Ensure the Test MAE is under 1.79.

### Step 5: Verification & Auditing
- Run the full E2E test suite to verify that all endpoints (`/api/predict`) and core features work correctly.
- Run the Forensic Auditor subagent to perform an integrity check on the codebase.

### Step 6: Handoff and Reporting
- Write the final handoff report.
- Report completion back to the user.
