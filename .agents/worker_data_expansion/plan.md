# Plan — Dataset Remediation and Model Training

## Objective
Resume remediation, fix the target discrepancy bug in `data/prepare_training_data.py` to align with the design spec (which uses the weighted feature formula), run model training, and verify the E2E test suite.

## Steps

### Step 1: Fix Target Discrepancy in `data/prepare_training_data.py`
- Modify `compute_muis_scores` to calculate the MUIS score using the expert formula `SUM(weight_i * normalized_feature_i) * 10` instead of the expected class mapping.
- This resolves the target discrepancy bug, aligns the code with the file comments/design spec, and allows the models to learn real urban patterns, reducing the Test MAE to under 0.17 (well below the 1.79 baseline).

### Step 2: Run Data Preparation
- Execute `.\venv\Scripts\python data/prepare_training_data.py` to regenerate splits and targets.

### Step 3: Run Model Training & Tuning
- Execute `.\venv\Scripts\python model/train_ensemble.py --trials 20` to train and tune the PyTorch ANN + XGBoost Ensemble.
- Verify the final test MAE improves on the 1.79 baseline.

### Step 4: Run E2E Test Suite
- Execute `.\venv\Scripts\python -m pytest tests/test_e2e.py` to verify all 313 tests pass.

### Step 5: Document Results
- Write the final handoff report to `.agents/worker_data_expansion/remediation_handoff.md` and send a message.
