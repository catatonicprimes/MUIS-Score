# Original User Request

## Initial Request — 2026-06-27T08:44:52Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Scale the MUIS dataset by fetching additional geographic regions, and implement advanced neural network architectures (e.g. TabNet, Ensemble) to find the absolute best predictive model.

Working directory: c:\Users\swast\Downloads\INTERNSHIP-II\muis_project
Integrity mode: benchmark

## Requirements

### R1. Expand Dataset
Use the existing data collection scripts (or write new automation) to fetch data for new geographic regions. The dataset must be expanded from 660 rows to at least 2,000 rows. 
**CRITICAL DATA CONSTRAINTS:** 
- The project currently rotates between 16 different API keys to avoid rate limits. Ensure the fetching logic strictly respects this rotation and implements robust handling for timeouts, rate limits, and server-side errors.
- Actively monitor and validate that the pulled data is structurally correct and actually useful (no empty/corrupt geographic pulls).
- Ensure the new data respects the train/test split rules previously established.
- Remove all temporary, useless, or clutter files generated during the fetching process once complete.

### R2. Build Advanced Architectures
Implement an advanced tabular architecture (like TabNet) or an Ensemble approach (combining ANN and XGBoost). Do not rely solely on a basic MLP. 

### R3. Rigorous Hyperparameter Tuning
Use Optuna to heavily tune the chosen architectures. Ensure heavy regularization (L2, Dropout) is evaluated to prevent overfitting.

## Acceptance Criteria

### Data Scale & Quality
- [ ] The `data` directory contains normalized arrays with `X_train.shape[0] + X_test.shape[0]` > 2000.
- [ ] No data leakage occurs (scaling is applied *after* train/test splitting).

### Model Execution & Performance
- [ ] The new training script executes successfully from start to finish without errors.
- [ ] The model's Test MAE, RMSE, and R2 are evaluated on a held-out test set and saved to `model/training_history.json`.
- [ ] The final test MAE improves upon the baseline (1.79 MAE) achieved by the simple ANN.

## Resume Request — 2026-06-27T18:57:51Z

# Teamwork Project Prompt — RESUMING PREVIOUS WORK

Scale the MUIS dataset by fetching additional geographic regions, and implement advanced neural network architectures (e.g. TabNet, Ensemble) to find the absolute best predictive model.

Working directory: c:\Users\swast\Downloads\INTERNSHIP-II\muis_project
Integrity mode: benchmark

**NOTE:** This is a resume of a previous attempt that hit an API quota limit. The previous agents already created `TEST_INFRA.md`, `tests/test_e2e.py`, and `data/expand_dataset.py`. You should resume from this point: run the data expansion, then proceed to the modeling.

## Requirements

### R1. Expand Dataset
Use the existing data collection scripts (like `data/expand_dataset.py`) to fetch data for new geographic regions. The dataset must be expanded from 660 rows to at least 2,000 rows.
**CRITICAL DATA CONSTRAINTS:** 
- The project currently rotates between 16 different API keys to avoid rate limits. Ensure the fetching logic strictly respects this rotation and implements robust handling for timeouts, rate limits, and server-side errors.
- Actively monitor and validate that the pulled data is structurally correct and actually useful (no empty/corrupt geographic pulls).
- Ensure the new data respects the train/test split rules previously established.
- Remove all temporary, useless, or clutter files generated during the fetching process once complete.

### R2. Build Advanced Architectures
Implement an advanced tabular architecture (like TabNet) or an Ensemble approach (combining ANN and XGBoost). Do not rely solely on a basic MLP. 

### R3. Rigorous Hyperparameter Tuning
Use Optuna to heavily tune the chosen architectures. Ensure heavy regularization (L2, Dropout) is evaluated to prevent overfitting.

## Acceptance Criteria

### Data Scale & Quality
- [ ] The `data` directory contains normalized arrays with `X_train.shape[0] + X_test.shape[0]` > 2000.
- [ ] No data leakage occurs (scaling is applied *after* train/test splitting).

### Model Execution & Performance
- [ ] The new training script executes successfully from start to finish without errors.
- [ ] The model's Test MAE, RMSE, and R2 are evaluated on a held-out test set and saved to `model/training_history.json`.
- [ ] The final test MAE improves upon the baseline (1.79 MAE) achieved by the simple ANN.
