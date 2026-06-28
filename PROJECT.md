# Project: muis_project

## Architecture
- Tabular dataset expanded to >= 2000 rows. Inputs are normalized numpy arrays (`X_train.npy`, `X_val.npy`, `X_test.npy`).
- Targets are regression scores (`y_train_score.npy`, `y_val_score.npy`, `y_test_score.npy`).
- Models: PyTorch ANN and XGBoost combined in a Weighted Ensemble.
- Hyperparameter Tuning: Optuna optimization for both ANN and XGBoost.
- Saves:
  - `model/train_ensemble.py` (Script to run tuning + train ensemble)
  - `model/ann_model.pth` (PyTorch model weights)
  - `model/xgb_model.json` (XGBoost model weights)
  - `model/training_history.json` (Parity with previous history, includes best parameters for both models)
  - `backend/inference.py` (Updated to run prediction using the Ensemble model)

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | E2E Testing | Design and implement Tier 1-4 test cases and test runner. Publish TEST_READY.md. | None | DONE |
| 2 | Data Expansion | Resolve skipped geocodings, scale data to >= 2000 rows with grid sampling (min 1.6km spacing to avoid leakage). Save normalized arrays to `data/` and `model/`. | None | DONE |
| 3 | Advanced Models | Implement Ensemble (PyTorch ANN + XGBoost) with heavy regularization, Optuna hyperparameter tuning, and training metrics logging. | 2 | DONE |
| 4 | Verification & Audit| Verify all E2E test cases pass, run Forensic Auditor, perform ponytail audit. | 3 | DONE |

## Interface Contracts
### Data Preparation ↔ Model Training
- Inputs: `data/raw/training_locations.csv` expanded to >= 2000 rows.
- Intermediate: `data/processed/features.csv` and `data/processed/features_labelled.csv`.
- Outputs: `model/X_train.npy`, `model/X_val.npy`, `model/X_test.npy`, `model/y_train_score.npy`, `model/y_val_score.npy`, `model/y_test_score.npy` and their copies in `data/`.
- Shapes: `X_train` has shape `(N_train, 27)`. `y_train_score` has shape `(N_train,)` representing the MUIS score in `[0.0, 10.0]`.
- Feature normalisation scaling fitted on train split and transformed on validation/test splits to avoid data leakage.

### Model Outputs ↔ Inference
- Inference loads ensemble model config and weights/estimators.
- Endpoint `/api/predict` returns predictions for any (lat, lon) by geocoding or scraping features dynamically and running predictions.

## Code Layout
- `data/collect_data.py` - Collects OSM & Google Places features.
- `data/prepare_training_data.py` - Normalizes features and generates splits.
- `model/train_ensemble.py` - New script to run tuning + training of both models and the ensemble.
- `model/training_history.json` - Output metrics of ensemble and component models.
- `model/ann_model.pth` - Saved weights of the PyTorch ANN component.
- `model/xgb_model.json` - Saved weights of the XGBoost component.
- `backend/inference.py` - Real-time inference using the ensemble model.
