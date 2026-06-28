# Context — MUIS Project Orchestration

## Current Status
- Project resumes from a state where E2E tests are implemented (`tests/test_e2e.py`), features are mapped (`TEST_INFRA.md`), and `data/expand_dataset.py` exists.
- The dataset needs expansion from its current size to >= 2,000 unique rows across train/test splits.
- An advanced tabular architecture or PyTorch ANN + XGBoost Ensemble must be built, tuned via Optuna, and must achieve a Test MAE better than 1.79.

## Codebase Context
- **Paths**:
  - `data/collect_data.py`: Handles fetching OSM and Google Places features. Has key rotation.
  - `data/expand_dataset.py`: Coordinates dataset expansion.
  - `data/prepare_training_data.py`: Handles data splits and feature normalisation.
  - `tests/test_e2e.py`: Main E2E test suite.
  - `.env`: Contains 16 Google API keys.
- **Normalisation**: Split-then-scale strategy must be preserved to prevent leakage.
- **Model baseline**: Old simple ANN achieved ~1.79 MAE.

## Key Decisions
- Build a Weighted Ensemble of PyTorch ANN and XGBoost Regressor.
- Use Optuna for tuning hyper-parameters of both models.
