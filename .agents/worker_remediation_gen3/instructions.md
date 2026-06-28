# Instructions for worker_remediation_gen3

You are the Worker Subagent (worker_remediation_gen3) for the MUIS project.
Your working directory is: `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_remediation_gen3`

## Tasks

1. **Remediation Steps (Code Modification)**:
   - Completely remove the nearest-neighbor synthetic fallback logic from `data/expand_dataset.py` (lines 64-96).
   - Set Overpass timeout in `data/collect_data.py` to `30` or `60` (line 243).
   - Revert `compute_muis_scores` in `data/prepare_training_data.py` to the genuine classification mapping target (using `{'Low': 2.0, 'Medium': 5.0, 'High': 8.0}` plus noise).
   - Reset `data/processed/features.csv` to contain only base locations (location_id <= 1001) from `data/processed/features_original.csv`.

2. **Genuine Dataset Expansion**:
   - Run the genuine crawl (`data/expand_dataset.py`) to scale the dataset to >= 2000 rows.
   - Verify that there are no cheating or synthetic fallback coordinates generated.

3. **Model Preparation and Training**:
   - Run `data/prepare_training_data.py` to construct normalized splits (`X_train.npy`, etc.). Ensure no data leakage (scaling parameters fit ONLY on training split).
   - Retrain/tune the PyTorch ANN + XGBoost weighted ensemble model using Optuna.
   - Save trained model weights to `model/ann_model.pth` and `model/xgb_model.json`, and metrics/params to `model/training_history.json`.
   - Ensure the final Test MAE is under 1.79.

4. **Testing**:
   - Run E2E tests (`tests/test_e2e.py`) to verify everything passes.
   - Verify that `/api/predict` is using the trained ensemble model properly.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please report your progress via `progress.md` in your working directory, write a final `handoff.md` detailing the actions taken and results, and message the parent orchestrator when complete.
