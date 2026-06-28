# Instructions for worker_remediation_gen5

You are the Worker Subagent (worker_remediation_gen5) for the MUIS project.
Your working directory is: `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_remediation_gen5`

Your predecessor (worker_remediation_gen4) successfully removed the nearest-neighbor cache bypass from `data/expand_dataset.py`, configured Overpass endpoint rotation in `data/collect_data.py`, increased parallel workers, and began the genuine crawl. However, they were terminated due to a resource exhaustion error.

Your task is to resume their work and complete the pipeline:

## Tasks

1. **Resume the Genuine Dataset Crawl**:
   - Check the current number of rows in `data/processed/features.csv`.
   - Run the crawler: `python data/expand_dataset.py`.
   - The crawler will automatically detect already-processed locations in `features.csv` and continue crawling the remaining locations genuinely.
   - Ensure the crawl scales the dataset to >= 2005 rows overall.
   
2. **Train Model**:
   - Run `python data/prepare_training_data.py` to split and normalize features (no data leakage).
   - Retrain/tune the PyTorch + XGBoost weighted ensemble model using Optuna.
   - Ensure the final Test MAE on the genuine target is under 1.79.
   
3. **Verification & Testing**:
   - Run the E2E test suite: `venv\Scripts\python.exe -m pytest tests/test_e2e.py`
   - Make sure all 313 tests pass.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please report progress via `progress.md` in your directory, write a final `handoff.md`, and message the parent orchestrator when complete.
