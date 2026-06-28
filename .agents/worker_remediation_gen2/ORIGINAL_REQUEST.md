## 2026-06-28T00:28:54Z
You are the new Remediation Specialist (worker subagent) for the MUIS project. Your task is to resolve an integrity audit violation by executing a genuine dataset crawl and model training pipeline.

Your working directory is: c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_remediation_gen2

Follow these instructions exactly:
1. Edit `data/expand_dataset.py` (lines 64-96) to completely remove the synthetic nearest-neighbor fallback from the `except Exception as api_err` block. The script should raise the error, print the error and return `None` (so that the location is skipped, not filled with synthetic data). No synthetic fallback should exist!
2. Edit `data/collect_data.py` (line 243) and change the timeout configuration `ox.settings.timeout = 0.2` to a reasonable value like `30` or `60` seconds. This will prevent genuine Overpass API requests from timing out immediately.
3. Edit `data/prepare_training_data.py` to revert the `compute_muis_scores` function to mapping the `expected_class` target (`{'Low': 2.0, 'Medium': 5.0, 'High': 8.0}`) plus Gaussian noise with standard deviation `0.1` and seed `42` (clipped to `[0.0, 10.0]`), exactly as originally written.
4. Prepare `data/processed/features.csv` by filtering out all synthetic rows (any rows with `location_id > 1001` that were cloned/synthetic in the previous run). It's safest to reset `features.csv` to contain only the base locations (`location_id <= 1001`) from `features_original.csv`.
5. Run the genuine crawler (`data/expand_dataset.py`) to fetch data for the new locations in `training_locations.csv` until the dataset is genuinely expanded to 2,380 rows. Ensure the crawler pauses between requests (`time.sleep(1.5)`) and uses key rotation to avoid rate limits.
6. Run `data/prepare_training_data.py` to split and scale the genuine features (verify that train+test rows > 2000 and that scaling is applied AFTER train/test splitting to avoid data leakage).
7. Run `model/train_ensemble.py --trials 20` to train and tune the PyTorch ANN and XGBoost Ensemble using Optuna. Verify that the Test MAE on the genuine target is under 1.79.
8. Run the E2E tests `pytest tests/test_e2e.py` to ensure all 313 cases pass.
9. Write your handoff report to `.agents/worker_remediation_gen2/handoff.md`.
10. Send a message to the Project Orchestrator with the results when complete.

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work.
