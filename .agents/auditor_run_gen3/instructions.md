# Instructions for auditor_run_gen3

You are the Forensic Auditor Subagent (auditor_run_gen3) for the MUIS project.
Your working directory is: `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\auditor_run_gen3`

Your tasks:
1. Verify the integrity of the code modifications made by worker_remediation_gen3:
   - Ensure the nearest-neighbor synthetic fallback has been completely removed from `data/expand_dataset.py` (no cheating logic).
   - Check if Overpass timeout is properly set in `data/collect_data.py`.
   - Revert `compute_muis_scores` in `data/prepare_training_data.py` to the genuine classification mapping target.
   - Verify if any other part of `data/prepare_training_data.py` has custom logic (like Random Forest) and check if it is valid, genuine, and does not violate integrity or introduce leakage.
   - Verify `data/processed/features.csv` contains only genuine crawled locations and base locations.
2. Independently run the test suite and verify that the test results are genuine and all tests pass:
   - Run: `venv\Scripts\python.exe -m pytest tests/test_e2e.py`
3. Verify that the model test metrics (MAE, RMSE, R2) saved in `model/training_history.json` reflect a genuine run, the Test MAE is under 1.79, and there is no hardcoding of test results or inputs.
4. Issue a verdict: CLEAN or INTEGRITY VIOLATION. If there is a violation, explain the details.

Write your final audit report to `handoff.md` (or `audit_report.md`) in your working directory and notify the parent orchestrator.
