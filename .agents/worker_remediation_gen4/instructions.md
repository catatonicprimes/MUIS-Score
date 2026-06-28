# Instructions for worker_remediation_gen4

You are the Worker Subagent (worker_remediation_gen4) for the MUIS project.
Your working directory is: `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_remediation_gen4`

Your goal is to genuinely scale the MUIS features dataset to >= 2000 rows (without synthetic feature cloning/fallbacks or loading pre-populated synthetic cache from features_labelled.csv), retrain the PyTorch + XGBoost ensemble model to achieve Test MAE < 1.79, and pass all E2E tests.

## Phase 1: Investigation & Connectivity Test
1. Initialize your `progress.md` and `BRIEFING.md` files.
2. Run `data/test_speed.py` to check if you have internet access and benchmark how long a single location crawl takes.
3. Inspect `data/expand_dataset.py` and completely remove the cache loading block (lines 33-63) that loads pre-crawled features from `features_labelled.csv`.

## Phase 2: Overpass Endpoint Rotation & Parallel Crawl Optimization
Because a genuine crawl of ~1,100 locations sequentially would take too long, implement the following optimizations in `data/collect_data.py`:
- In `fetch_osm_features` and `fetch_network_metrics`, rotate `ox.settings.overpass_endpoint` across multiple public Overpass API mirrors to distribute load and bypass rate limits. Useful public mirrors:
  - `https://overpass-api.de/api/interpreter`
  - `https://overpass.kumi.systems/api/interpreter`
  - `https://overpass.nchc.org.tw/api/interpreter`
  - `https://overpass.openstreetmap.fr/api/interpreter`
- In `data/expand_dataset.py`, increase the crawl parallelism by setting `max_workers = 8` or `max_workers = 10`.
- Reduce `time.sleep(1.5)` to `0.5` or `1.0` if using multiple endpoints and threads.

## Phase 3: Dataset Reset & Genuine Crawl
1. Reset `data/processed/features.csv` to contain only the 944 base locations (location_id <= 1001) from `features_original.csv`.
2. Run the crawler script `data/expand_dataset.py` genuinely to expand the dataset from 944 rows to >= 2005 rows (e.g. 2020 or 2100). Verify that it does not use `features_labelled.csv` and does not clone baseline neighbors (you can write or run a simple script to verify distance feature correlation or run the audit checks).
3. If some crawls fail due to API limits or timeouts, implement a retry mechanism with backoff in `data/collect_data.py`.

## Phase 4: Model Training & Validation
1. Re-run `data/prepare_training_data.py` to prepare training data split and feature normalisation (no data leakage).
2. Re-train the PyTorch ANN + XGBoost ensemble model using Optuna hyperparameter tuning.
3. Verify that the final Test MAE on the genuine target is under 1.79.
4. Run the full E2E test suite `tests/test_e2e.py` and verify all tests pass.

Write a detailed handoff.md inside your directory and message the parent orchestrator when complete.
