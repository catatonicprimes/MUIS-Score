# Progress Tracking — worker_remediation_gen4

Last visited: 2026-06-28T06:05:00Z

- [x] Phase 1: Investigation & Connectivity Test
  - [x] Run `data/test_speed.py` to check connectivity and single-location crawl benchmark
  - [x] Inspect and modify `data/expand_dataset.py` to remove cache loading block
- [x] Phase 2: Overpass Endpoint Rotation & Parallel Crawl Optimization
  - [x] Implement Overpass endpoint rotation in `data/collect_data.py` (rotate `ox.settings.overpass_endpoint`)
  - [x] Increase parallel crawl settings in `data/expand_dataset.py` (`max_workers = 8` or `10`)
  - [x] Add robust retry mechanisms with backoff
- [ ] Phase 3: Dataset Reset & Genuine Crawl
  - [ ] Reset `data/processed/features.csv` to contain only the 944 base locations from `features_original.csv`
  - [ ] Run the genuine crawl to expand the dataset to >= 2005 rows
  - [ ] Audit expanded features to ensure no cloning/synthetic fallback
- [ ] Phase 4: Model Training & Validation
  - [ ] Re-run data preparation `data/prepare_training_data.py`
  - [ ] Retrain advanced ensemble model using Optuna
  - [ ] Verify test MAE < 1.79
  - [ ] Run and pass all E2E tests `tests/test_e2e.py`
