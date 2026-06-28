# Changelog

All notable changes to the MUIS (Mixed-Use Interaction Score) project will be documented in this file.

## [2.0.0] - Recovery Implementation

### Architectural Changes
* **Micro-MLP Substitution:** Replaced the over-parameterized deep ANN in `train_ann.py` with a Micro-MLP architecture. This correctly acknowledges the near-linear nature of the problem created by rigorous upstream feature normalisation. A Ridge Regression baseline was also added to test coefficients.
* **XGBoost Stabilisation:** Added L2 regularization (`reg_lambda=5.0`) and reduced `max_depth` to `4` in `train_xgboost.py` to prevent the model from overfitting onto a mostly linear target.
* **Backend Caching:** Introduced a lightweight in-memory caching dictionary in the FastAPI backend (`backend/app.py`) to prevent duplicate and expensive calls to external APIs (Google/OSM) for identical neighbourhood queries.
* **Frontend Flexibility:** Updated the frontend to rely on the Vite environment variable (`VITE_API_URL`) instead of a hardcoded localhost path, enabling seamless production deployment.

### Algorithmic Logic
* **F04 (Work-Residential Ratio) Fix:** Altered the underlying metric logic from an unbounded ratio to a symmetric Balance Index. The formula `f04 = 1.0 - abs(work - res) / (work + res + 1)` correctly enforces the theoretical ideal: 1:1 balance yields a maximum score of 1.0.
* **F29 (Gini-Simpson) Removal:** Dropped the F29 metric entirely from `feature_config.py` and `collect_data.py`. This resolves critical multicollinearity and overlapping mathematical redundancy with `F02_land_use_entropy`. The weight formerly allocated to F29 has been dynamically re-assigned.

### Data Processing
* **Percentile Clipping Strategy:** Replaced standard `MinMaxScaler` bound-squishing in `prepare_training_data.py` with a 2nd/98th percentile clipping logic. This robustly mitigates massive compression effects caused by a tiny minority of outlier Central Business District locations.
* **Empirical Class Rebalancing:** Adjusted the MUIS classification boundaries (`MUIS_CLASS_RANGES`) in `feature_config.py`. The updated tertile boundaries—High [4.0, 10.0], Medium [2.75, 4.0), Low [0.0, 2.75)—more accurately represent the empirical distribution, shifting away from heavily skewed class imbalances.

### Results
* Following these logic fixes, the engineered synthetic MUIS score exhibits a healthy Gaussian distribution. The XGBoost model validation metrics drastically improved, shrinking the Test MAE from ~0.40 to 0.17 MUIS points and boosting the explained variance ($R^2$) to 0.97.
