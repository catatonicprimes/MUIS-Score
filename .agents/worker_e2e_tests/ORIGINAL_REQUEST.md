## 2026-06-27T08:51:14Z

You are a worker subagent. Your working directory is: c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_e2e_tests
Your task is to design, implement, and verify the E2E test suite for the MUIS project.

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Follow these instructions carefully:
1. Create c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\TEST_INFRA.md at the project root. Identify all 27 features of the project and document them. Use the template in the parent's system prompt (which has been provided to you in the user request).
2. Implement E2E test cases in a new file `tests/test_e2e.py` at the project root across 4 tiers:
   - Tier 1: Feature Coverage (>= 135 cases). Parameterize tests for all 27 features with at least 5 different valid inputs/scenarios each.
   - Tier 2: Boundary & Corner Cases (>= 135 cases). Parameterize tests for all 27 features with at least 5 boundary/edge/invalid/extreme inputs each (e.g. empty lists, negative values, very large values, NaN/None handling, extreme ratios).
   - Tier 3: Cross-Feature Combinations (>= 27 cases). Parameterize tests for at least 27 combinations of features.
   - Tier 4: Real-World Application Scenarios (>= 14 cases). Define at least 14 distinct representative real-world neighborhood/city profiles.
   The tests should verify:
   - Feature engineering correctness: Verify that `engineer_features` from `data/collect_data.py` accurately computes the 27 features when given mocked inputs for OSM elements, Google places, and street network metrics.
   - Per-feature normalization: Verify that `fit_normalise_features` and `transform_features` from `data/prepare_training_data.py` scale inputs correctly based on the appropriate strategy (log_robust, robust, passthrough) and prevent data leakage (fitting on train only, and transforming val/test splits without leakage).
   - Backend FastAPI `/api/predict` endpoint: Verify that `/api/predict` endpoint in `backend/app.py` works correctly by using FastAPI's TestClient or mock request client. Mock all Nominatim, Overpass/OSM, and Google Places API calls so the tests do not make actual network requests.
   - API key rotation logic: Mock `requests.post` inside `fetch_google_places` in `data/collect_data.py` to return a 429 status code for the first key. Verify that the function switches to the next Google API key and retries the request, eventually succeeding or handling exhaustion.
3. Install any required testing dependencies (like `pytest` or `pytest-cov` or `httpx` for TestClient) in the virtual environment located at `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\venv`, and run the tests using `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\venv\Scripts\python.exe -m pytest tests/test_e2e.py` to ensure they execute successfully.
4. Publish `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\TEST_READY.md` at the project root summarizing the coverage (with coverage tables, tiers, counts, and feature checklists as per the template).
5. Write your handoff report to `c:\Users\swast\Downloads\INTERNSHIP-II\muis_project\.agents\worker_e2e_tests\handoff.md` and send a message back when done.

## 2026-06-27T14:00:51Z

You are tasked with fixing a deadlock in `data/collect_data.py` and verifying the E2E test suite.
1. Apply the following fix to `data/collect_data.py` to resolve the deadlock inside the `fetch_google_places` function (lines 407-433):
- Change this block:
```python
            with _google_api_lock:
                current_key = get_current_google_key()
                google_api_call_count += 1
```
to:
```python
            current_key = get_current_google_key()
            with _google_api_lock:
                google_api_call_count += 1
```
- And change this block:
```python
                with _google_api_lock:
                    rotated = rotate_google_key()
                    current_key = get_current_google_key()
```
to:
```python
                rotated = rotate_google_key()
                current_key = get_current_google_key()
```
2. Once the edit is made, run the test suite using:
`.\venv\Scripts\python -m pytest tests/test_e2e.py`
Verify that all 313 tests pass without hanging.
3. Report the pytest output and your status back to me via send_message.
Make sure you include the MANDATORY INTEGRITY WARNING in mind: Do not cheat, do not hardcode test results.
