# MLOPS-PROJECT-1

VEHICLE INSURANCE DATA PIPELINE

---

### Model deployment notes (added)

- Models are now re-serialized with `cloudpickle` before being uploaded by the ModelPusher to improve cross-version compatibility.
- The app logs the runtime `scikit-learn` version when loading a model to help diagnose mismatches.
- `requirements.txt` now includes `cloudpickle` — ensure CI installs dependencies before running the app.

Recommended permanent fix: pin `scikit-learn` in `requirements.txt` to the version used for training (e.g., `scikit-learn==1.4.x` or whichever your training environment used) and re-run model pusher to upload a cloudpickle model and metadata.

Quick verification steps:

- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest -q tests/test_model_integration.py` (if AWS creds are present, it will test S3 load too)
- Re-serialize & upload latest local model: `python scripts/reserialize_and_upload.py` (requires AWS creds to upload)
