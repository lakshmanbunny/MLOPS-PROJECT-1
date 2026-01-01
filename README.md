# MLOPS-PROJECT-1

VEHICLE INSURANCE DATA PIPELINE - MLOPS 1ST PROJECT

---

### Model deployment notes (added)

- Models are now re-serialized with `cloudpickle` before being uploaded by the ModelPusher to improve cross-version compatibility.
- The app logs the runtime `scikit-learn` version when loading a model to help diagnose mismatches.
- `requirements.txt` now includes `cloudpickle` — ensure CI installs dependencies before running the app.

Recommended permanent fix: pin `scikit-learn` in `requirements.txt` to the exact version used for training (e.g., `scikit-learn==1.6.0`) and re-run model pusher to upload a cloudpickle model and metadata.

**Critical deployment steps**:

- Ensure `requirements.txt` contains at least:
  - `scikit-learn==1.8.0`
  - `numpy`
  - `pandas`
- Force Docker to rebuild without cache in CI: use `docker build --no-cache` so the new sklearn is installed and cached layers are not reused.
- In CD, pull the latest image before running: `docker pull $ECR_REGISTRY/$ECR_REPOSITORY:latest`.
- After deployment, verify runtime sklearn inside the container:
  - `docker exec -it mlops-app python -c "import sklearn; print(sklearn.__version__)"` (should be `1.8.0`)

Note: Rebuilding with `--no-cache` is mandatory to avoid Docker reusing an old layer that contains a different `sklearn` version.

Quick verification steps:

- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest -q tests/test_model_integration.py` (if AWS creds are present, it will test S3 load too)
- Re-serialize & upload latest local model: `python scripts/reserialize_and_upload.py` (requires AWS creds to upload)
