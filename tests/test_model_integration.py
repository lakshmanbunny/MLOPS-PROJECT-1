import os
import json
import pytest
import pandas as pd
from src.components.model_pusher import ModelPusher
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.s3_estimator import Proj1Estimator

# A small integration test: try to load model locally (artifact) and run sample predict.
# If local artifact is missing, attempt to load from S3 if AWS credentials are available.

SAMPLE_INPUT = {
    "Gender": ["M"],
    "Age": [30],
    "Driving_License": [1],
    "Region_Code": [1],
    "Previously_Insured": [0],
    "Annual_Premium": [1000],
    "Policy_Sales_Channel": [1],
    "Vintage": [30],
    "Vehicle_Age_lt_1_Year": [0],
    "Vehicle_Age_gt_2_Years": [1],
    "Vehicle_Damage_Yes": [1]
}


def _get_local_model_path():
    # look for artifact/*/model_trainer/trained_model/model.pkl
    root = os.path.join(os.path.dirname(__file__), '..')
    artifact_dir = os.path.join(root, 'artifact')
    if not os.path.exists(artifact_dir):
        return None
    # search directories sorted by name descending (latest first)
    candidates = []
    for name in os.listdir(artifact_dir):
        model_path = os.path.join(artifact_dir, name, 'model_trainer', 'trained_model', 'model.pkl')
        if os.path.exists(model_path):
            candidates.append((name, model_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def test_local_model_load_and_predict():
    model_path = _get_local_model_path()
    if not model_path:
        pytest.skip("No local artifact model found; skip local model test")

    # attempt to load using SimpleStorageService.load_model via local file
    # to avoid S3 dependency we use pickle/cloudpickle directly
    import pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        import cloudpickle
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)

    assert hasattr(model, 'predict')

    df = pd.DataFrame(SAMPLE_INPUT)
    # If model is MyModel wrapper, call predict with dataframe
    preds = None
    try:
        preds = model.predict(df)
    except Exception as e:
        pytest.fail(f"Model loaded but prediction failed: {e}")

    assert preds is not None


def test_s3_model_load_if_available():
    # If AWS credentials present, try to load model from S3 using Proj1Estimator
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        pytest.skip("AWS credentials not present; skip S3 model load test")
    # Use model config from code defaults
    from src.entity.config_entity import ModelPusherConfig
    cfg = ModelPusherConfig()
    estimator = Proj1Estimator(bucket_name=cfg.bucket_name, model_path=cfg.s3_model_key_path)
    # will raise if load fails
    model = estimator.load_model()
    assert hasattr(model, 'predict')
    df = pd.DataFrame(SAMPLE_INPUT)
    preds = model.predict(df)
    assert preds is not None
