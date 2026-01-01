"""Small utility to re-serialize the latest local model using cloudpickle and upload it to S3 via ModelPusher.

Usage: python scripts/reserialize_and_upload.py

This will locate the latest model under `artifact/*/model_trainer/trained_model/model.pkl` and call ModelPusher to re-serialize and upload.
"""
import os
import sys
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.components.model_pusher import ModelPusher


def find_latest_local_model():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    artifact_dir = os.path.join(root, 'artifact')
    if not os.path.exists(artifact_dir):
        return None
    candidates = []
    for name in os.listdir(artifact_dir):
        model_path = os.path.join(artifact_dir, name, 'model_trainer', 'trained_model', 'model.pkl')
        if os.path.exists(model_path):
            candidates.append((name, model_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    model_path = find_latest_local_model()
    if not model_path:
        print("No local model found under artifact/, ensure training ran first.")
        sys.exit(1)

    print(f"Using model: {model_path}")
    # Create a minimal ModelEvaluationArtifact just for pushing
    eval_artifact = ModelEvaluationArtifact(is_model_accepted=True, changed_accuracy=0.0, s3_model_path='model.pkl', trained_model_path=model_path)
    cfg = ModelPusherConfig()
    pusher = ModelPusher(model_evaluation_artifact=eval_artifact, model_pusher_config=cfg)
    pusher.initiate_model_pusher()
    print("Re-serialized model and uploaded to S3 (if credentials are configured).")


if __name__ == '__main__':
    main()
