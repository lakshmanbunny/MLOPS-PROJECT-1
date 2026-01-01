import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Uploading artifacts folder to s3 bucket")
            
            logging.info("Uploading new model to S3 bucket....")

            # Load the trained model from local file
            try:
                import pickle
                with open(self.model_evaluation_artifact.trained_model_path, 'rb') as f:
                    model_obj = pickle.load(f)
            except Exception as e:
                logging.error("Failed to load local trained model file for re-serialization", exc_info=True)
                raise

            # Re-serialize the model using cloudpickle for more robust cross-version loading
            try:
                import cloudpickle, json, platform, time
                temp_cloud_file = self.model_evaluation_artifact.trained_model_path + '.cloud.pkl'
                with open(temp_cloud_file, 'wb') as f:
                    cloudpickle.dump(model_obj, f)

                # Create metadata containing sklearn & python versions
                meta = {
                    'sklearn_version': None,
                    'python_version': platform.python_version(),
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }
                try:
                    import sklearn
                    meta['sklearn_version'] = sklearn.__version__
                except Exception:
                    meta['sklearn_version'] = 'unknown'

                meta_file = self.model_evaluation_artifact.trained_model_path + '.meta.json'
                with open(meta_file, 'w') as mf:
                    json.dump(meta, mf)

                # Upload cloudpickle model and metadata
                self.proj1_estimator.save_model(from_file=temp_cloud_file)
                self.s3.upload_file(from_file=meta_file,
                                    to_filename=self.model_pusher_config.s3_model_key_path + '.meta.json',
                                    bucket_name=self.model_pusher_config.bucket_name,
                                    remove=True)

                model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                            s3_model_path=self.model_pusher_config.s3_model_key_path)

            except Exception as e:
                logging.error("Failed to re-serialize/upload model using cloudpickle", exc_info=True)
                raise

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e