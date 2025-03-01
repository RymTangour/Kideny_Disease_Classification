import dagshub
import tensorflow as tf 
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from mlflow import MlflowClient
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml ,create_directories, save_json

tf.config.run_functions_eagerly(True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client=MlflowClient()

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

    def assign_alias_to_stage(self, model_name: str, stage: str, alias: str):
        """
        Assign an alias to the latest model version in a specific stage.

        Args:
            model_name (str): Name of the registered model.
            stage (str): Stage of the model (e.g., "Staging", "Production").
            alias (str): Alias to assign (e.g., "champion")."""
        
        latest_mv = self.client.get_latest_versions(model_name, stages=[stage])[0]
        self.client.set_registered_model_alias(model_name, alias, latest_mv.version)
        