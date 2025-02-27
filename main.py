from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_Ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PreapreBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
STAGE_NAME='Data Ingestion stage'

try:
    logger.info(f" stage {STAGE_NAME}  started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Prepare base model"

try:
    logger.info(f" stage {STAGE_NAME}  started")
    obj=PreapreBaseModelTrainingPipeline()
    obj.main()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e
  

STAGE_NAME="Training"

try:
    logger.info(f" stage {STAGE_NAME}  started")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
    raise e   