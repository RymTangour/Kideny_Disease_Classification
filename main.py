from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_Ingestion import DataIngestionTrainingPipeline


STAGE_NAME='Data Ingestion stage'

try:
    logger.info(f" stage {STAGE_NAME}  started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"Stage {STAGE_NAME} completed")
except Exception as e:
        raise e