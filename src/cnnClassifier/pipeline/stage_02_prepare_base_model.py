from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


class PreapreBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
   

STAGE_NAME="Prepare base model"


if __name__=='__main__':
    try:
        logger.info(f" stage {STAGE_NAME}  started")
        obj=PreapreBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"Stage {STAGE_NAME} completed")
    except Exception as e:
        raise e
  
