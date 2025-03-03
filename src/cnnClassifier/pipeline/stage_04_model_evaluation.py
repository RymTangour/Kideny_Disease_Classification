from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evauation_mlflow import Evaluation
from cnnClassifier import logger



class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        #evaluation.log_into_mlflow()
        #evaluation.assign_alias_to_stage("VGG16Model", "Staging", "champion")


STAGE_NAME="Evaluation stage"

if __name__=='__main__':
    try:
        logger.info(f" stage {STAGE_NAME}  started")
        obj=EvaluationPipeline()
        obj.main()
        logger.info(f"Stage {STAGE_NAME} completed")
    except Exception as e:
        raise e


