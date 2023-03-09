from steps.inference_dataloader import inference_data_loader
from steps.predictor import predictor
from steps.training_dataloader import training_data_loader
from zenml.integrations.mlflow.steps.mlflow_deployer import MLFlowDeployerParameters, mlflow_model_registry_deployer_step
from pipelines.inference_pipeline import inference_pipeline
from steps.drift_detector import drift_detector

inference = inference_pipeline(
        inference_data_loader=inference_data_loader(),
        mlflow_model_deployer=mlflow_model_registry_deployer_step(
            params=MLFlowDeployerParameters(
                registry_model_name="zenml-quickstart-model",
                registry_model_version="1",
                # or you can use the model stage if you have set it in the MLflow registry
                # registered_model_stage="None" # "Staging", "Production", "Archived"
            )
        ),
        predictor=predictor(),
        training_data_loader=training_data_loader(),
        drift_detector=drift_detector,
    )


if __name__ == "__main__":
    inference.run(unlisted=True)