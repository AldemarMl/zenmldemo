from pipelines.training_pipeline import training_pipeline
from steps.evaluator import evaluator
from steps.training_dataloader import training_data_loader
from steps.trainer import svc_trainer_mlflow
from zenml.integrations.mlflow.steps.mlflow_registry import MLFlowRegistryParameters, mlflow_register_model_step
from zenml.model_registries.base_model_registry import ModelRegistryModelMetadata


train_pipeline = training_pipeline(
    training_data_loader=training_data_loader(),
    trainer=svc_trainer_mlflow(),
    evaluator=evaluator(),
    model_register=mlflow_register_model_step(
            params=MLFlowRegistryParameters(
                name="zenml-quickstart-model",
                metadata=ModelRegistryModelMetadata(
                    gamma=0.01, arch="svc"
                ),
                description=f"The first run of the Quickstart pipeline.",
            )
        ),
)

if __name__ == "__main__":
    train_pipeline.run(unlisted=True)