from zenml.services import BaseService
from zenml.client import Client
from zenml.steps import Output, step
import pandas as pd



@step(enable_cache=False)
def prediction_service_loader() -> BaseService:
    """Load the model service of our train_evaluate_deploy_pipeline."""
    client = Client()
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="training_pipeline",
        pipeline_step_name="model_deployer",
        running=True,
    )
    service = services[0]
    return service

@step
def predictor(
    service: BaseService,
    data: pd.DataFrame,
) -> Output(predictions=list):
    """Run a inference request against a prediction service"""
    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(data.to_numpy())
    prediction = prediction.argmax(axis=-1)
    print(f"Prediction is: {[prediction.tolist()]}")
    return [prediction.tolist()]