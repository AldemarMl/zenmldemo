from zenml.pipelines import pipeline

@pipeline
def inference_pipeline(
    inference_data_loader,
    mlflow_model_deployer,
    predictor,
    training_data_loader,
    drift_detector,
):
    """Inference pipeline with skew and drift detection."""
    inference_data = inference_data_loader()
    model_deployment_service = mlflow_model_deployer()
    predictor(model_deployment_service, inference_data)
    training_data, _, _, _ = training_data_loader()
    drift_detector(training_data, inference_data)