from zenml.pipelines import pipeline


@pipeline
def training_pipeline(
    training_data_loader,
    trainer,
    evaluator,
    model_register,
):
    """Train, evaluate, and deploy a model."""
    X_train, X_test, y_train, y_test = training_data_loader()
    model = trainer(X_train=X_train, y_train=y_train)
    test_acc = evaluator(X_test=X_test, y_test=y_test, model=model)
    model_register(model)