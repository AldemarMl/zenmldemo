zenml integration install sklearn mlflow evidently -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-registry register mlflow_registry --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml data-validator register evidently_validator --flavor=evidently
zenml stack register quickstart_stack -a default\
                                       -o default\
                                       -d mlflow_deployer\
                                       -e mlflow_tracker\
                                       -r mlflow_registry\
                                       -dv evidently_validator\
                                       --set
zenml stack describe