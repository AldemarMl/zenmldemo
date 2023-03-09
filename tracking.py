from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


print(f'mlflow ui --backend-store-uri="{get_tracking_uri()}"')