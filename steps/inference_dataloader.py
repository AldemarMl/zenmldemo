import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from zenml.steps import step

@step
def inference_data_loader() -> pd.DataFrame:
    """Load some (random) inference data."""
    return pd.DataFrame(
        data=np.random.rand(10, 4) * 10,  # assume range [0, 10]
        columns=load_iris(as_frame=True).data.columns,
    )