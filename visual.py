from zenml.integrations.evidently.visualizers import EvidentlyVisualizer
from pipelines.inference_pipeline import inference_pipeline

inference_run = inference_pipeline.get_runs()[0]
drift_detection_step = inference_run.get_step(step="drift_detector")

EvidentlyVisualizer().visualize(drift_detection_step)