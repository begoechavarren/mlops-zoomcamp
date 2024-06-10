import joblib
import mlflow
from mlflow.tracking import MlflowClient

if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(artifacts, *args, **kwargs):
    dv, lr = artifacts

    # Set the tracking URI to the local MLflow server
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Create or get the experiment
    experiment_name = "TaxiTripDurationPrediction"
    mlflow_client = MlflowClient()
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Start a new MLflow run to log the artifacts
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        dv_path = "dict_vectorizer.pkl"
        joblib.dump(dv, dv_path)
        mlflow.log_artifact(dv_path, artifact_path="models")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("intercept", lr.intercept_)

        print(
            f"Model and DictVectorizer are logged to MLflow under run ID: {run.info.run_id}"
        )
