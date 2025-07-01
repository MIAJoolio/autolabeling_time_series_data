import mlflow

def setup_mlflow(tracking_uri="http://localhost:5000", experiment_name="ts_clustering"):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    mlflow.set_experiment(experiment_name)