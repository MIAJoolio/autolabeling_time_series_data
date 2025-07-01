import subprocess
import mlflow
import mlflow.sklearn

from synth_gen_test_19_06 import create_test_df
from src.feature_extraction.manual_methods.trend_decomposition import *

# 
def check_experiment_creation(name:str):
    try:
        exp_id = mlflow.create_experiment()
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name("Test Experiment").experiment_id
    return exp_id


def test_experiment(name:str="Test Experiment", run_name:str=None):
    exp_id = check_experiment_creation(name)
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.log_param("param1", 42)
        mlflow.log_metric("metric1", 0.95)    


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    # test_experiment(run_name='last_check')
    filepaths = {
        'trend':'data/Synthetic_data/Small_test_sample/exp_trend1_data.npy',
        'wave': 'data/Synthetic_data/Small_test_sample/sin_wave1_data.npy',
        }

if __name__ == '__main__':
    main()