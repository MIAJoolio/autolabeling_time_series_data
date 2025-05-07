import os 
import sys 
import pandas as pd

sys.path.append(os.path.abspath("ts2vec/"))

from src.clustering import *
from src.feature_extraction import *
from src.utils import * 


def pipeline2(experiment_name='exp1', config_file='configs/fe_default/ts2vec_config.yaml'):
    """
    Предобработка через TS2VEC    
    """ 
    os.makedirs(f'experiments/pipe2/{experiment_name}/', exist_ok=True)
    
    features, test_labels = get_ts2vec_feat(model_save_path=f'experiments/pipe2/{experiment_name}/ts2vec_model.pt', config_file=config_file)
    test_labels = test_labels.reshape(-1,)
    test_labels[test_labels==1] = 0
    test_labels[test_labels==2] = 1
    
    models = [DBSCAN_model(), AC_model(), Sklearn_kmeans_model()]
    configs = ["configs/parameters_search/dbscan.yaml", "configs/parameters_search/agglomerative.yaml", "configs/parameters_search/sklearn_kmeans.yaml"]
    
    res_df = None
    for model, config in zip(models, configs):
        if res_df is None:
            res_df = model.evaluate_results(features, test_labels, config)
        else:
            res_df = pd.concat([res_df, model.evaluate_results(features, test_labels, config)])

      
    res_df.to_excel(f'experiments/pipe2/{experiment_name}/evaluation_result.xlsx')


def pipeline3():
    return None

def main():
    # pipeline2('exp1')
    # pipeline2('exp2', 'configs/fe_default/ts2vec_config1.yaml')
    pipeline3()
if __name__ == '__main__':
    main()