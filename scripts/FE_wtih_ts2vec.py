"""

"""
from src.feature_extraction import ts2vec_extract_features, visualize_latent_space 

def get_embeddings():
    train_repr, train_labels, test_repr, test_labels = ts2vec_extract_features(dataset_name= DATASET_NAME, train_val_ratio=TRAIN_VAL_SPLIT, model_config=MODEL_CONFIG) 
    
    return train_repr, train_labels, test_repr, test_labels
    
def main():
    global DATASET_NAME, TRAIN_VAL_SPLIT, MODEL_CONFIG 
     
    DATASET_NAME = "Chinatown" # "ItalyPowerDemand" # "ECG200"
    MODEL_CONFIG = 'configs/fe_default/ts2vec_config1.yaml'

    for split in [0.4,0.5,0.6,0.7,0.8]:
        TRAIN_VAL_SPLIT = split
        train_repr, train_labels, test_repr, test_labels = get_embeddings()

        visualize_latent_space(train_repr, train_labels, test_repr, test_labels, f'experiments/FE_with_ts2vec/latent_{DATASET_NAME}_{MODEL_CONFIG.split(".")[0][-1]}_{TRAIN_VAL_SPLIT}.png')
    
if __name__ == '__main__':
    main()