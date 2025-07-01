from typing import Literal, Callable

from src.utils.parse_UCR import load_UCR
from src.utils.transform_tools import split_train_test


def load_data(dataset_name:str, split_ratio:float=0.8, source_data:Literal['UCR']='UCR', model_name:Literal['manual']='manual', random_state=42):
    
    # space for validity check of source_data, model_name
    
    if source_data == 'UCR':
        
        X, y, metadata = load_UCR(dataset_name)
        X_train, X_test, y_train, y_test = split_train_test(X, y, split_ratio, random_state)

        return X_train, X_test, y_train, y_test

    