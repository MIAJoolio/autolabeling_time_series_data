from typing import List, Optional
from pathlib import Path

from src.generation import *
from src.utils import *


def generate_Linear_dataset(config_paths:List[str], save_path:str='data/trend/trend_100.json', num_series:int=100):
    """
    Функция для генерации toy-dataset с трендами
    """
    generate_synthetic_dataset(config_paths, save_path, num_series)

def main():
    NUM_SERIES = 100
    
    generate_Linear_dataset(config_paths='configs/trend_encoder/trend_100_full', save_path= f'data/synthetic/trend_100_full/linear_{NUM_SERIES}.json',num_series=NUM_SERIES)

    generate_Linear_dataset(config_paths='configs/trend_encoder/trend_100_test', save_path= f'data/synthetic/trend_100_test/linear_{NUM_SERIES}.json',num_series=NUM_SERIES)
    
    dataset = Basic_dataset(f'data/synthetic/trend_100_full/linear_{NUM_SERIES}.json')
    
    print(dataset[0])
    
if __name__ == "__main__":
    main()