from typing import List, Optional
from pathlib import Path

from src.generation import *
from src.utils import *


def generate_Linear_dataset(config_paths:List[str], save_path:str='data/linear/Linear_dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset линейных данных с 4 кластерами
      1. Восходящий тренд 
      2. Нисходящий тренд
      3. Восх. тренд + Нисх. тренд
      4. Нисх. тренд + Восх. тренд
    """
    generate_synthetic_dataset(config_paths, save_path, num_series)


def generate_Seasonal_dataset(config_paths:List[str], save_path:str='data/seasonal/Seasonal_dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset сезонных данных с 2 кластерами
      1. кластер с частотой 50-100
      2. кластер с частотой 110-200
      
    Для сезонных данных амплитуда и фаза не так сильно влияют на сезонность как частота. Поэтому два кластера с интервалами частоты 50-100, 110-200   
    """
    
    generate_synthetic_dataset(config_paths, save_path, num_series)


def generate_ts_MNIST_dataset(config_paths:List[str], save_path:str='data/ts_MNIST/ts_MNIST_dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset со всеми типами распределений
    """
    generate_synthetic_dataset(config_paths, save_path, num_series)

def main():
    NUM_SERIES = 300
    
    # generate_Linear_dataset(config_paths='configs/scripts/create_learning_datasets/Linear_dataset', save_path= f'data/linear_{NUM_SERIES}/Linear_dataset.json',num_series=NUM_SERIES)
    
    # generate_Seasonal_dataset('configs/scripts/create_learning_datasets/Seasonal_dataset', save_path= f'data/seasonal_{NUM_SERIES}/Seasonal_dataset.json', num_series=NUM_SERIES)

    # generate_ts_MNIST_dataset('configs/scripts/create_learning_datasets/ts_MNIST', save_path= f'data/ts_MNIST_{NUM_SERIES}/ts_MNIST_dataset.json', num_series=NUM_SERIES)

    # dataset = Basic_dataset('data/linear_100/Linear_dataset.json')
    
    # print(dataset[0])
    
if __name__ == "__main__":
    main()