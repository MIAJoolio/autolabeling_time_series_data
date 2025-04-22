from typing import List, Optional
from pathlib import Path

from src.generation import *
from src.utils import *


def generate_Linear_dataset(config_paths:List[str], save_path:str='data/linear_100/Linear_dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset линейных данных с 4 кластерами
      1. Восходящий тренд 
      2. Нисходящий тренд
      3. Восх. тренд + Нисх. тренд
      4. Нисх. тренд + Восх. тренд
    """
    
    full_data = []
    save_path = Path(save_path)
    config_paths = config_paths if isinstance(config_paths, list) else Path(config_paths).iterdir()
    
    for inx, path in enumerate(config_paths):
        cluster_generator = Basic_generator(config_path=path)
        cluster_data = cluster_generator.generate_multiple(num_series=num_series)
        
        plot_series(cluster_data[:10], [i for i in range(10)], plot_title=f'Cluster {inx+1}', save_path=save_path.parent / f'cluster_{inx+1}.png')
        full_data.append([{'class_id':inx+1, 'row':list(row)} for row in cluster_data])
    
    save_generated_data(full_data, save_path=save_path)

def generate_Seasonal_dataset(config_paths:List[str], save_path:str='data/seasonal_100/Seasonal_dataset.json', num_series:int=100):
    """
    Функция для генерации toy-dataset сезонных данных с 2 кластерами
      1. кластер с частотой 50-100
      2. кластер с частотой 110-200
      
    Для сезонных данных амплитуда и фаза не так сильно влияют на сезонность как частота. Поэтому два кластера с интервалами частоты 50-100, 110-200   
    """
    
    full_data = []
    save_path = Path(save_path)
    config_paths = config_paths if isinstance(config_paths, list) else Path(config_paths).iterdir()
    
    for inx, path in enumerate(config_paths):
        cluster_generator = Basic_generator(config_path=path)
        cluster_data = cluster_generator.generate_multiple(num_series=num_series)
        
        plot_series(cluster_data[:10], [i for i in range(10)], plot_title=f'Cluster {inx+1}', save_path=save_path.parent / f'cluster_{inx+1}.png')
        full_data.append([{'class_id':inx+1, 'row':list(row)} for row in cluster_data])
    
    save_generated_data(full_data, save_path=save_path)


def main():
    generate_Linear_dataset(config_paths='configs/scripts/create_learning_datasets/Linear_dataset', num_series=100)
    
    generate_Seasonal_dataset('configs/scripts/create_learning_datasets/Seasonal_dataset', num_series=100)
    
if __name__ == "__main__":
    main()