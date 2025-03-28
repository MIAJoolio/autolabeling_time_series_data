from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from core.generation import *
from core.utils import *

__all__ = [
    "Synthetic_dataset_generator",
    "Synthetic_dataset"
]   

class Synthetic_dataset_generator:
    """
    Класс для генерации синтетического датасета временных рядов.
    Каждый класс в датасете имеет свою конфигурацию блоков с определенными распределениями параметров.
    """
    def __init__(self, num_classes: int, series_per_class: int, num_blocks: int, block_length: int, random_state: Optional[int] = None):
        """
        Инициализация генератора датасета.

        Args:
            num_classes: Количество классов в датасете
            series_per_class: Количество временных рядов для каждого класса
            num_blocks: Количество блоков в каждом временном ряду
            block_length: Длина каждого блока
            random_state: Seed для генерации случайных чисел
        """
        self.num_classes = num_classes
        self.series_per_class = series_per_class
        self.num_blocks = num_blocks
        self.block_length = block_length
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Словарь доступных функций генерации
        self.generator_functions = {
            "linear": linear_trend,
            "quadratic": quadratic_trend,
            "exponential": exponential_trend,
            "seasonal": seasonal_series,
            "harmonic": harmonic_oscillator,
            "sawtooth": sawtooth_wave,
            "random_walk": random_walk
        }
        
        # Словарь функций генерации параметров
        self.param_functions = {
            "linear": linear_trend_params,
            "quadratic": quadratic_trend_params,
            "exponential": exponential_trend_params,
            "seasonal": seasonal_series_params,
            "harmonic": harmonic_oscillator_params,
            "sawtooth": sawtooth_wave_params,
            "random_walk": random_walk_params
        }
        
        # Конфигурации классов
        self.class_configs = {}
        
    def set_class_config(self, class_id: int, blocks_config: List[Dict]):
        """
        Установка конфигурации для конкретного класса.

        Args:
            class_id: ID класса
            blocks_config: Список конфигураций блоков, где каждый блок - словарь с параметрами:
                {
                    'type': str,  # тип блока (linear, quadratic и т.д.)
                    'param_config': Dict,  # конфигурация для функции генерации параметров
                }
        """
        if len(blocks_config) != self.num_blocks:
            raise ValueError(f"Количество блоков в конфигурации ({len(blocks_config)}) не соответствует заданному количеству блоков ({self.num_blocks})")
                
        self.class_configs[class_id] = blocks_config
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация датасета на основе конфигураций классов.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Массив временных рядов и массив меток классов.
        """
        series_list = []
        labels_list = []
        
        for class_id in range(1, self.num_classes + 1):
            if class_id not in self.class_configs:
                raise ValueError(f"Конфигурация для класса {class_id} не установлена")
            
            class_config = self.class_configs[class_id]
            
            for _ in range(self.series_per_class):
                # Создаем пустой массив для всего временного ряда
                full_series = np.zeros(self.num_blocks * self.block_length)
                
                # Генерируем каждый блок
                for i, block_config in enumerate(class_config):
                    block_type = block_config['type']
                    params = block_config['param_config']
                    
                    # Создаем генератор для текущего блока
                    generator = Generator(self.block_length)
                    generator.add_block(block_type, **params)
                    
                    # Генерируем блок и помещаем его в нужную позицию
                    block_series = generator.generate()
                    start_idx = i * self.block_length
                    end_idx = (i + 1) * self.block_length
                    full_series[start_idx:end_idx] = block_series
                
                series_list.append(full_series)
                labels_list.append(class_id)
        
        return np.array(series_list), np.array(labels_list)
    
    def save_dataset(self, series: List[np.ndarray], labels: List[int], save_path: Union[str, Path]):
        """
        Сохранение датасета в JSON файл.

        Args:
            series: Список временных рядов
            labels: Список меток классов
            save_path: Путь для сохранения файла
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Преобразование данных в формат для JSON
        data = {
            'series': [s.tolist() for s in series], 
            'labels': labels.tolist(), 
            'configs': self.class_configs,
            'metadata': {
                'num_classes': self.num_classes,
                'series_per_class': self.series_per_class,
                'num_blocks': self.num_blocks,
                'block_length': self.block_length
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f)

class Synthetic_dataset(Dataset):
    """
    Класс для загрузки и использования синтетического датасета в PyTorch.
    """
    def __init__(self, data_path: Union[str, Path], n_dims: Optional[int] = None):
        """
        Инициализация датасета.

        Args:
            data_path: Путь к JSON файлу с данными
            n_dims: Количество измерений для преобразования (если None, оставляет одномерным)
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        self.series = [np.array(series) for series in data['series']]
        self.labels = data['labels']
        self.configs = data['configs']
        self.metadata = data['metadata']
        
        if n_dims is not None:
            self._transform_to_multidimensional(n_dims)
    
    def _transform_to_multidimensional(self, n_dims: int):
        """
        Преобразует одномерные ряды в многомерные путем добавления измерений.
        Каждое новое измерение - это сдвинутая версия исходного ряда.

        Args:
            n_dims: Количество измерений
        """
        transformed_series = []
        for series in self.series:
            multi_series = np.zeros((len(series), n_dims))
            multi_series[:, 0] = series
            
            for dim in range(1, n_dims):
                shift = dim * len(series) // n_dims
                multi_series[:, dim] = np.roll(series, shift)
            
            transformed_series.append(multi_series)
        
        self.series = transformed_series
        
    def __len__(self) -> int:
        """Возвращает количество временных рядов в датасете."""
        return len(self.series)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получение элемента датасета.

        Args:
            idx: Индекс элемента

        Returns:
            Кортеж (временной ряд как тензор, метка класса)
        """
        series = self.series[idx]
        # Проверяем, что длина последовательности соответствует ожидаемой
        expected_length = self.metadata['num_blocks'] * self.metadata['block_length']
        if len(series) != expected_length:
            raise ValueError(f"Несоответствие длины последовательности: ожидалось {expected_length}, получено {len(series)}")
        
        series = torch.FloatTensor(series)
        label = self.labels[idx]
        return series, label

def main():
    # Генерация датасета, который бы состоял из всех видов генераторов
    generator = Synthetic_dataset_generator(
        num_classes=3,
        series_per_class=5,
        num_blocks=1,
        block_length=40,
        random_state=42
    )
    
    # Конфигурация для первого класса (линейный тренд)
    generator.set_class_config(1, [
        {
            'type': 'linear',
            'param_config': generator.param_functions['linear'](random_state=23)
        }
    ])
    
    # Конфигурация для второго класса (сезонные данные)
    generator.set_class_config(2, [
        {
            'type': 'seasonal',
            'param_config': generator.param_functions['seasonal'](random_state=23)
        }
    ])
    
    # Конфигурация для третьего класса (сезонные данные)
    generator.set_class_config(3, [
        {
            'type': 'seasonal',
            'param_config': generator.param_functions['seasonal'](random_state=74)
        }
    ])
    
    # Генерация датасета
    series, labels = generator.generate_dataset()
    
    # Сохранение датасета
    generator.save_dataset(series, labels, 'data/test_dataset1.json')
    
    # Загрузка датасета (одномерный вариант)
    dataset_1d = Synthetic_dataset('data/test_dataset1.json')
    
    # Создание загрузчиков данных
    dataloader_1d = DataLoader(dataset_1d, batch_size=32, shuffle=True)

    # Проверка загрузки данных
    for batch_series, batch_labels in dataloader_1d:
        print("1D данные:")
        print(f"Batch shape: {batch_series.shape}")
        print(f"Labels: {batch_labels}")
        break
        
    # Визуализация разных классов (для одномерного случая)
    class_series = []
    class_labels = []
    for class_id in range(1, len(np.unique(dataset_1d.labels)) + 1):
        class_indices = [i for i, label in enumerate(labels) if label == class_id]
        if class_indices:
            class_series.append(series[class_indices[0]])
            class_labels.append(f"Класс {class_id}")

    # Визуализация 5 рядов для каждого класса
    for class_id in range(1, len(np.unique(dataset_1d.labels)) + 1):
        class_indices = [i for i, label in enumerate(labels) if label == class_id][:5]
        if class_indices:
            class_series = [series[i] for i in class_indices]
            class_labels = [f"Ряд {i+1}" for i in range(5)]
            
            plot_series_grid(
                series_list=class_series,
                labels=class_labels,
                plot_title=f"5 представителей класса {class_id}",
                ylabel="Значение",
                xlabel="Время",
                figsize=(15, 3),
                layout='horizontal'
            )

if __name__ == '__main__':
    main() 