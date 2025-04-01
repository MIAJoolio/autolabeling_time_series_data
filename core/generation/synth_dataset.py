from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import itertools

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from core.generation import *
from core.utils import *

__all__ = [
    "Synthetic_dataset_generator",
    "Synthetic_dataset",
    "Parametric_dataset_generator",
    "Parametric_dataset"
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
    def __init__(self, data_path: Union[str, Path], n_dims: Optional[int] = None, normalize: bool = False, norm_type: str = 'minmax'):
        """
        Инициализация датасета.

        Args:
            data_path: Путь к JSON файлу с данными
            n_dims: Количество измерений для преобразования (если None, оставляет одномерным)
            normalize: Флаг для нормализации данных
            norm_type: Тип нормализации ('minmax' или 'zscore')
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        self.series = [np.array(series) for series in data['series']]
        self.labels = data['labels']
        self.configs = data['configs']
        self.metadata = data['metadata']
        
        self.normalize = normalize
        self.norm_type = norm_type
        
        # Вычисляем статистики для нормализации
        if normalize:
            if n_dims is not None:
                # Для многомерных данных нормализуем каждое измерение отдельно
                self.norm_stats = []
                for dim in range(n_dims):
                    stats = self._compute_norm_stats(np.array([s[:, dim] for s in self.series]))
                    self.norm_stats.append(stats)
            else:
                # Для одномерных данных
                self.norm_stats = self._compute_norm_stats(np.array(self.series))
        
        if n_dims is not None:
            self._transform_to_multidimensional(n_dims)
    
    def _compute_norm_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Вычисление статистик для нормализации.

        Args:
            data: Массив данных

        Returns:
            Словарь со статистиками (min, max для minmax или mean, std для zscore)
        """
        if self.norm_type == 'minmax':
            return {
                'min': np.min(data),
                'max': np.max(data)
            }
        elif self.norm_type == 'zscore':
            return {
                'mean': np.mean(data),
                'std': np.std(data)
            }
        else:
            raise ValueError(f"Неизвестный тип нормализации: {self.norm_type}")
    
    def _normalize_data(self, data: np.ndarray, stats: Union[Dict[str, float], List[Dict[str, float]]]) -> np.ndarray:
        """
        Нормализация данных.

        Args:
            data: Массив данных для нормализации
            stats: Статистики для нормализации

        Returns:
            Нормализованный массив данных
        """
        if self.norm_type == 'minmax':
            if isinstance(stats, list):
                # Для многомерных данных
                normalized = np.zeros_like(data)
                for dim in range(data.shape[1]):
                    min_val = stats[dim]['min']
                    max_val = stats[dim]['max']
                    normalized[:, dim] = (data[:, dim] - min_val) / (max_val - min_val + 1e-8)
                return normalized
            else:
                # Для одномерных данных
                min_val = stats['min']
                max_val = stats['max']
                return (data - min_val) / (max_val - min_val + 1e-8)
                
        elif self.norm_type == 'zscore':
            if isinstance(stats, list):
                # Для многомерных данных
                normalized = np.zeros_like(data)
                for dim in range(data.shape[1]):
                    mean = stats[dim]['mean']
                    std = stats[dim]['std']
                    normalized[:, dim] = (data[:, dim] - mean) / (std + 1e-8)
                return normalized
            else:
                # Для одномерных данных
                mean = stats['mean']
                std = stats['std']
                return (data - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Неизвестный тип нормализации: {self.norm_type}")
    
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
        
        # Нормализация данных
        if self.normalize:
            if isinstance(self.series[0], np.ndarray) and len(self.series[0].shape) > 1:
                # Для многомерных данных
                series = self._normalize_data(series, self.norm_stats)
            else:
                # Для одномерных данных
                series = self._normalize_data(series, self.norm_stats)
        
        series = torch.FloatTensor(series)
        label = self.labels[idx]
        return series, label

class Parametric_dataset_generator:
    """
    Универсальный класс для генерации датасета временных рядов с заданными параметрами.
    Поддерживает все типы генераторов из ts_generators.py.
    """
    def __init__(self, generator_type: str, param_ranges: Dict[str, List[float]], 
                 series_length: int, random_state: Optional[int] = None):
        """
        Инициализация генератора датасета.

        Args:
            generator_type: Тип генератора ('linear', 'quadratic', 'exponential', 'seasonal', 
                           'harmonic', 'sawtooth', 'random_walk')
            param_ranges: Словарь с диапазонами параметров для выбранного генератора
            series_length: Длина генерируемых временных рядов
            random_state: Seed для генерации случайных чисел
        """
        self.generator_type = generator_type
        self.param_ranges = param_ranges
        self.series_length = series_length
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Словарь доступных функций генерации
        self.generator_functions = {
            'linear': linear_trend,
            'quadratic': quadratic_trend,
            'exponential': exponential_trend,
            'seasonal': seasonal_series,
            'harmonic': harmonic_oscillator,
            'sawtooth': sawtooth_wave,
            'random_walk': random_walk
        }
        
        if generator_type not in self.generator_functions:
            raise ValueError(f"Неизвестный тип генератора: {generator_type}")
            
        # Создаем все возможные комбинации параметров
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        self.param_combinations = list(itertools.product(*param_values))
        
        # Создаем группы (комбинации параметров без учета noise_level)
        self.groups = {}
        for params in self.param_combinations:
            # Создаем ключ группы без учета noise_level
            group_params = []
            for name, value in zip(param_names, params):
                if name != 'noise_level':
                    group_params.append((name, value))
            group_key = tuple(sorted(group_params))
            
            if group_key not in self.groups:
                self.groups[group_key] = len(self.groups)
    
    def generate_dataset(self) -> Tuple[np.ndarray, List[Tuple[List[float], int]]]:
        """
        Генерация датасета.
        
        Returns:
            Tuple[np.ndarray, List[Tuple[List[float], int]]]: 
                - Массив временных рядов
                - Список кортежей (параметры, группа) для каждого ряда
        """
        series_list = []
        labels_list = []
        param_names = list(self.param_ranges.keys())
        
        for params in self.param_combinations:
            # Создаем словарь параметров
            param_dict = dict(zip(param_names, params))
            param_dict['length'] = self.series_length
            param_dict['random_state'] = self.random_state
            param_dict['only_array'] = True
            
            # Создаем ключ группы
            group_params = []
            for name, value in zip(param_names, params):
                if name != 'noise_level':
                    group_params.append((name, value))
            group_key = tuple(sorted(group_params))
            group_id = self.groups[group_key]
            
            # Генерация временного ряда
            series = self.generator_functions[self.generator_type](**param_dict)
            
            series_list.append(series)
            labels_list.append((list(params), group_id))
        
        return np.array(series_list), labels_list
    
    def save_dataset(self, series: np.ndarray, labels: List[Tuple[List[float], int]], save_path: Union[str, Path]):
        """
        Сохранение датасета в JSON файл.

        Args:
            series: Массив временных рядов
            labels: Список кортежей (параметры, группа)
            save_path: Путь для сохранения файла
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'series': series.tolist(),
            'labels': labels,
            'groups': {str(k): v for k, v in self.groups.items()},
            'metadata': {
                'generator_type': self.generator_type,
                'param_ranges': self.param_ranges,
                'series_length': self.series_length
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f)

class Parametric_dataset(Dataset):
    """
    Класс для загрузки и использования параметрического датасета в PyTorch.
    """
    def __init__(self, data_path: Union[str, Path], n_dims: Optional[int] = None, normalize: bool = False, norm_type: str = 'minmax'):
        """
        Инициализация датасета.

        Args:
            data_path: Путь к JSON файлу с данными
            n_dims: Количество измерений для преобразования (если None, оставляет одномерным)
            normalize: Флаг для нормализации данных
            norm_type: Тип нормализации ('minmax' или 'zscore')
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        self.series = np.array(data['series'])
        self.labels = data['labels']
        self.groups = data['groups']
        self.metadata = data['metadata']
        
        self.normalize = normalize
        self.norm_type = norm_type
        
        # Вычисляем статистики для нормализации
        if normalize:
            if n_dims is not None:
                # Для многомерных данных нормализуем каждое измерение отдельно
                self.norm_stats = []
                for dim in range(n_dims):
                    stats = self._compute_norm_stats(self.series[:, :, dim])
                    self.norm_stats.append(stats)
            else:
                # Для одномерных данных
                self.norm_stats = self._compute_norm_stats(self.series)
        
        if n_dims is not None:
            self._transform_to_multidimensional(n_dims)
    
    def _compute_norm_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Вычисление статистик для нормализации.

        Args:
            data: Массив данных

        Returns:
            Словарь со статистиками (min, max для minmax или mean, std для zscore)
        """
        if self.norm_type == 'minmax':
            return {
                'min': np.min(data),
                'max': np.max(data)
            }
        elif self.norm_type == 'zscore':
            return {
                'mean': np.mean(data),
                'std': np.std(data)
            }
        else:
            raise ValueError(f"Неизвестный тип нормализации: {self.norm_type}")
    
    def _normalize_data(self, data: np.ndarray, stats: Union[Dict[str, float], List[Dict[str, float]]]) -> np.ndarray:
        """
        Нормализация данных.

        Args:
            data: Массив данных для нормализации
            stats: Статистики для нормализации

        Returns:
            Нормализованный массив данных
        """
        if self.norm_type == 'minmax':
            if isinstance(stats, list):
                # Для многомерных данных
                normalized = np.zeros_like(data)
                for dim in range(data.shape[1]):
                    min_val = stats[dim]['min']
                    max_val = stats[dim]['max']
                    normalized[:, dim] = (data[:, dim] - min_val) / (max_val - min_val + 1e-8)
                return normalized
            else:
                # Для одномерных данных
                min_val = stats['min']
                max_val = stats['max']
                return (data - min_val) / (max_val - min_val + 1e-8)
                
        elif self.norm_type == 'zscore':
            if isinstance(stats, list):
                # Для многомерных данных
                normalized = np.zeros_like(data)
                for dim in range(data.shape[1]):
                    mean = stats[dim]['mean']
                    std = stats[dim]['std']
                    normalized[:, dim] = (data[:, dim] - mean) / (std + 1e-8)
                return normalized
            else:
                # Для одномерных данных
                mean = stats['mean']
                std = stats['std']
                return (data - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Неизвестный тип нормализации: {self.norm_type}")
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[List[float], int]]:
        """
        Получение элемента датасета.

        Args:
            idx: Индекс элемента

        Returns:
            Tuple[torch.Tensor, Tuple[List[float], int]]: 
                - Временной ряд как тензор
                - Кортеж (список параметров, ID группы)
        """
        series = self.series[idx]
        
        # Нормализация данных
        if self.normalize:
            if isinstance(self.series[0], np.ndarray) and len(self.series[0].shape) > 1:
                # Для многомерных данных
                series = self._normalize_data(series, self.norm_stats)
            else:
                # Для одномерных данных
                series = self._normalize_data(series, self.norm_stats)
        
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
    
    for i in range(1,101):
        num = np.random.randint(1, 100000)
        # # Конфигурация для первого класса (линейный тренд)
        # generator.set_class_config(i, [
        #     {
        #         'type': 'linear',
        #         'param_config': generator.param_functions['linear'](random_state=23)
        #     }
        # ])
        
        # Конфигурация для второго класса (сезонные данные)
        generator.set_class_config(i, [
            {
                'type': 'seasonal',
                'param_config': generator.param_functions['seasonal'](random_state=num)
            }
        ])
    
    # Генерация датасета
    series, labels = generator.generate_dataset()
    
    # Сохранение датасета
    generator.save_dataset(series, labels, 'data/seasonal_synth.json')
    
    # Загрузка датасета (одномерный вариант с нормализацией)
    dataset_1d = Synthetic_dataset(
        'data/seasonal_synth.json',
        normalize=False,
    )
    
    # Создание загрузчиков данных
    dataloader_1d = DataLoader(dataset_1d, batch_size=32, shuffle=True)

    # Проверка загрузки данных
    for batch_series, batch_labels in dataloader_1d:
        print("1D данные (normalized):")
        print(f"Batch shape: {batch_series.shape}")
        print(f"Labels: {batch_labels}")
        print(f"Data range: {batch_series.min().item():.3f} to {batch_series.max().item():.3f}")
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

    # Пример использования параметрического генератора для сезонных данных с нормализацией
    seasonal_generator = Parametric_dataset_generator(
        generator_type='seasonal',
        param_ranges={
            'amplitude': [-0.5, 0.5 , 1, -1, 5, -5, 10, -10, 100, -100],
            'frequency': [0.5, 1, 3 , 5, 7, 10, 15, 20, 30],
            'noise_level': [0, 0.1], # , 0.3, 0.5, 0.7, 1, 2, 3],
            'phase': [0.5, 1, 2, 3, 4, 5]
        },
        series_length=100,
        random_state=42
    )
    
    # Генерация датасета
    series, labels = seasonal_generator.generate_dataset()
    
    # Сохранение датасета
    seasonal_generator.save_dataset(series, labels, 'data/seasonal_dataset2.json')
    
    # Загрузка датасета с нормализацией
    dataset = Parametric_dataset(
        'data/seasonal_dataset2.json',
        normalize=False,
        norm_type='minmax'  # или 'zscore'
    )
    
    # Создание загрузчика данных
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Проверка загрузки данных
    for batch_series, batch_labels in dataloader:
        print("Seasonal dataset (normalized):")
        print("Batch shape:", batch_series.shape)
        print("Labels:", batch_labels)
        break
        
    # Пример использования параметрического генератора для линейного тренда
    linear_generator = Parametric_dataset_generator(
        generator_type='linear',
        param_ranges={
            'slope': [-1, -0.5, 0, 0.5, 1],
            'noise_level': [0.1, 0.3, 0.5, 0.7, 1]
        },
        series_length=100,
        random_state=42
    )
    
    # Генерация датасета
    series, labels = linear_generator.generate_dataset()
    
    # Сохранение датасета
    linear_generator.save_dataset(series, labels, 'data/linear_dataset.json')
    
    # Загрузка датасета
    dataset = Parametric_dataset('data/linear_dataset.json')
    
    # Создание загрузчика данных
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Проверка загрузки данных
    for batch_series, batch_labels in dataloader:
        print("\nLinear dataset:")
        print("Batch shape:", batch_series.shape)
        print("Labels:", batch_labels)
        break

if __name__ == '__main__':
    main() 