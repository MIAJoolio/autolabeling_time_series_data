from typing import Callable, Union, Dict, Optional, List
import inspect
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import ParameterGrid

from src.utils.files_helper import load_config_file, save_config_file


class TS_feature_extractor:
    """
    Класс для извлечения признаков из временного ряда с помощью заданной функции.
    
    Параметры:
        feature_func (Callable): Функция, которая принимает временной ряд и возвращает вектор признаков.
        config (Union[dict, str], optional): Конфигурация параметров функции. По умолчанию используется сигнатура функции.
    """

    def __init__(self, feature_func: Callable, config: Union[dict, str] = None):
        self.feature_func = feature_func
        if isinstance(config, dict):
            self.config = config.copy()
        elif isinstance(config, str):
            self.config = load_config_file(config)
        else:
            self.config = {}

    def generate_params(self, round_val: int = 3, all_values: bool = False,
                        random_state: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Генерирует параметры для функции извлечения признаков на основе сигнатуры или конфига.

        Args:
            round_val (int): Число знаков после запятой для численных параметров.
            all_values (bool): Если True, возвращаются все значения по каждому параметру.
            random_state (Optional[int]): Фиксирует случайность выбора значений.

        Returns:
            Dict[str, Union[float, np.ndarray]]: Сгенерированные параметры.
        """
        if random_state is not None:
            np.random.seed(random_state)
        elif self.config.get('random_state', None) is not None:
            np.random.seed(self.config.get('random_state'))

        sig = inspect.signature(self.feature_func)
        params = {}

        for name, param in sig.parameters.items():
            if name == "series":
                continue  # Пропускаем аргумент time_series — это входной ряд

            d = self.config.get(f"{name}_d", param.default)
            u = self.config.get(f"{name}_u", param.default)
            q = self.config.get(f"{name}_q", 1)

            self.config[f"{name}_d"] = d
            self.config[f"{name}_u"] = u
            self.config[f"{name}_q"] = q

            value = self.config[f"{name}_d"] if all_values else np.random.choice(self.config[f"{name}_d"])
            params[name] = value

        return params

    def extract(self, time_series: np.ndarray, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Извлекает признаки из временного ряда.

        Args:
            time_series (np.ndarray): Входной временной ряд.
            **kwargs: Дополнительные параметры для feature_func или generate_params.

        Returns:
            np.ndarray: Вектор признаков (или список, если all_values=True).
        """
        if kwargs.get("all_values"):
            params_grid = self.generate_params(**kwargs)
            from itertools import product

            # Генерируем все комбинации параметров
            keys = list(params_grid.keys())
            values = list(params_grid.values())
            combinations = [dict(zip(keys, combo)) for combo in product(*values)]

            return [self.feature_func(time_series, **params) for params in combinations]

        params = self.generate_params(**kwargs)
        return self.feature_func(time_series, **params)

    def save_config(self, config_path: str = 'configs/config_file.yaml'):
        """
        Сохраняет текущий конфиг в указанный путь.
        """
        split_name = config_path.split('.')
        name = '_'.join([split_name[0], str(self.config.get('length', 'default'))])
        fmt = split_name[-1]
        full_name = f"{name}.{fmt}"
        save_config_file(self.config, full_name)
        return full_name
    

class Feature_extraction_method:
    def __init__(self, method_func, config_path=None):
        self.method_func = method_func
        self.config_path = config_path
        self.param_grid = None
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """Загружает конфигурационный файл и формирует сетку параметров"""
        config = load_config_file(config_path)
            
        param_grid = {}
        for key, val in config.get('param_grid', {}).items():
            if val['type'] == 'Integer' or val['type'] == 'Real':
                low, high = val.get('low'), val.get('high')
                step = val.get('step', 1)
                param_grid[key] = list(range(low, high + 1, step))
            elif val['type'] == 'Categorical':
                param_grid[key] = val.get('values', [])
        self.param_grid = param_grid

    def infer(self, series, params=None):
        """
        Применяет функцию извлечения признаков либо с конкретными параметрами, 
        либо с гридом, если параметры не заданы.
        """
        if params is not None:
            return self.method_func(series, **params)
        
        if self.param_grid is None:
            raise ValueError("Не загружена сетка параметров.")

        results = []
        for param_set in ParameterGrid(self.param_grid):
            try:
                features = self.method_func(series, **param_set)
                results.append((param_set, features))
            except Exception as e:
                print(f"Ошибка при использовании {param_set}: {e}")
        return results


class Feature_extractor_pipeline:
    def __init__(self, methods):
        self.methods = methods  # Список Feature_extraction_method объектов

    def parall_extract(self, series, verbose=False):
        """
        Применяет все методы к одному временному ряду.
        Возвращает:
        - features_list: список массивов признаков (по одному на метод)
        - params_log: информация о применённых параметрах
        """
        features_list = []
        params_log = []

        for method in self.methods:
            name = method.method_func.__name__
            if verbose:
                print(f"Обработка методом: {name}")
            try:
                result = method.infer(series)  # может быть list[(params, feats)] или feats
                if isinstance(result, list):  # если grid_search вернул несколько вариантов
                    method_results = []
                    method_params = []
                    for param_set, feats in result:
                        method_results.append(feats)
                        method_params.append({
                            'method': name,
                            'params': param_set
                        })
                    features_list.append(method_results)
                    params_log.append(method_params)
                else:  # если один результат
                    features_list.append([result])
                    params_log.append([{
                        'method': name,
                        'params': {}
                    }])
            except Exception as e:
                if verbose:
                    print(f"Ошибка в методе {name}: {e}")
                continue

        return features_list, params_log

    def batch_extract(self, dataset, mode='concat', verbose=False):
        if mode not in ['concat', 'grid']:
            raise ValueError("mode должен быть 'concat' или 'grid'")

        all_features_by_row = []
        all_features_by_param = []
        all_params = []

        for i, series in enumerate(dataset):
            if verbose:
                print(f"Ряд {i + 1}/{len(dataset)}")
            features_list, feature_params = self.parall_extract(series, verbose=verbose)

            # Сохраняем параметры
            all_params.append(feature_params)

            # Для режима concat: объединяем все признаки по всем методам и параметрам
            flat_features = []
            for method_idx in range(len(features_list)):
                for param_variant in features_list[method_idx]:
                    flat_features.append(param_variant.ravel())
            all_features_by_row.append(np.concatenate(flat_features))

            # Для grid: сохраняем как есть
            all_features_by_param.append(features_list)

        if mode == 'concat':
            return np.array(all_features_by_row, dtype=np.float32), all_params

        elif mode == 'grid':
            # Определяем максимальное число параметров среди всех методов
            max_n_params = max(len(method) for method in all_features_by_param[0])

            # Группируем по каждому набору параметров
            X_by_param = [[] for _ in range(max_n_params)]

            for sample_idx in range(len(dataset)):
                for param_idx in range(max_n_params):
                    combined = []
                    for method_idx in range(len(self.methods)):
                        # Используем последний доступный параметр, если меньше max_n_params
                        n_method_params = len(all_features_by_param[sample_idx][method_idx])
                        use_idx = min(param_idx, n_method_params - 1)
                        combined.append(all_features_by_param[sample_idx][method_idx][use_idx])
                    X_by_param[param_idx].append(np.concatenate(combined))

            return np.array(X_by_param, dtype=np.float32), all_params