from typing import Dict, Optional, Union, Callable
import inspect

import numpy as np

__all__ = [
    "normal_noise",
    "normal_noise_params",
    "poisson_noise",
    "poisson_noise_params",
    "uniform_noise",
    "exponential_noise",
    "exponential_noise_params",
    "Noise_generators_catalog"
]

class Noise_generators_catalog:
    """
    Класс для хранения и управления генераторами шума.
    """
    def __init__(self):
        self._generators = {
            'normal': {
                'generator': normal_noise,
                'params_generator': normal_noise_params
            },
            'poisson': {
                'generator': poisson_noise,
                'params_generator': poisson_noise_params
            },
            'uniform': {
                'generator': uniform_noise,
                'params_generator': uniform_noise_params
            },
            'exponential': {
                'generator': normal_noise,
                'params_generator': normal_noise_params
            }
        }

    def get_generator(self, generator_type: str) -> Dict[str, Callable]:
        """
        Получение генератора шума по типу.
        """
        if generator_type not in self._generators:
            raise ValueError(f"Unknown noise generator type: {generator_type}")
        return self._generators[generator_type]

    def add_generator(
        self,
        generator_type: str,
        generator: Callable,
        params_generator: Callable
    ):
        """
        Добавление нового генератора шума.
        """
        self._generators[generator_type] = {
            'generator': generator,
            'params_generator': params_generator
        }
    
    
    def update_generator_params(self, generator_type: str, new_params: dict):
        """
        Обновление параметров генераторов через YAML-конфигурацию.
        
        Args:
            generator_type: Тип генератора (например, 'linear', 'quadratic' и т.д.)
            new_params: Словарь с новыми параметрами для генератора.
                      Пример для линейного тренда:
                      {
                          'slope_d': 1,
                          'slope_u': 5,
                          'slope_q': 100
                      }
        """
        if generator_type not in self._generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        # Получаем текущую функцию генерации параметров
        current_params_generator = self._generators[generator_type]['params_generator']
        
        # Получаем сигнатуру оригинальной функции
        original_sig = inspect.signature(current_params_generator)
        
        # Создаем новую функцию с сохранением сигнатуры и обновленными значениями по умолчанию
        def new_params_generator(*args, **kwargs):
            # Объединяем новые параметры с переданными
            updated_params = {**new_params, **kwargs}
            return current_params_generator(**updated_params)
        
        # Обновляем значения по умолчанию в сигнатуре
        new_parameters = []
        for param_name, param in original_sig.parameters.items():
            if param_name in new_params:
                # Создаем новый параметр с обновленным значением по умолчанию
                new_param = param.replace(default=new_params[param_name])
                new_parameters.append(new_param)
            else:
                new_parameters.append(param)
        
        # Создаем новую сигнатуру с обновленными значениями по умолчанию
        new_sig = original_sig.replace(parameters=new_parameters)
        new_params_generator.__signature__ = new_sig
        
        # Обновляем функцию генерации параметров
        self._generators[generator_type]['params_generator'] = new_params_generator

### 
#Функции генерации шума
###

def normal_noise(data: np.ndarray, noise_pct: float, random_state: Optional[int] = None) -> np.ndarray:
    """
    Генерация нормального шума.
    """
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.normal(0, np.std(data) * noise_pct, data.shape)
    return noise

def poisson_noise(data: np.ndarray, lambda_: float, random_state: Optional[int] = None) -> np.ndarray:
    """
    Генерация шума Пуассона.
    """
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.poisson(lambda_, data.shape)
    return noise

def uniform_noise(data: np.ndarray, low: float, high: float, random_state: Optional[int] = None) -> np.ndarray:
    """
    Генерация равномерного шума.
    """
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.uniform(low, high, data.shape)
    return noise

def exponential_noise(data: np.ndarray, scale: float, random_state: Optional[int] = None) -> np.ndarray:
    """
    Генерация экспоненциального шума.
    """
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.exponential(scale, data.shape)
    return noise

### 
#Функции генерации параметров ВР
###

def _generate_params(
    config: dict,
    param_ranges: dict,
    k: int = 1,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Универсальная функция для генерации параметров.
    Args:
        config: Словарь с переопределенными значениями параметров (например, из YAML).
        param_ranges: Словарь с диапазонами значений параметров.
        k: Коэффициент масштабирования.
        random_state: Фиксация случайного состояния.
        round_val: Количество знаков после запятой.
        all_values: Если True, возвращает все возможные значения.
    Returns:
        Словарь с параметрами.
    """
    if random_state is not None:
        np.random.seed(random_state)

    params = {}
    for param_name, param_config in param_ranges.items():
        # Используем значение из config, если оно есть, иначе берем из param_ranges
        d = config.get(f"{param_name}_d", param_config["d"])
        u = config.get(f"{param_name}_u", param_config["u"])
        q = config.get(f"{param_name}_q", param_config["q"])

        values = (np.linspace(d, u, q) / k).round(round_val)
        params[param_name] = values if all_values else np.random.choice(values)

    return params

def normal_noise_params(
    config: dict = {},
    k: int = 1,
    noise_pct_d: float = 0.01,
    noise_pct_u: float = 0.5,
    noise_pct_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "noise_pct": {"d": noise_pct_d, "u": noise_pct_u, "q": noise_pct_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def poisson_noise_params(
    config: dict = {},
    k: int = 1,
    lambda_d: float = 0.1,
    lambda_u: float = 5.0,
    lambda_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "lambda": {"d": lambda_d, "u": lambda_u, "q": lambda_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def uniform_noise_params(
    config: dict = {},
    k: int = 1,
    low_d: float = -1.0,
    low_u: float = 0.0,
    low_q: int = 10,
    high_d: float = 0.0,
    high_u: float = 1.0,
    high_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "low": {"d": low_d, "u": low_u, "q": low_q},
        "high": {"d": high_d, "u": high_u, "q": high_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def exponential_noise_params(
    config: dict = {},
    k: int = 1,
    scale_d: float = 0.1,
    scale_u: float = 5.0,
    scale_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "scale": {"d": scale_d, "u": scale_u, "q": scale_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)