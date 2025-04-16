from typing import List, Dict, Optional, Tuple, Literal, Union, Callable
import inspect

import numpy as np

__all__ = [
    "linear_trend",
    "linear_trend_params",
    "quadratic_trend",
    "quadratic_trend_params",
    "exponential_trend",
    "exponential_trend_params",
    "seasonal_series",
    "seasonal_series_params",   
    "sawtooth_wave",
    "sawtooth_wave_params",
    "harmonic_oscillator",
    "harmonic_oscillator_params",
    "random_walk",
    "random_walk_params",
    'Time_series_generators_catalog'
]

class Time_series_generators_catalog:
    """
    Класс для хранения и управления генераторами временных рядов.
    """
    def __init__(self):
        self._generators = {
            'linear': {
                'generator': linear_trend,
                'params_generator': linear_trend_params
            },
            'quadratic': {
                'generator': quadratic_trend,
                'params_generator': quadratic_trend_params
            },
            'exponential': {
                'generator': exponential_trend,
                'params_generator': exponential_trend_params
            },
            'seasonal': {
                'generator': seasonal_series,
                'params_generator': seasonal_series_params
            },
            'sawtooth': {
                'generator': sawtooth_wave,
                'params_generator': sawtooth_wave_params
            },
            'harmonic': {
                'generator': harmonic_oscillator,
                'params_generator': harmonic_oscillator_params
            },
            'random_walk': {
                'generator': random_walk,
                'params_generator': random_walk_params
            }
        }

    def get_generator(self, generator_type: str) -> Dict[str, Callable]:
        """
        Получение генератора временного ряда по типу.
        """
        if generator_type not in self._generators:
            raise ValueError(f"Unknown time series generator type: {generator_type}")
        return self._generators[generator_type]

    def add_generator(self, generator_type: str, generator: Callable, params_generator: Callable):
        """
        Добавление нового генератора временного ряда.
        """
        self._generators[generator_type] = {
            'generator': generator,
            'params_generator': params_generator
        }

    # def update_generator_params(self, generator_type: str, new_params: dict):
    #     if generator_type not in self._generators:
    #         raise ValueError(f"Unknown generator type: {generator_type}")
        
    #     current_params_generator = self._generators[generator_type]['params_generator']
        
    #     def new_params_generator(*args, **kwargs):
    #         updated_params = {**new_params, **kwargs}
    #         return current_params_generator(**updated_params)
        
    #     self._generators[generator_type]['params_generator'] = new_params_generator

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
#Функции генерации Временных рядов (ВР)
###

def linear_trend(slope: float, length: int) -> np.ndarray:
    return slope * np.arange(length)

def quadratic_trend(a: float, b: float, c: float, length: int) -> np.ndarray:
    x = np.arange(length)
    return a * x**2 + b * x + c

def exponential_trend(alpha: float, length: int) -> np.ndarray:
    x = np.arange(length)
    return np.exp(alpha * x)

def seasonal_series(amplitude: float, frequency: float, phase: float, length: int) -> np.ndarray:
    x = np.arange(length)
    return amplitude * np.sin(2 * np.pi * frequency * x / length + phase)

def sawtooth_wave(amplitude: float, frequency: float, length: int) -> np.ndarray:
    t = np.arange(length)
    period = length / frequency
    return amplitude * (t % period) / period

def harmonic_oscillator(amplitude: float, frequency: float, damping: float, length: int) -> np.ndarray:
    t = np.arange(length)
    return amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t / length)

def random_walk(initial_value: float, noise_std: float, length: int) -> np.ndarray:
    series = np.zeros(length)
    series[0] = initial_value
    noise = np.random.normal(0, noise_std, length)
    for t in range(1, length):
        series[t] = series[t - 1] + noise[t]
    return series

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

def linear_trend_params(
    config: dict = {},
    k: int = 1,
    slope_d: float = -1,
    slope_u: float = 1,
    slope_q: int = 20,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "slope": {"d": slope_d, "u": slope_u, "q": slope_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def quadratic_trend_params(
    config: dict = {},
    k: int = 1,
    a_d: float = -0.5,
    a_u: float = 0.5,
    a_q: int = 10,
    b_d: float = -1,
    b_u: float = 1,
    b_q: int = 10,
    c_d: float = -1,
    c_u: float = 1,
    c_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "a": {"d": a_d, "u": a_u, "q": a_q},
        "b": {"d": b_d, "u": b_u, "q": b_q},
        "c": {"d": c_d, "u": c_u, "q": c_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def exponential_trend_params(
    config: dict = {},
    k: int = 1,
    alpha_d: float = -0.2,
    alpha_u: float = 0.2,
    alpha_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "alpha": {"d": alpha_d, "u": alpha_u, "q": alpha_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def seasonal_series_params(
    config: dict = {},
    k: int = 1,
    amplitude_d: float = 1,
    amplitude_u: float = 10,
    amplitude_q: int = 10,
    frequency_d: float = 0.1,
    frequency_u: float = 2.0,
    frequency_q: int = 10,
    phase_d: float = 0,
    phase_u: float = 2 * np.pi,
    phase_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "amplitude": {"d": amplitude_d, "u": amplitude_u, "q": amplitude_q},
        "frequency": {"d": frequency_d, "u": frequency_u, "q": frequency_q},
        "phase": {"d": phase_d, "u": phase_u, "q": phase_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def sawtooth_wave_params(
    config: dict = {},
    k: int = 1,
    amplitude_d: float = 1,
    amplitude_u: float = 10,
    amplitude_q: int = 10,
    frequency_d: float = 0.1,
    frequency_u: float = 2.0,
    frequency_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "amplitude": {"d": amplitude_d, "u": amplitude_u, "q": amplitude_q},
        "frequency": {"d": frequency_d, "u": frequency_u, "q": frequency_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def harmonic_oscillator_params(
    config: dict = {},
    k: int = 1,
    amplitude_d: float = 1,
    amplitude_u: float = 10,
    amplitude_q: int = 10,
    frequency_d: float = 0.1,
    frequency_u: float = 2.0,
    frequency_q: int = 10,
    damping_d: float = 0.01,
    damping_u: float = 0.5,
    damping_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "amplitude": {"d": amplitude_d, "u": amplitude_u, "q": amplitude_q},
        "frequency": {"d": frequency_d, "u": frequency_u, "q": frequency_q},
        "damping": {"d": damping_d, "u": damping_u, "q": damping_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)

def random_walk_params(
    config: dict = {},
    k: int = 1,
    initial_value_d: float = 0,
    initial_value_u: float = 10,
    initial_value_q: int = 10,
    noise_std_d: float = 0.1,
    noise_std_u: float = 2.0,
    noise_std_q: int = 10,
    random_state: Optional[int] = None,
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    param_ranges = {
        "initial_value": {"d": initial_value_d, "u": initial_value_u, "q": initial_value_q},
        "noise_std": {"d": noise_std_d, "u": noise_std_u, "q": noise_std_q}
    }
    return _generate_params(config, param_ranges, k, random_state, round_val, all_values)


def main():
    # import inspect
    # from src.utils import load_config_file
    
    # catalog = Time_series_generators_catalog()
    
    # def check_params(func):
    #     # Получение сигнатуры функции
    #     signature = inspect.signature(func)

    #     # Вывод параметров
    #     print("Параметры функции:")
    #     for name, param in signature.parameters.items():
    #         print(f" - {name}: {param}")
    
    # check_params(catalog.get_generator('linear')['params_generator'])
    # config = load_config_file('configs/scripts/datasets/all_gen_1_seg/linear_generator.yaml')['blocks'][0]
    # generator_type = config['ts_generator']
    # new_params = config['ts_params']
    # print(config)
    # catalog.update_generator_params(generator_type, new_params)
    # check_params(catalog.get_generator('linear')['params_generator'])
    
    return None

if __name__ == '__main__':
    main()