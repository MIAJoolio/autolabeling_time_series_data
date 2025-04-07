import numpy as np
from typing import List, Dict, Optional, Tuple, Literal, Union
from src.utils import *

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
    "random_walk_params"
]

def linear_trend(slope: float, length: int) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда с линейным трендом.

    Args:
        slope: Наклон тренда.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    
    # генерация линейного тренда
    trend = slope * np.arange(length)

    return trend

def linear_trend_params(
    k: int = 1, 
    slope_d: float = -1, 
    slope_u: float = 1, 
    slope_q: int = 20, 
    random_state: Optional[int] = None, 
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Генерация параметров для линейного тренда.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    slopes = (np.linspace(slope_d, slope_u, slope_q) / k).round(round_val)

    if all_values:
        return {'slope': slopes}
    else:
        return {'slope': np.random.choice(slopes)}
    
def quadratic_trend(a: float, b: float, c: float, length: int) -> np.ndarray:
    """
    Генерация временного ряда с квадратичным трендом.

    Args:
        a: Коэффициент при квадратичном члене.
        b: Коэффициент при линейном члене.
        c: Свободный член.
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # генерация временной оси
    x = np.arange(length)
    # вычисление квадратичного тренда
    trend = a * x**2 + b * x + c

    return trend
def quadratic_trend_params(
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
    """
    Генерация параметров для квадратичного тренда.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    a_values = (np.linspace(a_d, a_u, a_q) / k).round(round_val)
    b_values = (np.linspace(b_d, b_u, b_q) / k).round(round_val)
    c_values = (np.linspace(c_d, c_u, c_q) / k).round(round_val)

    if all_values:
        return {
            'a': a_values,
            'b': b_values,
            'c': c_values
        }
    else:
        return {
            'a': np.random.choice(a_values),
            'b': np.random.choice(b_values),
            'c': np.random.choice(c_values)
        }

def exponential_trend(alpha: float, length: int) -> np.ndarray:
    """
    Генерация временного ряда с экспоненциальным трендом.

    Args:
        alpha: Коэффициент экспоненты.
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # генерация временной оси
    x = np.arange(length)
    # вычисление экспоненциального тренда
    trend = np.exp(alpha * x)

    return trend

def exponential_trend_params(
    k: int = 1, 
    alpha_d: float = -0.2, 
    alpha_u: float = 0.2, 
    alpha_q: int = 10, 
    random_state: Optional[int] = None, 
    round_val: int = 3,
    all_values: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Генерация параметров для экспоненциального тренда.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    alpha_values = (np.linspace(alpha_d, alpha_u, alpha_q) / k).round(round_val)

    if all_values:
        return {'alpha': alpha_values}
    else:
        return {'alpha': np.random.choice(alpha_values)}

def seasonal_series(amplitude: float, frequency: float, phase: float, length: int) -> np.ndarray:
    """
    Генерация временного ряда с сезонностью.

    Args:
        amplitude: Амплитуда сезонности.
        frequency: Частота сезонности (количество циклов за период).
        phase: Фаза сезонности (сдвиг по горизонтали).
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # генерация временной оси
    x = np.arange(length)
    # вычисление сезонности
    seasonality = amplitude * np.sin(2 * np.pi * frequency * x / length + phase)

    return seasonality

def seasonal_series_params(
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
    """
    Генерация параметров для сезонного тренда.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = (np.linspace(amplitude_d, amplitude_u, amplitude_q) / k).round(round_val)
    frequency_values = (np.linspace(frequency_d, frequency_u, frequency_q) / k).round(round_val)
    phase_values = (np.linspace(phase_d, phase_u, phase_q) / k).round(round_val)

    if all_values:
        return {
            'amplitude': amplitude_values,
            'frequency': frequency_values,
            'phase': phase_values
        }
    else:
        return {
            'amplitude': np.random.choice(amplitude_values),
            'frequency': np.random.choice(frequency_values),
            'phase': np.random.choice(phase_values)
        }

def sawtooth_wave(amplitude: float, frequency: float, length: int) -> np.ndarray:
    """
    Генерация временного ряда, моделирующего пилообразный сигнал.

    Args:
        amplitude: Амплитуда пилообразного сигнала.
        frequency: Частота сигнала (количество пилообразных циклов за период).
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # генерация временной оси
    t = np.arange(length)
    # вычисление пилообразного сигнала
    period = length / frequency
    sawtooth = amplitude * (t % period) / period

    return sawtooth

def sawtooth_wave_params(
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
    """
    Генерация параметров для пилообразного сигнала.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = (np.linspace(amplitude_d, amplitude_u, amplitude_q) / k).round(round_val)
    frequency_values = (np.linspace(frequency_d, frequency_u, frequency_q) / k).round(round_val)

    if all_values:
        return {
            'amplitude': amplitude_values,
            'frequency': frequency_values
        }
    else:
        return {
            'amplitude': np.random.choice(amplitude_values),
            'frequency': np.random.choice(frequency_values)
        }

def harmonic_oscillator(amplitude: float, frequency: float, damping: float, length: int) -> np.ndarray:
    """
    Генерация временного ряда, моделирующего гармонический осциллятор.

    Args:
        amplitude: Амплитуда осциллятора.
        frequency: Частота осциллятора (количество колебаний за период).
        damping: Коэффициент затухания.
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # генерация временной оси
    t = np.arange(length)
    # вычисление осциллятора
    oscillator = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t / length)

    return oscillator

def harmonic_oscillator_params(
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
    """
    Генерация параметров для гармонического осциллятора.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = (np.linspace(amplitude_d, amplitude_u, amplitude_q) / k).round(round_val)
    frequency_values = (np.linspace(frequency_d, frequency_u, frequency_q) / k).round(round_val)
    damping_values = (np.linspace(damping_d, damping_u, damping_q) / k).round(round_val)

    if all_values:
        return {
            'amplitude': amplitude_values,
            'frequency': frequency_values,
            'damping': damping_values
        }
    else:
        return {
            'amplitude': np.random.choice(amplitude_values),
            'frequency': np.random.choice(frequency_values),
            'damping': np.random.choice(damping_values)
        }

def random_walk(initial_value: float, noise_std:float, length: int) -> np.ndarray:
    """
    Генерация временного ряда, моделирующего случайное блуждание.

    Args:
        initial_value: Начальное значение временного ряда.
        length: Длина временного ряда.
       
    Returns:
        Возвращает получившийся временной ряд
    """
    # инициализация временного ряда
    series = np.zeros(length)
    series[0] = initial_value
    # Генерация шума
    noise = np.random.normal(0, noise_std, length)
    
    # Моделирование случайного блуждания
    for t in range(1, length):
        series[t] = series[t - 1] + noise[t]
    
    return series

def random_walk_params(
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
    """
    Генерация параметров для случайного блуждания.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    initial_value_values = (np.linspace(initial_value_d, initial_value_u, initial_value_q) / k).round(round_val)
    noise_std_values = (np.linspace(noise_std_d, noise_std_u, noise_std_q) / k).round(round_val)

    if all_values:
        return {
            'initial_value': initial_value_values,
            'noise_std': noise_std_values
        }
    else:
        return {
            'initial_value': np.random.choice(initial_value_values),
            'noise_std': np.random.choice(noise_std_values)
        }

def main():
    return None

if __name__ == '__main__':
    main()