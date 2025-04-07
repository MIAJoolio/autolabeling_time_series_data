import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from src.utils import *

__all__ = [
    "normal_noise",
    "normal_noise_params",
]

def normal_noise(data:np.ndarray, noise_pct:float, random_state:int=42)->np.ndarray:
    """
    Функция генерации нормального шума. 
    
    Args:
        data: временной ряд 
        noise_pct: процент от стандартного отклонения ряда 

    Returns:
        Возвращает шум для данного временного ряда
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    data_with_noise = np.random.normal(0, np.std(data)*noise_pct, data.shape)
    
    return data_with_noise

def normal_noise_params(k: int = 1, noise_pct_d: float = 0.01, noise_pct_u: float = 0.5, noise_pct_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для шума.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        noise_pct_d: Нижняя граница для уровня шума.
        noise_pct_u: Верхняя граница для уровня шума.
        noise_pct_q: Количество точек в диапазоне для уровня шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для шума: noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    noise_pct_values = np.linspace(noise_pct_d, noise_pct_u, noise_pct_q) / k

    return {
        'noise_pct': np.random.choice(noise_pct_values)
    }
