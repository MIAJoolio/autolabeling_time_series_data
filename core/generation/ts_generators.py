import numpy as np
from typing import List, Dict, Optional, Tuple
from core.utils import plot_series_grid

def linear_trend(slope: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда с линейным трендом.

    Args:
        slope: Наклон тренда.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, тренд, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация линейного тренда
    trend = slope * np.arange(length)
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = trend + noise

    if only_array:
        return time_series

    return time_series, trend, noise

def linear_trend_params(k: int = 1, slope_d: float = -1, slope_up: float = 1, slope_q: int = 20, noise_u: float = 0.01, noise_d: float = 3, noise_q: int = 100, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для линейного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        slope_d: Нижняя граница для наклона.
        slope_up: Верхняя граница для наклона.
        slope_q: Количество точек в диапазоне для наклона.
        noise_u: Верхняя граница для уровня шума.
        noise_d: Нижняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для линейного тренда: slope, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    slopes = np.linspace(slope_d, slope_up, slope_q) / k
    noises = np.linspace(noise_d, noise_u, noise_q) / k

    return {'slope': np.random.choice(slopes), 'noise_level': np.random.choice(noises)}

def quadratic_trend(a: float, b: float, c: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда с квадратичным трендом.

    Args:
        a: Коэффициент при квадратичном члене.
        b: Коэффициент при линейном члене.
        c: Свободный член.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, тренд, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация временной оси
    x = np.arange(length)
    # вычисление квадратичного тренда
    trend = a * x**2 + b * x + c
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = trend + noise

    if only_array:
        return time_series

    return time_series, trend, noise

def quadratic_trend_params(k: int = 1, a_d: float = -0.5, a_up: float = 0.5, a_q: int = 10, 
                           b_d: float = -1, b_up: float = 1, b_q: int = 10, 
                           c_d: float = -1, c_up: float = 1, c_q: int = 10, 
                           noise_u: float = 5, noise_d: float = 10, noise_q: int = 100, 
                           random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для квадратичного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        a_d: Нижняя граница для коэффициента a.
        a_up: Верхняя граница для коэффициента a.
        a_q: Количество точек в диапазоне для a.
        b_d: Нижняя граница для коэффициента b.
        b_up: Верхняя граница для коэффициента b.
        b_q: Количество точек в диапазоне для b.
        c_d: Нижняя граница для коэффициента c.
        c_up: Верхняя граница для коэффициента c.
        c_q: Количество точек в диапазоне для c.
        noise_u: Верхняя граница для уровня шума.
        noise_d: Нижняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для квадратичного тренда: a, b, c, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    a_values = np.linspace(a_d, a_up, a_q) / k
    b_values = np.linspace(b_d, b_up, b_q) / k
    c_values = np.linspace(c_d, c_up, c_q) / k
    noise_values = np.linspace(noise_d, noise_u, noise_q) / k

    return {'a': np.random.choice(a_values),
            'b': np.random.choice(b_values),
            'c': np.random.choice(c_values),
            'noise_level': np.random.choice(noise_values)}

def exponential_trend(alpha: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда с экспоненциальным трендом.

    Args:
        alpha: Коэффициент экспоненты.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, тренд, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация временной оси
    x = np.arange(length)
    # вычисление экспоненциального тренда
    trend = np.exp(alpha * x)
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = trend + noise

    if only_array:
        return time_series

    return time_series, trend, noise

def exponential_trend_params(k: int = 1, alpha_d: float = -0.2, alpha_up: float = 0.2, alpha_q: int = 10, 
                             noise_d: float = 2, noise_up: float = 10, noise_q: int = 10, 
                             random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для экспоненциального тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        alpha_d: Нижняя граница для коэффициента alpha.
        alpha_up: Верхняя граница для коэффициента alpha.
        alpha_q: Количество точек в диапазоне для alpha.
        noise_d: Нижняя граница для уровня шума.
        noise_up: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для экспоненциального тренда: alpha, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    alpha_values = np.linspace(alpha_d, alpha_up, alpha_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {'alpha': np.random.choice(alpha_values),
            'noise_level': np.random.choice(noise_values)}

def seasonal_series(amplitude: float, frequency: float, phase: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда с сезонностью.

    Args:
        amplitude: Амплитуда сезонности.
        frequency: Частота сезонности (количество циклов за период).
        phase: Фаза сезонности (сдвиг по горизонтали).
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, сезонность, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация временной оси
    x = np.arange(length)
    # вычисление сезонности
    seasonality = amplitude * np.sin(2 * np.pi * frequency * x / length + phase)
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = seasonality + noise

    if only_array:
        return time_series

    return time_series, seasonality, noise

def seasonal_series_params(k: int = 1, amplitude_d: float = 1, amplitude_up: float = 10, amplitude_q: int = 10, 
                           frequency_d: float = 0.1, frequency_up: float = 2.0, frequency_q: int = 10, 
                           phase_d: float = 0, phase_up: float = 2 * np.pi, phase_q: int = 10, 
                           noise_d: float = 0.1, noise_up: float = 1.0, noise_q: int = 10, 
                           random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для сезонного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_up: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_up: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        phase_d: Нижняя граница для фазы.
        phase_up: Верхняя граница для фазы.
        phase_q: Количество точек в диапазоне для фазы.
        noise_d: Нижняя граница для уровня шума.
        noise_up: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для сезонного тренда: amplitude, frequency, phase, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    phase_values = np.linspace(phase_d, phase_up, phase_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'phase': np.random.choice(phase_values),
        'noise_level': np.random.choice(noise_values)
    }

def harmonic_oscillator(amplitude: float, frequency: float, damping: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда, моделирующего гармонический осциллятор.

    Args:
        amplitude: Амплитуда осциллятора.
        frequency: Частота осциллятора (количество колебаний за период).
        damping: Коэффициент затухания.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, осциллятор, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация временной оси
    t = np.arange(length)
    # вычисление осциллятора
    oscillator = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t / length)
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = oscillator + noise

    if only_array:
        return time_series

    return time_series, oscillator, noise

def harmonic_oscillator_params(k: int = 1, amplitude_d: float = 1, amplitude_up: float = 10, amplitude_q: int = 10, 
                               frequency_d: float = 0.1, frequency_up: float = 2.0, frequency_q: int = 10, 
                               damping_d: float = 0.01, damping_up: float = 0.5, damping_q: int = 10, 
                               noise_d: float = 0.1, noise_up: float = 1.0, noise_q: int = 10, 
                               random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для гармонического осциллятора.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_up: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_up: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        damping_d: Нижняя граница для коэффициента затухания.
        damping_up: Верхняя граница для коэффициента затухания.
        damping_q: Количество точек в диапазоне для затухания.
        noise_d: Нижняя граница для уровня шума.
        noise_up: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для гармонического осциллятора: amplitude, frequency, damping, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    damping_values = np.linspace(damping_d, damping_up, damping_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'damping': np.random.choice(damping_values),
        'noise_level': np.random.choice(noise_values)
    }

def sawtooth_wave(amplitude: float, frequency: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда, моделирующего пилообразный сигнал.

    Args:
        amplitude: Амплитуда пилообразного сигнала.
        frequency: Частота сигнала (количество пилообразных циклов за период).
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, пилообразный сигнал, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация временной оси
    t = np.arange(length)
    # вычисление пилообразного сигнала
    period = length / frequency
    sawtooth = amplitude * (t % period) / period
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = sawtooth + noise

    if only_array:
        return time_series

    return time_series, sawtooth, noise

def sawtooth_wave_params(k: int = 1, amplitude_d: float = 1, amplitude_up: float = 10, amplitude_q: int = 10, 
                          frequency_d: float = 0.1, frequency_up: float = 2.0, frequency_q: int = 10, 
                          noise_d: float = 0.1, noise_up: float = 1.0, noise_q: int = 10, 
                          random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для пилообразного сигнала.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_up: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_up: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        noise_d: Нижняя граница для уровня шума.
        noise_up: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для пилообразного сигнала: amplitude, frequency, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'noise_level': np.random.choice(noise_values)
    }

def random_walk(initial_value: float, noise_level: float, length: int, random_state: Optional[int] = None, only_array: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Генерация временного ряда, моделирующего случайное блуждание.

    Args:
        initial_value: Начальное значение временного ряда.
        noise_level: Уровень шума (стандартное отклонение).
        length: Длина временного ряда.
        random_state: Seed для генерации случайных чисел.
        only_array: Если True, возвращает только временной ряд.

    Returns:
        Возвращает получившийся временной ряд, шум. Если параметр only_array=True, то вернет только сгенерированный временной ряд.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация шума
    noise = np.random.normal(0, noise_level, length)
    # инициализация временного ряда
    series = np.zeros(length)
    series[0] = initial_value
    # моделирование случайного блуждания
    for t in range(1, length):
        series[t] = series[t - 1] + noise[t]

    if only_array:
        return series

    return series, noise

def random_walk_params(k: int = 1, initial_value_d: float = 0, initial_value_up: float = 10, initial_value_q: int = 10, 
                        noise_d: float = 0.1, noise_up: float = 1.0, noise_q: int = 10, 
                        random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для случайного блуждания.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        initial_value_d: Нижняя граница для начального значения.
        initial_value_up: Верхняя граница для начального значения.
        initial_value_q: Количество точек в диапазоне для начального значения.
        noise_d: Нижняя граница для уровня шума.
        noise_up: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для случайного блуждания: initial_value, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    initial_value_values = np.linspace(initial_value_d, initial_value_up, initial_value_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {
        'initial_value': np.random.choice(initial_value_values),
        'noise_level': np.random.choice(noise_values)
    }

class Generator:
    """Класс для генерации временных рядов из блоков."""
    
    def __init__(self, block_length: int):
        # список для хранения блоков
        self.blocks = []
        self.block_length = block_length
    
    def add_block(self, block_type: str, **params):
        """
        Добавление блока к генератору.
        
        Args:
            block_type: Тип блока (linear, quadratic, exponential, seasonal, harmonic, sawtooth, random_walk).
            **params: Параметры для генерации блока.
        """
        self.blocks.append({
            'type': block_type,
            'params': params
        })
    
    def remove_block(self, index: int):
        """
        Удаление блока по индексу.
        
        Args:
            index: Индекс блока для удаления.
        """
        if 0 <= index < len(self.blocks):
            self.blocks.pop(index)
    
    def update_block_params(self, index: int, new_params: Dict):
        """
        Обновление параметров блока.
        
        Args:
            index: Индекс блока для обновления.
            new_params: Новые параметры.
        """
        if 0 <= index < len(self.blocks):
            self.blocks[index]['params'].update(new_params)
    
    def generate(self) -> np.ndarray:
        """
        Генерация временного ряда из блоков.
        
        Returns:
            np.ndarray: Сгенерированный временной ряд.
        """
        if not self.blocks:
            raise ValueError("Нет блоков для генерации")
        
        # Словарь функций генерации
        generator_functions = {
            'linear': linear_trend,
            'quadratic': quadratic_trend,
            'exponential': exponential_trend,
            'seasonal': seasonal_series,
            'harmonic': harmonic_oscillator,
            'sawtooth': sawtooth_wave,
            'random_walk': random_walk
        }
        
        # Генерация временного ряда
        time_series = np.zeros(self.block_length)
        for block in self.blocks:
            block_type = block['type']
            params = block['params']
            params['length'] = self.block_length
            params['only_array'] = True
            
            if block_type not in generator_functions:
                raise ValueError(f"Неизвестный тип блока: {block_type}")
            
            block_series = generator_functions[block_type](**params)
            time_series += block_series
        
        return time_series

def show_all_generators_work():
    """Функция по визуализации работы всех генераторов"""
    # ex 1
    slope = 0.7  
    noise_level = 1.0  
    length = 10 

    series, trend, noise = linear_trend(slope, noise_level, length)
    plot_series_grid(
        series_list=[series, trend, noise], 
        labels=["Временной ряд", 'Тренд', 'Шум'], 
        plot_title="Линейный тренд с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 2
    a = 0.01
    b = 0.1
    c = 2
    noise_level = 0.5
    length = 10
    random_state = 42  

    series, trend, noise = quadratic_trend(a, b, c, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, trend, noise], 
        labels=["Временной ряд", 'Тренд', 'Шум'], 
        plot_title="Квадратичный тренд с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 3
    alpha = 0.5
    noise_level = 4
    length = 10
    random_state = 42  

    series, trend, noise = exponential_trend(alpha, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, trend, noise], 
        labels=["Временной ряд", 'Тренд', 'Шум'], 
        plot_title="Экспоненциальный тренд с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 4
    amplitude = 5.0
    frequency = 2.0
    phase = np.pi / 2
    noise_level = 4
    length = 100
    random_state = 42

    series, seasonality, noise = seasonal_series(amplitude, frequency, phase, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, seasonality, noise], 
        labels=["Временной ряд", 'Сезонность', 'Шум'], 
        plot_title="Сезонность (sin/cos) с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 5
    amplitude = 10.0
    frequency = 2.0
    damping = 0.05
    noise_level = 0.2
    length = 200
    random_state = 42

    series, oscillator, noise = harmonic_oscillator(amplitude, frequency, damping, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, oscillator, noise], 
        labels=["Временной ряд", 'Гармонический осциллятор', 'Шум'], 
        plot_title="Гармонический осциллятор", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 6
    amplitude = 5.0
    frequency = 4.0
    noise_level = 0.1
    length = 200
    random_state = 42

    series, sawtooth, noise = sawtooth_wave(amplitude, frequency, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, sawtooth, noise], 
        labels=["Временной ряд", 'Пилообразный сигнал', 'Шум'], 
        plot_title="Пилообразный сигнал", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 7
    initial_value = 10.0
    noise_level = 0.5
    length = 100
    random_state = 42

    series, noise = random_walk(initial_value, noise_level, length, random_state)
    plot_series_grid(
        series_list=[series, noise], 
        labels=["Временной ряд", 'Шум'], 
        plot_title="Случайное блуждание", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

def main():
    if SHOW_GENS == True:
        show_all_generators_work()
    
    if SHOW_GENERATOR == True:
        # создаем генератор
        generator = Generator(block_length=20)

        # добавляем блоки
        generator.add_block("linear", slope=0.1, noise_level=0.01)
        generator.add_block("seasonal", amplitude=10.0, frequency=1.0, phase=0.0, noise_level=0.01)

        # Генерация временного ряда
        time_series = generator.generate()

        # Вывод результата
        print("Generated time series:", time_series)
        
        # Визуализация результатов
        plot_series_grid(
            [time_series],
            ['full series'],
            plot_title="Сгенерированный временной ряд",
            xlabel="Время",
            ylabel="Значение",
            figsize=(14, 3),
            layout='horizontal'
        )

if __name__ == '__main__':
    
    SHOW_GENS = False
    SHOW_GENERATOR = True
    
    main()