import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from core.utils import *

__all__ = [
    "linear_trend",
    "linear_trend_params",
    "quadratic_trend",
    "quadratic_trend_params",
    "exponential_trend",
    "exponential_trend_params",
    "seasonal_series",
    "seasonal_series_params",   
    "harmonic_oscillator",
    "harmonic_oscillator_params",
    "sawtooth_wave",
    "sawtooth_wave_params",
    "random_walk",
    "random_walk_params",
    "noise_generator",
    "noise_generator_params",
    "Generator"
]

# добавим logger
logger = setup_logger(__name__, low_lvl='debug')

def noise_generator(data:np.ndarray, noise_lvl, noise_type:Literal['normal']='normal')->np.ndarray:
    """
    Функция генерации шума. 
    
    Args:
        data (np.ndarray): _description_
        noise_type (Literal[&#39;normal&#39;], optional): _description_. Defaults to 'normal'.

    Returns:
        : шум
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if noise_type == 'normal': 
        return np.random.normal(0, np.std(data)*noise_lvl, data.shape)
    

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

def linear_trend_params(k: int = 1, slope_d: float = -1, slope_u: float = 1, slope_q: int = 20, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для линейного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        slope_d: Нижняя граница для наклона.
        slope_u: Верхняя граница для наклона.
        slope_q: Количество точек в диапазоне для наклона.
        noise_d: Нижняя граница для уровня шума.
        noise_u: Верхняя граница для уровня шума.
        noise_q: Количество точек в диапазоне для шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для линейного тренда: slope, noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    slopes = np.linspace(slope_d, slope_u, slope_q) / k

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

def quadratic_trend_params(k: int = 1, a_d: float = -0.5, a_u: float = 0.5, a_q: int = 10, b_d: float = -1, b_u: float = 1, b_q: int = 10, c_d: float = -1, c_u: float = 1, c_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для квадратичного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        a_d: Нижняя граница для коэффициента a.
        a_u: Верхняя граница для коэффициента a.
        a_q: Количество точек в диапазоне для a.
        b_d: Нижняя граница для коэффициента b.
        b_u: Верхняя граница для коэффициента b.
        b_q: Количество точек в диапазоне для b.
        c_d: Нижняя граница для коэффициента c.
        c_u: Верхняя граница для коэффициента c.
        c_q: Количество точек в диапазоне для c.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для квадратичного тренда: a, b, c.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    a_values = np.linspace(a_d, a_u, a_q) / k
    b_values = np.linspace(b_d, b_u, b_q) / k
    c_values = np.linspace(c_d, c_u, c_q) / k

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

def exponential_trend_params(k: int = 1, alpha_d: float = -0.2, alpha_u: float = 0.2, alpha_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для экспоненциального тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        alpha_d: Нижняя граница для коэффициента alpha.
        alpha_u: Верхняя граница для коэффициента alpha.
        alpha_q: Количество точек в диапазоне для alpha.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для экспоненциального тренда: alpha.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    alpha_values = np.linspace(alpha_d, alpha_u, alpha_q) / k

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

def seasonal_series_params(k: int = 1, amplitude_d: float = 1, amplitude_u: float = 10, amplitude_q: int = 10, frequency_d: float = 0.1, frequency_u: float = 2.0, frequency_q: int = 10, phase_d: float = 0, phase_u: float = 2 * np.pi, phase_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для сезонного тренда.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_u: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_u: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        phase_d: Нижняя граница для фазы.
        phase_u: Верхняя граница для фазы.
        phase_q: Количество точек в диапазоне для фазы.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для сезонного тренда: amplitude, frequency, phase.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_u, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_u, frequency_q) / k
    phase_values = np.linspace(phase_d, phase_u, phase_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'phase': np.random.choice(phase_values)
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

def harmonic_oscillator_params(k: int = 1, amplitude_d: float = 1, amplitude_u: float = 10, amplitude_q: int = 10, frequency_d: float = 0.1, frequency_u: float = 2.0, frequency_q: int = 10, damping_d: float = 0.01, damping_u: float = 0.5, damping_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для гармонического осциллятора.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_u: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_u: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        damping_d: Нижняя граница для коэффициента затухания.
        damping_u: Верхняя граница для коэффициента затухания.
        damping_q: Количество точек в диапазоне для затухания.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для гармонического осциллятора: amplitude, frequency, damping.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_u, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_u, frequency_q) / k
    damping_values = np.linspace(damping_d, damping_u, damping_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'damping': np.random.choice(damping_values)
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

def sawtooth_wave_params(k: int = 1, amplitude_d: float = 1, amplitude_u: float = 10, amplitude_q: int = 10, frequency_d: float = 0.1, frequency_u: float = 2.0, frequency_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для пилообразного сигнала.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        amplitude_d: Нижняя граница для амплитуды.
        amplitude_u: Верхняя граница для амплитуды.
        amplitude_q: Количество точек в диапазоне для амплитуды.
        frequency_d: Нижняя граница для частоты.
        frequency_u: Верхняя граница для частоты.
        frequency_q: Количество точек в диапазоне для частоты.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для пилообразного сигнала: amplitude, frequency.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    amplitude_values = np.linspace(amplitude_d, amplitude_u, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_u, frequency_q) / k

    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values)
    }

def random_walk(initial_value: float, length: int) -> np.ndarray:
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

    return series

def random_walk_params(k: int = 1, initial_value_d: float = 0, initial_value_u: float = 10, initial_value_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для случайного блуждания.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        initial_value_d: Нижняя граница для начального значения.
        initial_value_u: Верхняя граница для начального значения.
        initial_value_q: Количество точек в диапазоне для начального значения.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для случайного блуждания: initial_value.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    initial_value_values = np.linspace(initial_value_d, initial_value_u, initial_value_q) / k

    return {
        'initial_value': np.random.choice(initial_value_values)
    }

def noise_generator_params(k: int = 1, noise_level_d: float = 0.01, noise_level_u: float = 0.5, noise_level_q: int = 10, random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Генерация параметров для шума.

    Args:
        k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
        noise_level_d: Нижняя граница для уровня шума.
        noise_level_u: Верхняя граница для уровня шума.
        noise_level_q: Количество точек в диапазоне для уровня шума.
        random_state: Seed для генерации случайных чисел.

    Returns:
        Словарь с параметрами для шума: noise_level.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    noise_level_values = np.linspace(noise_level_d, noise_level_u, noise_level_q) / k

    return {
        'noise_level': np.random.choice(noise_level_values)
    }

class Generator:
    """
    Класс для генерации временного ряда из блоков.
    """
    
    def __init__(self, block_length: int):
        # список для хранения блоков
        self.blocks = []
        self.block_length = block_length
        # Словарь функций генерации
        self.generator_functions = {
            'linear': linear_trend,
            'quadratic': quadratic_trend,
            'exponential': exponential_trend,
            'seasonal': seasonal_series,
            'harmonic': harmonic_oscillator,
            'sawtooth': sawtooth_wave,
            'random_walk': random_walk
        }
        # Словарь функций генерации параметров
        self.params_generator_functions = {
            'linear': linear_trend_params,
            'quadratic': quadratic_trend_params,
            'exponential': exponential_trend_params,
            'seasonal': seasonal_series_params,
            'harmonic': harmonic_oscillator_params,
            'sawtooth': sawtooth_wave_params,
            'random_walk': random_walk_params,
            'noise': noise_generator_params
        }
    
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
        logger.debug(f'New block added. Blocks:{self.blocks}')
    
    def remove_block(self, index: int):
        """
        Удаление блока по индексу.
        
        Args:
            index: Индекс блока для удаления.
        """
        if 0 <= index < len(self.blocks):
            self.blocks.pop(index)
            
        logger.debug(f'Remove {index} block. Blocks:{self.blocks}')
    
    def update_block_params(self, index: int, new_params: Dict):
        """
        Обновление параметров блока.
        
        Args:
            index: Индекс блока для обновления.
            new_params: Новые параметры.
        """
        if 0 <= index < len(self.blocks):
            self.blocks[index]['params'].update(new_params)

        logger.debug(f'Update {index} block params. Params:{new_params}')
    
    def generate(self) -> np.ndarray:
        """
        Генерация временного ряда из блоков.
        
        Returns:
            np.ndarray: Сгенерированный временной ряд.
        """
        if not self.blocks:
            logger.error("There not any blocks to generate!")
            raise ValueError("Нет блоков для генерации")
        
        # инициализация полного ряда
        time_series = np.array([], dtype=np.float64)
        # инициализация последних значений каждого блока
        end_pts = {}
        
        # Генерация временного ряда
        for inx, block in enumerate(self.blocks):
            logger.debug(f'generate function block: {block}')
            
            block_type = block['type']
            params = block['params']
            params['length'] = self.block_length if inx == 0 else self.block_length+1
            
            if block_type not in self.generator_functions:
                logger.error(f'Invalid block type: {block_type}')
                raise ValueError(f"Неизвестный тип блока: {block_type}")
            
            block_series = np.array(self.generator_functions[block_type](**params)) + (0 if end_pts.get(inx-1) is None else end_pts.get(inx-1))
            block_series = block_series[1:] if inx >= 1 else block_series
            time_series = np.concatenate([time_series, block_series])
            logger.debug(f'Time series updates: {time_series.shape}, {time_series}')
            # Обработка последних значений
            end_pts[inx] = block_series[-1]
            logger.debug(f'Last points dict: {end_pts}')
        
        return time_series

    def generate_with_noise(self, noise_level: float=None, noise_type: Literal['normal'] = None, random_state: Optional[int] = None) -> np.ndarray:
        """
        Добавление шума к сгенерированному временному ряду.
        
        Args:
            noise_level: Уровень шума (множитель стандартного отклонения).
            noise_type: Тип шума (в настоящее время поддерживается только 'normal').
            
        Returns:
            np.ndarray: Временной ряд с добавленным шумом.
        """
        if not self.blocks:
            logger.error("Нет блоков для генерации!")
            raise ValueError("Нет блоков для генерации")
        
        # Генерируем временной ряд без шума
        time_series = self.generate()
        
        if noise_level is None or noise_type is None:
            noise_level = self.params_generator_functions['noise'](random_state=random_state)['noise_level']
            logger.debug(f"Уровень шума: {noise_level}")

        # Добавляем шум
        noise = noise_generator(time_series, noise_level)
        noisy_series = time_series + noise
        
        logger.debug(f'Добавлен шум уровня {noise_level} типа {noise_type}')
        return noisy_series

def main():
    return None

if __name__ == '__main__':
    main()