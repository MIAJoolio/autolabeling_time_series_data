import numpy as np
import visuals_tools as vis


def linear_trend(slope, noise_level, length, random_state=None, only_array=False):
    """
    Генерация временного ряда с линейным трендом.

    Параметры:
    - slope: Наклон тренда (float).
    - noise_level: Уровень шума (стандартное отклонение, float).
    - length: Длина временного ряда (int).

    Возвращает:
    - Временной ряд (numpy array).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # генерация линейного тренда
    trend = slope * np.arange(length)
    # добавление шума
    noise = np.random.normal(0, noise_level, length)
    # итоговый временной ряд
    time_series = trend + noise

    if only_array == True:
        return time_series

    return time_series, trend, noise


def linear_trend_params(k:int, slope_d = -1, slope_up=1, slope_q=20, noise_u=0.01, noise_d=3, noise_q=100):
    """
    n - количество параметров
    k - параметр регуляризации (при высоком значении кластеры будут очень близко, а при высоком далеко)

    """

    slopes = np.linspace(slope_d, slope_up, slope_q)/k
    noises = np.linspace(noise_d, noise_u, noise_q)/k

    return {'slope':np.random.choice(slopes), 'noise_level':np.random.choice(noises)}



def quadratic_trend(a, b, c, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд с квадратичным трендом и шумом.

    Параметры:
    a (float): Коэффициент при квадратичном члене (x^2).
    b (float): Коэффициент при линейном члене (x).
    c (float): Свободный член.
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.

    Возвращает:
    np.array: Временной ряд с квадратичным трендом и шумом.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация временной оси (x)
    x = np.arange(length)
    
    # Вычисление квадратичного тренда
    trend = a * x**2 + b * x + c
    
    # Добавление шума
    noise = np.random.normal(0, noise_level, length)
    
    # Временной ряд с шумом
    series = trend + noise
    
    if only_array == True:
        return series
    
    return series, trend, noise 



def quadratic_trendparams(k: int, a_d=-0.5, a_up=0.5, a_q=10, 
                                    b_d=-1, b_up=1, b_q=10, 
                                    c_d=-1, c_up=1, c_q=10, 
                                    noise_u=5, noise_d=10, noise_q=100):
    """
    Генерация параметров для квадратичного тренда.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - a_d, a_up: Диапазон для коэффициента a (квадратичный член).
    - a_q: Количество точек в диапазоне для a.
    - b_d, b_up: Диапазон для коэффициента b (линейный член).
    - b_q: Количество точек в диапазоне для b.
    - c_d, c_up: Диапазон для коэффициента c (свободный член).
    - c_q: Количество точек в диапазоне для c.
    - noise_u, noise_d: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для квадратичного тренда.
    """
    # Генерация диапазонов для коэффициентов
    a_values = np.linspace(a_d, a_up, a_q) / k
    b_values = np.linspace(b_d, b_up, b_q) / k
    c_values = np.linspace(c_d, c_up, c_q) / k
    noise_values = np.linspace(noise_d, noise_u, noise_q) / k

    return {'a': a_values,
            'b': np.random.choice(b_values),
            'c': np.random.choice(c_values),
            'noise_level': np.random.choice(noise_values)}


def exponential_trend(alpha, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд с экспоненциальным трендом и шумом.

    Параметры:
    alpha (float): Коэффициент экспоненты.
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.
    only_array (bool, optional): Если True, возвращает только временной ряд. Иначе возвращает ряд, тренд и шум.

    Возвращает:
    np.array: Временной ряд с экспоненциальным трендом и шумом.
    или
    tuple: (series, trend, noise) — временной ряд, тренд и шум, если only_array=False.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация временной оси (x)
    x = np.arange(length)
    
    # Вычисление экспоненциального тренда
    trend = np.exp(alpha * x)
    
    # Добавление шума
    noise = np.random.normal(0, noise_level, length)
    
    # Временной ряд с шумом
    series = trend + noise
    
    if only_array:
        return series
    
    return series, trend, noise


def exponential_trend_params(k: int, alpha_d=-0.2, alpha_up=0.2, alpha_q=10, noise_d=2, noise_up=10, noise_q=10):
    """
    Генерация параметров для экспоненциального тренда.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - alpha_d, alpha_up: Диапазон для коэффициента alpha (экспоненциальный коэффициент).
    - alpha_q: Количество точек в диапазоне для alpha.
    - noise_d, noise_up: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для экспоненциального тренда.
    """
    # Генерация диапазонов для коэффициентов
    alpha_values = np.linspace(alpha_d, alpha_up, alpha_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    return {'alpha': np.random.choice(alpha_values),
            'noise_level': np.random.choice(noise_values)}


def seasonal_series(amplitude, frequency, phase, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд с сезонностью (синусоида) и шумом.

    Параметры:
    amplitude (float): Амплитуда сезонности.
    frequency (float): Частота сезонности (количество циклов за период).
    phase (float): Фаза сезонности (сдвиг по горизонтали).
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.
    only_array (bool, optional): Если True, возвращает только временной ряд. Иначе возвращает ряд, сезонность и шум.

    Возвращает:
    np.array: Временной ряд с сезонностью и шумом.
    или
    tuple: (series, seasonality, noise) — временной ряд, сезонность и шум, если only_array=False.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация временной оси (x)
    x = np.arange(length)
    
    # Вычисление сезонности (синусоида)
    seasonality = amplitude * np.sin(2 * np.pi * frequency * x / length + phase)
    
    # Добавление шума
    noise = np.random.normal(0, noise_level, length)
    
    # Временной ряд с шумом
    series = seasonality + noise
    
    if only_array:
        return series
    
    return series, seasonality, noise


def seasonal_series_params(k: int, amplitude_d=1, amplitude_up=10, amplitude_q=10, frequency_d=0.1, frequency_up=2.0, frequency_q=10, 
phase_d=0, phase_up=2 * np.pi, phase_q=10, noise_d=0.1, noise_up=1.0, noise_q=10):
    """
    Генерация параметров для сезонного тренда.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - amplitude_d, amplitude_up: Диапазон для амплитуды сезонности.
    - amplitude_q: Количество точек в диапазоне для амплитуды.
    - frequency_d, frequency_up: Диапазон для частоты сезонности.
    - frequency_q: Количество точек в диапазоне для частоты.
    - phase_d, phase_up: Диапазон для фазы сезонности.
    - phase_q: Количество точек в диапазоне для фазы.
    - noise_d, noise_up: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для сезонного тренда.
    """
    # Генерация диапазонов для параметров
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    phase_values = np.linspace(phase_d, phase_up, phase_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    # Случайный выбор параметров
    return {
            'amplitude': np.random.choice(amplitude_values),
            'frequency': np.random.choice(frequency_values),
            'phase': np.random.choice(phase_values),
            'noise_level': np.random.choice(noise_values)
        }


def harmonic_oscillator(amplitude, frequency, damping, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд, моделирующий гармонический осциллятор с затуханием и шумом.

    Параметры:
    amplitude (float): Амплитуда осциллятора.
    frequency (float): Частота осциллятора (количество колебаний за период).
    damping (float): Коэффициент затухания (чем больше, тем быстрее затухание).
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.
    only_array (bool, optional): Если True, возвращает только временной ряд. Иначе возвращает ряд, осциллятор и шум.

    Возвращает:
    np.array: Временной ряд, моделирующий гармонический осциллятор с затуханием и шумом.
    или
    tuple: (series, oscillator, noise) — временной ряд, осциллятор и шум, если only_array=False.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация временной оси (t)
    t = np.arange(length)
    
    # Моделирование гармонического осциллятора с затуханием
    oscillator = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t / length)
    
    # Добавление шума
    noise = np.random.normal(0, noise_level, length)
    
    # Временной ряд с шумом
    series = oscillator + noise
    
    if only_array:
        return series
    
    return series, oscillator, noise


def harmonic_oscillator_params(k: int, amplitude_d=1, amplitude_up=10, amplitude_q=10, frequency_d=0.1, frequency_up=2.0, frequency_q=10, 
damping_d=0.01, damping_up=0.5, damping_q=10, noise_d=0.1, noise_up=1.0, noise_q=10):
    """
    Генерация параметров для затухающего гармонического осциллятора.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - amplitude_d, amplitude_up: Диапазон для амплитуды осциллятора.
    - amplitude_q: Количество точек в диапазоне для амплитуды.
    - frequency_d, frequency_up: Диапазон для частоты осциллятора.
    - frequency_q: Количество точек в диапазоне для частоты.
    - damping_d, damping_up: Диапазон для коэффициента затухания.
    - damping_q: Количество точек в диапазоне для затухания.
    - noise_d, noise_up: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для затухающего гармонического осциллятора.
    """
    # Генерация диапазонов для параметров
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    damping_values = np.linspace(damping_d, damping_up, damping_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    # Генерация списка параметров с использованием list comprehension
    return {
        'amplitude': np.random.choice(amplitude_values),
        'frequency': np.random.choice(frequency_values),
        'damping': np.random.choice(damping_values),
        'noise_level': np.random.choice(noise_values)
    } 


def sawtooth_wave(amplitude, frequency, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд, моделирующий пилообразный сигнал (линейное возрастание с резким спадом) и шум.

    Параметры:
    amplitude (float): Амплитуда пилообразного сигнала.
    frequency (float): Частота сигнала (количество пилообразных циклов за период).
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.
    only_array (bool, optional): Если True, возвращает только временной ряд. Иначе возвращает ряд, пилообразный сигнал и шум.

    Возвращает:
    np.array: Временной ряд, моделирующий пилообразный сигнал с шумом.
    или
    tuple: (series, sawtooth, noise) — временной ряд, пилообразный сигнал и шум, если only_array=False.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация временной оси (t)
    t = np.arange(length)
    
    # Моделирование пилообразного сигнала
    period = length / frequency  # Период пилообразного сигнала
    sawtooth = amplitude * (t % period) / period
    
    # Добавление шума
    noise = np.random.normal(0, noise_level, length)
    
    # Временной ряд с шумом
    series = sawtooth + noise
    
    if only_array:
        return series
    
    return series, sawtooth, noise


def sawtooth_wave_params(k: int, amplitude_d=1, amplitude_up=10, amplitude_q=10, frequency_d=0.1, frequency_up=2.0, frequency_q=10, 
noise_d=0.1, noise_up=1.0, noise_q=10):
    """
    Генерация параметров для пилообразного сигнала.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - amplitude_d, amplitude_up: Диапазон для амплитуды пилообразного сигнала.
    - amplitude_q: Количество точек в диапазоне для амплитуды.
    - frequency_d, frequency_up: Диапазон для частоты пилообразного сигнала.
    - frequency_q: Количество точек в диапазоне для частоты.
    - noise_d, noise_up: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для пилообразного сигнала.
    """
    # Генерация диапазонов для параметров
    amplitude_values = np.linspace(amplitude_d, amplitude_up, amplitude_q) / k
    frequency_values = np.linspace(frequency_d, frequency_up, frequency_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k

    fin_ampl = np.random.choice(amplitude_values)
    # Генерация списка параметров с использованием list comprehension
    return {
        'amplitude': fin_ampl,
        'frequency': np.random.choice(frequency_values),
        'noise_level': np.random.choice(noise_values)
    }


def random_walk(initial_value, noise_level, length, random_state=None, only_array=False):
    """
    Генерирует временной ряд, моделирующий случайное блуждание.

    Параметры:
    initial_value (float): Начальное значение временного ряда.
    noise_level (float): Уровень шума (стандартное отклонение нормального распределения).
    length (int): Количество точек в временном ряде.
    random_state (int, optional): Seed для генерации случайных чисел. Если None, seed не устанавливается.
    only_array (bool, optional): Если True, возвращает только временной ряд. Иначе возвращает ряд и шум.

    Возвращает:
    np.array: Временной ряд, моделирующий случайное блуждание.
    или
    tuple: (series, noise) — временной ряд и шум, если only_array=False.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Генерация шума
    noise = np.random.normal(0, noise_level, length)
    
    # Инициализация временного ряда
    series = np.zeros(length)
    series[0] = initial_value  # Начальное значение
    
    # Моделирование случайного блуждания
    for t in range(1, length):
        series[t] = series[t - 1] + noise[t]
    
    if only_array:
        return series
    
    return series, noise


def random_walk_params(k: int, initial_value_d=0, initial_value_up=10, initial_value_q=10, noise_d=0.1, noise_up=1.0, noise_q=10):
    """
    Генерация параметров для случайного блуждания.

    Параметры:
    - n: Количество наборов параметров (int).
    - k: Параметр регуляризации (при высоком значении кластеры будут очень близко, а при низком — далеко).
    - initial_value_d, initial_value_up: Диапазон для начального значения временного ряда.
    - initial_value_q: Количество точек в диапазоне для начального значения.
    - noise_d, noise_up: Диапазон для уровня шума.
    - noise_q: Количество точек в диапазоне для шума.

    Возвращает:
    - Список словарей с параметрами для случайного блуждания.
    """
    # Генерация диапазонов для параметров
    initial_value_values = np.linspace(initial_value_d, initial_value_up, initial_value_q) / k
    noise_values = np.linspace(noise_d, noise_up, noise_q) / k
    
    fin_noise = np.random.choice(noise_values)

    # Генерация списка параметров с использованием list comprehension
    return {
        'initial_value': np.random.choice(initial_value_values),
        'noise_level':fin_noise 
    }


def main():
    
    # ex 1
    slope = 0.7  
    noise_level = 1.0  
    length = 10 

    series, trend, noise = linear_trend(slope, noise_level, length)
    vis.multi_plot_each(series_list=[series, trend, noise], labels=["Временной ряд", 'Тренд', 'Шум'], plot_title="Линейный тренд с шумом", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 2
    
    a = 0.01
    b = 0.1
    c = 2
    noise_level = 0.5
    
    length = 10
    random_state = 42  

    series, trend, noise = quadratic_trend(a, b, c, noise_level, length, random_state)
    vis.multi_plot_each(series_list=[series, trend, noise], labels=["Временной ряд", 'Тренд', 'Шум'], plot_title="Квадратичный тренд с шумом", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 3

    alpha = 0.5  # Коэффициент экспоненты
    noise_level = 4  # Уровень шума
    length = 10
    random_state = 42  

    series, trend, noise = exponential_trend(alpha, noise_level, length, random_state)
    vis.multi_plot_each(series_list=[series, trend, noise], labels=["Временной ряд", 'Тренд', 'Шум'], plot_title="Экспоненциальный тренд с шумом", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 4

    amplitude = 5.0  # Амплитуда сезонности
    frequency = 2.0  # Частота сезонности (количество циклов за период)
    # phase = 0.0  # Фаза сезонности (сдвиг по горизонтали)
    phase = np.pi / 2  # Добавляем pi/2 для косинуса
    noise_level = 4  # Уровень шума

    length = 100  # Количество точек
    random_state = 42  # Seed для воспроизводимости

    series, seasonality, noise = seasonal_series(amplitude, frequency, phase, noise_level, length, random_state)

    vis.multi_plot_each(series_list=[series, seasonality, noise], labels=["Временной ряд", 'Сезонность', 'Шум'], plot_title="Сезонность (sin/cos) с шумом", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 5

    amplitude = 10.0  # Амплитуда осциллятора
    frequency = 2.0  # Частота осциллятора (количество колебаний за период)
    damping = 0.05  # Коэффициент затухания
    noise_level = 0.2  # Уровень шума
    length = 200  # Количество точек
    random_state = 42  # Seed для воспроизводимости

    series, oscillator, noise = harmonic_oscillator(amplitude, frequency, damping, noise_level, length, random_state)

    vis.multi_plot_each(series_list=[series, oscillator, noise], labels=["Временной ряд", 'Гармонический осциллятор', 'Шум'], plot_title="Гармонический осциллятор", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 6

    amplitude = 5.0  # Амплитуда пилообразного сигнала
    frequency = 4.0  # Частота сигнала (количество пилообразных циклов за период)
    noise_level = 0.1  # Уровень шума
    length = 200  # Количество точек
    random_state = 42  # Seed для воспроизводимости

    series, sawtooth, noise = sawtooth_wave(amplitude, frequency, noise_level, length, random_state)

    vis.multi_plot_each(series_list=[series, sawtooth, noise], labels=["Временной ряд", 'Пилообразный сигнал', 'Шум'], plot_title="Пилообразный сигнал", xlabel="Время", ylabel="Значение", figsize=(14,3))

    # ex 7

    # Параметры
    initial_value = 10.0  # Начальное значение
    noise_level = 0.5  # Уровень шума
    length = 100  # Количество точек
    random_state = 42  # Seed для воспроизводимости

    # Генерация временного ряда
    series, noise = random_walk(initial_value, noise_level, length, random_state)

    vis.multi_plot_each(series_list=[series, noise], labels=["Временной ряд", 'Шум'], plot_title="Случайное блуждание", xlabel="Время", ylabel="Значение", figsize=(14,3))

if __name__ == '__main__':
    main()