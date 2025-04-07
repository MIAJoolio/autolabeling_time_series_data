from typing import Tuple, List, Dict, Optional, Union
import numpy as np

from scipy.signal import (
    detrend, savgol_filter, medfilt, butter, filtfilt, iirnotch, welch, spectrogram, find_peaks, peak_prominences, correlate, coherence, cwt, hilbert, ricker, argrelextrema, lfilter, wiener
)
from scipy.stats import skew, kurtosis

__all__ = [
    'trend_removal',
    'smoothing_trend',
    'filter_trend',
    'spectral_features_trend',
    'peaks_analys_trend',
    'correlation_trend'
]

def trend_removal(time_series: np.ndarray, get_features: bool = False, window_size: int = 5, poly_order: int = 2) -> Dict[str, np.ndarray]:
    """
    Удаление тренда методами scipy.signal. В случае get_features=True, возвращает словарь с признаками тренда.
    
    Эта функция позволяет удалять тренды из временных рядов, используя методы, такие как линейное вычитание и фильтрацию Савицкого-Голея. 
    Удаление тренда помогает выявить более мелкие колебания и паттерны в данных.

    Используйте эту функцию, если вам нужно:
    - Удалить линейные или полиномиальные тренды из временного ряда.
    - Оценить характеристики тренда, такие как сила, линейность и гладкость.

    Args:       
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков тренда.
        window_size: Размер окна для savgol_filter.
        poly_order: Порядок полинома для savgol_filter.

    Returns:
        Словарь с результатами различных методов удаления тренда или признаками тренда.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if window_size <= poly_order:
        raise ValueError("window_size должен быть больше poly_order.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend
    detrended = detrend(time_series)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter
    savgol_detrended = time_series - savgol_filter(time_series, window_size, poly_order)
    
    if get_features:
        # Вычисление признаков
        return {
            'trend_strength': np.std(time_series) / np.std(detrended),  # Сила тренда
            'trend_linearity': np.corrcoef(np.arange(len(time_series)), time_series)[0, 1],  # Линейность тренда
            'trend_smoothness': np.std(savgol_detrended) / np.std(time_series),  # Гладкость тренда
            'trend_direction': np.sign(np.mean(np.diff(time_series))),  # Направление тренда
            'trend_magnitude': np.abs(np.mean(np.diff(time_series))),  # Величина тренда
            'trend_reisd_detrend': np.abs(np.mean(np.diff(time_series - detrended))),  # Остаток тренда после detrend
            'trend_resid_savgol': np.abs(np.mean(np.diff(time_series - savgol_detrended))),  # Остаток тренда после savgol_filter
            'trend_resid_diff': np.abs(np.mean(np.diff(time_series - savgol_detrended - detrended))),  # Остаток тренда после detrend и savgol_filter
        }
        
    return {
        'original': time_series,
        'detrended': detrended,
        'savgol_detrended': savgol_detrended
    }

def smoothing_trend(time_series: np.ndarray, get_features: bool = False, window_size: int = 5, poly_order: int = 2) -> Dict[str, np.ndarray]:
    """
    Сглаживание тренда методами scipy.signal. При get_features=True возвращает словарь с признаками сглаживания.
    
    Эта функция позволяет сглаживать временные ряды, чтобы уменьшить шум и выделить основные тренды. 
    Используются методы, такие как фильтрация Савицкого-Голея, медианная фильтрация и адаптивная фильтрация.

    Используйте эту функцию, если вам нужно:
    - Уменьшить шум в данных.
    - Выделить основные тренды и паттерны в временном ряде.
    - Применить адаптивные методы фильтрации.

    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков сглаживания.
        window_size: Размер окна для фильтров.
        poly_order: Порядок полинома для savgol_filter.

    Returns:
        Словарь с результатами различных методов сглаживания или признаками сглаживания.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if window_size <= poly_order:
        raise ValueError("window_size должен быть больше poly_order.")
    if window_size % 2 == 0:
        raise ValueError("window_size должен быть нечетным.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter
    savgol_smoothed = savgol_filter(time_series, window_size, poly_order)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html#scipy.signal.medfilt
    medfilt_smoothed = medfilt(time_series, kernel_size=window_size)
    
    # Адаптивная фильтрация
    b = np.ones(window_size) / window_size
    a = 1
    fir_filtered = lfilter(b, a, time_series)
    wiener_filtered = wiener(time_series, mysize=window_size)
    
    if get_features:
        return {
            'smoothing_savgol_ratio': np.std(savgol_smoothed) / np.std(time_series),  # Эффективность сглаживания Савицкого-Голея
            'smoothing_median_ratio': np.std(medfilt_smoothed) / np.std(time_series),  # Эффективность медианного сглаживания
            'smoothing_difference': np.std(savgol_smoothed - medfilt_smoothed),  # Разница между методами сглаживания
            'smoothing_consistency': np.corrcoef(savgol_smoothed, medfilt_smoothed)[0, 1],  # Согласованность методов
            # Признаки адаптивной фильтрации
            'fir_filter_ratio': np.std(fir_filtered) / np.std(time_series),  # Эффективность FIR-фильтра
            'wiener_filter_ratio': np.std(wiener_filtered) / np.std(time_series),  # Эффективность фильтра Винера
            'fir_wiener_difference': np.std(fir_filtered - wiener_filtered),  # Разница между методами адаптивной фильтрации
            'fir_wiener_consistency': np.corrcoef(fir_filtered, wiener_filtered)[0, 1],  # Согласованность методов адаптивной фильтрации
        }
    
    return {
        'original': time_series,
        'savgol_smoothed': savgol_smoothed,
        'medfilt_smoothed': medfilt_smoothed,
        'fir_filtered': fir_filtered,
        'wiener_filtered': wiener_filtered
    }

def filter_trend(time_series: np.ndarray, get_features: bool = False, fs: float = 100.0, lowcut: float = 5.0, highcut: float = 20.0, notch_freq: float = 49.0) -> Dict[str, np.ndarray]:
    """
    Анализ фильтрации сигнала различными методами. При get_features=True возвращает словарь с признаками фильтрации.
    
    Эта функция позволяет применять фильтрацию к временным рядам для удаления нежелательных частот и выделения интересующих частот. 
    Полосовой фильтр Баттерворта используется для удаления частот вне заданного диапазона, а режекторный фильтр 
    позволяет удалить определенные частоты (например, шум). Также включает анализ огибающей сигнала для изучения амплитудной модуляции.

    Используйте эту функцию, если вам нужно:
    - Удалить высокочастотный или низкочастотный шум из временного ряда.
    - Выделить определенные частоты для дальнейшего анализа.
    - Оценить влияние фильтрации на временной ряд.
    - Проанализировать амплитудную модуляцию сигнала.

    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков фильтрации.
        fs: Частота дискретизации (по умолчанию 100.0).
        lowcut: Нижняя частота среза для полосового фильтра (по умолчанию 5.0).
        highcut: Верхняя частота среза для полосового фильтра (по умолчанию 20.0).
        notch_freq: Частота для режекторного фильтра (по умолчанию 49.0).

    Returns:
        Словарь с результатами различных методов фильтрации или признаками фильтрации.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if fs <= 0:
        raise ValueError("fs должна быть положительным числом.")
    if lowcut <= 0 or highcut <= 0 or lowcut >= highcut:
        raise ValueError("lowcut и highcut должны быть положительными числами, причем lowcut < highcut.")
    if notch_freq <= 0:
        raise ValueError("notch_freq должна быть положительным числом.")
    if notch_freq >= fs / 2:
        raise ValueError("notch_freq должна быть меньше половины частоты дискретизации.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    bandpass_filtered = filtfilt(b, a, time_series)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html#scipy.signal.iirnotch
    Q = 30.0
    w0 = notch_freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    notch_filtered = filtfilt(b, a, time_series)
    
    # Анализ огибающей
    analytic_signal = hilbert(time_series)
    envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    if get_features:
        return {
            'filtering_bandpass_ratio': np.std(bandpass_filtered) / np.std(time_series),  # Эффективность полосовой фильтрации
            'filtering_notch_ratio': np.std(notch_filtered) / np.std(time_series),  # Эффективность режекторной фильтрации
            'filtering_bandwidth': highcut - lowcut,  # Ширина полосы пропускания
            'filtering_center_freq': (highcut + lowcut) / 2,  # Центральная частота
            'filtering_bandpass_std': np.std(bandpass_filtered),  # Стандартное отклонение после полосовой фильтрации
            'filtering_notch_std': np.std(notch_filtered),  # Стандартное отклонение после режекторной фильтрации
            'filtering_bandpass_mean': np.mean(bandpass_filtered),  # Среднее значение после полосовой фильтрации
            'filtering_notch_mean': np.mean(notch_filtered),  # Среднее значение после режекторной фильтрации
            # Признаки огибающей
            'envelope_mean': np.mean(envelope),  # Среднее значение огибающей
            'envelope_std': np.std(envelope),  # Стандартное отклонение огибающей
            'envelope_max': np.max(envelope),  # Максимальное значение огибающей
            'envelope_min': np.min(envelope),  # Минимальное значение огибающей
            'envelope_ratio': np.max(envelope) / np.min(envelope),  # Отношение максимального к минимальному значению
            'phase_mean': np.mean(instantaneous_phase),  # Среднее значение фазы
            'phase_std': np.std(instantaneous_phase),  # Стандартное отклонение фазы
        }
    
    return {
        'original': time_series,
        'bandpass_filtered': bandpass_filtered,
        'notch_filtered': notch_filtered,
        'envelope': envelope,
        'instantaneous_phase': instantaneous_phase
    }

def spectral_features_trend(time_series: np.ndarray, get_features: bool = False, fs: float = 100.0, nperseg: int = 256, widths: np.ndarray = np.arange(1, 20), wavelet: callable = ricker) -> Dict[str, np.ndarray]:
    """
    Спектральный анализ сигнала. При get_features=True возвращает словарь со спектральными признаками.
    
    Эта функция позволяет анализировать частотные компоненты временного ряда, что помогает понять, как энергия распределена по различным частотам. 
    Спектральный анализ используется для выявления периодических паттернов, шумов и других характеристик сигнала, которые могут быть не видны в 
    временной области. Включает в себя как классический спектральный анализ (Фурье), так и вейвлет-анализ для выявления локальных особенностей.

    Используйте эту функцию, если вам нужно:
    - Определить основные частоты, присутствующие в сигнале.
    - Оценить спектральные характеристики, такие как ширина спектра, асимметрия и эксцесс.
    - Выявить периодические компоненты и шумы в данных.
    - Проанализировать локальные особенности сигнала с помощью вейвлет-преобразования.

    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения спектральных признаков.
        fs: Частота дискретизации (по умолчанию 100.0).
        nperseg: Количество точек на сегмент для метода Welch (по умолчанию 256).
        widths: Ширины для вейвлет-преобразования (по умолчанию от 1 до 20).
        wavelet: Функция вейвлета для анализа (по умолчанию ricker).

    Returns:
        Словарь с результатами спектрального анализа или спектральными признаками.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if fs <= 0:
        raise ValueError("fs должна быть положительным числом.")
    if nperseg <= 0 or nperseg > len(time_series):
        raise ValueError("nperseg должен быть положительным числом и не превышать длину временного ряда.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch
    f, Pxx = welch(time_series, fs=fs, nperseg=nperseg)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html#scipy.signal.spectrogram
    f, t, Sxx = spectrogram(time_series, fs=fs, nperseg=nperseg)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html#scipy.signal.cwt
    cwt_matrix = cwt(time_series, wavelet, widths)
    cwt_energy = np.sum(cwt_matrix**2, axis=1)
    
    if get_features:
        spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
        return {
            'spectral_centroid': spectral_centroid,  # Спектральный центроид
            'spectral_bandwidth': np.sqrt(np.sum(((f - spectral_centroid) ** 2) * Pxx) / np.sum(Pxx)),  # Спектральная ширина
            'spectral_flatness': np.exp(np.mean(np.log(Pxx + 1e-10))) / np.mean(Pxx),  # Спектральная плоскость
            'spectral_rolloff': np.percentile(Pxx, 85),  # Спектральный роллоф
            'spectral_skewness': skew(Pxx),  # Асимметрия спектра
            'spectral_kurtosis': kurtosis(Pxx),  # Эксцесс спектра
            'spectral_energy': np.sum(Pxx),  # Спектральная энергия
            'spectral_entropy': -np.sum(Pxx * np.log2(Pxx + 1e-10)),  # Спектральная энтропия
            'spectral_peak_freq': f[np.argmax(Pxx)],  # Частота основного пика
            'spectral_peak_magnitude': np.max(Pxx),  # Величина основного пика
            # Вейвлет-признаки
            'wavelet_energy_mean': np.mean(cwt_energy),  # Средняя энергия вейвлет-коэффициентов
            'wavelet_energy_std': np.std(cwt_energy),  # Стандартное отклонение энергии
            'wavelet_energy_max': np.max(cwt_energy),  # Максимальная энергия
            'wavelet_energy_min': np.min(cwt_energy),  # Минимальная энергия
            'wavelet_energy_ratio': np.max(cwt_energy) / np.min(cwt_energy),  # Отношение максимальной к минимальной энергии
        }
    
    return {
        'freq': f,
        'psd': Pxx,
        'time': t,
        'spectrogram': Sxx,
        'cwt_matrix': cwt_matrix,
        'cwt_energy': cwt_energy
    }

def peaks_analys_trend(time_series: np.ndarray, get_features: bool = False, distance: int = 10, prominence: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Анализ пиков в сигнале. При get_features=True возвращает словарь с признаками пиков.
    
    Эта функция позволяет выявлять пики в временных рядах, что может быть полезно для анализа событий, 
    таких как максимумы и минимумы в данных. Параметры, такие как расстояние и значимость, помогают 
    контролировать, какие пики будут считаться значительными. Включает как поиск пиков по значимости,
    так и поиск локальных экстремумов.

    Используйте эту функцию, если вам нужно:
    - Выявить важные пики в данных.
    - Оценить характеристики пиков, такие как количество, высота и расстояние между ними.
    - Найти локальные максимумы и минимумы в сигнале.

    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков пиков.
        distance: Минимальное расстояние между пиками.
        prominence: Минимальная значимость пика.

    Returns:
        Словарь с результатами анализа пиков или признаками пиков.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if distance <= 0:
        raise ValueError("distance должен быть положительным числом.")
    if prominence <= 0:
        raise ValueError("prominence должен быть положительным числом.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    peaks, properties = find_peaks(time_series, distance=distance, prominence=prominence)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences
    prominences = peak_prominences(time_series, peaks)[0]
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html#scipy.signal.argrelextrema
    maxima = argrelextrema(time_series, np.greater, order=distance)[0]
    minima = argrelextrema(time_series, np.less, order=distance)[0]
    
    if get_features:
        return {
            'peak_count': len(peaks),  # Количество пиков
            'peak_mean_prominence': np.mean(prominences),  # Средняя значимость пиков
            'peak_std_prominence': np.std(prominences),  # Стандартное отклонение значимости
            'peak_mean_distance': np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,  # Среднее расстояние между пиками
            'peak_std_distance': np.std(np.diff(peaks)) if len(peaks) > 1 else 0,  # Стандартное отклонение расстояния
            'peak_max_prominence': np.max(prominences) if len(prominences) > 0 else 0,  # Максимальная значимость
            'peak_min_prominence': np.min(prominences) if len(prominences) > 0 else 0,  # Минимальная значимость
            'peak_mean_height': np.mean(time_series[peaks]) if len(peaks) > 0 else 0,  # Средняя высота пиков
            'peak_std_height': np.std(time_series[peaks]) if len(peaks) > 0 else 0,  # Стандартное отклонение высоты
            # Признаки локальных экстремумов
            'extrema_maxima_count': len(maxima),  # Количество локальных максимумов
            'extrema_minima_count': len(minima),  # Количество локальных минимумов
            'extrema_maxima_mean': np.mean(time_series[maxima]) if len(maxima) > 0 else 0,  # Среднее значение максимумов
            'extrema_minima_mean': np.mean(time_series[minima]) if len(minima) > 0 else 0,  # Среднее значение минимумов
            'extrema_maxima_std': np.std(time_series[maxima]) if len(maxima) > 0 else 0,  # Стандартное отклонение максимумов
            'extrema_minima_std': np.std(time_series[minima]) if len(minima) > 0 else 0,  # Стандартное отклонение минимумов
        }
    
    return {
        'original': time_series,
        'peaks': peaks,
        'prominences': prominences,
        'maxima': maxima,
        'minima': minima
    }

def correlation_trend(time_series1: np.ndarray, time_series2: np.ndarray, get_features: bool = False, fs: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Корреляционный анализ двух сигналов. При get_features=True возвращает словарь с корреляционными признаками.
    
    Эта функция позволяет анализировать взаимосвязь между двумя временными рядами, что может быть полезно для 
    выявления коррелирующих паттернов или сигналов. Корреляция и когерентность помогают оценить, насколько 
    два сигнала связаны друг с другом.

    Используйте эту функцию, если вам нужно:
    - Оценить степень корреляции между двумя временными рядами.
    - Выявить задержки и пиковые значения корреляции.

    Args:
        time_series1: Первый временной ряд (одномерный массив numpy).
        time_series2: Второй временной ряд (одномерный массив numpy).
        get_features: Флаг для получения корреляционных признаков.
        fs: Частота дискретизации (по умолчанию 100.0).

    Returns:
        Словарь с результатами корреляционного анализа или корреляционными признаками.
    """
    # Проверка входных данных
    if not isinstance(time_series1, np.ndarray) or not isinstance(time_series2, np.ndarray):
        raise ValueError("Входные данные должны быть массивами numpy.")
    if time_series1.ndim != 1 or time_series2.ndim != 1:
        raise ValueError("Входные данные должны быть одномерными массивами.")
    if len(time_series1) != len(time_series2):
        raise ValueError("Входные данные должны иметь одинаковую длину.")
    if fs <= 0:
        raise ValueError("fs должна быть положительным числом.")
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate
    correlation = correlate(time_series1, time_series2, mode='full')
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html#scipy.signal.coherence
    f, Cxy = coherence(time_series1, time_series2, fs=fs)
    
    if get_features:
        return {
            'correlation_max': np.max(correlation),  # Максимальная корреляция
            'correlation_min': np.min(correlation),  # Минимальная корреляция
            'correlation_mean': np.mean(correlation),  # Средняя корреляция
            'correlation_std': np.std(correlation),  # Стандартное отклонение корреляции
            'coherence_mean': np.mean(Cxy),  # Средняя когерентность
            'coherence_std': np.std(Cxy),  # Стандартное отклонение когерентности
            'correlation_peak_lag': np.argmax(correlation) - len(correlation) // 2,  # Задержка пика корреляции
            'correlation_peak_value': np.max(correlation),  # Значение пика корреляции
            'coherence_peak_freq': f[np.argmax(Cxy)],  # Частота пика когерентности
            'coherence_peak_value': np.max(Cxy),  # Значение пика когерентности
        }
    
    return {
        'correlation': correlation,
        'freq': f,
        'coherence': Cxy
    }

if __name__ == '__main__':
    None 