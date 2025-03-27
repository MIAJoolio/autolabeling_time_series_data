from typing import Tuple, List, Dict, Optional, Union
import numpy as np

from scipy.signal import (
    detrend, savgol_filter, medfilt, butter, filtfilt, iirnotch, welch, spectrogram, find_peaks, peak_prominences, correlate, coherence, cwt, hilbert, ricker, argrelextrema, lfilter, wiener
)
from scipy.stats import skew, kurtosis

__all__ = [
    'seasonality_removal',
    'seasonality_smoothing',
    'seasonality_filtering',
    'seasonality_spectral',
    'seasonality_peaks',
    'seasonality_correlation'
]

def seasonality_removal(time_series: np.ndarray, get_features: bool = False, window_size: int = 5, poly_order: int = 2) -> Dict[str, np.ndarray]:
    """
    Удаление сезонных компонентов из временного ряда. При get_features=True возвращает словарь с признаками сезонности.
    
    Эта функция позволяет удалять сезонные компоненты из временных рядов, используя различные методы, такие как 
    вычитание тренда и фильтрация. Удаление сезонности помогает выявить более мелкие колебания и паттерны в данных.

    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков сезонности.
        window_size: Размер окна для savgol_filter.
        poly_order: Порядок полинома для savgol_filter.

    Returns:
        Словарь с результатами различных методов удаления сезонности или признаками сезонности.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if window_size <= poly_order:
        raise ValueError("window_size должен быть больше poly_order.")
    
    # Удаление тренда
    detrended = detrend(time_series)
    
    # Удаление сезонности с помощью фильтра Савицкого-Голея
    savgol_deseasoned = time_series - savgol_filter(time_series, window_size, poly_order)
    
    # Удаление сезонности с помощью медианной фильтрации
    medfilt_deseasoned = time_series - medfilt(time_series, kernel_size=window_size)
    
    if get_features:
        return {
            'seasonality_strength': np.std(time_series) / np.std(detrended),  # Сила сезонности
            'seasonality_consistency': np.corrcoef(np.arange(len(time_series)), time_series)[0, 1],  # Согласованность сезонности
            'seasonality_smoothness': np.std(savgol_deseasoned) / np.std(time_series),  # Гладкость сезонности
            'seasonality_direction': np.sign(np.mean(np.diff(time_series))),  # Направление сезонности
            'seasonality_magnitude': np.abs(np.mean(np.diff(time_series))),  # Величина сезонности
            'seasonality_resid_detrend': np.abs(np.mean(np.diff(time_series - detrended))),  # Остаток сезонности после detrend
            'seasonality_resid_savgol': np.abs(np.mean(np.diff(time_series - savgol_deseasoned))),  # Остаток сезонности после savgol_filter
            'seasonality_resid_medfilt': np.abs(np.mean(np.diff(time_series - medfilt_deseasoned))),  # Остаток сезонности после medfilt
        }
    
    return {
        'original': time_series,
        'detrended': detrended,
        'savgol_deseasoned': savgol_deseasoned,
        'medfilt_deseasoned': medfilt_deseasoned
    }

def seasonality_smoothing(time_series: np.ndarray, get_features: bool = False, window_size: int = 5, poly_order: int = 2) -> Dict[str, np.ndarray]:
    """
    Сглаживание сезонных компонентов. При get_features=True возвращает словарь с признаками сглаживания сезонности.
    
    Эта функция позволяет сглаживать сезонные компоненты временного ряда, чтобы уменьшить шум и выделить основные сезонные паттерны.
    
    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков сглаживания сезонности.
        window_size: Размер окна для фильтров.
        poly_order: Порядок полинома для savgol_filter.

    Returns:
        Словарь с результатами различных методов сглаживания сезонности или признаками сглаживания.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if window_size <= poly_order:
        raise ValueError("window_size должен быть больше poly_order.")
    if window_size % 2 == 0:
        raise ValueError("window_size должен быть нечетным.")
    
    # Сглаживание с помощью фильтра Савицкого-Голея
    savgol_smoothed = savgol_filter(time_series, window_size, poly_order)
    
    # Сглаживание с помощью медианной фильтрации
    medfilt_smoothed = medfilt(time_series, kernel_size=window_size)
    
    # Адаптивная фильтрация
    b = np.ones(window_size) / window_size
    a = 1
    fir_filtered = lfilter(b, a, time_series)
    wiener_filtered = wiener(time_series, mysize=window_size)
    
    if get_features:
        return {
            'seasonality_smoothing_savgol_ratio': np.std(savgol_smoothed) / np.std(time_series),
            'seasonality_smoothing_median_ratio': np.std(medfilt_smoothed) / np.std(time_series),
            'seasonality_smoothing_difference': np.std(savgol_smoothed - medfilt_smoothed),
            'seasonality_smoothing_consistency': np.corrcoef(savgol_smoothed, medfilt_smoothed)[0, 1],
            'seasonality_fir_filter_ratio': np.std(fir_filtered) / np.std(time_series),
            'seasonality_wiener_filter_ratio': np.std(wiener_filtered) / np.std(time_series),
            'seasonality_fir_wiener_difference': np.std(fir_filtered - wiener_filtered),
            'seasonality_fir_wiener_consistency': np.corrcoef(fir_filtered, wiener_filtered)[0, 1],
        }
    
    return {
        'original': time_series,
        'savgol_smoothed': savgol_smoothed,
        'medfilt_smoothed': medfilt_smoothed,
        'fir_filtered': fir_filtered,
        'wiener_filtered': wiener_filtered
    }

def seasonality_filtering(time_series: np.ndarray, get_features: bool = False, fs: float = 100.0, lowcut: float = 5.0, highcut: float = 20.0, notch_freq: float = 49.0) -> Dict[str, np.ndarray]:
    """
    Фильтрация сезонных компонентов. При get_features=True возвращает словарь с признаками фильтрации сезонности.
    
    Эта функция позволяет фильтровать сезонные компоненты временного ряда для выделения определенных частотных диапазонов.
    
    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков фильтрации сезонности.
        fs: Частота дискретизации (по умолчанию 100.0).
        lowcut: Нижняя частота среза для полосового фильтра (по умолчанию 5.0).
        highcut: Верхняя частота среза для полосового фильтра (по умолчанию 20.0).
        notch_freq: Частота для режекторного фильтра (по умолчанию 49.0).

    Returns:
        Словарь с результатами различных методов фильтрации сезонности или признаками фильтрации.
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
    
    # Полосовая фильтрация
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    bandpass_filtered = filtfilt(b, a, time_series)
    
    # Режекторная фильтрация
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
            'seasonality_filtering_bandpass_ratio': np.std(bandpass_filtered) / np.std(time_series),
            'seasonality_filtering_notch_ratio': np.std(notch_filtered) / np.std(time_series),
            'seasonality_filtering_bandwidth': highcut - lowcut,
            'seasonality_filtering_center_freq': (highcut + lowcut) / 2,
            'seasonality_filtering_bandpass_std': np.std(bandpass_filtered),
            'seasonality_filtering_notch_std': np.std(notch_filtered),
            'seasonality_filtering_bandpass_mean': np.mean(bandpass_filtered),
            'seasonality_filtering_notch_mean': np.mean(notch_filtered),
            'seasonality_envelope_mean': np.mean(envelope),
            'seasonality_envelope_std': np.std(envelope),
            'seasonality_envelope_max': np.max(envelope),
            'seasonality_envelope_min': np.min(envelope),
            'seasonality_envelope_ratio': np.max(envelope) / np.min(envelope),
            'seasonality_phase_mean': np.mean(instantaneous_phase),
            'seasonality_phase_std': np.std(instantaneous_phase),
        }
    
    return {
        'original': time_series,
        'bandpass_filtered': bandpass_filtered,
        'notch_filtered': notch_filtered,
        'envelope': envelope,
        'instantaneous_phase': instantaneous_phase
    }

def seasonality_spectral(time_series: np.ndarray, get_features: bool = False, fs: float = 100.0, nperseg: int = 256, widths: np.ndarray = np.arange(1, 20), wavelet: callable = ricker) -> Dict[str, np.ndarray]:
    """
    Спектральный анализ сезонности. При get_features=True возвращает словарь со спектральными признаками сезонности.
    
    Эта функция позволяет анализировать частотные компоненты сезонности во временном ряде.
    
    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения спектральных признаков сезонности.
        fs: Частота дискретизации (по умолчанию 100.0).
        nperseg: Количество точек на сегмент для метода Welch (по умолчанию 256).
        widths: Ширины для вейвлет-преобразования (по умолчанию от 1 до 20).
        wavelet: Функция вейвлета для анализа (по умолчанию ricker).

    Returns:
        Словарь с результатами спектрального анализа сезонности или спектральными признаками.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if fs <= 0:
        raise ValueError("fs должна быть положительным числом.")
    if nperseg <= 0 or nperseg > len(time_series):
        raise ValueError("nperseg должен быть положительным числом и не превышать длину временного ряда.")
    
    # Спектральный анализ
    f, Pxx = welch(time_series, fs=fs, nperseg=nperseg)
    f, t, Sxx = spectrogram(time_series, fs=fs, nperseg=nperseg)
    
    # Вейвлет-анализ
    cwt_matrix = cwt(time_series, wavelet, widths)
    cwt_energy = np.sum(cwt_matrix**2, axis=1)
    
    if get_features:
        spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
        return {
            'seasonality_spectral_centroid': spectral_centroid,
            'seasonality_spectral_bandwidth': np.sqrt(np.sum(((f - spectral_centroid) ** 2) * Pxx) / np.sum(Pxx)),
            'seasonality_spectral_flatness': np.exp(np.mean(np.log(Pxx + 1e-10))) / np.mean(Pxx),
            'seasonality_spectral_rolloff': np.percentile(Pxx, 85),
            'seasonality_spectral_skewness': skew(Pxx),
            'seasonality_spectral_kurtosis': kurtosis(Pxx),
            'seasonality_spectral_energy': np.sum(Pxx),
            'seasonality_spectral_entropy': -np.sum(Pxx * np.log2(Pxx + 1e-10)),
            'seasonality_spectral_peak_freq': f[np.argmax(Pxx)],
            'seasonality_spectral_peak_magnitude': np.max(Pxx),
            'seasonality_wavelet_energy_mean': np.mean(cwt_energy),
            'seasonality_wavelet_energy_std': np.std(cwt_energy),
            'seasonality_wavelet_energy_max': np.max(cwt_energy),
            'seasonality_wavelet_energy_min': np.min(cwt_energy),
            'seasonality_wavelet_energy_ratio': np.max(cwt_energy) / np.min(cwt_energy),
        }
    
    return {
        'freq': f,
        'psd': Pxx,
        'time': t,
        'spectrogram': Sxx,
        'cwt_matrix': cwt_matrix,
        'cwt_energy': cwt_energy
    }

def seasonality_peaks(time_series: np.ndarray, get_features: bool = False, distance: int = 10, prominence: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Анализ пиков сезонности. При get_features=True возвращает словарь с признаками пиков сезонности.
    
    Эта функция позволяет выявлять пики в сезонных компонентах временного ряда.
    
    Args:
        time_series: Входной временной ряд (одномерный массив numpy).
        get_features: Флаг для получения признаков пиков сезонности.
        distance: Минимальное расстояние между пиками.
        prominence: Минимальная значимость пика.

    Returns:
        Словарь с результатами анализа пиков сезонности или признаками пиков.
    """
    # Проверка входных данных
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy.")
    if distance <= 0:
        raise ValueError("distance должен быть положительным числом.")
    if prominence <= 0:
        raise ValueError("prominence должен быть положительным числом.")
    
    # Поиск пиков
    peaks, properties = find_peaks(time_series, distance=distance, prominence=prominence)
    prominences = peak_prominences(time_series, peaks)[0]
    
    # Поиск локальных экстремумов
    maxima = argrelextrema(time_series, np.greater, order=distance)[0]
    minima = argrelextrema(time_series, np.less, order=distance)[0]
    
    if get_features:
        return {
            'seasonality_peak_count': len(peaks),
            'seasonality_peak_mean_prominence': np.mean(prominences),
            'seasonality_peak_std_prominence': np.std(prominences),
            'seasonality_peak_mean_distance': np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,
            'seasonality_peak_std_distance': np.std(np.diff(peaks)) if len(peaks) > 1 else 0,
            'seasonality_peak_max_prominence': np.max(prominences) if len(prominences) > 0 else 0,
            'seasonality_peak_min_prominence': np.min(prominences) if len(prominences) > 0 else 0,
            'seasonality_peak_mean_height': np.mean(time_series[peaks]) if len(peaks) > 0 else 0,
            'seasonality_peak_std_height': np.std(time_series[peaks]) if len(peaks) > 0 else 0,
            'seasonality_extrema_maxima_count': len(maxima),
            'seasonality_extrema_minima_count': len(minima),
            'seasonality_extrema_maxima_mean': np.mean(time_series[maxima]) if len(maxima) > 0 else 0,
            'seasonality_extrema_minima_mean': np.mean(time_series[minima]) if len(minima) > 0 else 0,
            'seasonality_extrema_maxima_std': np.std(time_series[maxima]) if len(maxima) > 0 else 0,
            'seasonality_extrema_minima_std': np.std(time_series[minima]) if len(minima) > 0 else 0,
        }
    
    return {
        'original': time_series,
        'peaks': peaks,
        'prominences': prominences,
        'maxima': maxima,
        'minima': minima
    }

def seasonality_correlation(time_series1: np.ndarray, time_series2: np.ndarray, get_features: bool = False, fs: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Корреляционный анализ сезонности. При get_features=True возвращает словарь с корреляционными признаками сезонности.
    
    Эта функция позволяет анализировать взаимосвязь между сезонными компонентами двух временных рядов.
    
    Args:
        time_series1: Первый временной ряд (одномерный массив numpy).
        time_series2: Второй временной ряд (одномерный массив numpy).
        get_features: Флаг для получения корреляционных признаков сезонности.
        fs: Частота дискретизации (по умолчанию 100.0).

    Returns:
        Словарь с результатами корреляционного анализа сезонности или корреляционными признаками.
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
    
    # Корреляционный анализ
    correlation = correlate(time_series1, time_series2, mode='full')
    
    # Анализ когерентности
    f, Cxy = coherence(time_series1, time_series2, fs=fs)
    
    if get_features:
        return {
            'seasonality_correlation_max': np.max(correlation),
            'seasonality_correlation_min': np.min(correlation),
            'seasonality_correlation_mean': np.mean(correlation),
            'seasonality_correlation_std': np.std(correlation),
            'seasonality_coherence_mean': np.mean(Cxy),
            'seasonality_coherence_std': np.std(Cxy),
            'seasonality_correlation_peak_lag': np.argmax(correlation) - len(correlation) // 2,
            'seasonality_correlation_peak_value': np.max(correlation),
            'seasonality_coherence_peak_freq': f[np.argmax(Cxy)],
            'seasonality_coherence_peak_value': np.max(Cxy),
        }
    
    return {
        'correlation': correlation,
        'freq': f,
        'coherence': Cxy
    }

if __name__ == '__main__':
    None 