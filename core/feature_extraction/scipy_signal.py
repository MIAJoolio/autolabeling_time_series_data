from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from scipy.signal import (
    # Фильтрация и свертка
    convolve, correlate, correlation_lags, savgol_filter, detrend,
    # Фильтры
    butter, filtfilt, sosfilt,
    # Окна
    get_window,
    # Поиск пиков
    find_peaks, peak_prominences, argrelextrema,
    # Спектральный анализ
    welch, periodogram, spectrogram, csd, coherence,
    # Гильбертово преобразование
    hilbert
)
from scipy.stats import skew, kurtosis

__all__ = [
    'scipy_trend',
    'scipy_seasonality',
    'scipy_structural_changes',
    'scipy_noise',
    'scipy_cross_correlation'
]

def scipy_trend(time_series: np.ndarray, get_features: bool = False, window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Анализ тренда во временном ряде с использованием различных методов.
    
    Методы:
    1. Линейный тренд через корреляцию с временной осью
    2. Сглаживание с помощью полиномиального фильтра
    3. Спектральный анализ низкочастотных компонент
    4. Анализ локальных экстремумов
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна для сглаживания
        
    Returns:
        Словарь с результатами анализа или признаками тренда
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Линейный тренд
    time_axis = np.arange(len(time_series))
    trend_correlation = np.corrcoef(time_axis, time_series)[0, 1]
    
    # 2. Сглаживание с помощью savgol_filter
    smoothed = savgol_filter(time_series, window_length=window_size, polyorder=2)
    
    # 3. Спектральный анализ низкочастотных компонент
    f, Pxx = welch(time_series, nperseg=min(256, len(time_series)))
    low_freq_mask = f < 0.1  # Низкочастотные компоненты
    low_freq_energy = np.sum(Pxx[low_freq_mask])
    total_energy = np.sum(Pxx)
    
    # 4. Анализ локальных экстремумов
    maxima = argrelextrema(time_series, np.greater, order=window_size)[0]
    minima = argrelextrema(time_series, np.less, order=window_size)[0]
    
    if get_features:
        return {
            'trend_strength': abs(trend_correlation),  # Сила линейного тренда
            'trend_direction': np.sign(trend_correlation),  # Направление тренда
            'trend_smoothness': np.std(smoothed) / np.std(time_series),  # Гладкость тренда
            'trend_low_freq_ratio': low_freq_energy / total_energy,  # Доля низкочастотных компонент
            'trend_extrema_count': len(maxima) + len(minima),  # Количество экстремумов
            'trend_extrema_ratio': (len(maxima) - len(minima)) / (len(maxima) + len(minima) + 1e-10),  # Баланс экстремумов
            'trend_linearity': np.corrcoef(time_axis, smoothed)[0, 1],  # Линейность сглаженного тренда
            'trend_persistence': np.mean(np.diff(smoothed) > 0),  # Персистентность тренда
        }
    
    return {
        'original': time_series,
        'smoothed': smoothed,
        'trend_correlation': trend_correlation,
        'low_freq_energy': low_freq_energy,
        'maxima': maxima,
        'minima': minima
    }

def scipy_seasonality(time_series: np.ndarray, get_features: bool = False, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Анализ сезонности во временном ряде.
    
    Методы:
    1. Спектральный анализ для выявления периодических компонент
    2. Автокорреляция для определения периода
    3. Анализ пиков для определения регулярности
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        fs: Частота дискретизации
        
    Returns:
        Словарь с результатами анализа или признаками сезонности
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Спектральный анализ
    f, Pxx = welch(time_series, fs=fs, nperseg=min(256, len(time_series)))
    peak_freqs = f[find_peaks(Pxx)[0]]
    
    # 2. Автокорреляция
    autocorr = correlate(time_series, time_series, mode='full')
    lags = correlation_lags(len(time_series), len(time_series))
    peak_lags = lags[find_peaks(autocorr)[0]]
    
    # 3. Анализ пиков
    peaks, properties = find_peaks(time_series)
    peak_proms = peak_prominences(time_series, peaks)[0]
    
    if get_features:
        return {
            'seasonality_strength': np.max(Pxx) / np.mean(Pxx),  # Сила сезонности
            'seasonality_period': 1/peak_freqs[0] if len(peak_freqs) > 0 else 0,  # Основной период
            'seasonality_regularity': np.std(np.diff(peak_lags)) if len(peak_lags) > 1 else 0,  # Регулярность
            'seasonality_prominence': np.mean(peak_proms) if len(peak_proms) > 0 else 0,  # Выраженность
            'seasonality_spectral_entropy': -np.sum(Pxx * np.log2(Pxx + 1e-10)),  # Спектральная энтропия
            'seasonality_peak_count': len(peaks),  # Количество пиков
            'seasonality_peak_std': np.std(np.diff(peaks)) if len(peaks) > 1 else 0,  # Стандартное отклонение расстояний между пиками
        }
    
    return {
        'freq': f,
        'psd': Pxx,
        'peak_freqs': peak_freqs,
        'autocorr': autocorr,
        'peak_lags': peak_lags,
        'peaks': peaks,
        'peak_prominences': peak_proms
    }

def scipy_structural_changes(time_series: np.ndarray, get_features: bool = False, window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Анализ структурных изменений во временном ряде.
    
    Методы:
    1. Анализ локальной дисперсии
    2. Анализ локальных экстремумов
    3. Спектральный анализ в скользящем окне
    4. Анализ огибающей сигнала
    5. Использование sosfilt для каскадной фильтрации
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна для локального анализа
        
    Returns:
        Словарь с результатами анализа или признаками структурных изменений
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Локальная дисперсия
    local_std = np.array([np.std(time_series[i:i+window_size]) 
                         for i in range(0, len(time_series)-window_size+1)])
    
    # 2. Локальные экстремумы
    maxima = argrelextrema(time_series, np.greater, order=window_size)[0]
    minima = argrelextrema(time_series, np.less, order=window_size)[0]
    
    # 3. Спектральный анализ в окне
    window = get_window('hann', window_size)
    f, t, Sxx = spectrogram(time_series, window=window, nperseg=window_size)
    
    # 4. Анализ огибающей
    analytic_signal = hilbert(time_series)
    envelope = np.abs(analytic_signal)
    
    # 5. Применение sosfilt для каскадной фильтрации
    sos = butter(4, 0.1, btype='low', output='sos')
    filtered_sos = sosfilt(sos, time_series)
    
    if get_features:
        return {
            'structural_std_change': np.std(local_std),  # Изменчивость локальной дисперсии
            'structural_extrema_density': (len(maxima) + len(minima)) / len(time_series),  # Плотность экстремумов
            'structural_spectral_change': np.std(np.mean(Sxx, axis=0)),  # Изменчивость спектра
            'structural_envelope_change': np.std(envelope),  # Изменчивость огибающей
            'structural_local_std_ratio': np.max(local_std) / np.min(local_std),  # Отношение максимальной к минимальной локальной дисперсии
            'structural_spectral_entropy': -np.sum(np.mean(Sxx, axis=0) * np.log2(np.mean(Sxx, axis=0) + 1e-10)),  # Энтропия спектра
            'structural_envelope_ratio': np.max(envelope) / np.min(envelope),  # Отношение максимальной к минимальной огибающей
            'structural_change_points': len(find_peaks(local_std)[0]),  # Количество точек изменения
            'filtered_sos': filtered_sos  # Результат каскадной фильтрации
        }
    
    return {
        'local_std': local_std,
        'maxima': maxima,
        'minima': minima,
        'spectrogram': Sxx,
        'envelope': envelope
    }

def scipy_noise(time_series: np.ndarray, get_features: bool = False, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Анализ шума во временном ряде.
    
    Методы:
    1. Спектральный анализ высокочастотных компонент
    2. Анализ статистических характеристик
    3. Фильтрация и анализ остатков
    4. Анализ локальной нестабильности
    5. Использование фильтра IIR для подавления шума
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        fs: Частота дискретизации
        
    Returns:
        Словарь с результатами анализа или признаками шума
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Спектральный анализ
    f, Pxx = welch(time_series, fs=fs, nperseg=min(256, len(time_series)))
    high_freq_mask = f > 0.5  # Высокочастотные компоненты
    high_freq_energy = np.sum(Pxx[high_freq_mask])
    total_energy = np.sum(Pxx)
    
    # 2. Статистические характеристики
    noise_skewness = skew(time_series)
    noise_kurtosis = kurtosis(time_series)
    
    # 3. Фильтрация и остатки с использованием IIR фильтра
    b, a = butter(4, 0.1, btype='low')
    filtered = filtfilt(b, a, time_series)
    residuals = time_series - filtered
    
    # 4. Локальная нестабильность
    local_diff = np.diff(time_series)
    
    # 5. Использование periodogram для оценки спектральной плотности
    f_period, Pxx_period = periodogram(time_series, fs=fs)
    
    if get_features:
        return {
            'noise_high_freq_ratio': high_freq_energy / total_energy,  # Доля высокочастотных компонент
            'noise_skewness': noise_skewness,  # Асимметрия шума
            'noise_kurtosis': noise_kurtosis,  # Эксцесс шума
            'noise_residual_std': np.std(residuals),  # Стандартное отклонение остатков
            'noise_residual_ratio': np.std(residuals) / np.std(time_series),  # Отношение остатков к исходному сигналу
            'noise_local_instability': np.std(local_diff),  # Локальная нестабильность
            'noise_spectral_entropy': -np.sum(Pxx * np.log2(Pxx + 1e-10)),  # Спектральная энтропия
            'noise_whiteness': np.corrcoef(time_series[:-1], time_series[1:])[0, 1],  # Белый шум
            'periodogram_freq': f_period,  # Частоты для periodogram
            'periodogram_psd': Pxx_period  # Спектральная плотность для periodogram
        }
    
    return {
        'freq': f,
        'psd': Pxx,
        'high_freq_energy': high_freq_energy,
        'residuals': residuals,
        'local_diff': local_diff
    }

def scipy_cross_correlation(time_series1: np.ndarray, time_series2: np.ndarray, get_features: bool = False, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Анализ взаимной корреляции между двумя временными рядами.
    
    Методы:
    1. Кросс-корреляция
    2. Кросс-спектральная плотность
    3. Когерентность
    4. Анализ фазовых соотношений
    
    Args:
        time_series1: Первый временной ряд
        time_series2: Второй временной ряд
        get_features: Флаг для получения признаков
        fs: Частота дискретизации
        
    Returns:
        Словарь с результатами анализа или признаками взаимной корреляции
    """
    if not isinstance(time_series1, np.ndarray) or not isinstance(time_series2, np.ndarray):
        raise ValueError("Входные данные должны быть массивами numpy")
    if time_series1.ndim != 1 or time_series2.ndim != 1:
        raise ValueError("Входные данные должны быть одномерными массивами")
    if len(time_series1) != len(time_series2):
        raise ValueError("Входные данные должны иметь одинаковую длину")
    
    # 1. Кросс-корреляция
    cross_corr = correlate(time_series1, time_series2, mode='full')
    lags = correlation_lags(len(time_series1), len(time_series2))
    
    # 2. Кросс-спектральная плотность
    f, Pxy = csd(time_series1, time_series2, fs=fs)
    
    # 3. Когерентность
    f, Cxy = coherence(time_series1, time_series2, fs=fs)
    
    # 4. Фазовые соотношения
    analytic_signal1 = hilbert(time_series1)
    analytic_signal2 = hilbert(time_series2)
    phase_diff = np.angle(analytic_signal1) - np.angle(analytic_signal2)
    
    if get_features:
        return {
            'cross_corr_max': np.max(cross_corr),  # Максимальная корреляция
            'cross_corr_lag': lags[np.argmax(cross_corr)],  # Задержка максимальной корреляции
            'cross_corr_std': np.std(cross_corr),  # Стандартное отклонение корреляции
            'cross_spectral_energy': np.sum(np.abs(Pxy)),  # Энергия кросс-спектра
            'coherence_mean': np.mean(Cxy),  # Средняя когерентность
            'coherence_std': np.std(Cxy),  # Стандартное отклонение когерентности
            'phase_diff_mean': np.mean(phase_diff),  # Средняя разность фаз
            'phase_diff_std': np.std(phase_diff),  # Стандартное отклонение разности фаз
        }
    
    return {
        'cross_corr': cross_corr,
        'lags': lags,
        'freq': f,
        'cross_spectrum': Pxy,
        'coherence': Cxy,
        'phase_diff': phase_diff
    }

