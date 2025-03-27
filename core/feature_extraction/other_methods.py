from typing import Dict, Optional, Union, Tuple
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import stft, find_peaks, argrelmin, argrelmax, peak_prominences, peak_widths
from scipy.fft import fft, ifft
import pywt
from tslearn.piecewise import PiecewiseAggregateApproximation
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

__all__ = [
    'test_method_statistics',
    'test_method_peaks',
    'test_method_stft',
    'test_method_dft',
    'test_method_dwt',
    'test_method_paa',
    'noise_decomposition',
    'noise_statistics',
    'noise_autocorrelation',
    'noise_spectral_analysis',
    'noise_heteroscedasticity',
    'dwt_features',
    'dwt_signal',
    'extract_noise_features',
    'seasonality_removal',
    'seasonality_smoothing',
    'seasonality_filtering',
    'seasonality_spectral',
    'seasonality_peaks',
    'seasonality_correlation',
    'cusum_test',
    'rolling_window_analysis',
    'entropy_based_breaks',
    'regression_based_breaks',
    'signal_peaks_features',
    'paa_features',
    'extract_structural_breaks_features'
]

def test_method_statistics(time_series: np.ndarray, get_features: bool = False) -> Dict[str, Union[np.ndarray, float]]:
    """
    Анализ статистических характеристик временного ряда.
    
    Методы:
    1. Базовые статистики (среднее, дисперсия)
    2. Асимметрия и эксцесс
    3. Абсолютные разности
    4. Частота пересечения нуля
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Базовые статистики
    mean = np.mean(time_series)
    variance = np.var(time_series)
    
    # 2. Асимметрия и эксцесс
    skewness = skew(time_series)
    kurt = kurtosis(time_series)
    
    # 3. Абсолютные разности
    abs_diff = np.abs(np.diff(time_series))
    mean_abs_diff = np.mean(abs_diff)
    
    # 4. Частота пересечения нуля
    zero_crossings = np.sum(np.diff(time_series > 0))
    zero_crossing_rate = zero_crossings / len(time_series)
    
    if get_features:
        return {
            'stat_mean': mean,
            'stat_variance': variance,
            'stat_skewness': skewness,
            'stat_kurtosis': kurt,
            'stat_mean_abs_diff': mean_abs_diff,
            'stat_zero_crossing_rate': zero_crossing_rate
        }
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurt,
        'abs_diff': abs_diff,
        'zero_crossings': zero_crossings
    }

def test_method_peaks(time_series: np.ndarray, get_features: bool = False) -> Dict[str, Union[np.ndarray, float]]:
    """
    Анализ пиков во временном ряде.
    
    Методы:
    1. Относительные минимумы и максимумы
    2. Анализ пиков через find_peaks
    3. Анализ выступов пиков
    4. Анализ ширины пиков
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # 1. Относительные экстремумы
    rel_min_indices = argrelmin(time_series)[0]
    rel_max_indices = argrelmax(time_series)[0]
    
    # 2. Анализ пиков
    peaks, properties = find_peaks(time_series)
    num_peaks = len(peaks)
    
    # 3. Анализ выступов
    prominences = peak_prominences(time_series, peaks)[0] if num_peaks > 0 else np.array([])
    
    # 4. Анализ ширины
    widths, _, _, _ = peak_widths(time_series, peaks) if num_peaks > 0 else (np.array([]), np.array([]), np.array([]), np.array([]))
    
    if get_features:
        features = {
            'peak_num_rel_min': len(rel_min_indices),
            'peak_num_rel_max': len(rel_max_indices),
            'peak_num_peaks': num_peaks
        }
        
        if num_peaks > 0:
            features.update({
                'peak_mean_height': np.mean(time_series[peaks]),
                'peak_max_height': np.max(time_series[peaks]),
                'peak_min_height': np.min(time_series[peaks]),
                'peak_mean_prominence': np.mean(prominences),
                'peak_max_prominence': np.max(prominences),
                'peak_min_prominence': np.min(prominences),
                'peak_mean_width': np.mean(widths),
                'peak_max_width': np.max(widths),
                'peak_min_width': np.min(widths)
            })
        else:
            features.update({f'peak_{key}': 0 for key in [
                'mean_height', 'max_height', 'min_height',
                'mean_prominence', 'max_prominence', 'min_prominence',
                'mean_width', 'max_width', 'min_width'
            ]})
        
        return features
    
    return {
        'rel_min_indices': rel_min_indices,
        'rel_max_indices': rel_max_indices,
        'peaks': peaks,
        'peak_heights': time_series[peaks] if num_peaks > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths
    }

def test_method_stft(time_series: np.ndarray, get_features: bool = False, nperseg: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Анализ временного ряда с помощью короткого преобразования Фурье.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        nperseg: Размер сегмента для STFT
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    if nperseg is None:
        nperseg = len(time_series)
    
    _, _, Zxx = stft(time_series, nperseg=nperseg)
    magnitude = np.abs(Zxx)
    
    if get_features:
        return {
            'stft_magnitude': magnitude.flatten(),
            'stft_mean': np.mean(magnitude),
            'stft_std': np.std(magnitude),
            'stft_max': np.max(magnitude),
            'stft_min': np.min(magnitude)
        }
    
    return {
        'magnitude': magnitude,
        'phase': np.angle(Zxx)
    }

def test_method_dft(time_series: np.ndarray, get_features: bool = False, n_freqs: int = 10) -> Dict[str, np.ndarray]:
    """
    Анализ временного ряда с помощью дискретного преобразования Фурье.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        n_freqs: Количество значимых частотных компонент
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # Применяем FFT
    dft_values = fft(time_series)
    magnitude = np.abs(dft_values)
    phases = np.angle(dft_values)
    
    n = len(time_series)
    fs = 500  # Частота дискретизации
    freq = np.fft.fftfreq(n, d=1/fs)
    
    # Положительные частоты
    positive_freq = freq[:n // 2]
    positive_magnitude = magnitude[:n // 2]
    positive_phases = phases[:n // 2]
    
    # Значимые компоненты
    significant_indices = np.argsort(positive_magnitude)[-n_freqs:]
    
    if get_features:
        significant_components = np.array([[positive_magnitude[j], phases[j]] for j in significant_indices]).flatten()
        return {
            'dft_components': significant_components,
            'dft_mean_magnitude': np.mean(positive_magnitude),
            'dft_std_magnitude': np.std(positive_magnitude),
            'dft_max_magnitude': np.max(positive_magnitude),
            'dft_min_magnitude': np.min(positive_magnitude)
        }
    
    return {
        'freq': positive_freq,
        'magnitude': positive_magnitude,
        'phases': positive_phases,
        'significant_indices': significant_indices
    }

def test_method_dwt(time_series: np.ndarray, get_features: bool = False, 
                wavelet: str = 'db1', mode: str = 'symmetric', 
                level: int = 4, n_coeffs: int = 15) -> Dict[str, np.ndarray]:
    """
    Анализ временного ряда с помощью дискретного вейвлет-преобразования.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        wavelet: Тип вейвлета
        mode: Режим дополнения границ
        level: Уровень декомпозиции
        n_coeffs: Количество значимых коэффициентов
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # Вейвлет-преобразование
    coeffs = pywt.wavedec(time_series, wavelet, mode=mode, level=level)
    
    # Объединяем все коэффициенты
    all_coeffs = np.concatenate(coeffs)
    
    # Значимые коэффициенты
    significant_indices = np.argsort(np.abs(all_coeffs))[-n_coeffs:]
    significant_coeffs = all_coeffs[significant_indices]
    
    if get_features:
        return {
            'dwt_coeffs': significant_coeffs,
            'dwt_mean_coeffs': np.mean(coeffs, axis=1),
            'dwt_std_coeffs': np.std(coeffs, axis=1),
            'dwt_max_coeffs': np.max(coeffs, axis=1),
            'dwt_min_coeffs': np.min(coeffs, axis=1)
        }
    
    return {
        'coefficients': coeffs,
        'significant_coeffs': significant_coeffs,
        'significant_indices': significant_indices
    }

def test_method_paa(time_series: np.ndarray, get_features: bool = False, n_segments: int = 5) -> Dict[str, np.ndarray]:
    """
    Анализ временного ряда с помощью кусочно-агрегатного приближения.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        n_segments: Количество сегментов
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # Преобразуем в 2D массив для PAA
    if len(time_series.shape) == 1:
        time_series = time_series.reshape(1, -1)
    
    # Применяем PAA
    paa = PiecewiseAggregateApproximation(n_segments=n_segments)
    paa_result = paa.fit_transform(time_series).flatten()
    
    if get_features:
        return {
            'paa_segments': paa_result,
            'paa_mean': np.mean(paa_result),
            'paa_std': np.std(paa_result),
            'paa_max': np.max(paa_result),
            'paa_min': np.min(paa_result)
        }
    
    return {
        'paa_result': paa_result,
        'n_segments': n_segments
    }

def noise_decomposition(time_series: np.ndarray, get_features: bool = False, trend: Optional[np.ndarray] = None, 
                       seasonal: Optional[np.ndarray] = None, period: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Выделяет шум из временного ряда.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        trend: Тренд (если известен)
        seasonal: Сезонная составляющая (если известна)
        period: Период сезонности (если известен)
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    if trend is None or seasonal is None:
        decomposition = seasonal_decompose(time_series, period=period)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
    
    noise = time_series - trend - seasonal
    
    if get_features:
        return {
            'noise_component': noise,
            'noise_mean': np.mean(noise),
            'noise_std': np.std(noise),
            'noise_skewness': skew(noise),
            'noise_kurtosis': kurtosis(noise)
        }
    
    return {
        'noise': noise,
        'trend': trend,
        'seasonal': seasonal
    }

def noise_statistics(noise: np.ndarray, get_features: bool = False) -> Dict[str, Union[np.ndarray, float]]:
    """
    Вычисляет статистические характеристики шума.
    
    Args:
        noise: Шумовая составляющая
        get_features: Флаг для получения признаков
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(noise, np.ndarray) or noise.ndim != 1:
        raise ValueError("noise должен быть одномерным массивом numpy")
    
    # Базовые статистики
    mean = np.mean(noise)
    std = np.std(noise)
    skewness = scipy_stats.skew(noise)
    kurt = scipy_stats.kurtosis(noise)
    
    # Тест на нормальность
    _, p_value = scipy_stats.normaltest(noise)
    
    # Энтропия
    hist, _ = np.histogram(noise, bins='auto', density=True)
    ent = entropy(hist)
    
    if get_features:
        return {
            'noise_mean': mean,
            'noise_std': std,
            'noise_skewness': skewness,
            'noise_kurtosis': kurt,
            'noise_normality_p_value': p_value,
            'noise_entropy': ent
        }
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'normality_p_value': p_value,
        'entropy': ent
    }

def noise_autocorrelation(noise: np.ndarray, get_features: bool = False, max_lag: int = 20) -> Dict[str, np.ndarray]:
    """
    Анализирует автокорреляцию шума.
    
    Args:
        noise: Шумовая составляющая
        get_features: Флаг для получения признаков
        max_lag: Максимальный лаг
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(noise, np.ndarray) or noise.ndim != 1:
        raise ValueError("noise должен быть одномерным массивом numpy")
    
    from statsmodels.tsa.stattools import acf
    acf_values = acf(noise, nlags=max_lag)
    
    # Тест Льюнга-Бокса на независимость
    lb_test = acorr_ljungbox(noise, lags=range(1, max_lag+1))
    
    if get_features:
        return {
            'noise_acf': acf_values,
            'noise_lb_test': lb_test,
            'noise_acf_mean': np.mean(acf_values),
            'noise_acf_std': np.std(acf_values),
            'noise_acf_max': np.max(acf_values),
            'noise_acf_min': np.min(acf_values)
        }
    
    return {
        'acf': acf_values,
        'lb_test': lb_test
    }

def noise_spectral_analysis(noise: np.ndarray, get_features: bool = False, sampling_rate: float = 1) -> Dict[str, np.ndarray]:
    """
    Спектральный анализ шума.
    
    Args:
        noise: Шумовая составляющая
        get_features: Флаг для получения признаков
        sampling_rate: Частота дискретизации
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(noise, np.ndarray) or noise.ndim != 1:
        raise ValueError("noise должен быть одномерным массивом numpy")
    
    frequencies, power = welch(noise, fs=sampling_rate)
    
    if get_features:
        return {
            'noise_frequencies': frequencies,
            'noise_power': power,
            'noise_power_mean': np.mean(power),
            'noise_power_std': np.std(power),
            'noise_power_max': np.max(power),
            'noise_power_min': np.min(power)
        }
    
    return {
        'frequencies': frequencies,
        'power': power
    }

def noise_heteroscedasticity(noise: np.ndarray, get_features: bool = False, window_size: int = 20) -> Dict[str, Union[np.ndarray, float]]:
    """
    Анализ гетероскедастичности шума.
    
    Args:
        noise: Шумовая составляющая
        get_features: Флаг для получения признаков
        window_size: Размер окна
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(noise, np.ndarray) or noise.ndim != 1:
        raise ValueError("noise должен быть одномерным массивом numpy")
    
    variances = []
    for i in range(0, len(noise) - window_size + 1):
        window = noise[i:i + window_size]
        variances.append(np.var(window))
    
    variances = np.array(variances)
    hetero_score = np.std(variances) / np.mean(variances)
    
    if get_features:
        return {
            'noise_variances': variances,
            'noise_hetero_score': hetero_score,
            'noise_variance_mean': np.mean(variances),
            'noise_variance_std': np.std(variances),
            'noise_variance_max': np.max(variances),
            'noise_variance_min': np.min(variances)
        }
    
    return {
        'variances': variances,
        'hetero_score': hetero_score
    }

def dwt_features(series: np.ndarray, get_features: bool = False, wavelet: str = 'db4', level: int = 4) -> Dict[str, np.ndarray]:
    """
    Вычисляет признаки дискретного вейвлет-преобразования.
    
    Args:
        series: Временной ряд
        get_features: Флаг для получения признаков
        wavelet: Тип вейвлета
        level: Уровень декомпозиции
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        raise ValueError("series должен быть одномерным массивом numpy")
    
    coeffs = pywt.wavedec(series, wavelet, level=level)
    features = []
    
    for i, coeff in enumerate(coeffs):
        features.extend([
            np.mean(np.abs(coeff)),
            np.std(coeff),
            entropy(np.abs(coeff))
        ])
    
    if get_features:
        return {
            'dwt_features': np.array(features),
            'dwt_mean_coeffs': np.mean(coeffs, axis=1),
            'dwt_std_coeffs': np.std(coeffs, axis=1),
            'dwt_entropy_coeffs': [entropy(np.abs(coeff)) for coeff in coeffs]
        }
    
    return {
        'features': np.array(features),
        'coefficients': coeffs
    }

def dwt_signal(series: np.ndarray, get_features: bool = False, wavelet: str = 'db4', 
               level: int = 4, threshold: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Восстанавливает сигнал из DWT с пороговой обработкой.
    
    Args:
        series: Временной ряд
        get_features: Флаг для получения признаков
        wavelet: Тип вейвлета
        level: Уровень декомпозиции
        threshold: Порог для фильтрации
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        raise ValueError("series должен быть одномерным массивом numpy")
    
    coeffs = pywt.wavedec(series, wavelet, level=level)
    
    # Применяем пороговую обработку
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(np.abs(coeffs[i])))
    
    restored_signal = pywt.waverec(coeffs, wavelet)
    
    if get_features:
        return {
            'dwt_restored_signal': restored_signal,
            'dwt_original_signal': series,
            'dwt_difference': series - restored_signal,
            'dwt_rmse': np.sqrt(np.mean((series - restored_signal)**2)),
            'dwt_mae': np.mean(np.abs(series - restored_signal))
        }
    
    return {
        'restored_signal': restored_signal,
        'coefficients': coeffs
    }

def extract_noise_features(time_series: np.ndarray, get_features: bool = False, 
                         trend: Optional[np.ndarray] = None, 
                         seasonal: Optional[np.ndarray] = None, 
                         period: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Извлекает все признаки шума.
    
    Args:
        time_series: Входной временной ряд
        get_features: Флаг для получения признаков
        trend: Тренд (если известен)
        seasonal: Сезонная составляющая (если известна)
        period: Период сезонности (если известен)
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # Выделение шума
    noise = noise_decomposition(time_series, get_features=False, trend=trend, seasonal=seasonal, period=period)['noise']
    
    if get_features:
        return {
            'noise_component': noise,
            'noise_statistics': np.array([
                np.mean(noise),
                np.var(noise),
                np.std(noise),
                entropy(noise)
            ])
        }
    
    return {
        'noise': noise,
        'statistics': np.array([
            np.mean(noise),
            np.var(noise),
            np.std(noise),
            entropy(noise)
        ])
    }

def cusum_test(time_series: np.ndarray, get_features: bool = False, window_size: int = 10) -> Dict[str, np.ndarray]:
    """
    Тест на структурные сдвиги с помощью CUSUM.
    
    Args:
        time_series: Временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна для вычисления статистики
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # Вычисляем накопленную сумму отклонений от среднего
    mean = np.mean(time_series)
    cusum = np.cumsum(time_series - mean)
    
    # Нормализуем статистику
    std = np.std(time_series)
    cusum_stat = cusum / std
    
    # Находим точки разрыва
    break_points = np.where(np.abs(cusum_stat) > 2)[0]
    
    if get_features:
        return {
            'cusum_statistic': cusum_stat,
            'cusum_break_points': break_points,
            'cusum_mean': mean,
            'cusum_std': std,
            'cusum_max': np.max(cusum_stat),
            'cusum_min': np.min(cusum_stat)
        }
    
    return {
        'cusum_stat': cusum_stat,
        'break_points': break_points
    }

def rolling_window_analysis(time_series: np.ndarray, get_features: bool = False, window_size: int = 20) -> Dict[str, np.ndarray]:
    """
    Анализ структурных сдвигов с помощью скользящего окна.
    
    Args:
        time_series: Временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    means = []
    stds = []
    
    for i in range(0, len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        means.append(np.mean(window))
        stds.append(np.std(window))
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Находим точки разрыва как резкие изменения в статистиках
    mean_diff = np.diff(means)
    std_diff = np.diff(stds)
    
    break_points = np.where((np.abs(mean_diff) > 2*np.std(mean_diff)) | 
                          (np.abs(std_diff) > 2*np.std(std_diff)))[0]
    
    if get_features:
        return {
            'rolling_means': means,
            'rolling_stds': stds,
            'window_break_points': break_points,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'mean_diff_std': np.std(mean_diff),
            'std_diff_std': np.std(std_diff)
        }
    
    return {
        'means': means,
        'stds': stds,
        'break_points': break_points
    }

def entropy_based_breaks(time_series: np.ndarray, get_features: bool = False, window_size: int = 20) -> Dict[str, np.ndarray]:
    """
    Определение структурных сдвигов на основе энтропии.
    
    Args:
        time_series: Временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    entropies = []
    
    for i in range(0, len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        hist, _ = np.histogram(window, bins='auto', density=True)
        entropies.append(entropy(hist))
    
    entropies = np.array(entropies)
    entropy_diff = np.diff(entropies)
    break_points = np.where(np.abs(entropy_diff) > 2*np.std(entropy_diff))[0]
    
    if get_features:
        return {
            'entropies': entropies,
            'entropy_break_points': break_points,
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
            'entropy_diff_mean': np.mean(entropy_diff),
            'entropy_diff_std': np.std(entropy_diff)
        }
    
    return {
        'entropies': entropies,
        'break_points': break_points
    }

def regression_based_breaks(time_series: np.ndarray, get_features: bool = False, window_size: int = 20) -> Dict[str, np.ndarray]:
    """
    Определение структурных сдвигов на основе регрессионного анализа.
    
    Args:
        time_series: Временной ряд
        get_features: Флаг для получения признаков
        window_size: Размер окна
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    slopes = []
    
    for i in range(0, len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        x = np.arange(window_size)
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), window)
        slopes.append(model.coef_[0])
    
    slopes = np.array(slopes)
    slope_diff = np.diff(slopes)
    break_points = np.where(np.abs(slope_diff) > 2*np.std(slope_diff))[0]
    
    if get_features:
        return {
            'regression_slopes': slopes,
            'regression_break_points': break_points,
            'slope_mean': np.mean(slopes),
            'slope_std': np.std(slopes),
            'slope_diff_mean': np.mean(slope_diff),
            'slope_diff_std': np.std(slope_diff)
        }
    
    return {
        'slopes': slopes,
        'break_points': break_points
    }

def signal_peaks_features(series: np.ndarray, get_features: bool = False, height: Optional[float] = None, 
                         distance: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Вычисляет признаки пиков сигнала.
    
    Args:
        series: Временной ряд
        get_features: Флаг для получения признаков
        height: Минимальная высота пика
        distance: Минимальное расстояние между пиками
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        raise ValueError("series должен быть одномерным массивом numpy")
    
    peaks, properties = find_peaks(series, height=height, distance=distance)
    
    if len(peaks) == 0:
        if get_features:
            return {
                'peak_count': 0,
                'peak_mean_height': 0,
                'peak_std_height': 0,
                'peak_mean_distance': 0
            }
        return {
            'peaks': np.array([]),
            'properties': {}
        }
    
    if get_features:
        return {
            'peak_count': len(peaks),
            'peak_mean_height': np.mean(properties['peak_heights']) if 'peak_heights' in properties else 0,
            'peak_std_height': np.std(properties['peak_heights']) if 'peak_heights' in properties else 0,
            'peak_mean_distance': np.mean(np.diff(peaks))
        }
    
    return {
        'peaks': peaks,
        'properties': properties
    }

def paa_features(series: np.ndarray, get_features: bool = False, n_segments: int = 10) -> Dict[str, np.ndarray]:
    """
    Вычисляет признаки Piecewise Aggregate Approximation.
    
    Args:
        series: Временной ряд
        get_features: Флаг для получения признаков
        n_segments: Количество сегментов
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        raise ValueError("series должен быть одномерным массивом numpy")
    
    segment_size = len(series) // n_segments
    paa = np.array([np.mean(series[i:i+segment_size]) for i in range(0, len(series), segment_size)])
    
    if get_features:
        return {
            'paa_segments': paa,
            'paa_mean': np.mean(paa),
            'paa_std': np.std(paa),
            'paa_max': np.max(paa),
            'paa_min': np.min(paa)
        }
    
    return {
        'paa': paa,
        'n_segments': n_segments
    }

def extract_structural_breaks_features(time_series: np.ndarray, get_features: bool = False) -> Dict[str, np.ndarray]:
    """
    Извлекает все признаки структурных сдвигов.
    
    Args:
        time_series: Временной ряд
        get_features: Флаг для получения признаков
        
    Returns:
        Словарь с результатами анализа или признаками
    """
    if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
        raise ValueError("time_series должен быть одномерным массивом numpy")
    
    # CUSUM тест
    cusum_result = cusum_test(time_series, get_features=False)
    
    # Анализ скользящего окна
    window_result = rolling_window_analysis(time_series, get_features=False)
    
    # Анализ на основе энтропии
    entropy_result = entropy_based_breaks(time_series, get_features=False)
    
    # Регрессионный анализ
    regression_result = regression_based_breaks(time_series, get_features=False)
    
    # Объединяем все найденные точки разрыва
    all_breaks = np.unique(np.concatenate([
        cusum_result['break_points'],
        window_result['break_points'],
        entropy_result['break_points'],
        regression_result['break_points']
    ]))
    
    if get_features:
        return {
            'cusum_statistic': cusum_result['cusum_stat'],
            'cusum_break_points': cusum_result['break_points'],
            'rolling_means': window_result['means'],
            'rolling_stds': window_result['stds'],
            'window_break_points': window_result['break_points'],
            'entropies': entropy_result['entropies'],
            'entropy_break_points': entropy_result['break_points'],
            'regression_slopes': regression_result['slopes'],
            'regression_break_points': regression_result['break_points'],
            'all_break_points': all_breaks,
            'peaks': signal_peaks_features(time_series, get_features=True),
            'paa': paa_features(time_series, get_features=True)
        }
    
    return {
        'cusum_result': cusum_result,
        'window_result': window_result,
        'entropy_result': entropy_result,
        'regression_result': regression_result,
        'all_breaks': all_breaks
    } 