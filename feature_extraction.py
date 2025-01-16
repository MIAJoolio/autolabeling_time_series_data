
import numpy as np

# trend + autocorrelation, statistics
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis

# piecewise 
from tslearn.piecewise import PiecewiseAggregateApproximation

# анализ пиков 
from scipy.signal import argrelmin, argrelmax, find_peaks, find_peaks_cwt, peak_prominences, peak_widths
# 2
from scipy.signal import stft, find_peaks

# dft
from scipy.fft import fft

# dwt
import pywt


# # shapelets
# from tslearn.shapelets import ShapeletModel

# # tsfresh
# from tsfresh import extract_features, select_features
# from tsfresh.utilities.dataframe_functions import roll_time_series


# utils
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_italy_power_demand
from itertools import product

### trend + autocorrelation, statistics

def tsa_detrend(series):
    return detrend(series)

def tsa_acf(series, n_lags=10, thresh=0.5):
    acf_values = acf(series, nlags=n_lags+1)
    # significant_lags = np.where(acf_values > )[0]
    # if len(significant_lags) > 1:
    #     period = significant_lags[1]  # Первый значимый лаг (lag=0 пропускаем)
    #     return period
    # else:
    #     return None
    
    return acf_values[1:]  
    
# def find_seasonal_period(series, acf_lags=40):
#     # Автокорреляция (ACF)
#     acf_values = acf(series, nlags=acf_lags)
    
#     # Ищем значимые пики в ACF (например, значения выше 0.5)
#     significant_lags = np.where(acf_values > 0.5)[0]
#     if len(significant_lags) > 1:
#         period = significant_lags[1]  # Первый значимый лаг (lag=0 пропускаем)
#         return period
#     else:
#         return None


# Time-Domain Features
def statistical_features(time_series):
    return np.array([
        np.mean(time_series),
        np.var(time_series),
        skew(time_series),
        kurtosis(time_series),
        np.mean(np.abs(np.diff(time_series))),  # Mean absolute difference
        np.sum(np.diff(time_series) > 0) / len(time_series)  # Zero-crossing rate
    ])

### Shape-Based Features

def paa_features(time_series, n_segments=5):
    if len(time_series.shape) == 1:
        time_series = time_series.reshape(1, -1)

    paa = PiecewiseAggregateApproximation(n_segments=n_segments)
    return paa.fit_transform(time_series).flatten()


### анализ пиков
def signal_peaks_features(data):
    """
    Анализирует временной ряд и возвращает вектор признаков.

    Параметры:
    data (np.array): Временной ряд (1D массив).
    widths (np.array): Диапазон ширины пиков для find_peaks_cwt.

    Возвращает:
    features (dict): Словарь с признаками временного ряда.
    """
    
    widths=np.arange(1, len(data))
    
    
    features = []
    # 1. Нахождение относительных минимумов и максимумов
    rel_min_indices = argrelmin(data)[0]
    rel_max_indices = argrelmax(data)[0]

    features.append(len(rel_min_indices))  # Количество относительных минимумов
    features.append(len(rel_max_indices))  # Количество относительных максимумов

    # 2. Нахождение пиков с помощью find_peaks
    peaks, properties = find_peaks(data)
    num_peaks = len(peaks)
    features.append(num_peaks)  # Количество пиков

    if num_peaks > 0:
        # 3. Высота пиков
        peak_heights = data[peaks]
        features.append(np.mean(peak_heights))  # Средняя высота пиков
        features.append(np.max(peak_heights))   # Максимальная высота пиков
        features.append(np.min(peak_heights))   # Минимальная высота пиков

        # 4. Prominence (выдающаяся часть пика)
        prominences = peak_prominences(data, peaks)[0]
        features.append(np.mean(prominences))  # Средняя prominence
        features.append(np.max(prominences))   # Максимальная prominence
        features.append(np.min(prominences))   # Минимальная prominence

        # 5. Ширина пиков
        widths, _, _, _ = peak_widths(data, peaks)
        features.append(np.mean(widths))  # Средняя ширина пиков
        features.append(np.max(widths))   # Максимальная ширина пиков
        features.append(np.min(widths))   # Минимальная ширина пиков
    else:
        # Если пиков нет, добавляем нули
        features.extend([0] * 7)  # 7 признаков, связанных с пиками

    # 6. Нахождение пиков с помощью find_peaks_cwt
    cwt_peaks = find_peaks_cwt(data, widths)
    num_cwt_peaks = len(cwt_peaks)
    features.append(num_cwt_peaks)  # Количество пиков, найденных через CWT

    if num_cwt_peaks > 0:
        cwt_peak_heights = data[cwt_peaks]
        features.append(np.mean(cwt_peak_heights))  # Средняя высота CWT пиков
        features.append(np.max(cwt_peak_heights))   # Максимальная высота CWT пиков
        features.append(np.min(cwt_peak_heights))   # Минимальная высота CWT пиков
    else:
        # Если пиков нет, добавляем нули
        features.extend([0] * 3)  # 3 признака, связанных с CWT пиками

    # Преобразуем список признаков в np.ndarray
    return np.array(features, dtype=np.float32)

def stft_features(time_series, noverlap=None):
    nperseg= len(time_series)
    # time_series = time_series.reshape(time_series.shape[1], time_series.shape[0])
    _, _, Zxx = stft(time_series, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx).flatten()


### DFT

def dft_components(time_series, n_freqs=10):
    """
    Функция:
    * Применяет быстрое преобразование Фурье (FFT) к временному ряду.
    * Вычисляет амплитуды, частоты и фазы наиболее значимых компонент, т.е. n_freqs
    """
    dft_values = fft(time_series)
    magnitude = np.abs(dft_values)
    phases =  np.angle(dft_values)
    
    n = len(time_series)
    fs = 500
    freq = np.fft.fftfreq(n, d=1/fs) 

    # Только положительные частоты, т.к. в отрицательных нет информации
    positive_freq = freq[:n // 2]  
    # Соответствующие амплитуды
    positive_magnitude = magnitude[:n // 2] 
    # индексы наибольших значение мощности
    significant = np.argsort(positive_magnitude)[-n_freqs:]

    return np.array([[positive_magnitude[j], phases[j]] for j in significant]).flatten()

### DWT

def dwt_features(time_series, wavelet='db1', mode='symmetric', level=20, n_coeffs=15):
    """
    Функция:
    * Применяет дискретное вейвлет-преобразование (DWT) к временному ряду.
    * Вычисляет наиболее значимые коэффициенты (аппроксимации и детализации).
    * Возвращает признаки, основанные на этих коэффициентах.

    Параметры:
    - time_series: Входной временной ряд.
    - wavelet: Тип вейвлета (по умолчанию 'db4' — вейвлет Добеши 4-го порядка).
    - level: Уровень декомпозиции.
    - n_coeffs: Количество наиболее значимых коэффициентов для возврата.

    Возвращает:
    - Наиболее значимые коэффициенты DWT.
    """
    # Выполняем вейвлет-преобразование
    coeffs = pywt.wavedec(time_series, wavelet, mode=mode, level=level)
    
    # # # Объединяем все коэффициенты в один массив
    all_coeffs = np.concatenate(coeffs)
    
    # # Находим наиболее значимые коэффициенты (по абсолютной величине)
    significant_indices = np.argsort(np.abs(all_coeffs))[-n_coeffs:]
    significant_coeffs = all_coeffs[significant_indices]
    
    return significant_coeffs[:n_coeffs]

def mean_dwt_features(time_series, wavelet='db4', mode='periodic', level=5, n_coeffs=30):
    """
    Функция:
    * Применяет дискретное вейвлет-преобразование (DWT) к временному ряду.
    * Вычисляет наиболее значимые коэффициенты (аппроксимации и детализации).
    * Возвращает признаки, основанные на этих коэффициентах.

    Параметры:
    - time_series: Входной временной ряд.
    - wavelet: Тип вейвлета (по умолчанию 'db4' — вейвлет Добеши 4-го порядка).
    - level: Уровень декомпозиции.
    - n_coeffs: Количество наиболее значимых коэффициентов для возврата.

    Возвращает:
    - Наиболее значимые коэффициенты DWT.
    """
    # Выполняем вейвлет-преобразование
    coeffs = pywt.wavedec(time_series, wavelet, mode=mode, level=level)
    
    # # # Объединяем все коэффициенты в один массив
    # all_coeffs = np.concatenate(coeffs)
    
    # # Находим наиболее значимые коэффициенты (по абсолютной величине)
    # significant_indices = np.argsort(np.abs(all_coeffs))[-n_coeffs:]
    # significant_coeffs = all_coeffs[significant_indices]
    
    # return significant_coeffs[::-1][:10]

    return np.mean(coeffs, axis=1)

def dwt_features_with_info(time_series, wavelet='db1', mode='periodic', level=3, n_coeffs=10):
    """
    Функция:
    * Применяет DWT к временному ряду.
    * Возвращает наиболее значимые коэффициенты с информацией об их уровне и типе.

    Возвращает:
    - Список кортежей (коэффициент, уровень, тип), где тип может быть 'approximation' или 'detail'.
    """
    coeffs = pywt.wavedec(time_series, wavelet, mode=mode, level=level)
    
    # Создаем список всех коэффициентов с информацией об их уровне и типе
    all_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            coeff_type = 'approximation'
        else:
            coeff_type = 'detail'
        all_coeffs.extend([(c, i, coeff_type) for c in coeff])
    
    # Сортируем коэффициенты по абсолютной величине и выбираем наиболее значимые
    all_coeffs.sort(key=lambda x: abs(x[0]), reverse=True)
    significant_coeffs = all_coeffs[:n_coeffs]
    
    return significant_coeffs


if __name__ == "__main__":
    # Load data
    X, y = load_italy_power_demand()
    X = np.array(X)
    
    # Extract features
    # features = extract_all_features(X, y)

# ### Shapelet
# def shapelet_features(X_train, y_train, n_shapelets=3, shapelet_pct=0.15):
#     # похоже нейронка или я неправильно использовал зашита и поэтому очень долго считал
#     # from tslearn.utils import to_time_series_dataset

#     # Многомерный временной ряд (3 переменные, 10 точек)
#     time_series = np.array([
#         [1, 10, 100],
#         [2, 20, 200],
#         [3, 30, 300],
#         [4, 40, 400],
#         [5, 50, 500],
#         [6, 60, 600],
#         [7, 70, 700],
#         [8, 80, 800],
#         [9, 90, 900],
#         [10, 100, 1000]
#     ])

#     # Преобразование в формат, подходящий для tslearn
#     time_series = to_time_series_dataset([time_series])

#     # Инициализация модели
#     model = ShapeletModel(shapelet_length=0.15, verbose=0)

#     # Обучение модели
#     model.fit(time_series[0, :2,], np.array([0,1]).reshape(2,))

#     # Извлечение Shapelet'ов
#     shapelets = model.shapelets_
#     print(shapelets)


# # Feature Extraction and Selection tsfresh
# def tsfresh_features(X, y):
#     """
#     Extract and select relevant features using tsfresh.

#     Parameters:
#     X (np.ndarray): Input time-series data of shape (n_samples, n_timesteps).
#     y (np.ndarray): Target labels for feature selection.

#     Returns:
#     relevant_features (np.ndarray): Relevant features selected by tsfresh.
#     """
#     # Convert to DataFrame format required by tsfresh
#     df = pd.DataFrame(X)
#     df['id'] = df.index
#     df = df.melt(id_vars=['id'], var_name='time', value_name='value')

#     # Extract features
#     extracted_features = extract_features(df, column_id='id', column_sort='time')

#     # Select relevant features
#     relevant_features = select_features(extracted_features, y)
#     return relevant_features.to_numpy()



# # Main function to extract all features
# def extract_all_features(X, y, selected_features=None, save_path='extracted_feature'):
#     """
#     Extract selected features and save each transformation in separate files.

#     Parameters:
#     X (np.ndarray): Input time-series data of shape (n_samples, n_timesteps).
#     y (np.ndarray): Target labels for tsfresh feature selection.
#     selected_features (list): List of feature names to extract. If None, extract all features.

#     Returns:
#     None
#     # """
#     # # Define feature extraction functions and their corresponding file names
#     # feature_extractors = {
#     #     "dft_features": extract_dft_features,
#     #     "dwt_features": extract_dwt_features,
#     #     "stft_features": extract_stft_features,
#     #     "statistical_features": extract_statistical_features,
#     #     "peak_features": extract_peak_features,
#     #     "paa_features": extract_paa_features,
#     #     "shapelet_features": extract_shapelet_features,
#     #     "arima_features": extract_arima_features,
#     #     "entropy_features": extract_entropy_features,
#     #     "tsfresh_features": None,  # tsfresh is handled separately
#     # }

#     # If no features are selected, use all features
#     if selected_features is None:
#         selected_features = list(feature_extractors.keys())

#     # Initialize a dictionary to store feature lists
#     features = {key: [] for key in selected_features}

#     # Extract features for each sample
#     for sample in X:
#         for key in selected_features:
#             if key == "tsfresh_features":
#                 continue  # tsfresh_features is handled separately
#             features[key].append(feature_extractors[key](sample))

#     # Add tsfresh features if selected
#     if "tsfresh_features" in selected_features:
#         features["tsfresh_features"] = extract_tsfresh_features(X, y)

#     # Save each feature set to separate files
#     for key in selected_features:
#         feature_list = features[key]
#         if key != "tsfresh_features":  # tsfresh_features is already a numpy array
#             feature_list = np.array(feature_list)
#         np.save(f"{save_path}/{key}.npy", feature_list)

#     print("Selected feature sets saved to separate files.")