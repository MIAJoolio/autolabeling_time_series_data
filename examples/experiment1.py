import numpy as np
from core.generation.ts_generators import (
    Generator, linear_trend, exponential_trend, seasonal_series, 
    harmonic_oscillator, sawtooth_wave, linear_trend_params, 
    exponential_trend_params, seasonal_series_params, 
    harmonic_oscillator_params, sawtooth_wave_params
)
from core.utils import plot_series, plot_series_grid
from core.feature_extraction import TimeSeriesAnalyzer
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats

def analyze_time_series(series, metadata):
    """
    Анализирует временные ряды с помощью TimeSeriesAnalyzer.
    
    Параметры:
    series (list): Список временных рядов
    metadata (list): Метаданные рядов
    
    Возвращает:
    dict: Результаты анализа
    """
    analyzer = TimeSeriesAnalyzer()
    results = []
    
    for i, (s, m) in enumerate(zip(series, metadata)):
        try:
            # Определяем период с помощью автокорреляции
            acf_values = acf(s, nlags=len(s)//2)
            peaks = np.where(acf_values > 0.5)[0]
            period = peaks[1] if len(peaks) > 1 else None
            
            # Полный анализ
            features = analyzer.analyze(s, period=period)
            
            # Разложение на составляющие
            decomposition = analyzer.decompose(s, period=period)
            
            # Сохраняем результаты
            results.append({
                'series_id': i,
                'features': features,
                'decomposition': decomposition,
                'metadata': m
            })
            
        except Exception as e:
            print(f"Ошибка анализа ряда {i+1}: {e}")
            continue
    
    return results

def plot_analysis_results(series, results):
    """
    Визуализирует результаты анализа.
    
    Параметры:
    series (list): Список временных рядов
    results (list): Результаты анализа
    """
    for i, (s, r) in enumerate(zip(series, results)):
        # Создаем подграфики
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'Анализ временного ряда {i+1}')
        
        # Исходный ряд
        axes[0].plot(s, label='Исходный ряд')
        axes[0].set_title('Исходный временной ряд')
        axes[0].legend()
        
        # Разложение на составляющие
        decomp = r['decomposition']
        axes[1].plot(decomp['trend'], label='Тренд')
        axes[1].plot(decomp['seasonal'], label='Сезонность')
        axes[1].plot(decomp['noise'], label='Шум')
        axes[1].set_title('Разложение на составляющие')
        axes[1].legend()
        
        # Структурные сдвиги
        axes[2].plot(s, label='Исходный ряд')
        breaks = decomp['structural_breaks']
        if len(breaks) > 0:
            axes[2].vlines(breaks, min(s), max(s), colors='r', label='Точки разрыва')
        axes[2].set_title('Структурные сдвиги')
        axes[2].legend()
        
        # Спектральный анализ
        features = r['features']
        if 'frequencies' in features['noise']:
            axes[3].plot(features['noise']['frequencies'], features['noise']['power_spectrum'])
            axes[3].set_title('Спектральный анализ шума')
            axes[3].set_xlabel('Частота')
            axes[3].set_ylabel('Мощность')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим основные характеристики
        print(f"\nОсновные характеристики ряда {i+1}:")
        print(f"Сила тренда (tau): {features['trend']['trend_strength_tau']:.3f}")
        print(f"Сила сезонности: {features['seasonality']['seasonal_strength']:.3f}")
        print(f"Количество структурных сдвигов: {len(breaks)}")
        print(f"Энтропия шума: {features['noise']['entropy']:.3f}")
        print(f"Гетероскедастичность шума: {features['noise']['heteroscedasticity_score']:.3f}")

def generate_experiment1():
    # Создаем 5 генераторов
    generators = [Generator() for _ in range(5)]
    
    # Список всех функций для генерации
    external_functions = {
        "linear": linear_trend,
        "exponential": exponential_trend,
        "seasonal": seasonal_series,
        "harmonic": harmonic_oscillator,
        "sawtooth": sawtooth_wave
    }
    
    # Список функций для генерации параметров
    param_functions = {
        "linear": linear_trend_params,
        "exponential": exponential_trend_params,
        "seasonal": seasonal_series_params,
        "harmonic": harmonic_oscillator_params,
        "sawtooth": sawtooth_wave_params
    }
    
    # Генерируем 5 разных временных рядов
    all_series = []
    all_metadata = []
    
    # Типы блоков для каждого ряда
    block_types = [
        ["linear", "exponential", "seasonal", "harmonic", "sawtooth"],
        ["exponential", "seasonal", "harmonic", "sawtooth", "linear"],
        ["seasonal", "harmonic", "sawtooth", "linear", "exponential"],
        ["harmonic", "sawtooth", "linear", "exponential", "seasonal"],
        ["sawtooth", "linear", "exponential", "seasonal", "harmonic"]
    ]
    
    # Генерируем ряды
    for i, generator in enumerate(generators):
        # Добавляем 5 блоков для каждого ряда
        for j, block_type in enumerate(block_types[i]):
            # Генерируем параметры для блока с разными значениями k для разных рядов
            k = 1.0 + 0.2 * i  # Разные значения k для разных рядов
            params = param_functions[block_type](k=k)  # Убираем random_state
            length = 20  # 100 точек / 5 блоков
            generator.add_block(block_type, length=length, params=params, random_state=42 + i + j)
    
    # Генерируем все ряды
    for generator in generators:
        series, metadata = generator.generate(external_functions)
        all_series.append(series)
        all_metadata.append(metadata)
    
    return all_series, all_metadata

if __name__ == "__main__":
    # Флаги для управления отображением
    SHOW_COMBINED = True  # Показывать все ряды на одном графике
    SHOW_DETAILED = False  # Показывать детальные графики для каждого ряда
    SHOW_ANALYSIS = False  # Показывать результаты анализа
    
    # Генерируем ряды
    series, metadata = generate_experiment1()
    
    # Визуализируем исходные данные
    if SHOW_COMBINED:
        plot_series(
            series_list=series,
            labels=[f'Ряд {i+1}' for i in range(len(series))],
            plot_title='Все временные ряды',
            xlabel='Время',
            ylabel='Значение',
            figsize=(15, 8)
        )
    
    if SHOW_DETAILED:
        for i, (s, m) in enumerate(zip(series, metadata)):
            all_plots = [block['series'] for block in m]
            all_labels = [f'Блок_{j+1}' for j in range(len(m))]
            
            plot_series_grid(
                series_list=all_plots,
                labels=all_labels,
                plot_title=f'Временной ряд {i+1}',
                xlabel='Время',
                ylabel='Значение',
                figsize=(15, 3),
                layout='horizontal'
            )
            
            plot_series(s, labels=['Полный ряд'])
    
    # # Анализируем ряды
    # if SHOW_ANALYSIS:
    #     results = analyze_time_series(series, metadata)
    #     plot_analysis_results(series, results) 