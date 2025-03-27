import numpy as np
from core.generation.ts_generators import (
    Generator, linear_trend, seasonal_series, random_walk,
    linear_trend_params, seasonal_series_params, random_walk_params
)
from core.utils import plot_series, plot_series_grid

def generate_experiment3():
    # Создаем 10 генераторов
    generators = [Generator() for _ in range(10)]
    
    # Список всех функций для генерации
    external_functions = {
        "linear": linear_trend,
        "seasonal": seasonal_series,
        "random_walk": random_walk
    }
    
    # Список функций для генерации параметров
    param_functions = {
        "linear": linear_trend_params,
        "seasonal": seasonal_series_params,
        "random_walk": random_walk_params
    }
    
    # Генерируем 10 временных рядов
    all_series = []
    all_metadata = []
    
    # Типы блоков для каждого ряда
    block_types = [
        ["seasonal", "linear", "random_walk"],
        ["linear", "random_walk", "seasonal"],
        ["random_walk", "seasonal", "linear"],
        ["seasonal", "random_walk", "linear"],
        ["linear", "seasonal", "random_walk"],
        ["random_walk", "linear", "seasonal"],
        ["seasonal", "linear", "random_walk"],
        ["linear", "random_walk", "seasonal"],
        ["random_walk", "seasonal", "linear"],
        ["seasonal", "random_walk", "linear"]
    ]
    
    # Генерируем ряды
    for i, generator in enumerate(generators):
        # Добавляем блоки для каждого ряда
        for j, block_type in enumerate(block_types[i]):
            # Генерируем параметры для блока с разными значениями k для разных рядов
            k = 1.0 + 0.1 * i  # Разные значения k для разных рядов
            params = param_functions[block_type](k=k)  # Убираем random_state
            length = 33 if j < 2 else 34  # Распределяем 100 точек между тремя блоками
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
    SHOW_DETAILED = True  # Показывать детальные графики для каждого ряда
    
    # Генерируем ряды
    series, metadata = generate_experiment3()
    
    # Визуализируем результаты
    if SHOW_COMBINED:
        # Отображаем все ряды на одном графике
        plot_series(
            series_list=series,
            labels=[f'Ряд {i+1}' for i in range(len(series))],
            plot_title='Все временные ряды',
            xlabel='Время',
            ylabel='Значение',
            figsize=(15, 8)
        )
    
    if SHOW_DETAILED:
        # Показываем детальные графики для каждого ряда
        for i, (s, m) in enumerate(zip(series, metadata)):
            # Создаем список всех графиков для текущего ряда
            all_plots = [s] + [block['series'] for block in m]
            all_labels = ['Полный ряд'] + [f'Блок_{j+1}' for j in range(len(m))]
            
            # Отображаем графики в столбец
            plot_series_grid(
                series_list=all_plots,
                labels=all_labels,
                plot_title=f'Временной ряд {i+1}',
                xlabel='Время',
                ylabel='Значение',
                figsize=(12, 10),
                layout='vertical'
            ) 