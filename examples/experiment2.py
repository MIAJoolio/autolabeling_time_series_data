import numpy as np
from core.generation.ts_generators import (
    Generator, linear_trend, exponential_trend, seasonal_series, 
    harmonic_oscillator, sawtooth_wave, linear_trend_params, 
    exponential_trend_params, seasonal_series_params, 
    harmonic_oscillator_params, sawtooth_wave_params
)
from core.utils import plot_series, plot_series_grid

def generate_experiment2():
    # Создаем 10 генераторов (2 группы по 5)
    generators = [Generator() for _ in range(10)]
    
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
    
    # Генерируем 10 временных рядов (2 группы по 5)
    all_series = []
    all_metadata = []
    
    # Типы блоков для обеих групп
    block_types = ["linear", "exponential", "seasonal", "harmonic", "sawtooth"]
    
    # Генерируем первую группу
    for i in range(5):
        generator = generators[i]
        # Добавляем блоки с базовыми параметрами первой группы
        for j, block_type in enumerate(block_types):
            # Генерируем параметры для блока
            params = param_functions[block_type](k=1.0)  # Убираем random_state
            length = 20  # 100 точек / 5 блоков
            generator.add_block(block_type, length=length, params=params, random_state=42 + i + j)
    
    # Генерируем вторую группу
    for i in range(5):
        generator = generators[i + 5]
        # Добавляем блоки с базовыми параметрами второй группы
        for j, block_type in enumerate(block_types):
            # Генерируем параметры для блока с небольшими изменениями
            params = param_functions[block_type](k=1.2)  # Убираем random_state
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
    SHOW_DETAILED = True  # Показывать детальные графики для каждого ряда
    
    # Генерируем ряды
    series, metadata = generate_experiment2()
    
    # Визуализируем результаты
    if SHOW_COMBINED:
        # Отображаем все ряды на одном графике
        plot_series(
            series_list=series,
            labels=[f'Группа {i//5 + 1}, Ряд {i%5 + 1}' for i in range(len(series))],
            plot_title='Все временные ряды',
            xlabel='Время',
            ylabel='Значение',
            figsize=(15, 8)
        )
    
    if SHOW_DETAILED:
        # Показываем детальные графики для каждого ряда
        for i, (s, m) in enumerate(zip(series, metadata)):
            group_num = i // 5 + 1
            series_num = i % 5 + 1
            
            # Создаем список всех графиков для текущего ряда
            all_plots = [s] + [block['series'] for block in m]
            all_labels = ['Полный ряд'] + [f'Блок_{j+1}' for j in range(len(m))]
            
            # Отображаем графики в столбец
            plot_series_grid(
                series_list=all_plots,
                labels=all_labels,
                plot_title=f'Группа {group_num}, Ряд {series_num}',
                xlabel='Время',
                ylabel='Значение',
                figsize=(12, 15),
                layout='vertical'
            ) 