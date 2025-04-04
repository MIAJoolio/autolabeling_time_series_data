import numpy as np
from pathlib import Path
from core.generation.ts_datasets import Synthetic_dataset_generator
from core.generation import (
    linear_trend_params, quadratic_trend_params, exponential_trend_params,
    seasonal_series_params, harmonic_oscillator_params, sawtooth_wave_params,
    random_walk_params
)

def generate_4block_dataset():
    """Генерация датасета с 4 блоками и различными комбинациями"""
    generator = Synthetic_dataset_generator(
        num_classes=5,  # 5 классов
        series_per_class=100,
        num_blocks=4,
        block_length=50,
        random_state=42
    )
    
    # Класс 1: Растущий тренд + Сезонность + Random Walk
    generator.set_class_config(1, [
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=0.5, slope_up=1.0, random_state=1)
        },
        {
            'type': 'seasonal',
            'param_config': seasonal_series_params(k=1, amplitude_d=2.0, amplitude_up=3.0, random_state=2)
        },
        {
            'type': 'random_walk',
            'param_config': random_walk_params(k=1, initial_value_d=0.0, initial_value_up=0.1, random_state=3)
        },
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=0.2, slope_up=0.4, random_state=4)
        }
    ])
    
    # Класс 2: Убывающий тренд + Сезонность + Random Walk
    generator.set_class_config(2, [
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=-1.0, slope_up=-0.5, random_state=5)
        },
        {
            'type': 'seasonal',
            'param_config': seasonal_series_params(k=1, amplitude_d=1.5, amplitude_up=2.5, random_state=6)
        },
        {
            'type': 'random_walk',
            'param_config': random_walk_params(k=1, initial_value_d=-0.1, initial_value_up=0.0, random_state=7)
        },
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=-0.4, slope_up=-0.2, random_state=8)
        }
    ])

    # Класс 3: Экспоненциальный тренд + Гармонический осциллятор
    generator.set_class_config(3, [
        {
            'type': 'exponential',
            'param_config': exponential_trend_params(k=1, alpha_d=0.05, alpha_up=0.15, random_state=9)
        },
        {
            'type': 'harmonic',
            'param_config': harmonic_oscillator_params(k=1, amplitude_d=1.0, amplitude_up=2.0, random_state=10)
        },
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=0.1, slope_up=0.3, random_state=11)
        },
        {
            'type': 'seasonal',
            'param_config': seasonal_series_params(k=1, amplitude_d=1.0, amplitude_up=2.0, random_state=12)
        }
    ])

    # Класс 4: Квадратичный тренд + Пилообразный сигнал
    generator.set_class_config(4, [
        {
            'type': 'quadratic',
            'param_config': quadratic_trend_params(k=1, a_d=-0.5, a_up=0.5, random_state=13)
        },
        {
            'type': 'sawtooth',
            'param_config': sawtooth_wave_params(k=1, amplitude_d=1.0, amplitude_up=2.0, random_state=14)
        },
        {
            'type': 'random_walk',
            'param_config': random_walk_params(k=1, initial_value_d=0.0, initial_value_up=0.1, random_state=15)
        },
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=0.1, slope_up=0.3, random_state=16)
        }
    ])

    # Класс 5: Комбинация всех генераторов
    generator.set_class_config(5, [
        {
            'type': 'linear',
            'param_config': linear_trend_params(k=1, slope_d=0.5, slope_up=1.0, random_state=17)
        },
        {
            'type': 'exponential',
            'param_config': exponential_trend_params(k=1, alpha_d=0.05, alpha_up=0.15, random_state=18)
        },
        {
            'type': 'harmonic',
            'param_config': harmonic_oscillator_params(k=1, amplitude_d=1.0, amplitude_up=2.0, random_state=19)
        },
        {
            'type': 'random_walk',
            'param_config': random_walk_params(k=1, initial_value_d=0.0, initial_value_up=0.1, random_state=20)
        }
    ])
    
    # Генерация и сохранение датасета
    series, labels = generator.generate_dataset()
    generator.save_dataset(series, labels, 'data/4block_dataset.json')

def generate_3d_dataset():
    """Генерация 3D датасета с 2 блоками"""
    generator = Synthetic_dataset_generator(
        num_classes=20,  # 20 классов
        series_per_class=100,
        num_blocks=2,
        block_length=50,
        random_state=42
    )
    
    # Генерация 20 различных классов с разными комбинациями генераторов
    generator_types = ['linear', 'quadratic', 'exponential', 'seasonal', 'harmonic', 'sawtooth', 'random_walk']
    param_generators = {
        'linear': linear_trend_params,
        'quadratic': quadratic_trend_params,
        'exponential': exponential_trend_params,
        'seasonal': seasonal_series_params,
        'harmonic': harmonic_oscillator_params,
        'sawtooth': sawtooth_wave_params,
        'random_walk': random_walk_params
    }
    
    for class_id in range(1, 21):
        # Выбираем случайные типы генераторов для каждого блока
        block_types = np.random.choice(generator_types, size=2, replace=False)
        
        # Создаем конфигурацию для класса
        class_config = []
        for i, block_type in enumerate(block_types):
            param_generator = param_generators[block_type]
            class_config.append({
                'type': block_type,
                'param_config': param_generator(k=1, random_state=class_id * 100 + i)
            })
        
        generator.set_class_config(class_id, class_config)
    
    # Генерация и сохранение датасета
    series, labels = generator.generate_dataset()
    generator.save_dataset(series, labels, 'data/3d_dataset.json')

def main():
    # Создание директории для данных
    Path('data').mkdir(exist_ok=True)
    
    # Генерация датасетов
    print("Генерация датасета с 4 блоками...")
    generate_4block_dataset()
    print("Генерация 3D датасета...")
    generate_3d_dataset()
    print("Генерация завершена!")

if __name__ == '__main__':
    main() 