import numpy as np
import json
from pathlib import Path

def generate_simple_series(seq_len=200, num_samples=100):
    """Генерация простых временных рядов для тестирования"""
    series_list = []
    labels = []
    
    # Класс 1: Синусоида
    for i in range(num_samples):
        t = np.linspace(0, 10, seq_len)
        freq = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        series = amplitude * np.sin(2*np.pi*freq*t + phase)
        series_list.append(series)
        labels.append(1)
    
    # Класс 2: Линейный тренд с шумом
    for i in range(num_samples):
        t = np.linspace(0, 1, seq_len)
        slope = np.random.uniform(-2, 2)
        noise = np.random.normal(0, 0.1, seq_len)
        series = slope * t + noise
        series_list.append(series)
        labels.append(2)
    
    # Класс 3: Ступенчатая функция
    for i in range(num_samples):
        series = np.zeros(seq_len)
        num_steps = np.random.randint(2, 5)
        step_points = np.sort(np.random.choice(seq_len, num_steps, replace=False))
        current_level = 0
        for j in range(num_steps):
            current_level += np.random.uniform(-1, 1)
            series[step_points[j]:] = current_level
        series_list.append(series)
        labels.append(3)
    
    # Сохраняем датасет в формате, совместимом с Synthetic_dataset
    data = {
        'series': [s.tolist() for s in series_list],
        'labels': labels,
        'configs': {
            1: {'type': 'sine'},
            2: {'type': 'linear'},
            3: {'type': 'step'}
        },
        'metadata': {
            'num_classes': 3,
            'series_per_class': num_samples,
            'num_blocks': 1,
            'block_length': seq_len,
            'seq_len': seq_len,  # Добавляем явное указание длины последовательности
            'description': 'Toy dataset with simple patterns'
        }
    }
    
    Path('data').mkdir(exist_ok=True)
    with open('data/toy_dataset.json', 'w') as f:
        json.dump(data, f)
    
    print("Toy dataset generated and saved!")

if __name__ == '__main__':
    generate_simple_series() 