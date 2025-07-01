"""
Script to generate synthetic small sample for testing
"""

# external
from typing import Tuple, Any, Literal
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

# internal
from src.generation.ts_generators import *  
from src.generation.ts_noise_generators import normal_noise  
from src.generation.ts_creation import *
from src.utils import plot_series, load_config_file, save_config_file


def linear_generator(n_cluster:int=2, length:int=100, param_name:str='slope'):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(linear_trend, 'src/generation/base_configs/trend/basic_linear_trend.yaml')
    generator.config[f'{param_name}_q'] = n_cluster
    generator.config['length'] = length

    return generator


def quad_generator(n_cluster:int=2, length:int=100, param_name:str='a'):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(quadratic_trend, 'src/generation/base_configs/trend/basic_quadratic_trend.yaml')
    generator.config[f'{param_name}_q'] = n_cluster
    generator.config['length'] = length

    return generator


def exp_generator(n_cluster:int=2, length:int=100, param_name:str='alpha'):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(exponential_trend, 'src/generation/base_configs/trend/basic_exponential_trend.yaml')
    generator.config[f'{param_name}_q'] = n_cluster
    generator.config['length'] = length

    return generator

# функции сезонности
def sin_wave_generator(n_cluster:int=2, length:int=100,param_name:str='frequency'):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(sin_wave, 'src/generation/base_configs/wave/basic_sin_wave.yaml')
    generator.config[f'{param_name}_q'] = n_cluster
    generator.config['length'] = length

    return generator


def saw_generator(n_cluster:int=2, length:int=100,param_name:str='frequency'):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(sawtooth_wave, 'src/generation/base_configs/wave/basic_saw_wave.yaml')
    generator.config[f'{param_name}_q'] = n_cluster
    generator.config['length'] = length

    return generator
    
# функции сдвигов

def harmonic_shift_generator(length:int=100):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(harmonic_shift, 'configs/shifts/basic_harmonic_shift.yaml')

    # generator.config['frequency_q'] = 3 
    
    # generator.config['damping_d'] = 0.05
    # generator.config['damping_d'] = 0.2
    # generator.config['damping_q'] = 10
    
    generator.config['t_peak_d'] = int(0.2*length)
    generator.config['t_peak_u'] = int(0.8*length)
    generator.config['t_peak_q'] = int(length / 10)
    generator.config['length'] = length
     
    return generator

def saw_shift_generator(length:int=100):
    """
    Функция-генератор для линейного тренда с изменением кол-ва кластеров 
    """
    generator = TS_generator(sawtooth_shift, 'src/generation/base_configs/shift/basic_sawtooth_shift.yaml')

    # generator.config['frequency_d'] = 2
    # generator.config['frequency_u'] = 2
    # generator.config['frequency_q'] = int(length / 10) 
        
    return generator


# функция шума
def apply_noise(rows:Any, noise_int:int, freq:int=100):
    """
    Функция создает генератор шума
    """
    noise_gen = Noise_generator(rows, normal_noise, 'configs/noise/basic_normal_noise.yaml')
    noise_gen.generate_params()
    noise_gen.config['noise_pct_d'] = noise_int[0]
    noise_gen.config['noise_pct_u'] = noise_int[1]
    noise_gen.config['noise_pct_q'] = int(freq / 10)
    return noise_gen


def create_dist_params(config_path:str, return_generator:bool = True):
    """
    Функция генерирует сетку параметров для тестирования на синтетических данных согласно таблице 7 раздела 3.3
    """
    params = load_config_file(config_path)    

    # Генерация всех комбинаций
    param_grid = ParameterGrid(params)

    if return_generator:
        return param_grid

    return [grid for grid in param_grid]


def apply_scale(rows:Any, scale:Tuple[float, float]):
    """
    Нормирует все функции в этом диапазоне
    """
    if not isinstance(rows, np.ndarray):
        rows = np.asarray(rows)
    
    min_val, max_val = rows.min(), rows.max()
    
    return np.asarray([scale[0] + (row - min_val) * (scale[1] - scale[0]) / (max_val - min_val) for row in rows]) 


def generate_label(label):
    """
    
    """
    return '_'.join(label[0])


def get_labels(labels):
    """
    
    """
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    return labels_encoded


def create_dist(ts_gen:TS_generator, params: dict, shift_gen:TS_generator=None, show_pic:bool=False, save_path:str='data/Synthetic_data/example_generation/pic.png'):
    """
    Генерация датасетов с помощью  
    """
    rows,_ = ts_gen.generate()
    noise_gen = apply_noise(rows, params['noise_level'], params['sample_size']) 
    dist_gen = TS_merger()
    
    sample = []
    labels = []
    for _ in range(params['sample_size']):
        
        shift_gen_cicle = shift_gen
        # добавляем случайности добавлению структурного сдвига аномалий
        if params['shifts'] == 0:
            shift_gen_cicle = None
        if params['shifts'] == 1:
            coin = np.random.random()
            if coin < 0.5:
                shift_gen_cicle = None

        block1 = TS_block(trend=ts_gen, shifts=shift_gen_cicle, noise=noise_gen, length=params['ts_length'])
        dist_gen.add_block(block1)
        generation_res = dist_gen.generate()
        dist_gen.remove_block(0)
    
        sample.append(generation_res[0])
        labels.append(generate_label(generation_res[1]))
    
    sample_scaled = apply_scale(sample, params['value_scale'])
    labels_encoded = get_labels(labels)
    
    if show_pic:
        plot_series([series for series in sample_scaled], show_legend=False, save_path=save_path)

    return sample_scaled, labels_encoded


def sample_stats(X,y):
    print(X.shape)
    values, counts = np.unique(y, return_counts=True)
    frequencies = counts / len(y)
    for value, freq in zip(values, frequencies):
        print(f"{value} - {freq:.2f}")
        
        
def main():
    
    params_grid = {
        'noise_level': [0, 0.05],
        'sample_size': 50,
        'shifts': 0,
        'value_scale': [100, 200],
        'ts_length': 101,
        'n_clusters': 2
    }
    
    harmonic_shift = harmonic_shift_generator(params_grid['ts_length'])

    generator = linear_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/linear_trend.png')
    sample_stats(sample1, labels1)
    
    generator = quad_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/quad_trend.png')
    sample_stats(sample1, labels1)

    generator = exp_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/exponential_trend.png')
    sample_stats(sample1, labels1)
    
    generator = sin_wave_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/sin_wave.png')
    sample_stats(sample1, labels1)
    
    generator = saw_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/saw_wave.png')
    sample_stats(sample1, labels1)
    
    params_grid['shifts'] = 1
    
    generator = linear_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/linear_trend_with_shifts.png')
    sample_stats(sample1, labels1)
    
    generator = quad_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/quad_trend_with_shifts.png')
    sample_stats(sample1, labels1)

    generator = exp_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/exponential_trend_with_shifts.png')
    sample_stats(sample1, labels1)
    
    generator = sin_wave_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/sin_wave_with_shifts.png')
    sample_stats(sample1, labels1)
    
    generator = saw_generator(params_grid['n_clusters'], params_grid['ts_length'])
    sample1, labels1 = create_dist(generator, params_grid, harmonic_shift, True, 'data/Synthetic_data/example_generation/saw_wave_with_shifts.png')
    sample_stats(sample1, labels1)
    
if __name__ == '__main__':
    main()