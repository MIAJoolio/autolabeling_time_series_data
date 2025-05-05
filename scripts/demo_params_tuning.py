"""
Скрипт демонстрирует как можно настроить параметры и вид распределений из скрипта ts_generators.py модуля src.generation. Для удобства распределения были разбиты на 3 группы 

#TODO описать результаты
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.generation.ts_generators import *
from src.utils import *

# Директория для сохранения графиков
SAVE_PATH = Path('data/demo_params_tuning')
# Длина рядов
TS_LENGTH = 100


def show_trend_funcs(config_path: Path = 'configs/scripts/demo_params_tuning/trends.yaml'):
    """
    Функция по визуализации работы функций генерации временных рядов с трендом.
    """
    # Читаем файл конфигурации
    config_file = load_config_file(Path(config_path))
    general = config_file['general']
    linear_params = config_file['specific']['linear']
    quadratic_params = config_file['specific']['quadratic']
    exp_params = config_file['specific']['exponential']

    # Влияние k на функции генерации
    for length in general['length']:
        # Линейный тренд
        ts_linear = [linear_trend(**linear_trend_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_linear_lbl = [f'linear k = {k}' for k in general['k']]
        plot_series(ts_linear, ts_linear_lbl, save_path=(SAVE_PATH / f'trend/linear/k_range_len_{length}.png'))

        # Квадратичный тренд
        ts_quadratic = [quadratic_trend(**quadratic_trend_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_quadratic_lbl = [f'quadratic k = {k}' for k in general['k']]
        plot_series(ts_quadratic, ts_quadratic_lbl, save_path=(SAVE_PATH / f'trend/quadratic/k_range_len_{length}.png'))

        # Экспоненциальный тренд
        ts_exp = [exponential_trend(**exponential_trend_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_exp_lbl = [f'exponential k = {k}' for k in general['k']]
        plot_series(ts_exp, ts_exp_lbl, save_path=(SAVE_PATH / f'trend/exponential/k_range_len{length}.png'))
        
            
    # Линейный тренд
    all_linear_params = linear_trend_params(**linear_params, all_values=True)
    ts_linear = [linear_trend(slope=slope, length=50) for slope in all_linear_params['slope']]
    ts_linear_lbl = [f'slope = {slope}' for slope in all_linear_params['slope']]
    plot_series(ts_linear, ts_linear_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'trend/linear/linear_slope_range.png'))

    # Квадратичный тренд
    all_quadratic_params = quadratic_trend_params(**quadratic_params, all_values=True)

    fixed_a, fixed_b, fixed_c = np.mean(all_quadratic_params['a']),np.mean(all_quadratic_params['b']), np.mean(all_quadratic_params['c'])
    
    # Варьируем a, фиксируя b и c
    ts_a = [quadratic_trend(a=a, b=fixed_b, c=fixed_c, length=50) for a in all_quadratic_params['a']]
    ts_a_lbl = [f'a = {a}, b = {fixed_b}, c = {fixed_c}' for a in all_quadratic_params['a']]
    plot_series(ts_a, ts_a_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'trend/quadratic/quadratic_a_range.png'))

    # Варьируем b, фиксируя a и c
    ts_b = [quadratic_trend(a=fixed_a, b=b, c=fixed_c, length=50) for b in all_quadratic_params['b']]
    ts_b_lbl = [f'a = {fixed_a}, b = {b}, c = {fixed_c}' for b in all_quadratic_params['b']]
    plot_series(ts_b, ts_b_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'trend/quadratic/quadratic_b_range.png'))

    # Варьируем c, фиксируя a и b
    ts_c = [quadratic_trend(a=fixed_a, b=fixed_b, c=c, length=50) for c in all_quadratic_params['c']]
    ts_c_lbl = [f'a = {fixed_a}, b = {fixed_b}, c = {c}' for c in all_quadratic_params['c']]
    plot_series(ts_c, ts_c_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'trend/quadratic/quadratic_c_range.png'))

    # Экспоненциальный тренд
    all_exp_params = exponential_trend_params(**exp_params, all_values=True)
    ts_exp = [exponential_trend(alpha=alpha, length=50) for alpha in all_exp_params['alpha']]
    ts_exp_lbl = [f'alpha = {alpha}' for alpha in all_exp_params['alpha']]
    plot_series(ts_exp, ts_exp_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'trend/exponential/exponential_alpha_range.png'))

def show_periodic_funcs(config_path: Path = 'configs/scripts/demo_params_tuning/periodic.yaml'):
    """
    Функция по визуализации работы функций генерации временных рядов с сезонностью/периодичностью.
    """
    # Читаем файл конфигурации
    config_file = load_config_file(Path(config_path))
    general = config_file['general']
    sawtooth_params = config_file['specific']['sawtooth']
    seasonal_params = config_file['specific']['seasonal']
    harmonic_params = config_file['specific']['harmonic']

    # Влияние k на функции генерации
    for length in general['length']:
        # Пилообразный сигнал
        ts_sawtooth = [sawtooth_wave(**sawtooth_wave_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_sawtooth_lbl = [f'sawtooth k = {k}' for k in general['k']]
        plot_series(ts_sawtooth, ts_sawtooth_lbl, save_path=(SAVE_PATH / f'periodic/sawtooth/k_range_len{length}.png'))

        # Сезонность
        ts_seasonal = [seasonal_series(**seasonal_series_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_seasonal_lbl = [f'seasonal k = {k}' for k in general['k']]
        plot_series(ts_seasonal, ts_seasonal_lbl, save_path=(SAVE_PATH / f'periodic/seasonal/k_range_len{length}.png'))

        # Гармонический осциллятор
        ts_harmonic = [harmonic_oscillator(**harmonic_oscillator_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_harmonic_lbl = [f'harmonic k = {k}' for k in general['k']]
        plot_series(ts_harmonic, ts_harmonic_lbl, save_path=(SAVE_PATH / f'periodic/harmonic/k_range_len{length}.png'))
        
    # Пилообразный сигнал
    all_sawtooth_params = sawtooth_wave_params(**sawtooth_params, all_values=True)

    fixed_freq, fixed_amp = np.mean(all_sawtooth_params['frequency']), np.mean(all_sawtooth_params['amplitude'])
    # Варьируем amplitude, фиксируя frequency    
    ts_amp = [sawtooth_wave(amplitude=amp, frequency=fixed_freq, length=50) for amp in all_sawtooth_params['amplitude']]
    ts_amp_lbl = [f'amplitude = {amp}, frequency = {fixed_freq}' for amp in all_sawtooth_params['amplitude']]
    plot_series(ts_amp, ts_amp_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/sawtooth/sawtooth_amplitude_range.png'))

    # Варьируем frequency, фиксируя amplitude
    ts_freq = [sawtooth_wave(amplitude=fixed_amp, frequency=freq, length=50) for freq in all_sawtooth_params['frequency']]
    ts_freq_lbl = [f'amplitude = {fixed_amp}, frequency = {freq}' for freq in all_sawtooth_params['frequency']]
    plot_series(ts_freq, ts_freq_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/sawtooth/sawtooth_frequency_range.png'))

    # Сезонность
    all_seasonal_params = seasonal_series_params(**seasonal_params, all_values=True)
    fixed_amp, fixed_freq, fixed_phase = np.mean(all_seasonal_params['amplitude']), np.mean(all_seasonal_params['frequency']), np.mean(all_seasonal_params['phase'])
    
    # Варьируем amplitude, фиксируя frequency и phase
    ts_amp = [seasonal_series(amplitude=amp, frequency=fixed_freq, phase=fixed_phase, length=50) for amp in all_seasonal_params['amplitude']]
    ts_amp_lbl = [f'amplitude = {amp}, frequency = {fixed_freq}, phase = {fixed_phase}' for amp in all_seasonal_params['amplitude']]
    plot_series(ts_amp, ts_amp_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/seasonal/seasonal_amplitude_range.png'))

    # Варьируем frequency, фиксируя amplitude и phase
    ts_freq = [seasonal_series(amplitude=fixed_amp, frequency=freq, phase=fixed_phase, length=50) for freq in all_seasonal_params['frequency']]
    ts_freq_lbl = [f'amplitude = {fixed_amp}, frequency = {freq}, phase = {fixed_phase}' for freq in all_seasonal_params['frequency']]
    plot_series(ts_freq, ts_freq_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/seasonal/seasonal_frequency_range.png'))

    # Варьируем phase, фиксируя amplitude и frequency
    ts_phase = [seasonal_series(amplitude=fixed_amp, frequency=fixed_freq, phase=phase, length=50) for phase in all_seasonal_params['phase']]
    ts_phase_lbl = [f'amplitude = {fixed_amp}, frequency = {fixed_freq}, phase = {phase}' for phase in all_seasonal_params['phase']]
    plot_series(ts_phase, ts_phase_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/seasonal/seasonal_phase_range.png'))

    # Гармонический осциллятор
    all_harmonic_params = harmonic_oscillator_params(**harmonic_params, all_values=True)
    fixed_amp, fixed_freq, fixed_damping = np.mean(all_harmonic_params['amplitude']), np.mean(all_harmonic_params['frequency']), np.mean(all_harmonic_params['damping'])
    
    # Варьируем amplitude, фиксируя frequency и damping
    ts_amp = [harmonic_oscillator(amplitude=amp, frequency=fixed_freq, damping=fixed_damping, length=50) for amp in all_harmonic_params['amplitude']]
    ts_amp_lbl = [f'amplitude = {amp}, frequency = {fixed_freq}, damping = {fixed_damping}' for amp in all_harmonic_params['amplitude']]
    plot_series(ts_amp, ts_amp_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/harmonic/harmonic_amplitude_range.png'))

    # Варьируем frequency, фиксируя amplitude и damping
    ts_freq = [harmonic_oscillator(amplitude=fixed_amp, frequency=freq, damping=fixed_damping, length=50) for freq in all_harmonic_params['frequency']]
    ts_freq_lbl = [f'amplitude = {fixed_amp}, frequency = {freq}, damping = {fixed_damping}' for freq in all_harmonic_params['frequency']]
    plot_series(ts_freq, ts_freq_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/harmonic/harmonic_frequency_range.png'))

    # Варьируем damping, фиксируя amplitude и frequency
    ts_damping = [harmonic_oscillator(amplitude=fixed_amp, frequency=fixed_freq, damping=damp, length=50) for damp in all_harmonic_params['damping']]
    ts_damping_lbl = [f'amplitude = {fixed_amp}, frequency = {fixed_freq}, damping = {damp}' for damp in all_harmonic_params['damping']]
    plot_series(ts_damping, ts_damping_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'periodic/harmonic/harmonic_damping_range.png'))

def show_unstructured_funcs(config_path: Path = 'configs/scripts/demo_params_tuning/unstructured.yaml'):
    """
    Функция по визуализации работы функций генерации временных рядов со структурными сдвигами или просто не имеющих структуры.
    """
    # Читаем файл конфигурации
    config_file = load_config_file(Path(config_path))
    general = config_file['general']
    random_walk_prms = config_file['specific']['random_walk']

    # Влияние k на функции генерации
    for length in general['length']:
        # Гармонический осциллятор
        ts_harmonic = [harmonic_oscillator(**harmonic_oscillator_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_harmonic_lbl = [f'harmonic k = {k}' for k in general['k']]
        plot_series(ts_harmonic, ts_harmonic_lbl, save_path=(SAVE_PATH / f'unstr/harmonic/k_range_len{length}.png'))

        # Случайное блуждание
        ts_random_walk = [random_walk(**random_walk_params(k=k, random_state=general['random_state']), length=length) for k in general['k']]
        ts_random_walk_lbl = [f'random_walk k = {k}' for k in general['k']]
        plot_series(ts_random_walk, ts_random_walk_lbl, save_path=(SAVE_PATH / f'unstr/random_walk/k_range_len{length}.png'))
            
    # Случайное блуждание
    all_random_walk_params = random_walk_params(**random_walk_prms, all_values=True)  
    fixed_noise_std, fixed_init_val = np.mean(all_random_walk_params['noise_std']), np.mean(all_random_walk_params['initial_value'])
    
    # Варьируем initial_value, фиксируя noise_std
    ts_init_val = [random_walk(initial_value=init_val, noise_std=fixed_noise_std, length=50) for init_val in all_random_walk_params['initial_value']]
    ts_init_val_lbl = [f'initial_value = {init_val}, noise_std = {fixed_noise_std}' for init_val in all_random_walk_params['initial_value']]
    plot_series(ts_init_val, ts_init_val_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'unstr/random_walk/random_walk_initial_value_range.png'))

    # Варьируем noise_std, фиксируя initial_value
    ts_noise_std = [random_walk(initial_value=fixed_init_val, noise_std=noise_std, length=50) for noise_std in all_random_walk_params['noise_std']]
    ts_noise_std_lbl = [f'initial_value = {fixed_init_val}, noise_std = {noise_std}' for noise_std in all_random_walk_params['noise_std']]
    plot_series(ts_noise_std, ts_noise_std_lbl, figsize=(20, 10), save_path=(SAVE_PATH / 'unstr/random_walk/random_walk_noise_std_range.png'))

if __name__ == "__main__":
    show_trend_funcs()
    show_periodic_funcs()
    show_unstructured_funcs()