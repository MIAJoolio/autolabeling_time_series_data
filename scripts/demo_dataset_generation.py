"""
Демонстрация работы функций по созданию составных рядов и созданию синтетических датасетов из скрипта src.generation.ts_datasets.

#TODO описать результаты

"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from torch.utils.data import random_split, DataLoader

from src.generation import *
from src.utils import *

def show_Time_series_generator_works(show_one=False):
    
    generator = Time_series_generator(block_length=100)
    
    # блок без шума
    generator.add_block(linear_trend, linear_trend_params(random_state=2))
    # блок с шумом
    generator.add_block(linear_trend, linear_trend_params(random_state=2), normal_noise, normal_noise_params(noise_pct_d=0.1, noise_pct_u=0.2, noise_pct_q=10, random_state=2))
    ts1 = generator.generate()
    lbl1 = 'Составной ряд'
    
    # изменение параметров
    new_params = linear_trend_params(random_state=2)
    new_params['slope'] *= -1
    new_noise = normal_noise_params(noise_pct_d=0.1, noise_pct_u=0.2, noise_pct_q=10, random_state=2)
    new_noise['noise_pct'] = 0.05
    generator.update_block_params(1, new_params, new_noise)
    ts2 = generator.generate()
    lbl2 = 'Составной ряд после изменения параметров и размера шума'
    # уменьшение шума
    new_noise['noise_pct'] = 0.01
    generator.update_block_params(1, new_params, new_noise)
    ts3 = generator.generate()
    lbl3 = 'Составной ряд после изменения шума с 0.05 до 0.01'
    
    if not show_one:
        plot_series(ts1, [lbl1])
        plot_series(ts2, [lbl2])
        plot_series(ts3, [lbl3])
    else:
        plot_series_grid([ts1, ts2, ts3], [lbl1, lbl2, lbl3], layout='horizontal')

"""
Функции генерации
"""

def linear_generator1(config_path):
    """
    Линейный тренд с slope в диапазоне [1, 5], уровень шума [0, 0.05]
    """
    config = load_config_file(config_path)
    
    generator = Time_series_generator(block_length=100)
    
    for block in config['blocks']:
        # блок без шума
        generator.add_block(block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))
    
    return generator


def linear_generator2(config_path):
    """
    Линейный тренд с slope в диапазоне [1, 5], уровень шума [0, 0.05]
    """
    config = load_config_file(config_path)
    
    generator = Time_series_generator(block_length=100)
    
    for block in config['blocks']:
        # блок без шума
        generator.add_block(block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))
    
    return generator

"""
Создаем функции генераторы для датасетов
"""

def linear_dataset_generation(generators:Any, repeats:List[int]=[100,100]):
    """
    """
    
    result_dict = []
    for inx, q in enumerate(repeats):
        print(generators[inx].block_length)
        result_dict.append({
            'data':[generators[inx].generate() for _ in range(q)],
            'q_time_series':q, 
            'segments':[generators[inx].block_length]*len(generators[inx].blocks)
        })
            
    return result_dict     

def show_dataset_generation():
    # генерация датасета 
    generators = [linear_generator1('configs/scripts/demo_dataset_generation/linear_generator1.yaml'), linear_generator2('configs/scripts/demo_dataset_generation/linear_generator2.yaml')]
    
    res = linear_dataset_generation(generators)
    lbls = ['row'+str(m) for m in range(5)]

    for i, lt in enumerate(res):
        cluster = [res[i]['data'][m] for m in range(5)]
        plot_series_grid(cluster,lbls,plot_title=f'Cluster {i}')

        print(res[i]['q_time_series'], res[i]['segments'])

    
def linear_dataset_generation2(configs, repeats=[100, 100]):
    """
    1. Создаешь файл с параметрами генераторов
    2. Создаешь правило по размерности данных и длины + фиксируется информация по тому как создавались сегменты 
    3. Создаешь правило по разметке
    """
    
    result_dict = [{'data':[],'q_time_series':[], 'segments':[]} for _ in range(2)]
    
    for inx, (config_path, q) in enumerate(zip(configs, repeats)):
        
        config = load_config_file(config_path)
        
        for _ in range(q):
            
            generator = Time_series_generator(block_length=100)
            
            for block in config['blocks']:
                generator.add_block(block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))

            if result_dict[inx]['segments'] == []:
                result_dict[inx]['segments'] = [generator.block_length]*len(generator.blocks)
                
            result_dict[inx]['data'].append(generator.generate())

            for t in range(len(config['blocks'])):
                generator.remove_block(t)
        
        result_dict[inx]['q_time_series'] = q
    
    return result_dict

def show_dataset_generation2():
    # генерация датасета 
    
    configs = ['configs/scripts/demo_dataset_generation/linear_generator1.yaml', 'configs/scripts/demo_dataset_generation/linear_generator2.yaml']

    res = linear_dataset_generation2(configs)
    lbls = ['row'+str(m) for m in range(5)]

    for i, lt in enumerate(res):
        cluster = [res[i]['data'][m] for m in range(5)]
        plot_series_grid(cluster,lbls,plot_title=f'Cluster {i}')

        print(res[i]['q_time_series'], res[i]['segments'])

    return res

def main():
    # show_Time_series_generator_works(True)
    
    # show_dataset_generation()
    
    dt2 = show_dataset_generation2()
    
    save_generated_data(dt2, 'data/demo_dataset_generation/dt2.json') 
    
    # ts_dt = Time_series_dataset('data/demo_dataset_generation/dt2.json', normalize=False)
    ts_dt = Time_series_dataset('data/demo_dataset_generation/dt2.json', normalize=True, norm_type='minmax')
    
    series, lbl = ts_dt[0]
    plot_series(series, [lbl])
    
    # Определяем размеры выборок
    train_size = int(0.8 * len(ts_dt))  # 80% для обучения
    val_size = len(ts_dt) - train_size  # 20% для валидации

    # Разбиваем датасет
    train_dataset, val_dataset = random_split(ts_dt, [train_size, val_size])

    # Теперь можно создавать DataLoader для каждой выборки
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for (train_row, train_lbl), (val_row, val_lbl) in zip(train_loader, val_loader):
        plot_series([train_row.numpy()[0], val_row.numpy()[0]], [train_lbl[0].numpy(), val_lbl[0].numpy()])
        break
    
    
if __name__ == '__main__':
    main()