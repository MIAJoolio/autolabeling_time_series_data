"""
Демонстрация работы функций по созданию составных рядов и созданию синтетических датасетов из скрипта src.generation.ts_datasets.

#TODO описать результаты

"""

from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from src.generation import *
from src.utils import *

def show_Basic_generator_modes():
    """
    Иллюстрация работы класса Basic_generator.
    Класс поддерживает:
      * Ручное задание параметров.
      * Автоматическую генерацию параметров при None параметрах.
      * Настройку параметров через файл конфигурации.

    Также поддерживаются два режима генерации:
      * Генерация рядов с одинаковыми параметрами, но разным шумом.
      * Генерация рядов со случайными параметрами в рамках диапазонов.
    """

    # 1. Ручное задание параметров
    print("=== Режим 1: Ручное задание параметров ===")
    generator_manual = Basic_generator()
    generator_manual.add_block(
        block_length=50,
        ts_generator="linear",
        ts_params={"slope": 2},
        noise_generator="normal",
        noise_params={"noise_pct": 0.05}
    )
    series_manual = generator_manual.generate(random_state=42)
    print("Сгенерированный ряд (ручные параметры):", series_manual[:10], "...")
    lbl = 'manual generation.png'
    plot_series(series_manual, lbl, save_path=Path('data/demo_dataset_generation') / lbl)

    # 2. Автоматическая генерация параметров при None
    print("\n=== Режим 2: Автоматическая генерация параметров ===")
    generator_auto = Basic_generator()
    generator_auto.add_block(
        block_length=50,
        ts_generator="linear",
        ts_params=None,  # Параметры будут сгенерированы автоматически
        noise_generator="normal",
        noise_params=None  # Параметры будут сгенерированы автоматически
    )
    series_auto = generator_auto.generate(random_state=42)
    print("Сгенерированный ряд (автоматические параметры):", series_auto[:10], "...")
    lbl = 'auto_generation.png'
    plot_series(series_auto, lbl, save_path=Path('data/demo_dataset_generation') / lbl)

    # 3. Настройка параметров через файл конфигурации
    print("\n=== Режим 3: Настройка параметров через YAML-файл ===")

    generator_config = Basic_generator(config_path='configs/scripts/demo_dataset_generation/basic_generator_config.yaml')
    series_config = generator_config.generate(random_state=42)
    print("Сгенерированный ряд (параметры из YAML):", series_config[:10], "...")
    lbl = 'config_generation.png'
    plot_series(series_config, lbl, save_path=Path('data/demo_dataset_generation') / lbl)

    same_diff_noise,lbls = [],[]
    for i in range(5):
        series = generator_config.generate(mode='determinated')
        same_diff_noise.append(series)
        lbls.append(i)
    
    plot_series(same_diff_noise, lbls, plot_title='mode=determinated', save_path=Path('data/demo_dataset_generation')/ 'mode_determinated.png' )

    random_series = generator_config.generate_multiple(5)
    plot_series(random_series, lbls, plot_title='mode=random', save_path=Path('data/demo_dataset_generation')/ 'mode_random.png' )

    
def show_noise_addition(show_one=False):
    """
    Процесс добавления шума
    """
    generator = Basic_generator()
    length = 100
    
    # блок без шума
    generator.add_block(length, linear_trend, linear_trend_params(random_state=2))
    # блок с шумом
    generator.add_block(length, linear_trend, linear_trend_params(random_state=2), normal_noise, normal_noise_params(noise_pct_d=0.1, noise_pct_u=0.2, noise_pct_q=10, random_state=2))
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
        plot_series(ts1, [lbl1], save_path='/root/autolabeling_time_series_data/data/demo_dataset_generation/pic1.jpg')
        plot_series(ts2, [lbl2], save_path='/root/autolabeling_time_series_data/data/demo_dataset_generation/pic2.jpg')
        plot_series(ts3, [lbl3], save_path='/root/autolabeling_time_series_data/data/demo_dataset_generation/pic3.jpg')
    else:
        plot_series_grid([ts1, ts2, ts3], [lbl1, lbl2, lbl3], layout='horizontal', save_path='/root/autolabeling_time_series_data/data/demo_dataset_generation/all_pic.jpg')

    
# """
# Функции генерации
# """

# def linear_generator1(config_path):
#     """
#     Линейный тренд с slope в диапазоне [1, 5], уровень шума [0, 0.05]
#     """
#     length = 100
#     config = load_config_file(config_path)
    
#     generator = Basic_generator()
    
#     for block in config['blocks']:
#         # блок без шума
#         generator.add_block(length, block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))
    
#     return generator


# def linear_generator2(config_path):
#     """
#     Линейный тренд с slope в диапазоне [1, 5], уровень шума [0, 0.05]
#     """
#     length = 100
#     config = load_config_file(config_path)
    
#     generator = Basic_generator()
    
#     for block in config['blocks']:
#         # блок без шума
#         generator.add_block(length, block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))
    
#     return generator

# """
# Создаем функции генераторы для датасетов
# """

# def linear_dataset_generation(generators:Any, repeats:List[int]=[100,100]):
#     """
#     """
    
#     result_dict = []
#     for inx, q in enumerate(repeats):
#         print(generators[inx].blocks_length)
#         result_dict.append({
#             'data':[generators[inx].generate() for _ in range(q)],
#             'q_time_series':q, 
#             'segments':[generators[inx].blocks_length]*len(generators[inx].blocks)
#         })
            
#     return result_dict     

# def show_dataset_generation():
#     # генерация датасета 
#     generators = [linear_generator1('configs/scripts/demo_dataset_generation/linear_generator1.yaml'), linear_generator2('configs/scripts/demo_dataset_generation/linear_generator2.yaml')]
    
#     res = linear_dataset_generation(generators)
#     lbls = ['row'+str(m) for m in range(5)]

#     for i, lt in enumerate(res):
#         cluster = [res[i]['data'][m] for m in range(5)]
#         plot_series_grid(cluster,lbls,plot_title=f'Cluster {i}', save_path=f'/root/autolabeling_time_series_data/data/demo_dataset_generation/Cluster_{i}_gen1.jpg')

#         print(res[i]['q_time_series'], res[i]['segments'])

    
# def linear_dataset_generation2(configs, repeats=[100, 100]):
#     """
#     1. Создаешь файл с параметрами генераторов
#     2. Создаешь правило по размерности данных и длины + фиксируется информация по тому как создавались сегменты 
#     3. Создаешь правило по разметке
#     """
    
#     result_dict = [{'data':[],'q_time_series':[], 'segments':[]} for _ in range(2)]
    
#     for inx, (config_path, q) in enumerate(zip(configs, repeats)):
        
#         config = load_config_file(config_path)
        
#         for _ in range(q):
            
#             generator = Basic_generator()
#             length=100
            
#             for block in config['blocks']:
#                 generator.add_block(length, block['ts_generator'], linear_trend_params(**block['ts_params']), block['noise_generator'], normal_noise_params(**block['noise_params']))

#             if result_dict[inx]['segments'] == []:
#                 result_dict[inx]['segments'] = [generator.blocks_length[inx]]*len(generator.blocks)
                
#             result_dict[inx]['data'].append(generator.generate())

#             for t in range(len(config['blocks'])):
#                 generator.remove_block(t)
        
#         result_dict[inx]['q_time_series'] = q
    
#     return result_dict

# def show_dataset_generation2():
#     # генерация датасета 
    
#     configs = ['configs/scripts/demo_dataset_generation/linear_generator1.yaml', 'configs/scripts/demo_dataset_generation/linear_generator2.yaml']

#     res = linear_dataset_generation2(configs)
#     lbls = ['row'+str(m) for m in range(5)]

#     for i, lt in enumerate(res):
#         cluster = [res[i]['data'][m] for m in range(5)]
#         plot_series_grid(cluster,lbls,plot_title=f'Cluster {i}', save_path=f'/root/autolabeling_time_series_data/data/demo_dataset_generation/Cluster_{i}_gen2.jpg')

#         print(res[i]['q_time_series'], res[i]['segments'])

#     return res

def main():
    show_Basic_generator_modes()
    # show_noise_addition(True)
    
    # show_dataset_generation()
    
    # dt2 = show_dataset_generation2()
    
    # save_generated_data(dt2, 'data/demo_dataset_generation/dt2.json') 
    
    # # ts_dt = Time_series_dataset('data/demo_dataset_generation/dt2.json', normalize=False)
    # ts_dt = Time_series_dataset('data/demo_dataset_generation/dt2.json', normalize=True, norm_type='minmax')
    
    # series, lbl = ts_dt[0]
    # plot_series(series, [lbl])
    
    # # Определяем размеры выборок
    # train_size = int(0.8 * len(ts_dt))  # 80% для обучения
    # val_size = len(ts_dt) - train_size  # 20% для валидации

    # # Разбиваем датасет
    # train_dataset, val_dataset = random_split(ts_dt, [train_size, val_size])

    # # Теперь можно создавать DataLoader для каждой выборки
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # for (train_row, train_lbl), (val_row, val_lbl) in zip(train_loader, val_loader):
    #     plot_series([train_row.numpy()[0], val_row.numpy()[0]], [train_lbl[0].numpy(), val_lbl[0].numpy()])
    #     break
    
    
if __name__ == '__main__':
    main()