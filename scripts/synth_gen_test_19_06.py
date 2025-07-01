from pathlib import Path
import numpy as np

from scripts.synthetic_generators import *
from src.generation.ts_creation import *
from src.generation.ts_generators import *
from src.generation.ts_noise_generators import *
from src.utils.visuals import *


def gen_lin_trend(sample_name:str='lin_trend1'):
    gen_lin1 = linear_generator(10, 100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['slope'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_quad_trend(sample_name:str = 'quad_trend1'):
    gen_lin1 = quad_generator(10, 100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['a'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_exp_trend(sample_name:str = 'exp_trend1'):
    gen_lin1 = exp_generator(10, 100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['alpha'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_sin_wave(sample_name:str = 'sin_wave1'):
    gen_lin1 = sin_wave_generator(7, 100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['frequency'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_saw_wave(sample_name:str = 'saw_wave1'):
    gen_lin1 = saw_generator(7, 100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['frequency'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_harmonic_shift(sample_name:str = 'harmonic_shift1'):
    gen_lin1 = harmonic_shift_generator(100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['t_peak'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def gen_saw_shift(sample_name:str = 'saw_shift1'):
    gen_lin1 = saw_shift_generator(100)
    # print(gen_lin1.config)
    data = []
    lbls = []
    for i in range(10):
        row, lbl = gen_lin1.generate()
        data.append(row)
        lbls.append(lbl['frequency'])
    plot_series(data, lbls, save_path=f'data/Synthetic_data/Small_test_sample/{sample_name}.png')
    new_lbls = get_labels(lbls)
    print(new_lbls)
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_data.npy' ,np.asarray(data))
    np.save(f'data/Synthetic_data/Small_test_sample/{sample_name}_labels.npy' ,np.asarray(new_lbls))


def get_data(filepath:str):
    if filepath is None:
        return None
    
    data = np.load(filepath)
    inx = np.random.randint(0, data.shape[0])
    return data[inx]


def create_test_df(filepaths:dict, sample_num:int=100):
    
    data = []
    for _ in range(sample_num):
        block = TS_block({key: get_data(filepaths.get(key)) for key in ['trend', 'wave','shift']}, length=100)
        
        data.append(block)

    return data


def generate_small_sample_content():
    """
    Create distribution sample with random distribution of parameters
    """
    np.random.seed(21)
    # generate trends
    gen_lin_trend()
    gen_quad_trend()
    gen_exp_trend()
    # generate waves
    gen_sin_wave()
    gen_saw_wave()
    # generate shifts
    gen_harmonic_shift()
    gen_saw_shift()
    

def main():
    #
    generate_small_sample_content()
    #
    filepaths = {
        'trend':'data/Synthetic_data/Small_test_sample/exp_trend1_data.npy',
        'wave': 'data/Synthetic_data/Small_test_sample/sin_wave1_data.npy',
        }
    dt = create_test_df(filepaths, 20)
    plot_series([item.block() for item in dt], show_legend=False,save_path='data/Synthetic_data/Small_test_sample/create_test_df_1.png')
    

if __name__ == '__main__':
    main()