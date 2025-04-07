from typing import List, Dict, Optional, Tuple, Union, Literal
from pathlib import Path
import json
import itertools

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from torch.utils.data import Dataset, DataLoader
from src.generation.ts_generators import (
    linear_trend, linear_trend_params,
    quadratic_trend, quadratic_trend_params,
    exponential_trend, exponential_trend_params,
    seasonal_series, seasonal_series_params,
    sawtooth_wave, sawtooth_wave_params,
    harmonic_oscillator, harmonic_oscillator_params,
    random_walk, random_walk_params
)

from src.generation.noise_generators import normal_noise, normal_noise_params
from src.utils import *

# добавим logger
logger = setup_logger(__name__, level='debug')

__all__ = [
    "Time_series_generator",
    "save_generated_data",
    "Time_series_dataset"
]   

class Time_series_generator:    
    """
    Класс для генерации временного ряда из блоков.
    """
    
    def __init__(self, block_length: Union[int, dict], generators:dict=None, n_generators:dict=None):
        # список для хранения блоков
        self.blocks = []
        self.block_length = block_length
        logger.info(f"Initialized Generator with block_length={block_length}")

        if generators is None:
            self.generators = {
                'linear': {
                    'generator':linear_trend,
                    'params_generator':linear_trend_params
                },
                'quadratic': {
                    'generator':quadratic_trend,
                    'params_generator':quadratic_trend_params
                },
                'exponential': {
                    'generator':exponential_trend,
                    'params_generator':exponential_trend_params
                },
                'seasonal': {
                    'generator':seasonal_series,
                    'params_generator':seasonal_series_params
                },
                'sawtooth': {
                    'generator':sawtooth_wave,
                    'params_generator':sawtooth_wave_params
                },
                'harmonic': {
                    'generator':harmonic_oscillator,
                    'params_generator':harmonic_oscillator_params
                },
                'random': {
                    'generator':random_walk,
                    'params_generator':random_walk_params
                }
            }
            logger.debug('Time-series function are initialized!')
        
        if n_generators is None:   
            self.n_generators = {
                'normal': {
                    'generator':normal_noise,
                    'params_generator':normal_noise_params
                }, 
            } 
            logger.debug('Noise function are initialized!')

    def add_block(self, ts_generator, ts_params=None, noise_generator=None, noise_params=None, random_state=None):
        """
        Функция для добавления блока генерации 
        """
        
        use_ts_generator, auto_ts_params = False, False
        use_noise_generator, auto_noise_params = False, False
        
        # Если задана не функция генератор, а её тип в формате str
        if isinstance(ts_generator, str):
            use_ts_generator = True
            logger.debug('Used internal generator by name from self.generators')
        
        # Если параметры не заданы и используется self.generators
        if ts_params is None and use_ts_generator:
            auto_ts_params = True
            logger.debug('Used autogeneration parameters function')
        
        # Аналогично для noise_generator
        if isinstance(noise_generator, str):
            use_noise_generator = True
            logger.debug('Used internal generator by name from self.n_generators')
        
        if noise_params is None and use_noise_generator:
            auto_noise_params = True
            logger.debug('Used autogeneration parameters noise function')
        
        # Добавление блока
        self.blocks.append(
            {
            'ts_generator': self.generators[ts_generator]['generator'] if use_ts_generator else ts_generator, 
            'ts_params': self.generators[ts_generator]['params_generator'](random_state=random_state) if auto_ts_params else ts_params,
            'noise_generator': self.n_generators[noise_generator]['generator'] if use_noise_generator else noise_generator, 
            'noise_params': self.n_generators[noise_generator]['params_generator'](random_state=random_state) if auto_noise_params else noise_params
            }
        )
        
        logger.debug(f'Added block:\n  time-series params {self.blocks[-1]["ts_params"]}\n  noise params {self.blocks[-1]["noise_params"]}')

    def remove_block(self, index: int):
        """
        Удаление блока по индексу.
        
        Args:
            index: Индекс блока для удаления.
        """
        if 0 <= index < len(self.blocks):
            removed_block = self.blocks.pop(index)
            logger.debug(f'Removed block at index {index}: {removed_block}')
        else:
            logger.warning(f'Attempted to remove block at invalid index {index}')
    
    def update_block_params(self, index: int, new_ts_params: Dict=None, new_noise_params: Dict=None):
        """
        Обновление параметров блока.
        
        Args:
            index: Индекс блока для обновления.
            new_params: Новые параметры.
        """
        if 0 <= index < len(self.blocks):
            if new_ts_params is not None:
                old_params = self.blocks[index]['ts_params']
                self.blocks[index]['ts_params'].update(new_ts_params)
                logger.debug(f'Updated block {index} time-series params:\n   {old_params} -> {new_ts_params}')
            if new_noise_params is not None:
                old_params = self.blocks[index]['noise_params']
                self.blocks[index]['noise_params'].update(new_noise_params)
                logger.debug(f'Updated block {index} noise params:\n   {old_params} -> {new_noise_params}')
        else:
            logger.warning(f'Attempted to update block at invalid index {index}')
    
    def generate(self, random_state:int=None) -> np.ndarray:
        """
        Генерация временного ряда из блоков.
        
        Args:
            with_noise: Если True, то генерация будет использовать параметры шума 
        
        Returns:
            np.ndarray: Сгенерированный временной ряд.
        """

        if not self.blocks:
            logger.error("No blocks available for generation")
            raise ValueError("Нет блоков для генерации")
        
        with logger.start_run("generate_time_series"):
            # инициализация полного ряда
            time_series = np.array([], dtype=np.float64)
            # инициализация последних значений каждого блока
            end_pts = {}
            
            # Генерация временного ряда
            for inx, block in enumerate(self.blocks):
                logger.debug(f'Generating block {inx}: {block}')
                
                # проверка на существование параметров временного ряда
                if block['ts_params'] is None:
                    logger.error(f'Invalid block without parameters:\n  {block}')
                    raise ValueError(f"No parameters for block {inx}")

                # проверка на совместимость параметров для генерации ВР и шума
                try:
                    block['ts_generator'](**block['ts_params'], length=self.block_length)
                except:
                    logger.error(f'Invalid parameters for block ts generator: \n  {block["ts_params"]}, {block["ts_generator"]}')
                    raise ValueError(f"Invalid parameters for block {inx}")
                
                # генерация ряда
                ts_length = self.block_length if inx == 0 else self.block_length+1
                block_series = np.array(block['ts_generator'](**block['ts_params'], length=ts_length)) + (0 if end_pts.get(inx-1) is None else end_pts.get(inx-1))
                block_series = block_series[1:] if inx >= 1 else block_series
                # генерация и добавление шума к ВР
                if block.get('noise_generator') is not None and block.get('noise_params') is not None:
                    
                    try:
                        noise = block['noise_generator'](data=block_series, random_state=random_state, **block['noise_params'])
                    except Exception as e:
                        logger.error(f'Error in noise generator: {e}')
                        raise ValueError(f"Invalid parameters for block {inx}")

                    block_series += noise
                
                # соединяем блоки 
                time_series = np.concatenate([time_series, block_series])
                logger.debug(f'Time series shape: {time_series.shape}')
                # Обработка последних значений
                end_pts[inx] = block_series[-1]
            
            logger.info(f'Generated time series with shape: {time_series.shape}')
            
            return time_series

def save_generated_data(data:Dict[str, List[int]], save_path:Path):
    """
    Переводит данные из формата: 
    
    [
        метка:{
            data:[список временных рядов],
            q_time_series:[количество рядов],
            segments:{размеры каждого блока для каждого класса
            } 
    ]
    
    в json файл 
    
    Пример input для функции
    """

    ts_id = 0
    json_file = []
    for class_id, _ in enumerate(data):
        for item in data[class_id]['data']:
            json_file.append({
                'ts_id':ts_id,
                'class_id':class_id,
                'segments':data[class_id]['segments'],
                'row': list(item) if isinstance(item, np.ndarray) else item  
            }) 
            ts_id += 1
    
    with open(save_path, 'w') as file:
        json.dump(json_file, file)
        
    logger.debug(f'File was saved at {save_path}')

class Time_series_dataset(Dataset):
    """
    Класс для загрузки и использования синтетического датасета в PyTorch.
    """
    def __init__(self, data_path: Union[str, Path], n_dims: Optional[int] = None, normalize: bool = False, norm_type: Literal['minimax','zscore','robust_sklearn'] = 'minmax'):
        """
        Инициализация датасета.

        Args:
            data_path: Путь к JSON файлу с данными
            n_dims: Количество измерений для преобразования (если None, оставляет одномерным)
            normalize: Флаг для нормализации данных
            norm_type: Тип нормализации ('minmax' или 'zscore')
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        self.series = [np.array(data[inx]['row']) for inx, item in enumerate(data)]
        self.labels = [data[inx]['class_id'] for inx, item in enumerate(data)]
        
        self.normalize_funcs = {
            'zscore':StandardScaler,
            'minmax':MinMaxScaler,
            'robust_sklearn':RobustScaler
        }
        
        if normalize:
            scaler = self.normalize_funcs[norm_type]()
            self.series = scaler.fit_transform(self.series)
            
    def __len__(self) -> int:
        """Возвращает количество временных рядов в датасете."""
        return len(self.series)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получение элемента датасета.

        Args:
            idx: Индекс элемента

        Returns:
            Кортеж (временной ряд как тензор, метка класса)
        """
        series = self.series[idx]
        logger.debug(f'series {idx} length: {len(series)}')
        label = self.labels[idx]
        logger.debug(f'series {idx} label: {label}')
                
        return series, label
