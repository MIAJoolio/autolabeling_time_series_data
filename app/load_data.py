from pathlib import Path
import numpy as np
import json
from abc import ABC, abstractmethod

class TS_baseloader(ABC):
    """
    Базовый класс для загрузки временных рядов
    """
    @abstractmethod
    def load_data(self, filepath):
        """
        Загружает данные из файла
        
        Args:
            filepath (str): Путь к файлу с данными
            
        Returns:
            dict: Загруженные данные в стандартизированном формате
        """
        pass

class Format1_loader(TS_baseloader):
    """
    Загружает данные из JSON и преобразует списки обратно в numpy.ndarray.
    Реализует специфичный для format1 способ загрузки.
    """
    
    def __init__(self):
        return None
    
    def load_data(self, filepath):
        """
        Загружает данные из JSON и преобразует списки обратно в numpy.ndarray.
        
        Args:
            filepath (str): Путь к JSON-файлу
            
        Returns:
            dict: {'data': List[np.ndarray], 'label': List[...], 'meta': ...}
        """
        def ndarray_hook(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, list):
                        try:
                            if all(isinstance(x, (int, float)) for x in value):
                                d[key] = np.array(value)
                            else:
                                d[key] = value
                        except:
                            pass
                    elif isinstance(value, dict):
                        ndarray_hook(value)
            return d

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data = ndarray_hook(data)

        if 'data' in data and isinstance(data['data'], list) and data['data'] and isinstance(data['data'][0], list):
            data['data'] = [np.array(series) for series in data['data']]

        return data
    