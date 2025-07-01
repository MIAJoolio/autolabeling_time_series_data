from typing import Dict, Type, List
import inspect
from pathlib import Path
import json 
from graphviz import Digraph
        

from src.feature_extraction.manual_methods.manual_feature_extraction import *
from src.feature_extraction.fe_classes import *


class PipelineConfigurator:
    """
    Класс для управления конфигурацией пайплайна через строковые названия методов.
    Не изменяет оригинальный FE_pipeline, а создает конфигурацию для его инициализации.
    """
    
    # Реестр всех доступных экстракторов
    _EXTRACTOR_REGISTRY: Dict[str, Type[Base_extractor]] = {
        'linregress': Linregress_detrender,
        'stl': STL_detrender,
        'wavelet': Wavelet_detrender,
        'spline': Spline_detrender,
        'sma': SMA_detrender,
        'ema': EMA_detrender,
        'hpfilter': HP_filter_detrender,
        'bkfilter': BK_filter_detrender,
        'cffilter': CF_filter_detrender
    }
    
    def __init__(self):
        self._extractor_names: List[str] = []
        self._extractor_params: Dict[str, dict] = {}
    
    @classmethod
    def list_available_extractors(cls) -> List[str]:
        """Возвращает список всех зарегистрированных экстракторов"""
        return list(cls._EXTRACTOR_REGISTRY.keys())
    
    @classmethod
    def get_extractor_params_info(cls, extractor_name: str) -> dict:
        """
        Возвращает информацию о параметрах экстрактора
        Args:
            extractor_name: Имя экстрактора из реестра
        Returns:
            dict: Параметры и их значения по умолчанию
        """
        if extractor_name not in cls._EXTRACTOR_REGISTRY:
            raise ValueError(f"Экстрактор {extractor_name} не найден. Доступные: {cls.list_available_extractors()}")
        
        extractor_class = cls._EXTRACTOR_REGISTRY[extractor_name]
        sig = inspect.signature(extractor_class.extract)
        return {
            param.name: param.default 
            for param in sig.parameters.values() 
            if param.name != 'self' and param.default != inspect.Parameter.empty
        }
    
    def add_extractor(self, extractor_name: str, **params) -> None:
        """
        Добавляет экстрактор в конфигурацию пайплайна
        Args:
            extractor_name: Имя экстрактора из реестра
            **params: Параметры для инициализации экстрактора
        """
        if extractor_name not in self._EXTRACTOR_REGISTRY:
            raise ValueError(f"Экстрактор {extractor_name} не найден")
        
        # Валидация параметров
        valid_params = self.get_extractor_params_info(extractor_name)
        for param in params:
            if param not in valid_params:
                raise ValueError(f"Параметр {param} не поддерживается для экстрактора {extractor_name}")
        
        self._extractor_names.append(extractor_name)
        self._extractor_params[extractor_name] = params
    
    def remove_extractor(self, extractor_name: str) -> None:
        """
        Удаляет экстрактор из конфигурации
        Args:
            extractor_name: Имя экстрактора для удаления
        """
        if extractor_name in self._extractor_names:
            idx = self._extractor_names.index(extractor_name)
            self._extractor_names.pop(idx)
            self._extractor_params.pop(extractor_name, None)
    
    def clear_pipeline(self) -> None:
        """Очищает текущую конфигурацию пайплайна"""
        self._extractor_names = []
        self._extractor_params = {}
    
    def build_pipeline(self) -> FE_pipeline:
        """
        Создает экземпляр FE_pipeline на основе текущей конфигурации
        Returns:
            FE_pipeline: Готовый пайплайн с заданными экстракторами
        """
        extractors = []
        for name in self._extractor_names:
            extractor_class = self._EXTRACTOR_REGISTRY[name]
            params = self._extractor_params.get(name, {})
            extractors.append(extractor_class(**params))
        
        return FE_pipeline(extractors)
    
    def get_current_config(self) -> dict:
        """
        Возвращает текущую конфигурацию пайплайна
        Returns:
            dict: Словарь с именами экстракторов и их параметрами
        """
        return {
            'extractors': self._extractor_names.copy(),
            'params': self._extractor_params.copy()
        }
        
    def save_config(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.get_current_config(), f)

    @classmethod
    def load_config(cls, filepath: str) -> 'PipelineConfigurator':
        with open(filepath, 'r') as f:
            config = json.load(f)
        pc = cls()
        for name, params in zip(config['extractors'], config['params']):
            pc.add_extractor(name, **params)
        return pc
    
    def visualize(self):
        dot = Digraph()
        for i, name in enumerate(self._extractor_names):
            dot.node(str(i), name)
            if i > 0:
                dot.edge(str(i-1), str(i))
        return dot

# Пример использования
if __name__ == "__main__":
    # 1. Инициализация конфигуратора
    configurator = PipelineConfigurator()
    
    # 2. Просмотр доступных экстракторов
    print("Доступные экстракторы:", configurator.list_available_extractors())
    
    # 3. Добавление экстракторов в пайплайн
    configurator.add_extractor('wavelet', wavelet='db4', level=3)
    configurator.add_extractor('spline', s=0.1, k=3)
    
    # 4. Просмотр текущей конфигурации
    print("Текущая конфигурация:", configurator.get_current_config())
    
    # 5. Удаление экстрактора
    configurator.remove_extractor('spline')
    
    # 6. Создание пайплайна
    pipeline = configurator.build_pipeline()
    
    # 7. Применение к данным (пример)
    data = np.random.randn(100)
    processed = pipeline.apply(data)