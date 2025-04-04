from typing import Literal 
import logging
import datetime
from pathlib import Path

def setup_logger(name: str, log_dir: str = ".logs", low_lvl:Literal['info', 'debug']='info') -> logging.Logger:
    """Настройка логгера с выводом в консоль и файл."""
    
    logger_option = {
        'info':logging.INFO,
        'debug':logging.DEBUG
    }

    logger = logging.getLogger(name)
    logger.setLevel(logger_option[low_lvl])

    # Форматирование
    formatter = logging.Formatter(
        "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Создание директории для логов, если её нет
    Path(log_dir).mkdir(exist_ok=True)

    # Обработчики (только если их ещё нет)
    if not logger.handlers:
        # Консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Файл (с человекочитаемым именем)
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        file_handler = logging.FileHandler(
            Path(log_dir) / f"{name}_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger