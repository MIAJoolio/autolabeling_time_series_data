from pathlib import Path
import yaml

__all__ =  ['load_yaml_file', 'load_config_file']

def load_yaml_file(path_to_file:Path):
    with open(path_to_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_config_file(config_name: Path) -> dict:
    """
    Загружает конфигурационный файл из указанной директории.
    
    Args:
        config_name: Имя конфигурационного файла (c расширением .yaml)
        config_dir: Директория с конфигурационными файлами
        
    Returns:
        Словарь с конфигурацией
    """
    config_path = Path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден")
    
    return load_yaml_file(config_path)

def main():
    return None

if __name__ == '__main__':
    main()
