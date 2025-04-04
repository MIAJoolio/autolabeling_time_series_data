from pathlib import Path
import yaml

__all__ =  ['load_yaml_file']

def load_yaml_file(path_to_file:Path):
    with open(path_to_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    return None

if __name__ == '__main__':
    main()
