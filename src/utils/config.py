import os
from pathlib import Path
import yaml
import argparse
from types import SimpleNamespace

def get_project_root():
    current_file_path = Path(__file__).resolve()
    marker_dirs = {'src', 'scripts', "configs"}
    for parent in [current_file_path] + list(current_file_path.parents):
        if parent.is_dir():
            items = {item.name for item in parent.iterdir()}
            if marker_dirs.intersection(items):
                return parent
    return current_file_path.parent


def load_config(config_file):
    project_root = get_project_root()

    config_path = project_root / "configs" / config_file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k.endswith('_dir') and isinstance(v, str):
                    abs_path = project_root / v
                    d[k] = str(abs_path.resolve())
                d[k] = dict_to_namespace(d[k])
            return SimpleNamespace(**d)
        return d

    return dict_to_namespace(config_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="baseline-dreambooth_lora.yaml", help="Path to YAML config in configs/")
    return parser.parse_args()


def smart_convert_str_to_num(str_value):
    """ attemp to convert string to number (int or float).
    Examples:
        1. "1e-4" -> 0.0001 (float)
    2. "100" -> 100 (int)
    3. "00" -> "00" (reserved as string, to prevent ID from being converted to 0)
    4. "/path/to/file" -> reserved as string
    """
    if not isinstance(str_value, str):
        return str_value
        
    if '/' in str_value or '\\' in str_value:
        return str_value
    try:
        val_float = float(str_value)

        # check int
        if val_float.is_integer() and '.' not in str_value and 'e' not in str_value.lower():
            # special protection: "00", "01" ..
            if len(str_value) > 1 and str_value.startswith('0'):
                return str_value
            return int(val_float)
        
        return val_float
    except ValueError:
        return str_value