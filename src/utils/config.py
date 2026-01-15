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
                d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)
        return d

    return dict_to_namespace(config_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="baseline-dreambooth_lora.yaml", help="Path to YAML config in configs/")
    return parser.parse_args()