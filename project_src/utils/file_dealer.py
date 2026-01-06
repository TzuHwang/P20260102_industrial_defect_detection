import os
import yaml
from types import SimpleNamespace


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def load_yaml_as_ns(filepath):
    with open(filepath, 'r') as f:
        # safe_load is best practice to avoid code injection
        data = yaml.safe_load(f)
    return dict_to_namespace(data)
