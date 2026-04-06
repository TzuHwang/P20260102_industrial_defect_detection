from types import SimpleNamespace

import yaml
import json
import csv


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def namespace_to_dict(ns):
    """Convert a SimpleNamespace to a regular dict."""
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_yaml_as_ns(filepath):
    with open(filepath, 'r') as f:
        # safe_load is best practice to avoid code injection
        data = yaml.safe_load(f)
    return dict_to_namespace(data)


def json_loader(filepath):
    with open(filepath, 'r') as f:
        js_data = json.load(f)
    return js_data


def csv_loader(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data
