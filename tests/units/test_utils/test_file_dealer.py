from types import SimpleNamespace

from project_src.utils.file_dealer import load_yaml_as_ns


def test_load_yaml_as_ns():
    assert isinstance(
        load_yaml_as_ns('configs/yamls/template_config.yaml'), SimpleNamespace)
