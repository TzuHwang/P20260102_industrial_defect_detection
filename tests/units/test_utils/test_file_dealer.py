from types import SimpleNamespace

from project_src.utils.file_dealer import load_yaml_as_ns


def test_load_yaml_as_ns():
    assert isinstance(
        load_yaml_as_ns('data/test/integration/test_pipeline.yml'), SimpleNamespace)
