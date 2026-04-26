import argparse

from utils.file_dealer import load_yaml_as_ns


class AccessArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # parse pipeline contral args
        self.parser.add_argument(
            '--yml-config', type=str, help='',
        )

        self.parser.add_argument(
            '--task', type=str, help='',
            choices=['train', 'inference', 'deploy'],
        )

        args = self.parser.parse_args()
        self.args = load_yaml_as_ns(args.yml_config)

        self.args.task = args.task

    def get_args(self):
        return self.args
