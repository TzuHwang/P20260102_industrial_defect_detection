import argparse
import yaml


class AccessArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser

        # parse pipeline contral args
        self.parser.add_argument(
            '--yml-config', type=str, help='',
        )

        self.parser.add_argument(
            '--task', type=str, help='',
            choices=['train', 'inference'],
        )

        self.parser.add_argument(
            '--debug', action='store_ture', help=''
        )

        self.args = self.parser.parse_args()
