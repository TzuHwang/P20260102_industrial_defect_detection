import torch.optim as optim

from project_src.utils.file_dealer import namespace_to_dict


class OptimizerFactory:
    """Factory class to create optimizers."""

    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
    }

    def __init__(self, params, args):
        args_dict = namespace_to_dict(args)
        self.optimizer = self.optimizers.get(args_dict.pop('name', 'adam'), optim.Adam)(
            params, **args_dict
        )

    def get_optimizer(self):
        return self.optimizer
