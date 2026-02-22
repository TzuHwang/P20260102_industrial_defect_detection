import torch.nn as nn


class ActivationFactory:
    """Factory class to create activation functions."""

    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=-1),
        'log_softmax': nn.LogSoftmax(dim=-1),
    }

    def __init__(self, args):
        super().__init__()
        self.activation = self.activations.get(args.name, nn.Softmax(dim=-1))

    def get_activation(self):
        return self.activation
