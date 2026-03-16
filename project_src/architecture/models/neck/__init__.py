from torch.nn import Identity as TorchIdentity

__all__ = [
    'Identity',
]


class Identity(TorchIdentity):
    """Identity neck that returns the input as is."""
    def __init__(self, args):
        _ = args  # args parameter is unused but kept for consistency
        super(Identity, self).__init__()
