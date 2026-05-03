from torch.nn import Identity as TorchIdentity
from .detr_neck import DETRTransformerEncoder
from .cspnext_pafpn import RTMDetPAFPN
from .fpn import ResNet50FPNNeck

__all__ = [
    'Identity',
    'DETRTransformerEncoder',
    'RTMDetPAFPN',
    'ResNet50FPNNeck',
]


class Identity(TorchIdentity):
    """Identity neck that returns the input as is."""
    def __init__(self, args):
        _ = args  # args parameter is unused but kept for consistency
        super(Identity, self).__init__()
