
import torch.nn as nn

from project_src.post_process.wrapper.cam import CAMWrapper, CAM2BoxWrapper
from project_src.post_process.wrapper.nms import NMSWrapper

_WRAPPER_REGISTRY = {
    'CAM': CAMWrapper,
    'CAM2Box': CAM2BoxWrapper,
    'NMS': NMSWrapper,
}


class PostProcessFactory:
    def __init__(self, args, model):
        post_process_type = args.get('type') if isinstance(args, dict) else getattr(args, 'type', None)

        if not post_process_type:
            self._wrapped = model
            return

        if post_process_type not in _WRAPPER_REGISTRY:
            raise ValueError(
                f"Unsupported post process type: {post_process_type}. "
                f"Supported: {list(_WRAPPER_REGISTRY.keys())}"
            )

        wrapper_cls = _WRAPPER_REGISTRY[post_process_type]

        if issubclass(wrapper_cls, CAMWrapper):
            self._wrapped = wrapper_cls(args, model, self._resolve_target_layer(args, model))
        elif wrapper_cls is NMSWrapper:
            # Detection decoders are stateless callables — no model reference needed.
            kw = {k: v for k, v in vars(args).items() if k != 'type'} if hasattr(args, '__dict__') else {}
            self._wrapped = wrapper_cls(**kw)
        else:
            self._wrapped = wrapper_cls(args, model)

    def _resolve_target_layer(self, args, model):
        layer_name = args.get('target_layer') if isinstance(args, dict) else getattr(args, 'target_layer', None)
        if layer_name is not None:
            return dict(model.named_modules())[layer_name]
        # Fall back to last Conv2d in the model
        last_conv = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model; set 'target_layer' in post_process config")
        return last_conv

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def to(self, device):
        if hasattr(self._wrapped, 'to'):
            self._wrapped.to(device)
        return self

    def __getattr__(self, name):
        return getattr(self._wrapped, name)
