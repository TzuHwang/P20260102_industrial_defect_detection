import torch

from .models import ModelFactory
from .activation import ActivationFactory
from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory
from .loss import LossFuncs


class ArchitectureBuilder:
    """Builder class to create the architecture components."""

    def __init__(self, args):
        self.pretrained_path = getattr(args, 'pretrained_path', None)
        self.force_load = getattr(args, 'force_load', False)

        self.model = ModelFactory(args.model)
        self.activation = ActivationFactory(args.activation).get_activation()
        self.loss_funcs = LossFuncs(args.loss_funcs)

        self.optimizer = OptimizerFactory(self.model.parameters(), args.optimizer).get_optimizer()
        self.scheduler = SchedulerFactory(self.optimizer, args.scheduler).get_scheduler()

    def get_model(self):
        return self.model

    def get_activation(self):
        return self.activation

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_loss_funcs(self):
        return self.loss_funcs

    def load_pretrained(self) -> None:
        """Load pretrained weights, optimizer and scheduler state.

        Uses pretrained_path and force_load attributes from the builder.
        If force_load is True, loads pretrained weights even if layer params
        mismatch (skips mismatched layers).
        """
        if self.pretrained_path is None:
            return

        checkpoint = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)

        # Load model state dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        if self.force_load:
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            self.model.load_state_dict(filtered_state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict)

        # Load optimizer state if available
        if "optimizer" in checkpoint:
            self.optimizer.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state if available
        if "scheduler" in checkpoint:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
