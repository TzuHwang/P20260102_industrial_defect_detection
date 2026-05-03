import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from project_src.utils.file_dealer import namespace_to_dict


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm up the learning rate.

    Args:
        optimizer: Wrapped optimizer.
        total_iters: Number of warmup iterations.
        after_scheduler: Scheduler to use after warmup. If None, keep the learning rate constant after warmup.
        last_epoch: The index of the last epoch when resuming training.
    """

    def __init__(self, optimizer, total_iters, after_scheduler=None, last_epoch=-1):
        self.total_iters = total_iters
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.total_iters:
            if self.after_scheduler is not None:
                if not self.after_scheduler.finished:
                    return self.after_scheduler.get_lr()
                else:
                    return [base_lr for base_lr in self.base_lrs]
            else:
                return [base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.last_epoch + 1) / self.total_iters for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished:
            return super().step(epoch)
        else:
            if self.last_epoch >= self.total_iters:
                if self.after_scheduler is not None:
                    if isinstance(self.after_scheduler, ReduceLROnPlateau):
                        if metrics is None:
                            raise ValueError(
                                "When using ReduceLROnPlateau as after_scheduler, "
                                "you must pass metrics to step()."
                            )
                        self.after_scheduler.step(metrics, epoch)
                    else:
                        self.after_scheduler.step(epoch)
                    self.finished = True
                else:
                    self.finished = True
                    super().step(epoch)
            else:
                super().step(epoch)

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key not in ['optimizer', 'after_scheduler']}
        if self.after_scheduler is not None:
            state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop('after_scheduler', None)
        self.__dict__.update(state_dict)
        if after_scheduler_state is not None and self.after_scheduler is not None:
            self.after_scheduler.load_state_dict(after_scheduler_state)


class SchedulerFactory:
    """Factory class to create learning rate schedulers."""

    schedulers = {
        'StepLR': lr_scheduler.StepLR,
        'MultiStepLR': lr_scheduler.MultiStepLR,
        'ExponentialLR': lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        'LambdaLR': lr_scheduler.LambdaLR,
    }

    def __init__(self, optimizer, args):
        args_dict = namespace_to_dict(args)
        warmup_iters = args_dict.pop('warmup_iters', 0)
        base_scheduler = self.schedulers.get(args_dict.pop('name', 'ExponentialLR'), lr_scheduler.ExponentialLR)(
            optimizer, **args_dict
        )

        if warmup_iters > 0:
            self.scheduler = GradualWarmupScheduler(
                optimizer,
                total_iters=warmup_iters,
                after_scheduler=base_scheduler,
            )
        else:
            self.scheduler = base_scheduler

    def get_scheduler(self):
        return self.scheduler
