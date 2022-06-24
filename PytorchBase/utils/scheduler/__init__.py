import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .cosine import CosineAnnealingWarmUpRestarts
from .warmup_step import WarmupStep

def get_scheduler(scheduler:str, optimizer:Optimizer, **kwargs) -> _LRScheduler:
    """ get pytorch learning rate scheduler.

    Args:
        scheduler (str): shecudler name
        optimizer (Optimizer): pytorch optimizer

    Raises:
        NotImplementedError: Not supported scheduler

    Returns:
        _LRScheduler: _description_
    """
    print(f'[Notice] Scheduler - {scheduler}')
    if scheduler == "cosine":
        return CosineAnnealingWarmUpRestarts(optimizer, **kwargs)
    elif scheduler == "step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        
    elif scheduler == "warmup_step":
        return WarmupStep(optimizer, **kwargs)
    else:
        raise NotImplementedError
