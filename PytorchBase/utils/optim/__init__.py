import imp
import torch
from torch.optim import Optimizer

def get_optimizer(params, optimizer:str, lr:float=1e-3, weight_decay:float=0.) -> Optimizer:
    """_summary_

    Args:
        params (iterable): model params
        optimizer (str): Optimizer name
        lr (float, optional): learning rate for training. Defaults to 1e-3.
        weight_decay (float, optional): wegiht dacay value. Defaults to 0..

    Raises:
        NotImplementedError: Not supported optimizer.

    Returns:
        Optimizer: _description_
    """
    print(f'[Notice] Opimizer - {optimizer}')

    if optimizer == 'Adam':
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    # elif optimizer is 'AdamP':
    #     return AdamP(params=params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError