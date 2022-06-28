import torch
import numpy as np
import random

def seed_everything(seed:int): # seed 고정
    """ seed everything for pytorch

    Args:
        seed (int): seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor:torch.Tensor) -> np.ndarray:
    """convert torch.Tensor to numpy.ndarray.

    Args:
        tensor (torch.Tensor): Input tensor to convert.

    Returns:
        np.ndarray: Output result.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()