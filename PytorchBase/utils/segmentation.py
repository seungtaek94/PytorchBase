
import numpy as np
from .functions import to_numpy
import torch
from typing import Union, List,Tuple


def classMask2colorMask(class_mask:Union[torch.Tensor, np.ndarray], color:List[Tuple]) -> np.ndarray:
    """Covert the input 1 channel class mask to 3 channel color mask.

    Args:
        class_mask (Union[torch.Tensor, np.ndarray]): 
            Input 1 channel class mask. 
        color (List[int]): Output color list.
            e.g.) color = [
                    (255, 0, 25), 
                    (205, 10, 10)
                ]
    Raises:
        ValueError: 
            Class mask dimension should be 2 or 3.
            And the first channel of the 3 dimension class mask should be 1.
    Returns:
        np.ndarray: 
            Output 3 channel color mask.
    """

    if type(class_mask) == torch.Tensor:
        class_mask = to_numpy(class_mask)

    if not len(class_mask.shape) == 2 and not len(class_mask.shape) == 3:
        raise ValueError("class_mask dimension should be 2 or 3.")    

    if len(class_mask.shape) == 3:
        if class_mask.shape[0] == 1:
            class_mask = class_mask[0]
        else:
            raise ValueError("3-dimension class_mask must have the first channel 1.")

    r = np.zeros_like(class_mask, dtype=np.uint8)
    g = np.zeros_like(class_mask, dtype=np.uint8)
    b = np.zeros_like(class_mask, dtype=np.uint8)

    for idx, color in enumerate(color):
        r[class_mask == idx] = color[0]
        g[class_mask == idx] = color[1]
        b[class_mask == idx] = color[2]        
    
    return np.stack([r,g,b], axis=2)