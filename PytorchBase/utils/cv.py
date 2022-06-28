import os
import numpy as np
import cv2
from typing import Dict, Optional


def imwrite_kr(file:str, image:np.ndarray, params:Optional[Dict]=None) -> bool:
    """Support for write the image with paths that contain Korean characters.

    Args:
        file (str): path to save image
        image (np.ndarray): image array
        params (Optional[Dict], optional): additional param

    Returns:
        bool: succes(true) or fail(false)
    """
    try: 
        ext = os.path.splitext(file)[1] 
        result, n = cv2.imencode(ext, image, params) 
        
        if result: 
            with open(file, mode='w+b') as f:                
                n.tofile(f) 
            return True 
        
        else: 
            return False 
        
    except Exception as e:
        print(e) 
        return False


def imread_kr(file:str, dtype=np.uint8) -> np.ndarray:
    """Support for reading the image with paths that contain Korean characters.

    Args:
        file (str): Path to read image
        dtype (, optional): Data type of image to read. Defaults to np.uint8.

    Returns:
        np.ndarray: Output image array
    """
    return cv2.imdecode(np.fromfile(file, dtype=dtype), cv2.IMREAD_COLOR)