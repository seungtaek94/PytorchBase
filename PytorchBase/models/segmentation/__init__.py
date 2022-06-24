from .hrnet.model import HRNet

def get_model(name:str, n_classes:int):
    """ get segmentation model
    Args:
        name (str): _description_
        n_classes (int): _description_
    """

    if name[:5] == "hrnet":
        return HRNet(backbone=name, num_classes=n_classes)
    else:
        raise NotImplementedError

