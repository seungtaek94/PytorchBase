import torch

from .dice import DiceLoss, AuxDiceLoss

def get_loss(loss:str, weight=None, **kwargs):
        print(f'[Notice] Loss - {loss}')
        if loss == 'margin':
            return torch.nn.MultiLabelMarginLoss()
        elif loss == 'soft_margin':
            return torch.nn.MultiLabelSoftMarginLoss()
        elif loss == 'ce':
            return torch.nn.CrossEntropyLoss(weight=weight)
        elif loss == 'mse':
            return torch.nn.MSELoss()
        elif loss == 'bce_logit':
            return torch.nn.BCEWithLogitsLoss()
        elif loss == 'dice':
            return DiceLoss()
        elif loss == 'aux_dice':
            return AuxDiceLoss(**kwargs)
        else:
            raise NotImplementedError