from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import numpy as np

class DiceLoss(_Loss):
    def __init__(
        self, 
        classes: Optional[List[int]] = None,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        super(DiceLoss, self).__init__()

        
        classes = to_tensor(classes, dtype=torch.int32)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activation to get [0 .. 1] class prob
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if True: # For multi
                y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if True: # For Multi
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes) # N, H*W -> N, H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1) # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1) # H, C, H*W

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)


def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()

    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    return dice_score


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x



if __name__=="__main__":
    eps = 1e-5

    criterion = DiceLoss()

    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    print(y_pred.shape)
    y_true = torch.tensor([[0, 0, 1, 1]])
    print(y_true.shape)
    loss = criterion(y_pred, y_true)
    print("Ideal:", loss)

    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])
    loss = criterion(y_pred, y_true)
    print("Worst:", loss)

    y_pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])
    loss = criterion(y_pred, y_true)
    print(" 3/1 :", loss)

    #print(pytest.approx(1.0 - 1.0 / 3.0, abs=eps))