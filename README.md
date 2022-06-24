# PytorchBase

파이토치에서 많이 사용하는 함수들을 추상화한 레포지토리 입니다.

## Requirements

```python
albumentations 
wandb 
onnx 
onnxruntime
torch
torchvision
yacs
```

## Model
### Segmentation
```Python
from PytorchBase.models.segmentation import get_model
model = get_model(name:str, n_classes:int)
# model = get_model(name="hrnet18", n_classes=10)
```

- hrnet
    - "hrnet18"
    - "hrnet32"
    - "hrnet48"
    - "hrnet64"

## Loss
- Dice

## Optim
- Adam


