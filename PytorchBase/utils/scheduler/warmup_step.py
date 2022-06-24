import math
from typing import List
from torch.optim.lr_scheduler import _LRScheduler

class WarmupStep(_LRScheduler):
    def __init__(self, optimizer, warmup_step, steps:List[int], max_lr, gamma=0.1, last_epoch=-1):

        self.warmup_step = warmup_step
        self.max_lr = max_lr

        self.steps = steps
        self.step_idx = 0
        self.gamma = gamma
        self.drop_rate = 1
        
        super(WarmupStep, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs
        elif self.last_epoch <= self.warmup_step:
            return [(self.max_lr - base_lr)*self.last_epoch / self.warmup_step + base_lr for base_lr in self.base_lrs]
        else:
            return [self.max_lr * self.drop_rate]
            

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch == self.steps[self.step_idx]:
            self.drop_rate = self.gamma ** (self.step_idx + 1)
            
            if self.step_idx < len(self.steps) - 1:
                self.step_idx += 1

        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr