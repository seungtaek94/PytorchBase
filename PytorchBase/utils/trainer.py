from abc import *
from typing import Dict

from .dist import is_ddp_active

class BaseTrainer(metaclass=ABCMeta):
    def __init__(
        self, 
        model,
        criterion,
        optimizer,
        scheduler,
        start_epoch:int=0, 
        end_epoch:int=None,
        save_dir:str="./",
        save_term:int=10,
        exp_prefix:str='',
        use_wandb=False        
        ) -> None:
        super(BaseTrainer, self).__init__()

        self.model = model

        if is_ddp_active():            
            self.model_without_ddp = model.module
        else:
            self.model_without_ddp = model

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.save_dir = save_dir
        self.save_term = save_term
        self.exp_prefix = exp_prefix
        self.use_wandb = use_wandb
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def fit(self, train_loader, val_loader=None):
        """_summary_

        Args:
            train_loader (_type_): _description_
            val_loader (_type_): _description_
        """
        pass
    
    @abstractmethod
    def _train_one_epoch(self, epoch, train_loader) -> Dict[str, float]:
        pass

    @abstractmethod
    def _val_one_epoch(self, epoch, val_loader) -> Dict[str, float]:
        pass
    
    
        
        
    
    # get accu




    

