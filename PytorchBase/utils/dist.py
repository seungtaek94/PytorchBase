import os 
import numpy
import torch

import torch.distributed as dist

from typing import Dict


def init_dist_env(number_of_gpus:int):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = f'{number_of_gpus}'
    os.environ['RANK'] = '0'

    print(f"[Notice] MASTER_ADDR - {os.environ['MASTER_ADDR']}")
    print(f"[Notice] MASTER_PORT - {os.environ['MASTER_PORT']}")
    print(f"[Notice] WORLD_SIZE - {os.environ['WORLD_SIZE']}")
    print(f"[Notice] RANK - {os.environ['RANK']}")
    

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    import wandb

    builtin_print = __builtin__.print
    wandb_init = wandb.init
    wandb_finish = wandb.finish
    wandb_log = wandb.log
    wandb_watch = wandb.watch

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    def init(*args, **kwargs):
        if is_master:
            wandb_init(*args, **kwargs)

    def finish(*args, **kwargs):
        if is_master:
            wandb_finish(*args, **kwargs)

    def log(*args, **kwargs):
        if is_master:
            wandb_log(*args, **kwargs)

    def watch(*args, **kwargs):
        if is_master:
            wandb_watch(*args, **kwargs)

    __builtin__.print = print
    wandb.init = init
    wandb.finish = finish
    wandb.log = log  
    wandb.watch = watch


def is_ddp_active():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_ddp_active():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_ddp_active():
        return 1
    return dist.get_rank()


def is_main_process():
    if not is_ddp_active():
        return True
    return get_rank() == 0


def save_one_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def run_on_main_process(method):
    if is_main_process():
        return method()


def init_single_node_distributed_mode(rank, world_size):
    torch.cuda.set_device(rank)    
    
    print('[Notice] distributed init (rank {})'.format(
        rank), flush=True)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size, 
        rank=rank)

    torch.distributed.barrier()
    setup_for_distributed(rank == 0)


def barrier():
    if is_ddp_active():
        dist.barrier()


def cleanup():
    dist.destroy_process_group()


def reduce_avg_value(value:float, reduceOp=dist.ReduceOp.SUM):
    """
    모든 프로세스(GPU, rank)의 값을 모아 평균을 반환, single node에서 만 테스트됨.

    Args:
        value(float): value for reduce
        reduceOp(?): recude operation

    Returns:
        float: reduced avg value
    """
    value_tensor = torch.Tensor([value]).to(device=get_rank(), dtype=torch.float32, non_blocking=True)
    dist.all_reduce(value_tensor, reduceOp, async_op=False)
    value = value_tensor.item()       
    del value_tensor

    return value / dist.get_world_size()


def reduce_tensor_dict(src:Dict[str, torch.Tensor], reduceOp=dist.ReduceOp.SUM):        
    dst = {}
    for key in src.keys():
        dst[key] = reduce_avg_value(src[key])

    return dst