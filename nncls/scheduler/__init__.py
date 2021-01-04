import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler

def build_scheduler(cfg, optimizer):
    scheduler_name = ' '
    if cfg.TRAIN.SCHEDULER == 'step':
        scheduler_name = 'step'
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                   milestones=cfg.TRAIN.STEPS, 
                                                   gamma=0.2)
    elif cfg.TRAIN.SCHEDULER == 'cosine':
        scheduler_name = 'cosine'
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         cfg.TRAIN.EPOCHES)
    else:
        raise ValueError(
            'the lr scheduler name you entered is not supported yet')
    if cfg.DIST and dist.get_rank() == 0:
        print("use lr scheduler: ", scheduler_name)
    elif not cfg.DIST:
        print("use lr scheduler: ", scheduler_name)

    return scheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]