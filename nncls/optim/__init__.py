import torch
import torch.distributed as dist
from .sam import SAMSGD
from .lars import LARSOptimizer
from .sgd_gc import SGD_GCC, SGD_GC, SGDW, SGDW_GCC, SGDW_GC 

def build_optim(cfg, net):
    optim_name = ' '
    if cfg.TRAIN.OPTIM == 'sgd':
        optim_name = 'sgd'
        optimizer = torch.optim.SGD(net.parameters(), 
                              lr=cfg.TRAIN.LR, 
                              momentum=0.9, 
                              weight_decay=5e-4)
    elif cfg.TRAIN.OPTIM == 'adam':
        optim_name = 'adam'
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIM == 'sam':
        optim_name = 'sam'
        optimizer = SAMSGD(net.parameters(), 
                           lr=cfg.TRAIN.LR,
                           rho=0.05)
    elif cfg.TRAIN.OPTIM == 'sgd_gc':
        optim_name = 'sgd_gc'
        optimizer = SGD_GC(net.parameters(), 
                           lr=cfg.TRAIN.LR, 
                           momentum=0.9, 
                           weight_decay=5e-4)
    elif cfg.TRAIN.OPTIM == 'sgd_gcc':
        optim_name = 'sgd_gcc'
        optimizer = SGD_GCC(net.parameters(), 
                            lr=cfg.TRAIN.LR,
                            momentum=0.9, 
                            weight_decay=5e-4)
    elif cfg.TRAIN.OPTIM == 'lars':
        optim_name = 'lars'
        optimizer = LARSOptimizer(params,
                                  lr=cfg.TRAIN.LR,
                                  momentum=0.9,
                                  eps=1e-9,
                                  thresh=-e-2)
    else:
        raise ValueError(
            'the optimizer name you entered is not supported yet')
    if cfg.DIST and dist.get_rank() == 0:
        print("use optimizer: ",optim_name)
    elif not cfg.DIST:
        print("use optimizer: ",optim_name)

    return optimizer