from .cifar import resnet50, resnest50, mobilenetv3
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def build_network(cfg):
    net = eval(cfg.NET)()
    net.to(cfg.DEVICE)

    if cfg.GPU: 
        if not cfg.APEX and cfg.DIST and dist.is_available():
            local_rank = cfg.LOCAL_RANK
            if cfg.SYNC_BN:
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = net.to(local_rank)
            net = DDP(net,
                      device_ids=[local_rank],
                      output_device=local_rank)
            if dist.get_rank() == 0:
                print('use network: ', cfg.NET)
        elif cfg.APEX and cfg.DIST and dist.is_available():
            if dist.get_rank() == 0:
                print('use network: ', cfg.NET)
        else:
            net = torch.nn.DataParallel(net)
            print('use network: ', cfg.NET)
    return net