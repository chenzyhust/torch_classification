import torch
import torch.nn as nn
import torch.distributed as dist
from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingLoss
from .aug_loss import MixLoss, RicapLoss, DualCutoutLoss

def build_loss(cfg):
    loss_name = ' '
    aug_name = ' '
    if cfg.AUG.MIXUP:
        aug_name = 'Mixup'
        loss_name = 'MixLoss'
        train_loss = MixLoss()
    elif cfg.AUG.CUTMIX:
        aug_name = 'CutMix'
        loss_name = 'MixLoss'
        train_loss = MixLoss()
    elif cfg.AUG.RICAP:
        aug_name = 'Ricap'
        loss_name = 'RicapLoss'
        tain_loss = RicapLoss()
    elif cfg.AUG.D_CUTOUT:
        loss_name = 'DualCutoutLoss'
        tain_loss = DualCutoutLoss(cfg.AUG.DCUT_ALPHA)
    elif cfg.AUG.LABEL_SMOOTH:
        loss_name = 'LabelSmoothingLoss'
        train_loss = LabelSmoothingLoss(n_classes=cfg.TRAIN.CLASSES, 
                                        epsilon=cfg.AUG.SMOOTH_EPS)
    elif cfg.AUG.FOCAL_LOSS:
        loss_name = 'FocalLoss'
        train_loss = FocalLoss(gamma=cfg.AUG.GAMMA,
                               alpha=AUG.ALPHA)
    else:
        loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()
    if cfg.DIST and dist.get_rank() == 0:
        print("use extra aug:", aug_name)
        print("use loss function: ", loss_name)
    elif not cfg.DIST:
        print("use extra aug:", aug_name)
        print("use loss function: ", loss_name)

    return train_loss, val_loss