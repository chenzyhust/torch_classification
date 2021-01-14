import torch
import torch.nn as nn
import torch.distributed as dist
from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingLoss
from .aug_loss import MixLoss, RicapLoss, DualCutoutLoss

def build_loss(cfg):
    aug_loss_name = ' '
    train_loss_name = ' '
    aug_name = ' '
    
    if cfg.AUG.MIXUP:
        aug_name = 'Mixup'
        train_loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
        aug_loss_name = 'MixLoss'
        aug_loss = MixLoss()
    elif cfg.AUG.CUTMIX:
        aug_name = 'CutMix'
        train_loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
        aug_loss_name = 'MixLoss'
        aug_loss = MixLoss()
    elif cfg.AUG.RICAP:
        aug_name = 'Ricap'
        train_loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
        aug_loss_name = 'RicapLoss'
        aug_loss = RicapLoss()

    elif  cfg.AUG.D_CUTOUT:
        train_loss_name = 'DualCutoutLoss'
        tain_loss = DualCutoutLoss(cfg.AUG.DCUT_ALPHA)

    elif cfg.AUG.LABEL_SMOOTH:
        train_loss_name = 'LabelSmoothingLoss'
        train_loss = LabelSmoothingLoss(n_classes=cfg.TRAIN.CLASSES, 
                                        epsilon=cfg.AUG.SMOOTH_EPS)
        aug_loss_name = 'LabelSmoothingLoss'
        aug_loss = LabelSmoothingLoss(n_classes=cfg.TRAIN.CLASSES, 
                                        epsilon=cfg.AUG.SMOOTH_EPS)

    elif cfg.AUG.FOCAL_LOSS:
        train_loss_name = 'FocalLoss'
        train_loss = FocalLoss(gamma=cfg.AUG.GAMMA,
                               alpha=cfg.AUG.ALPHA)
        aug_loss_name = 'FocalLoss'
        aug_loss = FocalLoss(gamma=cfg.AUG.GAMMA,
                             alpha=cfg.AUG.ALPHA)
    else:
        aug_loss_name = 'CrossEntropyLoss'
        aug_loss = nn.CrossEntropyLoss()
        train_loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()
    if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
        print("use extra aug:", aug_name)
        print("use train loss function: ", train_loss_name)
        print("use aug loss function: ", aug_loss_name)
    
    return train_loss, aug_loss, val_loss