import torch
import numpy as np
import torch.distributed as dist
from .cutout import Cutout, DualCutout
from .random_erasing import RandomErasing

def build_transformer(cfg):
    transformer = []
    trans_name = []
    if cfg.AUG.CUTOUT:
        trans_name.appen('Cutout')
        transformer.append(Cutout(n_holes=cfg.AUG.N_HOLES, 
                                  length=cfg.AUG.LENGTH, 
                                  prob=cfg.AUG.PROB))
    if cfg.AUG.D_CUTOUT:
        trans_name.appen('DualCutout')
        transformer.append(DualCutout(n_holes=cfg.AUG.N_HOLES, 
                                      length=cfg.AUG.LENGTH, 
                                      prob=cfg.AUG.PROB))
    if cfg.AUG.R_ERASING:
        trans_name.appen('Random Erasing')
        transformer.append(RandomErasing(prob=cfg.AUG.PROB,
                                         max_attempt=cfg.AUG.M_ATTEMPT,
                                         area_ratio_range=cfg.AUG.A_RAT_RANGE,
                                         min_aspect_ratio=cfg.AUG.M_ASP_RATIO))
    if cfg.DIST and dist.get_rank() == 0:
        print("use extra transforms: ", [t for t in trans_name])
    elif not cfg.DIST:
        print("use extra transforms: ", [t for t in trans_name])
    return transformer

def aug_data(cfg, x, y):
    
    if cfg.AUG.MIXUP:
        images, labels = mixup_data(x, y, cfg.AUG.MIXUP_A)
    elif cfg.AUG.CUTMIX:
        images, labels = cutmix_data(x, y, cfg.AUG.CUTMIX_BETA)
    elif cfg.AUG.RICAP:
        images, labels = ricap_data(x, y, cfg.AUG.RICAP_BETA)
    else:
        images = x
        labels = y
    return images, labels
        
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    targets = [y_a, y_b, lam]
    return mixed_x, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, beta=1.0, use_cuda=True):
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    y_a = y
    y_b = y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    targets = [y_a, y_b, lam]

    return x, y_a, y_b, lam

def ricap_data(x, y, beta=0.3):
    I_x, I_y = x.size()[2:]

    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    cropped_images = {}
    c_ = []
    W_ = []
    for k in range(4):
        idx = torch.randperm(x.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = x[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_.append(y[idx].cuda())
        w_.append(w_[k] * h_[k] / (I_x * I_y))
        #c_[k] = y[idx].cuda()
        #W_[k] = w_[k] * h_[k] / (I_x * I_y)

    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
        torch.cat((cropped_images[2], cropped_images[3]), 2)),
    3)
    targets = [c_, w_]

    return patched_images, targets

