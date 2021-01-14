# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import *
from config.config import get_cfg_defaults
from nncls.models import build_network
from nncls.losses import build_loss
from nncls.optim import build_optim
from nncls.scheduler import build_scheduler, WarmUpLR
from nncls.transformer import build_transformer, aug_data
from nncls.utils import *

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def schedule(epoch):
    t = (epoch) / (cfg.TRAIN.SWA_START)
    lr_ratio = cfg.TRAIN.SWA_LR / cfg.TRAIN.LR 
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return cfg.TRAIN.LR * factor

def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if cfg.GPU:
            labels = labels.cuda()
            images = images.cuda()
        r = np.random.rand(1)
        #aug_images, aug_labels = images, labels
        if cfg.TRAIN.OPTIM == 'sam':
            def closure():
                optimizer.zero_grad()
                if r < cfg.AUG.PROB:
                    aug_images, aug_labels = aug_data(cfg, images, labels)
                    outputs = net(aug_images)
                    loss = aug_loss(outputs, aug_labels)
                else:
                    aug_images, aug_labels = images, labels
                    outputs = net(aug_images)
                    loss = train_loss(outputs, aug_labels)
                if cfg.APEX:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            if r < cfg.AUG.PROB:
                aug_images, aug_labels = aug_data(cfg, images, labels)
                outputs = net(aug_images)
                loss = aug_loss(outputs, aug_labels)
            else:
                aug_images, aug_labels = images, labels
                outputs = net(aug_images)
                loss = train_loss(outputs, aug_labels)
            
            # outputs = net(aug_images)
            # loss = train_loss(outputs, aug_labels)
            if cfg.APEX:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()
        if epoch <= cfg.TRAIN.WARM:
            warmup_scheduler.step()

        if cfg.DIST:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            loss_all_reduce.wait()
            loss.div_(dist.get_world_size())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not cfg.DIST or cfg.DIST and dist.get_rank() == 0:
            n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

            last_layer = list(net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
            if cfg.DIST:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * cfg.TRAIN.BATCH * dist.get_world_size() + len(images) * dist.get_world_size(),
                    total_samples=len(cifar100_training_loader.dataset)
                ))
            else:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * cfg.TRAIN.BATCH  + len(images),
                    total_samples=len(cifar100_training_loader.dataset)
                ))

            #update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)
    if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
   
@torch.no_grad()
def eval_training(net, epoch=0, tb=True, suffix=''):

    start = time.time()
    net.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    for (images, labels) in cifar100_test_loader:

        if cfg.GPU:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = val_loss(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
       
        if cfg.DIST:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc1_all_reduce = dist.all_reduce(acc1,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc5_all_reduce = dist.all_reduce(acc5,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            loss_all_reduce.wait()
            acc1_all_reduce.wait()
            acc5_all_reduce.wait()
            loss.div_(dist.get_world_size())
            acc1.div_(dist.get_world_size())
            acc5.div_(dist.get_world_size())
        loss = loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()
        num = images.size(0)
        loss_meter.update(loss, num)
        acc1_meter.update(acc1, num)
        acc5_meter.update(acc5, num)
        if torch.cuda.is_available():
            torch.cuda.synchronize()     
    if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
        finish = time.time()
        if suffix == '':
            if cfg.GPU:
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')
            print('Evaluating Network.....')
        print('{} Test set: Epoch: {}, Avg loss: {:.4f}, Top 1 Acc: {:.4f}, Top 5 Acc: {:.4f}, Time consumed:{:.2f}s'.format(
            suffix,
            epoch,
            loss_meter.avg,
            acc1_meter.avg,
            acc5_meter.avg,
            finish - start
        ))
        print()
        #add informations to tensorboard
        if tb:
            writer.add_scalar('Test/Average loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/Accuracy', acc1_meter.avg, epoch)
    
    return acc1_meter.avg, acc5_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.LOCAL_RANK = args.local_rank
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.GPU_ID
    torch.backends.cudnn.deterministic = cfg.TRAIN.DETEM 
    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN
    if cfg.DIST:
        torch.cuda.set_device(cfg.LOCAL_RANK)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        set_seed(1 + rank)
    else:
        set_seed(cfg.TRAIN.SEED)
    
    if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
        print('----------------config-----------------')
        print('use gpu id: ', cfg.TRAIN.GPU_ID)
    
    net = build_network(cfg)
    if cfg.APEX and cfg.DIST and cfg.SYNC_BN:
        net = apex.parallel.convert_syncbn_model(net)
        if dist.get_rank() == 0:
            print('using apex synced BN')
    elif not cfg.APEX and cfg.DIST and cfg.SYNC_BN:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)(net)
        if dist.get_rank() == 0:
            print('using torch ddp synced BN')
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        cfg,
        cfg.CIFAR100_TRAIN_MEAN,
        cfg.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=cfg.TRAIN.BATCH,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        cfg,
        cfg.CIFAR100_TRAIN_MEAN,
        cfg.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=cfg.TRAIN.BATCH,
        shuffle=True
    )
   
    train_loss, aug_loss, val_loss = build_loss(cfg)
    optimizer = build_optim(cfg, net)
    swa_model = None
    if cfg.TRAIN.SCHEDULER == 'step' and cfg.TRAIN.SWA:
        swa_model = deepcopy(net)
        swa_n = 0
    # 通过调整下面的opt_level实现半精度训练。
    # opt_level选项有：'O0', 'O1', 'O2', 'O3'.
    # 其中'O0'是fp32常规训练，'O1'、'O2'是fp16训练，'O3'则可以用来推断但不适合拿来训练（不稳定）
    # 注意，当选用fp16模式进行训练时，keep_batchnorm默认是None，无需设置；
    # scale_loss是动态模式，可以设置也可以不设置。
    if cfg.APEX:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level=cfg.OPT_LEVEL)
        net = DDP(net)

    train_scheduler = build_scheduler(cfg, optimizer)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * cfg.TRAIN.WARM)
    resume_epoch = 1
    writer = SummaryWriter(log_dir=os.path.join(
                cfg.LOG_DIR, cfg.NET, cfg.TIME_NOW))
    # setup exponential moving average of model weights, SWA could be used here too
        
    if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
        print('----------------config-----------------')
   
        if cfg.TRAIN.RESUME:
            recent_folder = most_recent_folder(os.path.join(cfg.CHECKPOINT_PATH, cfg.NET), fmt=cfg.DATE_FORMAT)
            if not recent_folder:
                raise Exception('no recent folder were found')

            checkpoint_path = os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder)

        else:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, cfg.TIME_NOW)

        #use tensorboard
        if not os.path.exists(cfg.LOG_DIR):
            os.mkdir(cfg.LOG_DIR)

        #since tensorboard can't overwrite old values
        #so the only way is to create a new tensorboard log
        #input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
        #writer.add_graph(net, input_tensor)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_acc1 = 0.0
        best_epoch = 1
        best_acc5 = 0.0
        if cfg.TRAIN.SWA:
            best_swa_acc1 = 0.0
            best_swa_acc5 = 0.0
        if cfg.TRAIN.RESUME:
            best_weights = best_acc_weights(os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder))
            if best_weights:
                weights_path = os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder, best_weights)
                print('found best acc weights file:{}'.format(weights_path))
                print('load best training file to test acc...')
                if cfg.DIST:
                    net.module.load_state_dict(torch.load(weights_path))
                else:
                    net.load_state_dict(torch.load(weights_path))
                best_acc = eval_training(tb=False)
                print('best acc is {:0.2f}'.format(best_acc))

            recent_weights_file = most_recent_weights(os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder))
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            if cfg.DIST:
                net.module.load_state_dict(torch.load(weights_path))
            else:
                net.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(os.path.join(cfg.CHECKPOINT_PATH, cfg.NET, recent_folder))

    for epoch in range(1, cfg.TRAIN.EPOCHES):
        if epoch > cfg.TRAIN.WARM:
            if cfg.TRAIN.SWA:
                lr = schedule(epoch)
                adjust_learning_rate(optimizer, lr)
            else:
                train_scheduler.step(epoch)

        if cfg.TRAIN.RESUME:
            if epoch <= resume_epoch:
                continue
        if cfg.DIST:
            cifar100_training_loader.sampler.set_epoch(epoch)
        train(epoch)
        acc1, acc5 = eval_training(net, epoch)
        if cfg.TRAIN.SWA and  epoch >= cfg.TRAIN.SWA_START:
            moving_average(swa_model, net, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(cifar100_training_loader, swa_model)
            swa_acc1, swa_acc5 = eval_training(swa_model, epoch, suffix='SWA')
            if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
                if best_swa_acc1 < swa_acc1:
                    best_swa_acc1 = swa_acc1
                    best_swa_acc5 = swa_acc5
                print('best swa acc1: {:.4f}, best swa acc5: {:.4f}'.format(best_swa_acc1, best_swa_acc5))
                print()
        #start to save best performance model after learning rate decay to 0.01
        if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
            if epoch >  cfg.TRAIN.STEPS[1] and best_acc1 < acc1:
                if not cfg.DIST:
                    torch.save(net.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='best'))
                else:
                    torch.save(net.module.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='best'))
                best_acc1 = acc1
                best_acc5 = acc5
                best_epoch = epoch
                print('best epoch: {}, best acc1: {:.4f}, acc5: {:.4f}'.format(best_epoch, best_acc1, best_acc5))
                print()
                continue
            if epoch > cfg.TRAIN.STEPS[1]:
                print('best epoch: {}, best acc1: {:.4f}, acc5: {:.4f}'.format(best_epoch, best_acc1, best_acc5))
                print()
            if not epoch % cfg.SAVE_EPOCH:
                if not cfg.DIST:
                    torch.save(net.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='regular'))
                else:
                    torch.save(net.module.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='regular'))
    writer.close()
