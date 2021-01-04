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

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import (get_training_dataloader, get_test_dataloader, accuracy, 
                   most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, 
                   AverageMeter, set_seed)
from config.config import get_cfg_defaults
from nncls.models import build_network
from nncls.losses import build_loss
from nncls.optim import build_optim
from nncls.scheduler import build_scheduler, WarmUpLR
from nncls.transformer import build_transformer, aug_data

def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if cfg.GPU:
            labels = labels.cuda()
            images = images.cuda()
        r = np.random.rand(1)
        if r < cfg.AUG.PROB:
            aug_images, aug_labels = aug_data(cfg, images, labels)
        
        optimizer.zero_grad()
        outputs = net(aug_images)
        loss = train_loss(outputs, aug_labels)
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
def eval_training(epoch=0, tb=True):

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
        if cfg.GPU:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Avg loss: {:.4f}, Top 1 Acc: {:.4f}, Top 5 Acc: {:.4f}, Time consumed:{:.2f}s'.format(
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
    
    return acc1_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.LOCAL_RANK = args.local_rank
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.GPU_ID
   
    cudnn.benchmark = cfg.TRAIN.CUDNN
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
   
    train_loss, val_loss = build_loss(cfg)
    optimizer = build_optim(cfg, net)
    train_scheduler = build_scheduler(cfg, optimizer)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * cfg.TRAIN.WARM)
    resume_epoch = 1
    writer = None
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
        writer = SummaryWriter(log_dir=os.path.join(
                cfg.LOG_DIR, cfg.NET, cfg.TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
        #writer.add_graph(net, input_tensor)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_acc = 0.0
        best_epoch = 1
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
            train_scheduler.step(epoch)

        if cfg.TRAIN.RESUME:
            if epoch <= resume_epoch:
                continue
        if cfg.DIST:
            cifar100_training_loader.sampler.set_epoch(epoch)
        train(epoch)
        acc = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
        if not cfg.DIST or (cfg.DIST and dist.get_rank() == 0):
           
            if epoch >  cfg.TRAIN.STEPS[1] and best_acc < acc:
                if not cfg.DIST:
                    torch.save(net.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='best'))
                else:
                    torch.save(net.module.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='best'))
                best_acc = acc
                best_epoch = epoch
                print('best epoch: {}, best acc: {:.4f}'.format(best_epoch, best_acc))
                print()
                continue
            if epoch > cfg.TRAIN.STEPS[1]:
                print('best epoch: {}, best acc: {:.4f}'.format(best_epoch, best_acc))
                print()
            if not epoch % cfg.SAVE_EPOCH:
                if not cfg.DIST:
                    torch.save(net.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='regular'))
                else:
                    torch.save(net.module.state_dict(), checkpoint_path.format(net=cfg.NET, epoch=epoch, type='regular'))
    writer.close()
