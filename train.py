# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:40:02 2021

@author: HXT
"""

from IQAtools.utils.utils import config_loader,config_writer,get_sampler
from IQAtools.models.builder import build_model
from IQAtools.dataset.builder import build_dataset
from IQAtools.optimizer.builder import build_optimizer,build_lr_scheduler
from IQAtools.loss.builder import build_loss
from IQAtools.runner.basic_runner import BasicRunner
from IQAtools.runner.IQA_runner import IQARunner
from torch.utils.data import DataLoader
import shutil
import os
import argparse
import torch
import torch.backends.cudnn as cudnn


def main(args):
    config_path = args.cfg_path
    cfg = config_loader(config_path)
    # 根据命令行修改配置文件,如果命令行中未指定如下参数，将按照当前配置文件中的对应参数
    # 否则优先使用命令行中的参数
    if args.checkpoint_dir is not None:
        cfg["train_config"]["checkpoint_dir"] = args.checkpoint_dir
    if args.train_img_info_file is not None:
        cfg["dataset"]["train"]["img_info_file"] = args.train_img_info_file
    if args.train_dis_img_path is not None:
        cfg["dataset"]["train"]["dis_img_path"] = args.train_dis_img_path
    if args.val_img_info_file is not None:
        cfg["dataset"]["val"]["img_info_file"] = args.val_img_info_file
    if args.val_dis_img_path is not None:
        cfg["dataset"]["val"]["dis_img_path"] = args.val_dis_img_path
    
    # 使用当前cfg配置生成新的配置文件存到checkpoint_dir文件夹
    checkpoint_dir = cfg["train_config"]["checkpoint_dir"]
    if args.local_rank<=0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        out_config_name = os.path.join(checkpoint_dir,"config.yaml")
        config_writer(out_config_name,cfg)
    # 是否distributed
    distributed = args.local_rank >= 0
    gpus = cfg['gpu_ids']
    if distributed:
        # 初始化
        device = torch.device('cuda:{}'.format(gpus[args.local_rank]))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    else:
        # 不distributed设置主卡
        device = torch.device('cuda:{}'.format(gpus[0]))
        torch.cuda.set_device(device)
    
    model = build_model(cfg["model"])
    if distributed:
        # 多进程bn设置
        if len(gpus)>1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        else:
            model = model.to(device)
        # 多进程模型ddp
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpus[args.local_rank]],
        )
        cudnn.benchmark = True
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpus,output_device=gpus[0])
    
    train_dataset = build_dataset(cfg["dataset"]["train"])
    if distributed:
        train_batch_size = cfg["train_config"]["train_batch_size"]
    else:
        train_batch_size = cfg["train_config"]["train_batch_size"] * len(gpus)
    train_sampler = get_sampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True if train_sampler is None else False,
                                  num_workers=4,pin_memory=True,drop_last=True,sampler=train_sampler)
    
    val_dataset = build_dataset(cfg["dataset"]["val"])
    val_batch_size = cfg["train_config"]["val_batch_size"]
    ###########  只在主卡上进行验证
    if args.local_rank <=0:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset,batch_size=val_batch_size, 
                                  shuffle=False,num_workers=4,pin_memory=True,sampler=val_sampler)
    else:
        val_dataloader = None
    optimizer = build_optimizer(model,cfg["optimizer"])
    if "lr_scheduler" in cfg.keys():
        scheduler = build_lr_scheduler(optimizer,cfg["lr_scheduler"])
    else:
        scheduler = None
    criterion = build_loss(cfg["loss"])
    local_rank = args.local_rank
    runner = IQARunner(cfg,model,
                 train_dataloader,
                 train_sampler,
                 val_dataloader,
                 optimizer,
                 criterion,
                 local_rank,
                 scheduler)
    runner.run()

def parse_args():
    parser = argparse.ArgumentParser(description='Train IQA network')
    parser.add_argument('--cfg_path',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--train_img_info_file', type=str, default=None)
    parser.add_argument('--train_dis_img_path', type=str, default=None)
    parser.add_argument('--val_img_info_file', type=str, default=None)
    parser.add_argument('--val_dis_img_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.seed > 0:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed) 
    main(args)