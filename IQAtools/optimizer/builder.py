# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 07:41:14 2021

@author: HXT
"""
from ..module_list import ModuleList
import torch
import inspect

Optimizer_List = ModuleList()

#copy from mmcv
def insert_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            Optimizer_List.insert_Module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = insert_torch_optimizers()

LR_scheduler_List = ModuleList()
def insert_torch_lr_scheduler():
    torch_lr_schedulers = []
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        _scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler,
                                                  torch.optim.lr_scheduler._LRScheduler):
            LR_scheduler_List.insert_Module()(_scheduler)
            torch_lr_schedulers.append(module_name)
    return torch_lr_schedulers

TORCH_SCHEDULERS = insert_torch_lr_scheduler()


def build_optimizer(model, cfg):
    cur_cfg = cfg.copy()
    optimizer_type = cur_cfg['type']
    obj_cls = Optimizer_List.get(optimizer_type)
    cur_cfg.pop("type")
    return obj_cls(model.parameters(),**cur_cfg)

def build_lr_scheduler(optimizer,cfg):
    cur_cfg = cfg.copy()
    scheduler_type = cur_cfg['type']
    obj_cls = LR_scheduler_List.get(scheduler_type)
    cur_cfg.pop("type")
    return obj_cls(optimizer,**cur_cfg)
