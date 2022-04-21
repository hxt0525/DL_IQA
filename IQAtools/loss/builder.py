# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 07:41:14 2021

@author: HXT
"""
from ..module_list import ModuleList
import torch
import inspect

Loss_List = ModuleList()

def insert_torch_losses():
    torch_losses = []
    for module_name in dir(torch.nn.modules.loss):
        if module_name.startswith('__'):
            continue
        _loss = getattr(torch.nn.modules.loss, module_name)
        #print(_loss)
        if inspect.isclass(_loss) and issubclass(_loss,torch.nn.modules.loss._Loss):
            Loss_List.insert_Module()(_loss)
            torch_losses.append(module_name)
    return torch_losses


TORCH_LOSSES = insert_torch_losses()

def build_loss(cfg):
    cur_cfg = cfg.copy()
    optimizer_type = cur_cfg['type']
    obj_cls = Loss_List.get(optimizer_type)
    cur_cfg.pop("type")
    return obj_cls(**cur_cfg)