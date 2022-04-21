# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 02:01:28 2021

@author: HXT
"""

from ..module_list import ModuleList

Dataset_List = ModuleList()
Transforms_List = ModuleList()

def build_transform(cfg):
    cur_cfg = cfg.copy()
    transform_type = cur_cfg.get("type")
    obj_cls = Transforms_List.get(transform_type)
    cur_cfg.pop("type")
    return obj_cls(**cur_cfg)

def build_dataset(cfg):
    cur_cfg = cfg.copy()
    model_type = cur_cfg["type"]
    cur_cfg.pop("type")
    obj_cls = Dataset_List.get(model_type)
    transforms = cur_cfg.get("IQA_transforms")
    cur_cfg.pop("IQA_transforms")
    IQA_transformers = []
    for transform in transforms:
        IQA_transformers.append(build_transform(transform))
    cur_cfg['IQA_transforms'] = IQA_transformers
    return obj_cls(**cur_cfg)
