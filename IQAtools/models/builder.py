# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:52:23 2021

@author: HXT
"""

from ..module_list import ModuleList

Model_List = ModuleList()

def build_model(cfg):
    cur_cfg = cfg.copy()
    model_type = cur_cfg["type"]
    cur_cfg.pop("type")
    obj_cls = Model_List.get(model_type)
    return obj_cls(**cur_cfg)
    