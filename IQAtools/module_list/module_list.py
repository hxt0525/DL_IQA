# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 01:55:08 2021

@author: HXT
"""

class ModuleList(object):
    def __init__(self):
        self._ModuleList = dict()
    def insert_Module(self,name=None,module=None):
        if module is not None:
            self._insert_Module(module_class=module,module_name=name)
            return module
        def _register(cls):
            self._insert_Module(module_class=cls,module_name=name)
            return cls
        return _register
    def _insert_Module(self,module_class,module_name=None):
        if module_name is None:
            module_name = module_class.__name__
        if module_name not in self._ModuleList:
            self._ModuleList[module_name] = module_class
    def get(self,module_name):
        return self._ModuleList.get(module_name)
    def getAllmodule(self):
        return list(self._ModuleList.keys())