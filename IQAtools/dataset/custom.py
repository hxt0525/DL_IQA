# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:02:39 2021

@author: HXT
"""

from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import numpy as np
from .builder import Dataset_List

'''
说明：常规IQA数据库
输入参数：
   IQA_type:  'NR'或者‘FR’
   img_info_file: 图片标签信息文件路径
     标签文件信息格式 请使用xlsx、txt格式
        对于全参考(FR): 第一列是失真图片名   第二列是参考图片名  第三列是分数
        对于无参考(NR): 第一列是失真图片名   第二列是分数
   dis_img_path:失真图片文件夹路径
   ref_img_path:参考图片文件夹路径
   IQA_transforms: 对图片的处理,使用列表方式存储IQAtransforms.py中的方法

'''
@Dataset_List.insert_Module()
class CustomDatabase(Dataset):
    def __init__(self,
                 IQA_type='NR',
                 nor_value = 5,
                 img_info_file = None,
                 img_info = None,
                 dis_img_path = None,
                 ref_img_path = None,
                 IQA_transforms =None):
        super(CustomDatabase,self).__init__()
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type
        self.img_info_file = img_info_file
        self.nor_value = float(nor_value)
        if img_info is None:
            self.label_file_type = self.img_info_file.split(".")[-1]
            assert self.label_file_type in ["xlsx","txt","XLSX","TXT"]
            assert os.path.exists(img_info_file)          
            if self.label_file_type in ["xlsx","XLSX"]:
                self.img_info = pd.read_excel(self.img_info_file)
                self.img_info = np.array(self.img_info)
            elif self.label_file_type in ["txt","TXT"]:             
                self.img_info = np.loadtxt(self.img_info_file,dtype=str)
        else:
            self.img_info = img_info
        if(self.IQA_type == 'NR'):
            assert self.img_info.shape[1] == 2
        else:
            assert self.img_info.shape[1] == 3
        self.dis_img_path = dis_img_path
        self.ref_img_path = ref_img_path
        self.IQA_transforms = IQA_transforms
    
    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        if self.IQA_type=='NR':
            img = os.path.join(self.dis_img_path,self.img_info[idx][0])
            for IQA_transform in self.IQA_transforms:
                img = IQA_transform(img)
            if(len(img.shape)==3):
                label = np.ones((1,1))
            elif(len(img.shape)==4):
                label = np.ones((img.shape[0],1))
            label *= float(self.img_info[idx][1])
            return img,label/self.nor_value
        else:
            img = os.path.join(self.dis_img_path,self.img_info[idx][0])
            ref = os.path.join(self.ref_img_path,self.img_info[idx][1])
            for IQA_transform in self.IQA_transforms:
                img,ref = IQA_transform(img,ref)
            label = np.ones((img.shape[0],1))
            label *= float(self.img_info[idx][2])
            return img,ref,label/self.nor_value
    
