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
import cv2

@Dataset_List.insert_Module()
class ErasePatchDatabase(Dataset):
    def __init__(self,
                 img_info_file = None,
                 img_info = None,
                 dis_img_path = None,
                 erase_ratio = 0.5,
                 erase_patchsize = 16,
                 IQA_transforms =None):
        super(ErasePatchDatabase,self).__init__()
        self.img_info_file = img_info_file
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
        self.dis_img_path = dis_img_path
        self.IQA_transforms = IQA_transforms
        self.erase_ratio = erase_ratio
        self.erase_patchsize = erase_patchsize
    
    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img = os.path.join(self.dis_img_path,self.img_info[idx][0])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.
        img = img.astype(np.float32)
        dis = img.copy()
        x, y, z = img.shape
        x_num = int(x / self.erase_patchsize)
        y_num = int(y / self.erase_patchsize)
        for i in range(x_num):
            for j in range(y_num):
                x_s, x_e = i*self.erase_patchsize, (i+1)*self.erase_patchsize
                y_s, y_e = j*self.erase_patchsize, (j+1)*self.erase_patchsize
                # print(x_s,x_e,y_s,y_e)
                if np.random.random() < self.erase_ratio:
                    dis[x_s:x_e,y_s:y_e,:] = 0.5
        for IQA_transform in self.IQA_transforms:
            dis,img = IQA_transform(dis,img)
        return dis,img
