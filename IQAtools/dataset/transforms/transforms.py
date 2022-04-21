# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:44:47 2021

@author: HXT
"""
import cv2
import numpy as np
from scipy.signal import convolve2d
from ..builder import Transforms_List
import torch
from torchvision import transforms as T

@Transforms_List.insert_Module()
class ToFloat(object):
    def __init__(self,IQA_type='NR'):
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        img = img.astype(np.float32) / 255
        if self.IQA_type == 'FR':
            ref = ref.astype(np.float32) / 255
            return img,ref
        else:
            return img

@Transforms_List.insert_Module()
class LocalNormalization(object):
    def __init__(self,IQA_type='NR'):
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None,P=3,Q=3,C=1):
        self.P = P
        self.Q = Q
        self.C = C
        for i in range(img.shape[2]):
            img[:,:,i] = self.__LocalNormalization(img[:,:,i])
            if self.IQA_type == 'FR':
               ref[:,:,i] = self.__LocalNormalization(ref[:,:,i])
        img = img.astype(np.float32) / 255
        if self.IQA_type == 'FR':
            ref = ref.astype(np.float32) / 255
            return img,ref
        else:
            return img
    def __LocalNormalization(self,patch):
        kernel = np.ones((self.P, self.Q)) / (self.P * self.Q)
        patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
        patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
        patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + self.C
        patch_ln = ((patch - patch_mean) / patch_std)
        return patch_ln
        

@Transforms_List.insert_Module()
class default_loader(object):    
    def __init__(self,IQA_type='NR'):
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        if self.IQA_type == 'NR':
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            img = cv2.imread(img)
            ref = cv2.imread(ref)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            return img,ref

@Transforms_List.insert_Module()
class default_loader2(object):    
    def __init__(self,IQA_type='NR'):
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        if self.IQA_type == 'NR':
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255.
            return img
        else:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
            ref = cv2.imread(ref)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)/255.
            return img,ref

def myLocalNormalization(patch, P=3, Q=3, C=1):
    for i in range(3):
        cur_patch = patch[:,:,i]
        kernel = np.ones((P,Q)) / (P * Q)
        patch_mean = convolve2d(cur_patch, kernel, boundary='symm', mode='same')
        patch_sm = convolve2d(np.square(cur_patch), kernel, boundary='symm', mode='same')
        patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
        patch_ln = torch.from_numpy((cur_patch - patch_mean) / patch_std).float().unsqueeze(0)
        patch[:,:,i] = patch_ln
    return patch
@Transforms_List.insert_Module()
class default_loader3(object):    
    def __init__(self,IQA_type='NR'):
        assert IQA_type in ['NR','FR']
        self.IQA_type = IQA_type

    def __call__(self,img,ref=None):
        if self.IQA_type == 'NR':
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = myLocalNormalization(img)
            img = img/255.
            return img
        else:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = myLocalNormalization(img)
            img = img/255.
            ref = cv2.imread(ref)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            ref = myLocalNormalization(ref)
            ref = ref/255.
            return img,ref


@Transforms_List.insert_Module()
class RandomRotate90(object):
    def __init__(self,p=0.5,IQA_type='NR'):
        self.IQA_type = IQA_type
        self.p = p
    def __call__(self,img,ref=None):
        if self.IQA_type == "NR":
            if np.random.random() < self.p:
                img = np.rot90(img)
            return img
        else:
            if np.random.random() < self.p:
                img = np.rot90(img)
                ref = np.rot90(ref)
            return img,ref

@Transforms_List.insert_Module()
class RandomVerticleFlip(object):
    def __init__(self,p=0.5,IQA_type='NR'):
        self.IQA_type = IQA_type
        self.p = p
    def __call__(self,img,ref=None):
        if self.IQA_type == "NR":
            if np.random.random() < self.p:
                img = cv2.flip(img, 0)
            return img
        else:
            if np.random.random() < self.p:
                img = cv2.flip(img, 0)
                ref = cv2.flip(ref, 0)
            return img,ref
@Transforms_List.insert_Module()
class RandomHorizontalFlip(object):
    def __init__(self,p=0.5,IQA_type='NR'):
        self.IQA_type = IQA_type
        self.p = p
    def __call__(self,img,ref=None):
        if self.IQA_type == "NR":
            if np.random.random() < self.p:
                img = cv2.flip(img, 1)
            return img
        else:
            if np.random.random() < self.p:
                img = cv2.flip(img, 1)
                ref = cv2.flip(ref, 1)
            return img,ref

@Transforms_List.insert_Module()
class Resize(object):
    def __init__(self,img_size,IQA_type='NR'):
        self.img_size = (img_size[0],img_size[1])
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        img = cv2.resize(img,self.img_size)
        if self.IQA_type == 'NR':
            return img
        else:
            ref = cv2.resize(ref,self.img_size)
            return img,ref
        
@Transforms_List.insert_Module()   
class RandomCropPatches(object):
    def __init__(self,IQA_type='NR',patch_size=(512,512),patch_num=1):
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.IQA_type = IQA_type
         
    def __call__(self,img,ref=None):
        w, h, z = img.shape
        patch_input_imgs = []
        if self.IQA_type == 'FR':
            patch_input_refs = []
        n = 1
        while n <= self.patch_num:
            top = np.random.randint(0, w - self.patch_size[0])
            left = np.random.randint(0, h - self.patch_size[1])
            patch_input_img = img[top:top+self.patch_size[0],left:left+self.patch_size[1],:]
            n += 1
            #patch_input_img = patch_input_img.transpose(2,0,1)
            patch_input_imgs.append(patch_input_img)
            if self.IQA_type == 'FR':
                patch_input_ref = ref[top:top+self.patch_size[0],left:left+self.patch_size[1],:]
                #patch_input_ref = patch_input_ref.transpose(2,0,1)
                patch_input_refs.append(patch_input_ref)
            
        
        patch_input_imgs = np.array(patch_input_imgs)
        if self.IQA_type == 'NR':
            return patch_input_imgs
        else:
            patch_input_refs = np.array(patch_input_refs)
            return patch_input_imgs,patch_input_refs


@Transforms_List.insert_Module()    
class CropPatches(object):
    
    def __init__(self,IQA_type='NR',patch_size=(512,512), stride=(512,512)):
        self.patch_size = patch_size
        self.stride = stride
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        x, y, z = img.shape
        x_num = ( x - self.patch_size[0] )//self.stride[0] + 1
        y_num = ( y - self.patch_size[1] )//self.stride[1] + 1
        while((x_num-1) * self.stride[0] + self.patch_size[0] < x): x_num = x_num + 1
        while((y_num-1) * self.stride[1] + self.patch_size[1] < y): y_num = y_num + 1
        patches = ()
        if self.IQA_type == 'FR':
            patches_ref = ()
        for i in range(x_num):
            for j in range(y_num):
                x_s, x_e = i*self.stride[0], i*self.stride[0]+self.patch_size[0]
                y_s, y_e = j*self.stride[1], j*self.stride[1]+self.patch_size[1]
                if(y_e > y):
                    y_e = y
                    y_s = y - self.patch_size[1]
                if(x_e > x):
                    x_e = x
                    x_s = x - self.patch_size[0]
                patch = img[x_s:x_e, y_s:y_e,:]
                if self.IQA_type == 'FR':
                    patch_ref = ref[x_s:x_e, y_s:y_e,:]
                #patch = patch.transpose(2,0,1)
                patches = patches + (patch,)
                if self.IQA_type == 'FR':
                    #patch_ref = patch_ref.transpose(2,0,1)
                    patches_ref = patches_ref + (patch_ref,)
        patches = np.array(patches)
        if self.IQA_type == 'NR':
            return patches
        else:
            patches_ref = np.array(patches_ref)
            return patches,patches_ref


     
@Transforms_List.insert_Module()
class ToTensor(object):
    def __init__(self,IQA_type='NR'):
        self.IQA_type = IQA_type
    def __call__(self,img,ref=None):
        dim = len(img.shape)
        img = np.ascontiguousarray(img)
        if(dim == 3):
            img = img.transpose(2,0,1)
        elif(dim == 4):
            img = img.transpose(0,3,1,2)
        elif(dim == 5):
            img = img.transpose(0,1,4,2,3)
        if self.IQA_type == 'NR':
            return torch.Tensor(img)
        else:
            ref = np.ascontiguousarray(ref)
            if(dim == 3):
                ref = ref.transpose(2,0,1)
            elif(dim == 4):
                ref = ref.transpose(0,3,1,2)
            elif(dim == 5):
                ref = ref.transpose(0,1,4,2,3)
            return torch.Tensor(img),torch.Tensor(ref)
        
@Transforms_List.insert_Module()
class Normalize(object):
    def __init__(self,mean,std,IQA_type='NR'):
        self.IQA_type = IQA_type
        self.mean = [mean[0],mean[1],mean[2]]
        self.std = [std[0],std[1],std[2]]
        self.normalize = T.Normalize(mean=self.mean,std=self.std)
    def __call__(self,img,ref=None):
        if self.IQA_type =='NR':
            img = self.normalize(img)
            return img
        else:
            img = self.normalize(img)
            ref = self.normalize(ref)
            return img,ref
            

if __name__ == "__main__":
    img = 'test.bmp'
    ref = 'ref.bmp'
    my_defaulter_loader = default_loader(IQA_type='FR')
    #my_LN = LocalNormalization(IQA_type='FR')
    my_float = ToFloat(IQA_type='FR')
    my_crop = RandomCropPatches(IQA_type='FR')    
    img,ref = my_defaulter_loader(img,ref)
    #img,ref = my_LN(img,ref)
    img,ref = my_float(img,ref)
    img,ref = my_crop(img,ref)
    import matplotlib.pyplot as plt
    import os
    img = img[0]
    img = img.transpose(1,2,0)
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    plt.imshow(img)
    
    ref = ref[0]
    ref = ref.transpose(1,2,0)
    plt.figure()
    plt.imshow(ref)
    
    print(my_crop)