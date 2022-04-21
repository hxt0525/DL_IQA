# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:29:57 2021

@author: HXT
"""

import time
import torch
from tqdm import tqdm
import numpy as np
from scipy import stats
import os
os.environ['KMP_DUPLICATE_LIB_OK']="True"
from torch.autograd import Variable
def IQA_perform(q,labels):
    q = np.array(q)
    sq = np.array(labels)
    srocc = stats.spearmanr(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    return srocc,plcc

class BasicRunner(object):
    def __init__(self,cfg,model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 criterion,
                 scheduler=None):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cfg = cfg
        self.train_cfg = cfg["train_config"]
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cur_epoch = 1
        self.checkpoint_dir = self.train_cfg["checkpoint_dir"]
        self.train_info_file = os.path.join(self.checkpoint_dir,"training_info.txt")
        f = open(self.train_info_file,mode='w')
        f.truncate(0)
        f.close()
    def logger(self,info):
        f = open(self.train_info_file,mode='a')
        f.write(info + '\n')
        f.close()  
    def train(self):
        print("-"*10 + "training" + "-"*10)
        self.model.train()
        train_loss = 0
        log_iter = self.train_cfg['log_iter']
        device_id = self.cfg["gpu_ids"][0]
        for batch_idx, (data,target) in enumerate(self.train_dataloader):
            data, target = Variable(data.cuda(device_id)), Variable(target.cuda(device_id))
            data = data.view(-1, 3, data.shape[-2], data.shape[-1])
            target = target.view(-1,1)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred,target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if((batch_idx+1) % log_iter == 0):
                train_loss /= log_iter
                display_info = "current epoch:%d,current_iter:%d,loss:%f"%(self.cur_epoch,batch_idx+1,train_loss)
                print(display_info)
                self.logger(display_info)
                train_loss = 0
        self.cur_epoch += 1
        self.scheduler.step()
    def val(self):
        print("-"*10 + "val" + "-"*10)
        time.sleep(1)
        self.model.eval()
        pred_results = []
        true_results = []
        with torch.no_grad():
            for batch_idx, (data,target) in enumerate(tqdm(self.val_dataloader)):
                data = data.view(-1, 3, data.shape[-2], data.shape[-1])
                target = target.view(-1)
                pred = self.model(data)
                pred = pred.view(-1)
                pred_results.extend(pred.cpu().numpy())
                true_results.extend(target.cpu().numpy())
        srocc,plcc = IQA_perform(true_results,pred_results)
        time.sleep(1)
        display_info = "current epoch:%d,plcc:%f,srocc:%f"%(self.cur_epoch-1,plcc,srocc)
        print(display_info)
        self.logger(display_info)
    
    def save(self):
        state = {'state_dict':self.model.module.state_dict()}
        checkpoint_dir = self.train_cfg["checkpoint_dir"]
        filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(self.cur_epoch-1))
        torch.save(state, filename,_use_new_zipfile_serialization=False)

    def run(self):
        total_epoch = self.train_cfg["epoch"]
        save_epoch = self.train_cfg["save_epoch"]
        val_epoch = self.train_cfg["val_epoch"]
        for i in range(total_epoch):
            self.train()
            if(self.cur_epoch % val_epoch == 0):
                self.val()
            if(self.cur_epoch % save_epoch == 0):
                self.save()
        print("training finish!!!")