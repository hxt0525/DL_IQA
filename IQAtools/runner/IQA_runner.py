import time
import torch
from tqdm import tqdm
import numpy as np
from scipy import stats
import os
from ..utils.utils import IQA_perform,IQA_perform_Tensor,MetricLogger,SmoothedValue
import logging

class IQARunner(object):
    def __init__(self,cfg,model,
                 train_dataloader,
                 train_sampler,
                 val_dataloader,
                 optimizer,
                 criterion,
                 local_rank,
                 scheduler=None):     
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.val_dataloader = val_dataloader
        self.cfg = cfg
        self.train_cfg = cfg["train_config"]
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cur_epoch = 1
        self.checkpoint_dir = self.train_cfg["checkpoint_dir"]
        self.local_rank = local_rank
        self.best_srocc = [0,0]
        self.best_plcc = [0,0]
        if self.local_rank <= 0:           
            self.train_info_file = os.path.join(self.checkpoint_dir,"training_info.txt")
            f = open(self.train_info_file,mode='w')
            f.truncate(0)
            f.close()
            fmt = '%(asctime)s %(name)s %(levelname)s %(message)s'
            self.logger = logging.getLogger("local training")
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.train_info_file, 'a')
            formatter = logging.Formatter(fmt)
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
            self.logger.propagate = False       
        else:
            self.logger = None
    def train(self):
        self.model.train()
        if self.local_rank <= 0:
            print("-"*10 + "training" + "-"*10)
            metric_logger = MetricLogger(logger=self.logger,is_print=True,delimiter="  ")
        else:
            metric_logger = MetricLogger(logger=None,is_print=False,delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
        header = 'Epoch: [{}]'.format(self.cur_epoch)
        log_iter = self.train_cfg['log_iter']
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.cur_epoch)
        for image,target in metric_logger.log_every(self.train_dataloader,print_freq=log_iter,header=header):
            start_time = time.time()
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            #print(self.local_rank)
            #print(target)
            image = image.view(-1, 3, image.shape[-2], image.shape[-1])
            target = target.view(-1)
            pred = self.model(image)
            pred = pred.view(-1)
            loss = self.criterion(pred,target)
            #if self.local_rank>=0:
            #    torch.distributed.barrier()
            self.optimizer.zero_grad()
            #print(str(self.local_rank)+"   begin")
            loss.backward()
            #print(str(self.local_rank)+"   over")
            self.optimizer.step()
            srocc,krocc,plcc,rmse,mae = IQA_perform_Tensor(pred,target)
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters['srocc'].update(srocc, n=batch_size)
            metric_logger.meters['krocc'].update(krocc, n=batch_size)
            metric_logger.meters['plcc'].update(plcc, n=batch_size)
            metric_logger.meters['rmse'].update(rmse, n=batch_size)
            metric_logger.meters['mae'].update(mae, n=batch_size)
            metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if self.local_rank >= 0:
            #print(str(self.local_rank)+"   begin")
            metric_logger.synchronize_between_processes()
            #print(str(self.local_rank)+"   over")
        if self.local_rank <= 0:
            print('SROCC: {sr.global_avg:.4f} KROCC: {kr.global_avg:.4f} PLCC: {pl.global_avg:.4f} RMSE: {rm.global_avg:.4f} MAE: {ma.global_avg:.4f} LOSS: {ls.global_avg:.4f}'.format(sr=metric_logger.srocc,kr=metric_logger.krocc,
                                                                    pl=metric_logger.plcc,rm=metric_logger.rmse,ma=metric_logger.mae,ls=metric_logger.loss))
            self.logger.info('SROCC: {sr.global_avg:.4f} KROCC: {kr.global_avg:.4f} PLCC: {pl.global_avg:.4f} RMSE: {rm.global_avg:.4f} MAE: {ma.global_avg:.4f} LOSS: {ls.global_avg:.4f}'.format(sr=metric_logger.srocc,kr=metric_logger.krocc,
                                                                    pl=metric_logger.plcc,rm=metric_logger.rmse,ma=metric_logger.mae,ls=metric_logger.loss))   
        if self.scheduler is not None:
            self.scheduler.step()
    def val(self):
        if self.local_rank <= 0: 
            print("-"*10 + "val" + "-"*10)
        time.sleep(1)
        self.model.eval()
        pred_results = []
        true_results = []
        val_loss = 0
        iter_count = 0
        with torch.no_grad():
            for batch_idx, (data,target) in enumerate(tqdm(self.val_dataloader)):
                data = data.cuda()
                data = data.view(-1, 3, data.shape[-2], data.shape[-1])
                target = target.cuda()
                target = target.view(-1)
                pred = self.model(data)
                pred = pred.view(-1)
                cur_loss = self.criterion(pred,target)
                val_loss += cur_loss.item()
                iter_count += 1
                pred_results.extend(pred.cpu().numpy())
                true_results.extend(target.cpu().numpy())
        srocc,krocc,plcc,rmse,mae = IQA_perform(true_results,pred_results)
        if srocc > self.best_srocc[1]:
            self.best_srocc[1] = srocc
            self.best_srocc[0] = self.cur_epoch
            self.best_plcc[1] = plcc
            self.best_plcc[0] = self.cur_epoch
            self.save(save_name="best_srocc_model")
        val_loss /= iter_count
        display_info = "current epoch:%d SROCC:%.4f  PLCC:%.4f  LOSS:%.4f"%(self.cur_epoch,srocc,plcc,val_loss)
        print(display_info)
        self.logger.info(display_info)
        display_info = "best epoch:%d    SROCC:%.4f  PLCC:%.4f"%(self.best_srocc[0],self.best_srocc[1],self.best_plcc[1])
        print(display_info)
        self.logger.info(display_info)
    
    def save(self,save_name=None):
        state = {'model':self.model.module.state_dict()}
        checkpoint_dir = self.train_cfg["checkpoint_dir"]
        if save_name is None:
            filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(self.cur_epoch))
        else:
            filename = os.path.join(checkpoint_dir, str(save_name) + ".pth")
        torch.save(state, filename,_use_new_zipfile_serialization=False)

    def run(self):
        total_epoch = self.train_cfg["epoch"]
        save_epoch = self.train_cfg["save_epoch"]
        val_epoch = self.train_cfg["val_epoch"]
        for i in range(total_epoch):
            self.train()
            if(self.cur_epoch % val_epoch == 0 and self.local_rank<=0):
                self.val()
            if(self.cur_epoch % save_epoch == 0 and self.local_rank<=0):
                self.save()
            self.cur_epoch += 1
        if(self.local_rank<=0):
            print("training finish!!!")