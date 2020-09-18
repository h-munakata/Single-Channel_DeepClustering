import sys
from model import loss
import torch
import os
import yaml
from tqdm import tqdm
import logging
import tensorboardX as tbx


class Trainer():
    def __init__(self,train_dataloader,val_dataloader,dpcl,optimizer,config):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_spks = config['num_spks']
        self.cur_epoch = 0
        self.total_epoch = config['train']['epoch']
        self.early_stop = config['train']['early_stop']
        self.checkpoint = config['train']['path']
        self.name = config['name']

        # about setting of machine
        if config['train']['is_gpu']:
            self.device = torch.device('cuda:0')
            self.dpcl = dpcl.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.dpcl = dpcl.to(self.device)

        
        
        # about restart
        if config['resume']['state']:    
            ckp = torch.load(config['resume']['path'],map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.dpcl = dpcl.load_state_dict(ckp['model_state_dict']).to(self.device)
            self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.dpcl = dpcl.to(self.device)
            self.optimizer = optimizer
        
        if config['optim']['clip_norm']:
            self.clip_norm = config['optim']['clip_norm']
        else:
            self.clip_norm = 0
        

    def train(self, epoch):
        self.dpcl.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        print('epoch{}:train'.format(epoch))
        for log_pow_mix, class_targets, non_silent in tqdm(self.train_dataloader):
            log_pow_mix = log_pow_mix.to(self.device)
            class_targets = class_targets.to(self.device)
            non_silent = non_silent.to(self.device)

            mix_embs = self.dpcl(log_pow_mix)
            epoch_loss = loss(mix_embs, class_targets, non_silent, self.num_spks, self.device)
            total_loss += epoch_loss.item()

            self.optimizer.zero_grad()
            epoch_loss.backward()
            
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.dpcl.parameters(),self.clip_norm)

            self.optimizer.step()

        total_loss = total_loss/num_batchs

        return total_loss

    def validation(self, epoch):
        self.dpcl.eval()
        num_batchs = len(self.val_dataloader)
        total_loss = 0.0
        print('epoch{}:validation'.format(epoch))
        with torch.no_grad():
            for log_pow_mix, class_targets, non_silent in tqdm(self.val_dataloader):
                log_pow_mix = log_pow_mix.to(self.device)
                class_targets = class_targets.to(self.device)
                non_silent = non_silent.to(self.device)

                mix_embs = self.dpcl(log_pow_mix)

                epoch_loss = loss(mix_embs, class_targets, non_silent, self.num_spks, self.device)
                total_loss += epoch_loss.item()
    
        total_loss = total_loss/num_batchs

        return total_loss
    
    def run(self):
        train_loss = []
        val_loss = []

        writer = tbx.SummaryWriter("test_tbx/exp1")

        with torch.cuda.device(0):
            self.save_checkpoint(self.cur_epoch,best=False)
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                v_loss = self.validation(self.cur_epoch)

                writer.add_scalar('t_loss',t_loss,self.cur_epoch)
                writer.add_scalar('v_loss',v_loss,self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch,best=True)
                
                if no_improve == self.early_stop:
                    break
            self.save_checkpoint(self.cur_epoch,best=False)
        
        writer.close()


    def save_checkpoint(self, epoch, best=True):
        os.makedirs(os.path.join(self.checkpoint,self.name),exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.dpcl.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        os.path.join(self.checkpoint,self.name,'{0}.pt'.format('best' if best else 'last')))