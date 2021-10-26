import torch
from model.network import DeepClustering
from model.loss import affinity_loss
from dataloader import make_dataloader
import yaml
import tensorboardX as tbx
import os
from tqdm import tqdm
import pathlib
import datetime
from pytorch_model_summary import summary
import shutil
import argparse


class Trainer():
    def __init__(self, model, config, name):
        self.model = model
        self.dir_save = pathlib.Path('./checkpoint') / name
        os.makedirs(self.dir_save, exist_ok=True)
        self.config = config

        self.set_optimizer()
        if config['resume']['state']:    
            self.load_checkpoint(config)
        else:
            self.cur_epoch = 0

        # setting about machine
        if config["device"]["gpu"]:
            self.device_id = config["device"]["cuda"]
            self.device = torch.device(f"cuda:{self.device_id[0]}")
            if len(self.device_id)>1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.device_id)

        self.model = self.model.to(self.device)

        self.max_epoch = config['stop']['max_epoch']
        self.early_stop = config['stop']['early_stop']


    def train(self, epoch, dataloader):
        self.model.train()

        total_loss = 0
        for x, s in tqdm(dataloader):
            self.optimizer.zero_grad()
            x = x.to(self.device).detach()
            s = s.to(self.device).detach()
            X_logp = self.model.logp(x)
            S_logp = self.model.logp(s)

            emb = self.model(x)
            epoch_loss = affinity_loss(emb, S_logp, X_logp, self.config["threshold"])

            epoch_loss.backward()
            if "clipping_norm" in self.config["optimizer"]:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.config["optimizer"]["clipping_norm"])
            self.optimizer.step()

            total_loss += epoch_loss.detach()

        return total_loss / len(dataloader)


    def validation(self, dataloader):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for x, s in tqdm(dataloader):
                x = x.to(self.device)
                s = s.to(self.device)
                X_logp = self.model.logp(x)
                S_logp = self.model.logp(s)

                emb = self.model(x)
                total_loss += affinity_loss(emb, S_logp, X_logp, self.config["threshold"])

        return total_loss / len(dataloader)

    
    def run(self,train_dataloader,valid_dataloader):
        train_loss = []
        val_loss = []
        print('cur_epoch',self.cur_epoch)

        writer = tbx.SummaryWriter(self.dir_save)
        self.save_checkpoint(self.cur_epoch,best=False)
        v_loss = self.validation(valid_dataloader)
        print(f'epoch{self.cur_epoch}: valid_loss = {v_loss}')
        best_loss = torch.tensor(float('inf'))
        no_improve = 0

        # starting training part
        while self.cur_epoch < self.max_epoch:
            self.cur_epoch += 1
            t_loss = self.train(self.cur_epoch, train_dataloader)
            print(f'epoch{self.cur_epoch}: train_loss = {t_loss}')
            v_loss = self.validation(valid_dataloader)
            print(f'epoch{self.cur_epoch}: valid_loss = {v_loss}')

            writer.add_scalar('t_loss', t_loss, self.cur_epoch)
            writer.add_scalar('v_loss', v_loss, self.cur_epoch)

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
        

    def set_optimizer(self):
        optimizer = getattr(torch.optim, config['optimizer']['name'])
        self.optimizer = optimizer(self.model.parameters(), **config['optimizer']['HyperParams'])
    

    def save_checkpoint(self, epoch, best=True):
        self.model.to('cpu')
        print('save model epoch:{0} as {1}'.format(epoch,"best" if best else "last"))
        path_save_model = os.path.join(self.dir_save,'{0}.pt'.format('best' if best else 'last'))

        if len(self.device_id)>1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optim_state_dict': self.optimizer.state_dict()
            },
            path_save_model)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict()
            },
            path_save_model)

        self.model.to(self.device)


    def load_checkpoint(self, config):
        print('load on:',self.device)

        ckp = torch.load(config['resume']['path'], map_location=torch.device('cpu'))
        self.cur_epoch = ckp['epoch']
        self.model.load_state_dict(ckp['model_state_dict'])
        if config['resume']["load_optim"]:
            self.optimizer.load_state_dict(ckp['optim_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        self.model = self.model.to(self.device)

        print('training resume epoch:',self.cur_epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training a model')
    parser.add_argument('path_config', help='path of config(.yaml)') 
    args = parser.parse_args()


    with open(args.path_config, 'r') as yml:
        config = yaml.safe_load(yml)

    dt_now = datetime.datetime.now()
    time = str(dt_now.strftime('%Y_%m_%d_%H-%M-%S'))
    dir_save = pathlib.Path("./checkpoint") / time
    os.makedirs(dir_save, exist_ok=True)

    shutil.copyfile(args.path_config, dir_save / "config.yaml")
    shutil.copytree("scp", dir_save / "scp")

    dc = DeepClustering(**config["HyperParams"])

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(summary(dc, torch.zeros((1,150000))))
    trainer = Trainer(dc,config,time)


    train_dataloader = make_dataloader(config, "./scp/tr_mix.scp", ["./scp/tr_s1.scp","./scp/tr_s2.scp"])
    valid_dataloader = make_dataloader(config, "./scp/cv_mix.scp", ["./scp/cv_s1.scp","./scp/cv_s2.scp"])

    trainer.run(train_dataloader,valid_dataloader)