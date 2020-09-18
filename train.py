import sys
from trainer import Trainer
import torch
import yaml
import model
from torch.utils.data import  DataLoader
from dataloader import wav_dataset,padding



def make_dataloader(config):
    path_scp_tr_mix = config['dataloader']['train']['path_scp_mix']
    path_scp_tr_targets = config['dataloader']['train']['path_scp_targets']
    tr_dataset  = wav_dataset(config, path_scp_tr_mix, path_scp_tr_targets)

    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    shuffle = config['dataloader']['shuffle']

    tr_dataloader = DataLoader(tr_dataset,batch_size=batch_size,num_workers=num_workers,
                                shuffle=shuffle,collate_fn=padding)
    
    path_scp_cv_mix = config['dataloader']['val']['path_scp_mix']
    path_scp_cv_targets = config['dataloader']['val']['path_scp_targets']
    cv_dataset  = wav_dataset(config, path_scp_cv_mix, path_scp_cv_targets)

    cv_dataloader = DataLoader(cv_dataset,batch_size=batch_size,num_workers=num_workers,
                                shuffle=shuffle,collate_fn=padding)
    return tr_dataloader, cv_dataloader


def make_optimizer(params,config):
    opt_name = config['optim']['name']
    weight_decay = config['optim']['weight_decay']
    lr = config['optim']['lr']
    momentum = config['optim']['momentum']

    optimizer = getattr(torch.optim, opt_name)
    if opt_name == 'Adam':
        optimizer = optimizer(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optimizer(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    return optimizer

def train():
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    dpcl = model.DeepClustering(config)
    optimizer = make_optimizer(dpcl.parameters(),config)

    train_dataloader, val_dataloader = make_dataloader(config)

    trainer = Trainer(train_dataloader, val_dataloader, dpcl, optimizer, config)
    trainer.run()


if __name__ == "__main__":
    train()