import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from Learning import utils
import yaml
import matplotlib.pyplot as plt


class transform():
    def __init__(self,config):
        self.num_spks = config['num_spks']
        self.wp = utils.wav_processor(config)
        self.mask = config['dataloader']['train']['mask']
    def __call__(self,y_mix,y_targets=None):
        Y_mix = self.wp.stft(y_mix)
        log_pow_mix = self.wp.log_power(Y_mix)
        log_pow_mix_normalized = self.wp.apply_normalize(log_pow_mix)

        non_silent = self.wp.non_silent(self.wp.stft(y_mix))

        if y_targets:
            Y_targets = [self.wp.stft(y_target) for y_target in y_targets]
            pow_targets = [np.abs(Y_target) for Y_target in Y_targets]

            T,F = non_silent.shape
            class_targets = np.zeros([T*F,self.num_spks])

            if self.mask=='IBM':
                mask_targets = np.argmax(np.array(pow_targets), axis=0)
                for i in range(self.num_spks):
                    mask_i = np.ones(non_silent.shape) * (mask_targets==i)
                    class_targets[:,i] = mask_i.reshape([T*F])


            elif self.mask=='IRM':
                eps = np.finfo(np.float64).eps
                sum_pow_targets = sum(pow_targets)
                
                for i,pow_target in enumerate(pow_targets):
                    class_targets[:,i] = (pow_target / (sum_pow_targets + eps)).reshape([T*F])
                print(class_targets)
            
            elif self.mask=='WM':
                eps = np.finfo(np.float64).eps
                pow_targets = [np.power(pow_target,2) for pow_target in pow_targets]
                sum_pow_targets = sum(pow_targets)

                for i,pow_target in enumerate(pow_targets):
                    class_targets[:,i] = (pow_target / (sum_pow_targets + eps)).reshape([T*F])
                        
                    
            return log_pow_mix_normalized, class_targets, non_silent

        return log_pow_mix_normalized, non_silent


class wav_dataset(Dataset):
    def __init__(self,config,path_scp_mix,path_scp_targets):
        self.wp = utils.wav_processor(config)

        self.scp_mix = self.wp.read_scp(path_scp_mix)
        self.scp_targets = [self.wp.read_scp(path_scp_target) \
                                for path_scp_target in path_scp_targets]
                  
        self.keys = [key for key in self.scp_mix.keys()]

        self.trans = transform(config)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        y_mix = self.wp.read_wav(self.scp_mix[key])
        y_targets = [self.wp.read_wav(scp_target[key]) \
                                for scp_target in self.scp_targets]
        
        return self.trans(y_mix,y_targets)


def padding(batch):
    batch_log_pow_mix,batch_class_targets,batch_non_silent = [],[],[]
    for log_pow_mix,class_targets,non_silent in batch:
        batch_log_pow_mix.append(torch.tensor(log_pow_mix,dtype=torch.float32))
        batch_class_targets.append(torch.tensor(class_targets,dtype=torch.int64))
        batch_non_silent.append(torch.tensor(non_silent,dtype=torch.float32))

    batch_log_pow_mix = pad_sequence(batch_log_pow_mix, batch_first=True)
    batch_class_targets = pad_sequence(batch_class_targets, batch_first=True)
    batch_non_silent = pad_sequence(batch_non_silent, batch_first=True)

    return batch_log_pow_mix,batch_class_targets,batch_non_silent


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