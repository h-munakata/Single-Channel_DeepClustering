import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import utils
import yaml
import matplotlib.pyplot as plt


class transform():
    def __init__(self,config):
        self.wp = utils.wav_processor(config)

    def __call__(self,y_mix,y_targets=None):
        Y_mix = self.wp.stft(y_mix)
        log_pow_mix = self.wp.log_power(Y_mix)
        log_pow_mix_normalized = self.wp.apply_normalize(log_pow_mix)

        non_silent = self.wp.non_silent(self.wp.stft(y_mix))

        if y_targets:
            Y_targets = [self.wp.stft(y_target) for y_target in y_targets]
            log_pow_targets = [self.wp.log_power(Y_target) for Y_target in Y_targets]
            class_targets = np.argmax(np.array(log_pow_targets), axis=0)

            return log_pow_mix_normalized, class_targets, non_silent

        return log_pow_mix_normalized, non_silent

class wav_dataset(Dataset):
    def __init__(self,config,path_scp_mix,path_scp_targets):
        self.wp = utils.wav_processor(config)

        self.scp_mix = utils.read_scp(path_scp_mix)
        self.scp_targets = [utils.read_scp(path_scp_target) \
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





if __name__=="__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    path_scp_mix = config['dataloader']['train']['path_scp_mix']
    path_scp_targets = config['dataloader']['train']['path_scp_targets']
    dataset = wav_dataset(config,path_scp_mix,path_scp_targets)

    dataloader = DataLoader(dataset,batch_size=5,collate_fn=padding)

    for batch in dataloader:
        print('log_pow_mix',batch[0].shape)
        print('class_targets',batch[1].shape)
        print('non_silent',batch[2].shape)