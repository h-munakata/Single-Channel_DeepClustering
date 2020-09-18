import torch
from sklearn.cluster import KMeans
import model
import numpy as np
from dataloader import transform,padding
import utils
import os
import yaml
from tqdm import tqdm


class Separation():
    def __init__(self, dpcl, config):
        self.wp = utils.wav_processor(config)
        self.dpcl = model.DeepClustering(config)
        if config['train']['is_gpu']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        path_model = config['test']['path_model']
        ckp = torch.load(path_model,map_location=self.device)
        self.dpcl.load_state_dict(ckp['model_state_dict'])
        self.dpcl.eval()

        path_scp_mix = config['test']['path_scp_mix']
        self.scp_mix = utils.read_scp(path_scp_mix)

        self.trans = transform(config)

        self.num_spks = config['num_spks']
        self.kmeans = KMeans(n_clusters=self.num_spks)
        self.path_separated = config['test']['path_separated']


    def make_mask(self, wave, non_silent):
        '''
            input: T x F
        '''
        # TF x D


        mix_emb = self.dpcl(torch.tensor(
            wave, dtype=torch.float32), is_train=False)
        mix_emb = mix_emb.detach().numpy()
        # N x D
        T, F = non_silent.shape
        non_silent = non_silent.reshape(-1)
        # print(non_silent)
        # mix_emb = (mix_emb.T*non_silent).T
        # N
        mix_cluster = self.kmeans.fit_predict(mix_emb)
        targets_mask = []
        for i in range(self.num_spks):
            mask = (mix_cluster == i)
            mask = mask.reshape(T,F)
            targets_mask.append(mask)

        return targets_mask

    def run(self):
        for key in tqdm(self.scp_mix.keys()):
            y_mix = self.wp.read_wav(self.scp_mix[key])
            log_pow_mix_normalized, non_silent = self.trans(y_mix)

            Y_mix = self.wp.stft(y_mix)
            target_mask = self.make_mask(log_pow_mix_normalized, non_silent)

            for i in range(len(target_mask)):
                Y_separated_i = target_mask[i] * Y_mix
                y_separated_i = self.wp.istft(Y_separated_i)
                self.wp.write_wav(self.path_separated,key + str(i+1) + '.wav',y_separated_i)


if __name__ == "__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    separation = Separation(config,config)
    separation.run()