import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from Learning import model,utils
from Learning.dataloader import transform,padding
import numpy as np
import os
import yaml
import logging
from tqdm import tqdm
import datetime


class Separation():
    def __init__(self,config):
        self.wp = utils.wav_processor(config)
        self.dpcl = model.DeepClustering(config)
        self.device = torch.device(config['gpu'])
        print('Processing on',config['gpu'])

        self.path_model = config['test']['path_model']
        ckp = torch.load(self.path_model,map_location=self.device)
        self.dpcl.load_state_dict(ckp['model_state_dict'])
        self.dpcl.eval()

        self.dir_wav_root = config['dir_wav_root']
        path_scp_mix = config['test']['path_scp_mix']
        self.scp_mix = self.wp.read_scp(path_scp_mix)

        self.trans = transform(config)

        self.num_spks = config['num_spks']
        self.kmeans = KMeans(n_clusters=self.num_spks)
        self.gmm = GMM(n_components=self.num_spks, max_iter=1000)
        dt_now = datetime.datetime.now()
        self.path_separated = config['test']['path_separated'] + '/'+str(dt_now.isoformat())
        self.type_mask = config['test']['type_mask']
        


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
        self.gmm.fit(mix_emb)
        mix_cluster_soft = self.gmm.predict_proba(mix_emb)
        targets_mask = []

        # hard clustering
        if self.type_mask == 'hard':
            for i in range(self.num_spks):
                mask = (mix_cluster == i)
                mask = mask.reshape(T,F)
                targets_mask.append(mask)

        # soft clustering
        elif self.type_mask == 'soft':
            for i in range(self.num_spks):
                mask = mix_cluster_soft[:,i]
                mask = mask.reshape(T,F)
                targets_mask.append(mask)

        return targets_mask

    def run(self):
        sdr_eval=True 
        os.makedirs(self.path_separated,exist_ok=True)
        logging.basicConfig(filename=self.path_separated+'/logger.log', level=logging.DEBUG)
        logging.info(self.dir_wav_root)
        logging.info(self.path_model)

        if sdr_eval:
            path_scp_targets = config['test']['path_scp_targets']
            scp_targets = [self.wp.read_scp(path_scp_target) \
                    for path_scp_target in path_scp_targets]


        list_sdr=[]
        for key in tqdm(self.scp_mix.keys()):
            y_mix = self.wp.read_wav(self.scp_mix[key])

            log_pow_mix_normalized, non_silent = self.trans(y_mix)
            Y_mix = self.wp.stft(y_mix)
            target_mask = self.make_mask(log_pow_mix_normalized, non_silent)

            Y_separated = []
            for i in range(len(target_mask)):
                Y_separated.append(target_mask[i] * Y_mix)

            if sdr_eval:
                y_targets = [self.wp.read_wav(scp_target[key]) for scp_target in scp_targets]
                Y_targets = [self.wp.stft(y_target) for y_target in y_targets]

                logging.info(key)
                sdr = self.wp.eval_sdr(Y_targets,Y_separated)
                list_sdr.append(sdr)
                logging.info(self.wp.eval_sdr(Y_targets,Y_separated))
            for i,Y_separated_i in enumerate(Y_separated):
                y_separated_i = self.wp.istft(Y_separated_i)
                self.wp.write_wav(self.path_separated+'/separated',key.replace('.wav','') + '_'
                                    + str(i+1) + '.wav',y_separated_i)
        logging.info('mean_sdr = '+str(np.mean(list_sdr)))



if __name__ == "__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    separation = Separation(config)
    separation.run()