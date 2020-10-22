import os
import yaml
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt

class wav_processor:
    def __init__(self,config):
        self.n_fft = config['transform']['n_fft']
        self.hop_length = config['transform']['hop_length']
        self.win_length = config['transform']['win_length']
        self.window = config['transform']['window']
        self.center = config['transform']['center']
        self.sr = config['transform']['sr']
        self.mask_threshold = config['transform']['mask_threshold']
        path_normalize = config['transform']['path_normalize']
        self.normalize = pickle.load(open(path_normalize, 'rb'))

    def stft(self,y):
        Y = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            center=self.center)
        return Y.T

    def log_power(self,Y):
        eps = np.finfo(float).eps
        log_power =  np.log(np.maximum(np.abs(Y),eps))

        return log_power

    def istft(self, Y):
        Y = Y.T
        y = librosa.istft(Y, hop_length=self.hop_length,win_length=self.win_length,
                            window=self.window,center=self.center)
        return y

    def non_silent(self,Y):
        eps = np.finfo(float).eps
        Y_db = 20 * np.log10(np.maximum(np.abs(Y),eps))
        max_db = np.max(Y_db)
        min_magnitude = 10**((max_db - self.mask_threshold) / 20)
        non_silent = np.array(Y > min_magnitude, dtype=np.float32)
        return non_silent

    def apply_normalize(self,Y):
        return (Y - self.normalize['mean']) / self.normalize['std']

    def read_wav(self,wav_path):
        y,_ = librosa.load(wav_path, sr=self.sr)
        return y

    def write_wav(self,dir_path, filename, y):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
        librosa.output.write_wav(file_path, y, self.sr)

    def read_scp(self,scp_path):
        files = open(scp_path, 'r')
        lines = files.readlines()
        scp_wav = {}
        for line in lines:
            line = line.split()
            if line[0] in scp_wav.keys():
                raise ValueError
            scp_wav[line[0]] = line[1]
        return scp_wav

    def calc_sdr(self,Y,Y_hat):

        Y_sum = np.sum(np.sum((np.abs(Y))**2,1),0)
        Y_hat_sum = np.sum(np.sum((np.abs(Y_hat))**2,1),0)
        
        c = (Y_sum/Y_hat_sum)**(0.5)

        dist = np.sum(np.sum((np.abs(Y) - c*np.abs(Y_hat))**2,1),0)
        sdr = 10*np.log10(Y_sum/dist)

        return sdr

    def calc_si_sdr(self,Y,Y_hat):

        Y_sum = np.sum(np.sum((np.abs(Y))**2,1),0)
        Y_hat_sum = np.sum(np.sum((np.abs(Y_hat))**2,1),0)
        
        c = (Y_sum/Y_hat_sum)**(0.5)

        dist = np.sum(np.sum((np.abs(Y) - c*np.abs(Y_hat))**2,1),0)
        sdr = 10*np.log10(Y_sum/dist)

        return sdr

    
    def eval_sdr(self,Y_list,Y_hat_list):
        num_spks = len(Y_list)

        sdr = []
        for i in range(num_spks):
            sdr_i = -np.inf
            for j in range(num_spks):
                sdr_j = self.calc_sdr(Y_list[i], Y_hat_list[j])
                if sdr_j > sdr_i:
                    sdr_i = sdr_j
            sdr.append(sdr_i)

        return sdr


