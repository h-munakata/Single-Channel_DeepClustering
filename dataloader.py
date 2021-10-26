import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import yaml


class wav_dataset(Dataset):
    def __init__(self, path_scp_mix, path_scp_targets, max_length=None):
        self.max_length = max_length

        self.scp_mix = read_scp(path_scp_mix)
        self.scp_targets = [read_scp(path_scp_target_i) \
                                for path_scp_target_i in path_scp_targets]

        self.keys = [key for key in self.scp_mix.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        y_mix = read_wav(self.scp_mix[key], self.max_length)
        y_targets = [read_wav(scp_target_i[key], self.max_length) \
                                for scp_target_i in self.scp_targets]
        
        return y_mix,y_targets


def padding(batch):
    batch_mix, batch_s = [],[]
    for mix, s in batch:
        batch_mix.append(torch.tensor(mix,dtype=torch.float32).unsqueeze(-1))
        batch_s.append(torch.tensor(s,dtype=torch.float32).transpose(0,1))

    batch_mix = pad_sequence(batch_mix, batch_first=True)
    batch_size = batch_mix.shape[0]
    batch_mix = batch_mix.view([batch_size,-1])

    batch_s = pad_sequence(batch_s, batch_first=True)

    return batch_mix, batch_s


def make_dataloader(config, path_scp_mix, path_scp_targets):
    dataset  = wav_dataset(path_scp_mix, path_scp_targets, config['max_length'])

    dataloader = DataLoader(dataset, **config['dataloader'], collate_fn=padding)

    return dataloader



def read_scp(scp_path):
    files = open(scp_path, 'r')
    lines = files.readlines()
    scp_wav = {}
    for line in lines:
        line = line.split()
        if line[0] in scp_wav.keys():
            raise ValueError
        scp_wav[line[0]] = line[1]
    return scp_wav


def read_wav(path_wav, max_length=None):
    y, sr = sf.read(path_wav)
    if max_length!=None:
        y = y[:int(sr*max_length)]
    return y

def write_wav(path_wav, y, sr):
    os.makedirs(os.path.dirname(path_wav), exist_ok=True)
    y = np.array(y)
    y = 0.9/np.max(y) * y
    sf.write(path_wav, y, sr)

def stft(s,n_fft=512):
    S = librosa.stft(s, n_fft=n_fft, hop_length=n_fft//4,
                            win_length=n_fft//2, window='hann',
                            center=False)
    return S

def istft(S,n_fft=512):
    s = librosa.istft(S, hop_length=n_fft//4, win_length=n_fft//2,
                            window='hann',center=False)
    return s


def save_scp(dir_wav,scp_name):
    os.makedirs("./scp", exist_ok=True)
    path_scp = "./scp" + "/" + scp_name

    print("making {0} from {1}".format(path_scp,dir_wav))

    if not os.path.exists(dir_wav):
        raise ValueError("directory of .wav doesn't exist")

    with open(path_scp,'w') as scp:
        for root, dirs, files in os.walk(dir_wav):
            files.sort()
            for file in files:
                scp.write(file+" "+root+'/'+file)
                scp.write('\n')


def wav2scp(dir_dataset,num_spks):
    print('making scp files')

    type_list = ['/tr','/cv', '/tt']

    for type_data in type_list:
        dir_type = dir_dataset + type_data + '/mix'
        scp_name = type_data + '_mix.scp'
        save_scp(dir_type,scp_name)

        for i in range(num_spks):
            dir_type = dir_dataset + type_data + '/s{0}'.format(i+1)
            scp_name = type_data + '_s{0}.scp'.format(i+1)
            save_scp(dir_type,scp_name)


if __name__ == "__main__":
    pass
