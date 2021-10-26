from model.network import DeepClustering
from model.eval_metrics import eval_SI_SDRi
import os
import yaml
from dataloader import wav_dataset
import torch
from tqdm import tqdm
import datetime
import csv
import sys
import pathlib
import argparse
import soundfile as sf
class Separation():
    def __init__(self, path_checkpoint):
        path_config = pathlib.Path(path_checkpoint) / "config.yaml"
        path_model = pathlib.Path(path_checkpoint) / "best.pt"
        with open(path_config, 'r') as yml:
            config = yaml.safe_load(yml)
        self.model = DeepClustering(**config["HyperParams"])
        self.C = 2
        self.sr = config["sampling_rate"]

        ckp = torch.load(path_model, map_location=torch.device('cpu'))

        self.model.load_state_dict(ckp['model_state_dict'],strict=False)

        dt_now = datetime.datetime.now()
        time = str(dt_now.strftime('%Y_%m_%d_%H-%M-%S'))
        self.dir_save = pathlib.Path('separated/') / time
        os.makedirs(self.dir_save,exist_ok=True)
        self.eval_dataset  = wav_dataset("./scp/tt_mix.scp", ["./scp/tt_s1.scp","./scp/tt_s2.scp"])

    def run(self,save_sound=False):
        with open(os.path.join(self.dir_save,'result.csv'), 'w') as f:
            writer = csv.writer(f)
            header = ['key']
            for c in range(self.C):
                header.append(f"SI-SDRi_{c+1}")
            header.append("self.path_model")
            writer.writerow(header)

            for idx in range(len(self.eval_dataset)):
                key = self.eval_dataset.keys[idx]
                filename = self.eval_dataset.keys[idx]

                with torch.no_grad():
                    x, s = self.eval_dataset[idx]
                    x = torch.tensor(x,dtype=torch.float32).reshape([1, -1]).detach()
                    s = torch.tensor(s,dtype=torch.float32).T.reshape([1, -1, self.C]).detach()
                    s_hat = self.model.infer(x, self.C)
                    s_hat = self.padding_end(s_hat, s)

                si_sdri = eval_SI_SDRi(s_hat, s, x)

                result = [key]
                result += si_sdri
                print(f"idx:{idx}. key:{key}, SI-SDRi:{si_sdri}")

                if save_sound:
                    for c in range(self.C):
                        s_hat_i = s_hat[:,:,c].reshape(-1)
                        s_hat_i = s_hat_i / torch.max(s_hat_i) * 0.9
                        path_wav = self.dir_save / filename.replace('.wav', f'_{str(c+1)}.wav')
                        sf.write(path_wav, s_hat_i, self.sr)

                writer.writerow(result)


    def padding_end(self, s_hat, x):
        T_origin = x.shape[1]
        Batchsize, T_conv, C = s_hat.shape
        T_pad = T_origin-T_conv

        padding = torch.zeros([Batchsize, T_pad, C]).to(s_hat.device)
        s_hat =  torch.cat([s_hat,padding],dim=1)

        return s_hat
        
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training a model')
    parser.add_argument('path_ckeckpoint', help='path of checkpint (e.g. 2021_01_01_12-00-00)') 
    parser.add_argument('-s', action='store_true', help='save the separation result') 
    parser.add_argument('-max_length', help='cut testdata upto max_length') 
    args = parser.parse_args()

    separation = Separation(args.path_ckeckpoint)
    separation.run(save_sound=args.s)