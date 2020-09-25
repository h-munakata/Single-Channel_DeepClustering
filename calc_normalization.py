import numpy as np
import yaml
from tqdm import tqdm
import utils
import pickle
from Learning import utils


with open('config.yaml', 'r') as yml:
    config = yaml.safe_load(yml)
wp = utils.wav_processor(config)

path_scp_mix = config['dataloader']['train']['path_scp_mix']
scp_mix = utils.read_scp(path_scp_mix)

f_bin = int(config['transform']['n_fft']/2+1)

mean_f = np.zeros(f_bin)
var_f = np.zeros(f_bin)

for key in tqdm(scp_mix.keys()):
    y = wp.read_wav(scp_mix[key])
    Y = wp.stft(y)
    log_pow = wp.log_power(Y)

    mean_f += np.mean(log_pow, 0)
    var_f += np.mean(log_pow**2, 0)

mean_f = mean_f / len(scp_mix.keys())
var_f = var_f / len(scp_mix.keys())

std_f = np.sqrt(var_f - mean_f**2)

with open(config['transform']['path_normalize'], "wb") as f:
    normalize_dict = {"mean": mean_f, "std": std_f}
    pickle.dump(normalize_dict, f)
print("Global mean: {}".format(mean_f))
print("Global std: {}".format(std_f))