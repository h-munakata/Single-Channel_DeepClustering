import os
import yaml


def save_scp(dir_wav,dir_scp,scp_name):
    os.makedirs(dir_scp, exist_ok=True)
    path_scp = dir_scp + '/' + scp_name
    print(dir_wav)
    if not os.path.exists(dir_wav):
        raise ValueError("directory of .wav doesn't exist")

    scp = open(path_scp,'w')
    for root, dirs, files in os.walk(dir_wav):
        files.sort()
        for file in files:
            scp.write(file+" "+root+'/'+file)
            scp.write('\n')
    
    print('{0} saved in {1}'.format(scp_name,dir_wav))


if __name__=="__main__":
    print('making scp files')
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    dir_scp = './scp'
    dir_wav_root = config['dir_wav_root']
    num_spks = config['num_spks']

    speks = range(1,num_spks+1)

    wav_types = ['/tr','/tt','/cv']
    for wav_type in wav_types:

        dir_type = dir_wav_root + wav_type
        save_scp(dir_type + '/mix', dir_scp, wav_type + '_mix.scp')
        for i in speks:
            save_scp(dir_type + '/s' + str(i), dir_scp, wav_type + '_s' + str(i) + '.scp')