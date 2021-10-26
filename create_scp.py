import sys
import os

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
    dir_dataset = sys.argv[1]
    wav2scp(dir_dataset,2)