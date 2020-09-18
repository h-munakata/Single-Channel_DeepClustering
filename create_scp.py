import os

dir_wav_root = '../../../data1/h_munakata/small/2speakers/wav8k/min/'
dir_scp = './scp'

dir_tr_mix = dir_wav_root + 'tr/mix'
dir_tr_s1 = dir_wav_root + 'tr/s1'
dir_tr_s2 = dir_wav_root + 'tr/s2'

dir_tt_mix = dir_wav_root + 'tt/mix'
dir_tt_s1 = dir_wav_root + 'tt/s1'
dir_tt_s2 = dir_wav_root + 'tt/s2'

dir_cv_mix = dir_wav_root + 'cv/mix'
dir_cv_s1 = dir_wav_root + 'cv/s1'
dir_cv_s2 = dir_wav_root + 'cv/s2'

def save_scp(dir_wav,dir_scp,scp_name):
    os.makedirs(dir_scp, exist_ok=True)
    path_scp = dir_scp + '/' + scp_name

    if not os.path.exists(dir_wav):
        raise ValueError("directory of .wav doesn't exist")

    scp = open(path_scp,'w')
    for root, dirs, files in os.walk(dir_wav):
        files.sort()
        for file in files:
            scp.write(file+" "+root+'/'+file)
            scp.write('\n')

save_scp(dir_tr_mix,dir_scp,'tr_mix.scp')
save_scp(dir_tr_s1,dir_scp,'tr_s1.scp')
save_scp(dir_tr_s2,dir_scp,'tr_s2.scp')

save_scp(dir_tt_mix,dir_scp,'tt_mix.scp')
save_scp(dir_tt_s1,dir_scp,'tt_s1.scp')
save_scp(dir_tt_s2,dir_scp,'tt_s2.scp')

save_scp(dir_cv_mix,dir_scp,'cv_mix.scp')
save_scp(dir_cv_s1,dir_scp,'cv_s1.scp')
save_scp(dir_cv_s2,dir_scp,'cv_s2.scp')
