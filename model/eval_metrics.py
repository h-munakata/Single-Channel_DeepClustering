import torch
import itertools
import numpy as np

def SI_SDR(est_s, s):
    e_target = vec_inner(est_s,s)/vec_inner(s,s) * s 
    e_res = e_target - est_s
    si_sdr = 10*torch.log10(vec_inner(e_target, e_target) / vec_inner(e_res, e_res))
    return si_sdr


def vec_inner(s1,s2):
    return torch.sum(s1*s2)


def eval_SI_SDRi(est_s, s, x):
    si_sdr = eval_SI_SDR(est_s, s)
    C = s.shape[2]
    x_repeat = torch.cat([x.unsqueeze(2) for _ in range(C)], dim=2)
    base_si_sdr = permutation_solver(x_repeat, s, SI_SDR)

    SI_SDRi = [si_sdr_c - base_si_sdr_c for si_sdr_c, base_si_sdr_c in zip(si_sdr, base_si_sdr)]

    SI_SDRi = list(map(lambda x: np.round(x.item(), 2), SI_SDRi))
    
    return SI_SDRi


def eval_SI_SDR(est_s, s):
    si_sdr = permutation_solver(est_s, s, SI_SDR)
    return si_sdr
    

def oracle_SI_SDRi(s, x):
    pass



# def IBM(S,C):
#     ibm = np.eye(C)[np.argmax(np.abs(S),axis=0)]
#     return ibm


# def eval_ideal(s,mix_s,precision):
#     s = s.detach().numpy()
#     mix_s = mix_s.detach().numpy()
#     C = s.shape[1]

#     mix_S = stft(mix_s)
#     S = np.array([stft(s[:,i]) for i in range(C)])
#     ibm = IBM(S,C)
    
#     ideal_s = []
#     for i in range(C):
#         ideal_s.append(istft(mix_S*ibm[:,:,i]))
#     ideal_s = torch.tensor(ideal_s).T

#     ideal_s = torch.cat([ideal_s, torch.zeros([s.shape[0]-ideal_s.shape[0],C])], dim=0)

#     return eval_SISDR(ideal_s,torch.tensor(s),precision)



# def eval_SISDR(est_s,s,precision):
#     _,C = s.shape
#     permutations = list(itertools.permutations(range(C)))
#     min_loss = 1e50
#     for p in permutations:
#         permutation_est_eval = []
#         for i in range(C):
#             value = SI_SDR(est_s[:,p[i]], s[:,i]).detach().numpy().tolist()
#             permutation_est_eval.append(np.round(value,precision))
#         loss = -sum(permutation_est_eval)/C
#         if loss < min_loss:
#             min_loss = loss
#             est_value = permutation_est_eval
    
#     return est_value

# def eval_base_SISDR(est_s,mix_s,precision):
#     base_value = []
#     _,C = est_s.shape
#     for i in range(C):
#         value = SI_SDR(mix_s, est_s[:,i]).detach().numpy().tolist()
#         base_value.append(np.round(value,precision))
    
#     return base_value



# def eval_each_s(est_s, s, mix_s,eval_idx = "sisdr",precision=3):
#     if s.shape[2] != est_s.shape[2]:
#         raise ValueError("C does not match with the number of speakers")

#     s = s.squeeze(0)
#     est_s = est_s.squeeze(0)
#     mix_s = mix_s.squeeze(0)

#     est_value = eval_SISDR(est_s,s,precision)
#     base_value = eval_base_SISDR(est_s,mix_s,precision)
#     ideal_value = eval_ideal(s,mix_s,precision)
#     improve_value = [np.round(est-base,precision) for est,base in zip(est_value,base_value)]
#     improve_ideal_value = [np.round(ideal-base,precision) for ideal,base in zip(ideal_value,base_value)]

#     return est_value, base_value, ideal_value, improve_value, improve_ideal_value



def permutation_solver(est_s, s, metrics):
    C = est_s.shape[2]
    permutations = list(itertools.permutations(range(C)))
    permutation_solved_loss = [-torch.tensor(float('inf'))]*C
    max_value = - torch.tensor(float('inf'))

    for p in permutations:
        loss = []
        for i in range(C):
            loss.append(metrics(est_s[0,:,i], s[0,:,p[i]]))
        if sum(loss) > max_value:
            permutation_solved_loss = loss
            max_value = sum(loss)

    return permutation_solved_loss

    