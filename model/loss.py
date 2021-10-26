import torch

def affinity_loss(emb, S_logp, X_logp, threshold_dB=-40):
    '''
    emb: B, T*F, D
    S_logP: B, T, F
    '''
    Batchsize, T, F  = X_logp.shape

    with torch.no_grad():
        non_sil = calc_non_sil(X_logp)
        non_sil.to(X_logp.device)
        size_slot = torch.sum(non_sil, dim=1, keepdim=True)

    v = emb * non_sil
    y = calc_y(S_logp) * non_sil

    vTv = torch.linalg.norm(torch.bmm(v.transpose(1,2), v) / size_slot, dim=[1,2])**2
    vTy = torch.linalg.norm(torch.bmm(v.transpose(1,2), y) / size_slot, dim=[1,2])**2
    yTy = torch.linalg.norm(torch.bmm(y.transpose(1,2), y) / size_slot, dim=[1,2])**2
    
    loss = (vTv - 2*vTy + yTy) * size_slot.view(-1)

    loss = torch.sum(loss) / Batchsize
    return loss


def calc_non_sil(X_logp, threshold_dB=-40):
    non_sil = torch.where(X_logp > threshold_dB, 1, 0)
    B, T, F  = X_logp.shape
    non_sil = non_sil.view(B, T*F, 1)
    non_sil.to(X_logp.device)
    
    return non_sil


def calc_y(S_logp):
    with torch.no_grad():
        B, T, F, C  = S_logp.shape
        max_spk = torch.argmax(S_logp, dim=3)

        y = []
        for c in range(C):
            y_c = torch.ones(max_spk.shape).to(S_logp.device) * torch.where(max_spk==c, 1, 0)
            y_c = y_c.view(B, -1, 1)
            y.append(y_c)

        y = torch.cat(y, dim=2)

    return y



if __name__ == "__main__":
    S_logp = torch.randn([1, 100, 200, 2])

    X_logp = torch.randn([1, 100, 200])
    emb = torch.randn([1, 20000, 40])

    print(affinity_loss(emb, S_logp, X_logp))