import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class DeepClustering(nn.Module):
    def __init__(self, D, hidden_size, num_layers, stft_ms, sr, activation="Tanh", dropout=0.0, bidirectional=True):
        '''
        explanation
        '''
        super(DeepClustering, self).__init__()
        
        self.D = D
        self.n_fft = int(stft_ms * sr/1000)
        input_size = int(self.n_fft/2 + 1)
        self.normalization = nn.GroupNorm(1, input_size, eps=1e-8)
        self.LSTM = nn.LSTM(input_size = input_size, 
                        hidden_size = hidden_size,
                        num_layers = num_layers,
                        batch_first = True,
                        dropout = dropout,
                        bidirectional = bidirectional)
        self.FC = nn.Linear((2*hidden_size if bidirectional else hidden_size), input_size * D)
        self.activation =  getattr(torch.nn, activation)()
        
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        """
        Args:
            x: [Batchsize, SignalLength] or [SignalLength]
        Returns:
            s_hat: [Batchsize, SignalLength, C]  or [1, SignalLength, C]
        """

        if x.dim()==1:
            x = x.unsqueeze(0)

        # [Batchsize, SignalLength] -> [Batchsize, T, F]
        X_logp = self.logp(x)
        Batchsize, T, F = X_logp.shape
        # [Batchsize, T, F] -> [Batchsize, T, hidden_size]
        X_logp = self.normalization(X_logp.transpose(1,2)).transpose(1,2)
        emb, _ = self.LSTM(X_logp)
        # [Batchsize, T, hidden_size] -> [Batchsize, T, F*D]
        emb = self.FC(emb)
        # [Batchsize, T, F*D] -> [Batchsize, T*F, D]
        emb = emb.reshape(Batchsize, T*F, self.D)
        emb = self.activation(emb)

        # |v|^2=1
        emb = emb / (torch.linalg.norm(emb, dim=2, keepdim=True) + torch.tensor(1e-8))

        return emb


    def logp(self, x):
        if x.dim()==1:
            x = x.unsqueeze(0)
        eps = torch.tensor(1e-8)
        power = torch.abs(self.stft(x))
        logp = 20*torch.log10(power + eps)

        return logp

    def stft(self, x):
        with torch.no_grad():
            if x.dim()==1:
                x = x.unsqueeze(0)
            
            if x.dim()==2:
                X = torch.stft(x, n_fft=self.n_fft, return_complex=True).transpose(1,2)

                
            elif x.dim()==3:
                X =[]
                for c in range(x.shape[2]):
                    X.append(torch.stft(x[:,:,c], n_fft=self.n_fft, return_complex=True).transpose(1,2).unsqueeze(3))
                X = torch.cat(X, dim=3)

        return X

    
    def istft(self, X):
        with torch.no_grad():
            if X.dim()==2:
                X = X.unsqueeze(0)
            x = torch.istft(X.transpose(1,2), n_fft=self.n_fft, onesided=True)

        return x

    
    def infer(self, x, C):
        with torch.no_grad():
            X = self.stft(x)
            _, T, F = X.shape
            emb = self.forward(x).squeeze(0)
            kmeans = KMeans(n_clusters=C)
            emb_cluster = kmeans.fit_predict(emb.numpy())

            s_hat = []
            for i in range(C):
                mask = (emb_cluster == i)
                mask = mask.reshape(1,T,F)
                S_hat = X*mask 
                s_hat.append(self.istft(S_hat))

            s_hat = torch.cat(s_hat, dim=0).transpose(0,1).unsqueeze(0)

        return s_hat


if __name__ == "__main__":
    D, hidden_size, num_layers, stft_ms, sr = 40, 600, 2, 32, 8000

    s = torch.randn([5,100001])

    mdc = DeepClustering(D, hidden_size, num_layers, stft_ms, sr)

    mdc.mask_inference(s[0,:], 2)