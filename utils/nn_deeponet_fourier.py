'''
DeepOnet
"UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS"
"Long-time integration of parametric evolution equations with physics-informed DeepONets"
'''


import torch
import torch.nn as nn
import rff
# import pywt
# class RandomWaveletEmbedding(nn.Module):
#     def __init__(self, input_dim, num_features, wavelet="mexh"):
#         super().__init__()
#         self.num_features = num_features
#         self.wavelet = wavelet
#         self.W = nn.Parameter(torch.randn(num_features, input_dim))  # Random projection
#         self.b = nn.Parameter(torch.rand(num_features) * 2 * torch.pi)  # Random shift
#
#     def forward(self, x):
#         projected = torch.matmul(x, self.W.T) + self.b
#         wavelet_transformed = torch.stack([torch.tensor(pywt.wavelet(self.wavelet).wavefun(level=1)[0](p.cpu().numpy())) for p in projected])
#         return wavelet_transformed.to(x.device)
#
# # Example usage
# x = torch.randn(100, 10)  # 100 samples, 10D input
# embedding = RandomWaveletEmbedding(10, 256, wavelet="mexh")  # 256 random wavelet features
# x_embed = embedding(x)

class ModifiedMLP_Block(nn.Module):
    def __init__(self, layers_size, in_layer_size, fourier_sigma): # activation= F.relu
        """
        Modified MLP model with two parallel paths U and V.

        Args:
            layers (list): List of layer sizes.
            activation (function): Activation function.
        """
        super(ModifiedMLP_Block, self).__init__()

        self.fourier_encode = rff.layers.GaussianEncoding(sigma=fourier_sigma, input_size=in_layer_size, encoded_size= int(layers_size[0]/2))
        self.activation = nn.Tanh()
        self.Hs = nn.ModuleList()

        # Residual connections
        for i in range(len(layers_size) - 1): #todo: no fourier
            H = nn.Linear(layers_size[i], layers_size[i + 1])
            # torch.nn.init.xavier_uniform_(H.weight)
            self.Hs.append(H)

        #  U and V
        self.U = nn.Linear(layers_size[0], layers_size[1])
        self.V = nn.Linear(layers_size[0], layers_size[1])

        # torch.nn.init.xavier_uniform_(self.U.weight)
        # torch.nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x):
        # Compute U and V parallel paths
        x = self.fourier_encode(x)
        U = self.activation(self.U(x))
        V = self.activation(self.V(x))

        for h in self.Hs[:-1]:
            Z = self.activation(h(x))
            # Blend U and V with Z
            x = (1 - Z) * U + Z * V

        x = self.Hs[-1](x)
        return x

class DeepOnet(nn.Module):
    def __init__(self, layer_size_branch, layer_size_trunk, pq_max, time_length, fourier_sigma):

        super(DeepOnet, self).__init__()

        self.branch = ModifiedMLP_Block(layer_size_branch, 2, fourier_sigma)
        self.trunk = ModifiedMLP_Block(layer_size_trunk, 1, fourier_sigma)
        self.pq_max = pq_max
        self.time_length = time_length

    def forward(self, t,  pq0):

        t = t/self.time_length * 2 -1
        pq0 = pq0 / self.pq_max #todo:0.35
        br = self.branch(pq0).squeeze()
        tr = self.trunk(t).squeeze()
        br1 = br[:,:100]
        br2 = br[:, 100:]
        tr1 = tr[:,:100]
        tr2 = tr[:, 100:]
        p = torch.sum(tr1 * br1, dim=1, keepdim=True)
        q = torch.sum(tr2 * br2, dim=1, keepdim=True)
        p = self.pq_max * p #todo:0.35
        q = self.pq_max * q #todo:0.35
        return p, q



