'''
Modified MLP
"UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS"
"Long-time integration of parametric evolution equations with physics-informed DeepONets"
'''


import torch
import torch.nn as nn
import rff


class ModifiedMLP_Block(nn.Module):
    def __init__(self, layers_size, sigma): # activation= F.relu
        """
        Modified MLP model with two parallel paths U and V.

        Args:
            layers (list): List of layer sizes.
            activation (function): Activation function.
        """
        super(ModifiedMLP_Block, self).__init__()

        self.fourier_encode = rff.layers.GaussianEncoding(sigma=sigma, input_size=1, encoded_size=50)
        self.activation = nn.Tanh()
        self.Hs = nn.ModuleList()

        # Residual connections
        for i in range(len(layers_size) - 1):
            H = nn.Linear(layers_size[i], layers_size[i + 1])
            torch.nn.init.xavier_uniform_(H.weight)
            self.Hs.append(H)

        #  U and V
        self.U = nn.Linear(layers_size[0], layers_size[1])
        self.V = nn.Linear(layers_size[0], layers_size[1])

        torch.nn.init.xavier_uniform_(self.U.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

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

class ModifiedMLP(nn.Module):
    def __init__(self, layers_size, T, pqmax, sigma):

        super(ModifiedMLP, self).__init__()

        self.nn1 = ModifiedMLP_Block(layers_size, sigma)
        self.nn2 = ModifiedMLP_Block(layers_size, sigma)
        self.T = T
        self.pqmax = pqmax

    def forward(self, x):
        x = x / self.T * 2 - 1
        p = self.nn1(x)
        q = self.nn2(x)
        p = self.pqmax * p
        q = self.pqmax * q
        return p, q



