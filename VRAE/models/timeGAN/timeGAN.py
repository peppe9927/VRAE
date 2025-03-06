import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Importing the modules of the architecture
from .embedder import Embedder
from .recovery import Recovery
from .generator import Generator
from .supervisor import Supervisor
from .discriminator import Discriminator

class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers, cell_type = 'GRU'):
        super(TimeGAN, self).__init__()
        # ori_data: (batch, seq_len, dim)
        self.f_dim = input_dim #
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        super(TimeGAN).__init__()

        # Embedder
        self.embedder = Embedder(
            inpt_dim= self.f_dim,
            hidden_dim= self.hidden_dim,
            num_layers= self.num_layers,
            cell_type= self.cell_type
        )
        # Recovery
        self.recovery = Recovery(
            hidden_dim= self.f_dim,
            feature_dim= self.hidden_dim,
            num_layers= self.num_layers,
            cell_type= self.cell_type
        )
        # Generator
        self.generator = Generator(
            hidden_dim= self.hidden_dim,
            num_layers= self.num_layers,
            cell_type= self.cell_type
        )
        # Supervisor
        self.supervisor = Supervisor(
            num_hidden= self.hidden_dim,
            num_layers= self.num_layers,
            cell_type= self.cell_type
        )
        # Discriminator
        self.discriminator = Discriminator(
            num_hidden= self.hidden_dim,
            num_layers= self.num_layers,
            cell_type= self.cell_type
        )

    def forward(self, X, T, z):
        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)

        E_hat = self.generator(z, T) # embedding of the generated data, to reproduce feature distribution
        H_hat = self.supervisor(E_hat, T) # supervised fake data, to reproduce temporal dynamics
        H_hat_supervise = self.supervisor(H, T) # supervised real data 

        X_hat = self.recovery(H_hat, T) # generated data
        
        Y_real = self.discriminator(H, T) # output of discriminator for real data
        Y_fake = self.discriminator(H_hat, T) # output of discriminator for fake data


        return X_tilde, X_hat, H_hat, H_hat_supervise, Y_real, Y_fake
