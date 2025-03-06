import torch
import torch.optim as optim
import torch.nn as nn


from VRAE import TimeGAN


model = TimeGAN(
    input_dim= 5, # input dimension
    hidden_dim= 100, # embedding dimension
    num_layers= 1, # number of layers
    cell_type= 'GRU'
    )
    
