# Description: This file contains the necessary imports for the training and testing of the VRAE model.
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import custom_collate_fn
from models.vae import VRAE
from torch.optim import Adam
from utils.loos_fcns import loss_function_vrae

# Device
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 0
learning_rate = 0.001
num_epochs = 50
batch_size = 128
torch.manual_seed(42) # Set the seed for reproducibility

# Load data
dataset=torch.load('./VRAE/datasets/training_data.pt') # Load the dataset for training
print(dataset)
train_loader = DataLoader(dataset, batch_size=40, shuffle=True, collate_fn=custom_collate_fn)

# Ottieni il primo batch dal train_loader e stampa il primo elemento
batch = next(iter(train_loader))
print(batch.keys()) # Stampa le chiavi del batch
print(batch['routes'].shape) # Stampa la forma del batch
print(batch['times'].shape) # Stampa la forma del batch
print(batch['classes'].shape) # Stampa la forma del batch
print(batch['lengths'].shape) # Stampa la forma del batch

# Architecture
sensor_dim = 1
num_routes = 32
route_emb_dim = 4
num_hidden = 64
num_latent = 16

# Model
model = VRAE(sensor_dim,
             num_routes,
             route_emb_dim,
             num_hidden,
             num_latent).to(device)  # Move the model to the device

# test the model
optimizer = Adam(model.parameters(), lr=learning_rate) # Optimizer
 # Loss function

start_time = time.time()


for epoch in range(num_epochs):
    for batch_idx, X in enumerate(train_loader):
        # Input
        routes = X['routes'].to(device)
        times = X['times'].to(device)
        labels = X['classes'].to(device)
        lengths = X['lengths'].to('cpu')
        

        model.train() # Set the model to training mode

        ### FORWARD AND BACK PROP
        z_mean, z_log_var, z, sensor_recon, route_recon = model(routes, times, lengths)
        

        # cost = reconstruction loss + Kullback-Leibler divergence
        cost = loss_function_vrae(sensor_recon, route_recon, times,routes, z_mean, z_log_var)
        
        optimizer.zero_grad()
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
