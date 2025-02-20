import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import custom_collate_fn
from models.cvrae.cvrae import CVRAE
from torch.optim import Adam
from utils.loos_fcns import loss_function_vrae
from pathlib import Path


# Architecture
sensor_dim = 1
num_routes = 32
route_emb_dim = 4
num_hidden = 64
num_latent = 16
num_classes = 3
num_hidden_lstm = 32

# Model



# hyperparameters
random_seed = 0
learning_rate = 0.001
num_epochs = 50
batch_size = 40
torch.manual_seed(42) # Set the seed for reproducibility

# Load data
dataset=torch.load('./VRAE/datasets/training_data.pt') # Load the dataset for training

train_loader = DataLoader(dataset, batch_size=40, shuffle=True, collate_fn=custom_collate_fn)

# Ottieni il primo batch dal train_loader e stampa il primo elemento
batch = next(iter(train_loader))
print(batch.keys()) # Stampa le chiavi del batch
print(batch['routes'].shape) # Stampa la forma del batch
print(batch['times'].shape) # Stampa la forma del batch
print(batch['classes'].shape) # Stampa la forma del batch
print(batch['lengths'].shape) # Stampa la forma del batch


 # Loss function

start_time = time.time()

epoch_losses = [] # List to keep track of the loss for every epoch
device = 'mps' if torch.mps.is_available() else 'cpu'
print('Device:', device)

model = CVRAE(sensor_dim, num_routes, route_emb_dim, num_hidden_lstm, num_hidden, num_latent, num_classes, device).to(device) # Move the model to the device
optimizer = Adam(model.parameters(), lr=learning_rate) # Optimizer

for epoch in range(num_epochs): 
    epoch_loss = 0.0
    num_batches = 0
    for batch_idx, X in enumerate(train_loader):
        # Input
        routes = X['routes'].to(device)
        times = X['times'].to(device)
        labels = X['classes'].to(device)
        lengths = X['lengths'].to('cpu')
        

        model.train() # Set the model to training mode

        ### FORWARD AND BACK PROP
        z_mean, z_log_var, z, sensor_recon, route_recon = model(routes, times, lengths, labels, device)
        

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


# Save model state and hyperparameters
model_info = {
    'state_dict': model.state_dict(),
    'hyperparameters': {
        'sensor_dim': sensor_dim,
        'num_routes': num_routes,
        'route_emb_dim': route_emb_dim,
        'num_hidden': num_hidden,  
        'num_latent': num_latent,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size
    }
}


# 1. Create models directory 
MODEL_PATH = Path("./VRAE/models/trained")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_info, # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label='Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()