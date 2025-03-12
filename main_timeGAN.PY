from VRAE import Generator, Discriminator, Supervisor, Recovery, Embedder
from VRAE import weights_init
from VRAE import random_generator
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
device = torch.device(device)
print(device)

# Set random seed for reproducibility

random_seed = 42
torch.manual_seed(random_seed)

# data loading

trajectory_dataset = torch.load("./trajectory_dataset.pt")

# Print basic information about the dataset
print(f"Type of loaded file: {type(trajectory_dataset)}")


# If it's a dictionary, print the keys
if isinstance(trajectory_dataset, dict):
    print(f"Keys: {trajectory_dataset.keys()}")
    # Print shapes of items
    for key, value in trajectory_dataset.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"{key} length: {len(value)}")
        else:
            print(f"{key} type: {type(value)}")
# If it's a tensor, print its shape
elif isinstance(trajectory_dataset, torch.Tensor):
    print(f"Tensor shape: {trajectory_dataset.shape}")
# If it's another container type, print its length
elif hasattr(trajectory_dataset, '__len__'):
    print(f"Dataset length: {len(trajectory_dataset)}")

max_seq_len = trajectory_dataset['max_length']

dataset = TensorDataset(trajectory_dataset['data'], trajectory_dataset['lengths'])

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# set the hyperparameters

EPOCHS = 100
BATCH_SIZE = 64
LATENT_DIM = 100

gen_lr = 0.001
dis_lr = 0.001
embedder_lr = 0.001
supervise_lr = 0.001

gamma = 0.1
seq_len = 50
input_dim = 5

# Set random seed for reproducibility
torch.manual_seed(random_seed)

# define the models
hidden_dim_generator = 100
num_layers_generator = 1
cell_type_generator = 'GRU'
generator = Generator(hidden_dim = hidden_dim_generator,
                      num_layers = num_layers_generator,
                      max_seq = max_seq_len,
                      cell_type = cell_type_generator)
generator.apply(weights_init)
generator.to(device)

num_layers_discriminator = 1
hidden_dim_discriminator = 100
cell_type_discriminator = 'GRU'

discriminator = Discriminator(num_hidden = hidden_dim_discriminator,
                              num_layers = num_layers_discriminator,
                                max_seq = max_seq_len,
                              cell_type = cell_type_discriminator)
discriminator.apply(weights_init)
discriminator.to(device)

num_layers_supervisor = 1
hidden_dim_supervisor = 100
cell_type_supervisor = 'GRU'

supervisor = Supervisor(num_hidden = hidden_dim_supervisor,
                        num_layers = num_layers_supervisor,
                        max_seq = max_seq_len,
                        cell_type = cell_type_supervisor)
supervisor.apply(weights_init)
supervisor.to(device)

num_layers_embedder = 1
hidden_dim_embedder = 100
cell_type_embedder = 'GRU'

embedder = Embedder(inpt_dim = input_dim,
                    hidden_dim = hidden_dim_embedder,
                    num_layers = num_layers_embedder,
                    max_seq = max_seq_len,
                    cell_type = cell_type_embedder)
embedder.apply(weights_init)
embedder.to(device)

hidden_dim_recovery = 100
num_layers_recovery = 1
cell_type_recovery = 'GRU'

recovery = Recovery(hidden_dim = hidden_dim_recovery,
                    feature_dim = input_dim,
                    num_layers = num_layers_recovery,
                    max_seq = max_seq_len,
                    cell_type = cell_type_recovery)
recovery.apply(weights_init)
recovery.to(device)

# Define optimizers (one per parameter group)
optimizer_E = optim.Adam(
    list(embedder.parameters()) + list(recovery.parameters()),
    lr=embedder_lr
)
optimizer_G = optim.Adam(
    list(generator.parameters()) + list(supervisor.parameters()),
    lr=gen_lr
)
optimizer_D = optim.Adam(discriminator.parameters(), lr=dis_lr)

# Define loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()


emb_costs = []
emb_rec_costs = []
PRE_TRAIN_EPOCHS = 100

for epoch in range(PRE_TRAIN_EPOCHS):
    # pre-training of the embedder and recovery
    for i, (real_samples, seq_lens) in enumerate(train_loader): 
        
        embedder.train()
        recovery.train()

        real_samples = real_samples.to(device) # Send data to GPU
        seq_lens = seq_lens.to(device) # Send sequence lengths to GPU

        real_labels = torch.ones(real_samples.size(0), 1).to(device)
        fake_labels = torch.zeros(real_samples.size(0), 1).to(device)

        # Train the embedder
        optimizer_E.zero_grad() # Reset gradients
        
        # create latent representation
        H = embedder(real_samples, seq_lens) 
        # refine latent representation
        X_tilde = recovery(H,seq_lens)
        
        # Calculate embedder loss
        E_loss_T0 = mse_loss(X_tilde, real_samples) # reconstruction loss
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss0.backward()

        # Update embedder and recovery parameters
        optimizer_E.step()
        
        emb_costs.append(E_loss0.item())

        if not i % 100: 
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Reconstruction Loss: %.4f' 
                    %(epoch+1, EPOCHS, i, 
                        len(train_loader), E_loss0))

    # supervised training of the embedder


supervisor_costs = []

for epoch in range(EPOCHS):
    # Step 2: Generator, supervised only
    for i, (real_samples, seq_lens) in enumerate(train_loader):  # loop over batches

        real_samples = real_samples.to(device)
        seq_lens = seq_lens.to(device)

        generator.train()
        supervisor.train()

        optimizer_G.zero_grad()

        # generate real data embeddings
        H = embedder(real_samples,seq_lens).detach()
        H_hat_Supervise = supervisor(H, seq_lens)

        G_loss_S = mse_loss(H[:, 1:, :], H_hat_Supervise[:, :-1, :]) # supervised loss
        G_loss_S.backward()
        
        optimizer_G.step()

        supervisor_costs.append(G_loss_S.item())

        if not i % 100: 
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Supervisor Loss: %.4f' 
                    %(epoch+1, EPOCHS, i, 
                        len(train_loader), G_loss_S))

generator_losses = []
discriminator_losses = []
emb_costs = []
supervisor_costs = []
for epoch in range(EPOCHS):
    for i,(real_samples, seq_lens) in enumerate(train_loader):
        
        # discriminator network
        discriminator.train()
        # generator network
        generator.train()
        supervisor.train()
        # embedding network 
        embedder.train()
        recovery.train()

        
        real_samples = real_samples.to(device)
        seq_lens = seq_lens.to(device)
        # train the generator twice as much as the discriminator
        for kk in range(2): 
            # Step 3.1: Generator training
            optimizer_G.zero_grad()

            
            # sample random noise
            Z_mb = random_generator(len(real_samples), hidden_dim_generator, seq_lens, seq_len)
            H = embedder(real_samples,seq_lens).detach() 
            # generate fake data embeddings
            E_hat = generator(Z_mb,seq_lens)
            # refine fake data embeddings
            H_hat = supervisor(E_hat,seq_lens)
            # reconstruct fake data
            X_hat = recovery(H_hat,seq_lens).detach()
            # discriminator outputs
            Y_fake = discriminator(H_hat, seq_lens).detach() # 
            Y_real = discriminator(H, seq_lens).detach()     
            Y_fake_e = discriminator(E_hat, seq_lens).detach()
            
            # unsupervised loss
            G_loss_U = bce_loss(torch.ones_like(Y_fake), Y_fake) # after supervisor
            G_loss_U_e = bce_loss(torch.ones_like(Y_fake_e), Y_fake_e) # before supervisor
            # supervised loss
            # H_hat = supervisor(generator(Z_mb,seq_lens),seq_lens)
            G_loss_S = mse_loss(H[:, 1:, :], H_hat[:, :-1, :]) # supervised loss
            
            G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var(real_samples, dim=0) + 1e-6) - torch.sqrt(torch.var(X_hat, dim=0) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs(torch.mean(real_samples, dim=0) - torch.mean(X_hat, dim=0)))
            G_loss_V = G_loss_V1 + G_loss_V2
            
            # Total generator loss
            G_loss = G_loss_U + G_loss_U_e + 100*torch.sqrt(G_loss_S) + 100*G_loss_V 
            G_loss.backward()
            optimizer_G.step()
            generator_losses.append(G_loss.item())
            supervisor_costs.append(G_loss_S.item())


            # step 3.2: embedder and recovery training
            optimizer_E.zero_grad()
            
            X_tilde = recovery(H, seq_lens) 
            # reconstrucion loss
            E_loss_T0 = mse_loss(X_tilde, real_samples)
            E_loss_T0.backward()
            optimizer_E.step()
            emb_costs.append(E_loss_T0.item())
        
        # Step 4: Discriminator training
        optimizer_D.zero_grad()  # Zero discriminator gradients
        
        # Get discriminator outputs for real data

        H = embedder(real_samples, seq_lens).detach()  # Embed real data

        E_hat = generator(Z_mb, seq_lens).detach()  # Generate fake data
        H_hat = supervisor(E_hat, seq_lens).detach()  # Supervise fake data
        X_hat = recovery(H_hat, seq_lens).detach()  # Recover fake data

        Y_fake = discriminator(H_hat, seq_lens)  # Prediction on supervised fake data
        Y_fake_e = discriminator(E_hat, seq_lens)  # Prediction on directly generated data
        Y_real = discriminator(H, seq_lens)  # Discriminator prediction on real data
        
        # Calculate discriminator loss - wants to predict real=1, fake=0
        D_loss_real = bce_loss(Y_real, torch.ones_like(Y_real))
    
        # For generated data, target is 0
        D_loss_fake = bce_loss(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = bce_loss(Y_fake_e, torch.zeros_like(Y_fake_e))
        
        # Total loss: sum of components, with gamma balancing the loss_fake_e term
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
    

        # Backpropagate and update discriminator weights
        D_loss.backward()
        
        optimizer_D.step()
        discriminator_losses.append(D_loss.item())

        if not i % 100:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Discriminator Loss: %.4f' 
                    %(epoch+1, EPOCHS, i, 
                        len(train_loader), D_loss))
            print ('Generator Loss: %.4f | Embedder Loss: %.4f | Supervisor Loss: %.4f'
                    %(G_loss, E_loss_T0, G_loss_S))
            