import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F  # Add this import for loss functions

from VRAE import TimeGAN

def discriminator_loss(Y_real, Y_fake, Y_fake_e, gamma=1.0):
    """
    Calculates the discriminator loss in TimeGAN architecture.
    
    The discriminator tries to distinguish between real and generated sequences.
    This loss function encourages the discriminator to correctly classify:
    - Real sequences as real (label=1)
    - Generated sequences as fake (label=0)
    
    Args:
        Y_real (torch.Tensor): Discriminator output on real data.
        Y_fake (torch.Tensor): Discriminator output on supervised generated data.
        Y_fake_e (torch.Tensor): Discriminator output on directly generated data.
        gamma (float): Weight factor for the Y_fake_e term.
        
    Returns:
        torch.Tensor: Total discriminator loss.
    """
    # For real data, target is 1
    D_loss_real = F.binary_cross_entropy(Y_real, torch.ones_like(Y_real))
    
    # For generated data, target is 0
    D_loss_fake = F.binary_cross_entropy(Y_fake, torch.zeros_like(Y_fake))
    D_loss_fake_e = F.binary_cross_entropy(Y_fake_e, torch.zeros_like(Y_fake_e))
    
    # Total loss: sum of components, with gamma balancing the loss_fake_e term
    D_loss_total = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
    return D_loss_total


def generator_loss(Y_fake, Y_fake_e, H, H_hat_supervise, X, X_hat, gamma=1.0):
    """
    Calculates the total generator loss in TimeGAN.
    
    This complex loss function has multiple components:
    1. Adversarial loss - Makes generated data look realistic to the discriminator
    2. Supervised loss - Ensures temporal coherence in generated sequences
    3. Moment matching loss - Aligns statistical properties between real and generated data
    
    Args:
        Y_fake (torch.Tensor): Discriminator output on supervised generated sequences.
        Y_fake_e (torch.Tensor): Discriminator output on directly generated sequences.
        H (torch.Tensor): Real latent sequence from the embedder.
        H_hat_supervise (torch.Tensor): Supervisor output applied to real latent sequence.
        X (torch.Tensor): Original real data.
        X_hat (torch.Tensor): Synthetic data reconstructed by applying Recovery to H_hat.
        gamma (float): Weight factor for the direct generator adversarial term.
    
    Returns:
        tuple: (Total generator loss, Supervised loss component)
    """
    # 1. Adversarial Loss: Forces generator to fool the discriminator
    G_loss_U = F.binary_cross_entropy(Y_fake, torch.ones_like(Y_fake))
    G_loss_U_e = F.binary_cross_entropy(Y_fake_e, torch.ones_like(Y_fake_e))
    
    # 2. Supervised Loss: Ensures temporal coherence by predicting next step in sequence
    # Compares H[:, 1:, :] (real sequence shifted by one step) with H_hat_supervise[:, :-1, :]
    G_loss_S = F.mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])
    
    # 3. Moment Matching Loss: Aligns statistical properties (mean and std) between real and synthetic data
    mean_real = torch.mean(X, dim=0)
    mean_fake = torch.mean(X_hat, dim=0)
    std_real = torch.sqrt(torch.var(X, dim=0) + 1e-6)
    std_fake = torch.sqrt(torch.var(X_hat, dim=0) + 1e-6)
    G_loss_V = F.l1_loss(mean_fake, mean_real) + F.l1_loss(std_fake, std_real)
    
    # 4. Total Generator Loss: Combines components with empirical weights
    # Square root of supervised loss and factor of 100 for supervision and moment matching
    loss_gen = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
    return loss_gen, G_loss_S


def embedder_loss(X, X_tilde, G_loss_S, eps=1e-6):
    """
    Calculates the embedder loss in TimeGAN.
    
    The embedder maps real data to latent space. This loss ensures:
    1. Reconstruction quality - Data can be properly reconstructed from latent space
    2. Temporal coherence - By incorporating the supervised loss from the generator
    
    Args:
        X (torch.Tensor): Original data.
        X_tilde (torch.Tensor): Reconstructed data via Recovery network.
        G_loss_S (torch.Tensor): Supervised loss (to reinforce temporal coherence).
        eps (float): Small value to avoid division by zero in square root.
    
    Returns:
        torch.Tensor: Total embedder loss.
    """
    # Reconstruction loss (MSE between original and reconstructed data)
    E_loss_T0 = F.mse_loss(X_tilde, X)
    
    # E_loss0: square root of MSE, multiplied by 10 for scaling
    E_loss_0 = 10 * torch.sqrt(E_loss_T0 + eps)
    
    # Total embedder loss: combines reconstruction with a fraction of supervised loss
    E_loss = E_loss_0 + 0.1 * G_loss_S
    return E_loss


EPOCHS = 100
NUM_BATCHES = 100
LEN_BATCH = 128

model = TimeGAN(
    input_dim= 5, # input dimension
    hidden_dim= 100, # embedding dimension
    num_layers= 1, # number of layers
    cell_type= 'GRU'
    )
    

# Define optimizers

# Embedder and recovery
Opt_E0 = optim.Adam(list(model.embedder.parameters(),model.recovery.parameters()), lr=0.001)
Opt_E = optim.Adam(list(model.embedder.parameters(),model.recovery.parameters()), lr=0.001)
# Discriminator
Opt_D = optim.Adam(model.discriminator.parameters(), lr=0.001)
# Generator and supervisor
Opt_G = optim.Adam(list(model.generator.parameters(),model.supervisor.parameters()), lr=0.001)
Opt_G_S = optim.Adam(list(model.generator.parameters(),model.supervisor.parameters()), lr=0.001)

# Define loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# Training loop

seq_len = 50
input_dim = 5
batch_size = 64
data = torch.randn(1000, seq_len, input_dim)

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(EPOCHS): # loop over the dataset multiple times
    
    for batch, X in enumerate(dataloader): # loop over batches
        
        model.train()

        # Step 1: Embedder
        Opt_E0.zero_grad()
        Opt_E.zero_grad()

        H = model.embedder(X,T) # create latent representation
        H_hat_Supervise = model.supervisor(H,T)
        X_tilde = model.recovery(H,T) # reconstruct data from latent representation
        
        # Calculate embedder loss
        G_loss_S = mse_loss(H[:, 1:, :], H_hat_Supervise[:, :-1, :]) # supervised loss
        E_loss_T0 = mse_loss(X_tilde, X) # reconstruction loss
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1*G_loss_S # total embedder loss

        # Backpropagate and update weights
        E_loss.backward()
        E_loss0.backward()

        Opt_E0.step()
        Opt_E.step()

    for batch, X in enumerate(dataloader): # loop over batches

        model.train()
        # Step 2: Generator, supervised only

        Opt_G_S.zero_grad()

        Z_mb = torch.randn(LEN_BATCH, seq_len, input_dim) # sample random noise
        H = model.embedder(X,T) # create latent representation
        H_hat_Supervise = model.supervisor(H,T)
        E_hat = model.generator(Z_mb,T)
        H_hat = model.supervisor(E_hat,T)

        G_loss_S = mse_loss(H[:, 1:, :], H_hat_Supervise[:, :-1, :]) # supervised loss
        G_loss_S.backward()
        Opt_G_S.step()

    for batch, X in enumerate(dataloader): # loop over batches

        model.train()
        # Step 3: Generator, joint training
        Opt_G.zero_grad()

        Z_mb = torch.randn(LEN_BATCH, seq_len, input_dim) # sample random noise
        E_hat = model.generator(Z_mb,T)
        H_hat = model.supervisor(E_hat,T)
        X_hat = model.recovery(H_hat,T)

        Y_fake_e = model.discriminator(E_hat,T) 
        Y_fake = model.discriminator(H_hat,T)

        G_loss_U = bce_loss(torch.ones_like(Y_fake), Y_fake)
        G_loss_U_e = bce_loss(torch.ones_like(Y_fake_e), Y_fake_e)
        G_loss_S = mse_loss(H[:, 1:, :], H_hat[:, :-1, :])
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var(X, dim=0) + 1e-6) - torch.sqrt(torch.var(X_hat, dim=0) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs(torch.mean(X, dim=0) - torch.mean(X_hat, dim=0)))
        G_loss_V = G_loss_V1 + G_loss_V2

        G_loss = G_loss_U + G_loss_U_e + 100*torch.sqrt(G_loss_S) + 100*G_loss_V
        G_loss.backward()
        Opt_G.step()

        # embedder loss

        Opt_E.zero_grad() 
        
        H = model.embedder(X,T) # create latent representation
        X_tilde = model.recovery(H,T) # reconstruct data from latent representation
       
        E_loss_T0 = mse_loss(X_tilde, X) # reconstruction loss
        E_loss_T0.backward()
        Opt_E.step()










