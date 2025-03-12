import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from VRAE import Generator, Discriminator, Supervisor, Recovery, Embedder
from VRAE import weights_init, random_generator


def main():
    # Imposta il dispositivo
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.mps.is_available() else
        'cpu'
    )
    device = torch.device(device)
    print(f"Dispositivo: {device}")

    # Imposta il seme random per la riproducibilità
    random_seed = 42
    torch.manual_seed(random_seed)

    # Caricamento del dataset
    trajectory_dataset = torch.load("./trajectory_dataset.pt")
    print(f"Tipo di file caricato: {type(trajectory_dataset)}")
    if isinstance(trajectory_dataset, dict):
        print(f"Chiavi: {trajectory_dataset.keys()}")
        for key, value in trajectory_dataset.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"{key} lunghezza: {len(value)}")
            else:
                print(f"{key} tipo: {type(value)}")
    elif isinstance(trajectory_dataset, torch.Tensor):
        print(f"Tensor shape: {trajectory_dataset.shape}")
    elif hasattr(trajectory_dataset, '__len__'):
        print(f"Dataset length: {len(trajectory_dataset)}")

    max_seq_len = trajectory_dataset['max_length']
    dataset = TensorDataset(trajectory_dataset['data'], trajectory_dataset['lengths'])
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Iperparametri
    EPOCHS = 100
    LATENT_DIM = 100
    gen_lr = 0.001
    dis_lr = 0.001
    embedder_lr = 0.001
    supervise_lr = 0.001
    gamma = 0.1
    seq_len = 50
    input_dim = 5

    # Inizializzazione dei modelli

    # Generator
    hidden_dim_generator = 100
    num_layers_generator = 1
    cell_type_generator = 'GRU'
    generator = Generator(
        hidden_dim=hidden_dim_generator,
        num_layers=num_layers_generator,
        max_seq=max_seq_len,
        cell_type=cell_type_generator
    )
    generator.apply(weights_init)
    generator.to(device)

    # Discriminator
    num_layers_discriminator = 1
    hidden_dim_discriminator = 100
    cell_type_discriminator = 'GRU'
    discriminator = Discriminator(
        num_hidden=hidden_dim_discriminator,
        num_layers=num_layers_discriminator,
        max_seq=max_seq_len,
        cell_type=cell_type_discriminator
    )
    discriminator.apply(weights_init)
    discriminator.to(device)

    # Supervisor
    num_layers_supervisor = 1
    hidden_dim_supervisor = 100
    cell_type_supervisor = 'GRU'
    supervisor = Supervisor(
        num_hidden=hidden_dim_supervisor,
        num_layers=num_layers_supervisor,
        max_seq=max_seq_len,
        cell_type=cell_type_supervisor
    )
    supervisor.apply(weights_init)
    supervisor.to(device)

    # Embedder
    num_layers_embedder = 1
    hidden_dim_embedder = 100
    cell_type_embedder = 'GRU'
    embedder = Embedder(
        inpt_dim=input_dim,
        hidden_dim=hidden_dim_embedder,
        num_layers=num_layers_embedder,
        max_seq=max_seq_len,
        cell_type=cell_type_embedder
    )
    embedder.apply(weights_init)
    embedder.to(device)

    # Recovery
    hidden_dim_recovery = 100
    num_layers_recovery = 1
    cell_type_recovery = 'GRU'
    recovery = Recovery(
        hidden_dim=hidden_dim_recovery,
        feature_dim=input_dim,
        num_layers=num_layers_recovery,
        max_seq=max_seq_len,
        cell_type=cell_type_recovery
    )
    recovery.apply(weights_init)
    recovery.to(device)

    # Definizione degli ottimizzatori
    optimizer_E = optim.Adam(
        list(embedder.parameters()) + list(recovery.parameters()),
        lr=embedder_lr
    )
    optimizer_G = optim.Adam(
        list(generator.parameters()) + list(supervisor.parameters()),
        lr=gen_lr
    )
    optimizer_D = optim.Adam(discriminator.parameters(), lr=dis_lr)

    # Definizione delle funzioni di loss
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # =============================================================================
    # Pre-training: Embedder & Recovery
    # =============================================================================
    PRE_TRAIN_EPOCHS = 100
    pretrain_emb_costs = []
    for epoch in range(PRE_TRAIN_EPOCHS):
        for i, (real_samples, seq_lens) in enumerate(train_loader):
            embedder.train()
            recovery.train()

            # Sposta i dati sul dispositivo
            real_samples = real_samples.to(device)
            seq_lens = seq_lens.to(device)

            optimizer_E.zero_grad()

            # Creazione della rappresentazione latente e ricostruzione
            H = embedder(real_samples, seq_lens)
            X_tilde = recovery(H, seq_lens)
            E_loss_T0 = mse_loss(X_tilde, real_samples)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            E_loss0.backward()
            optimizer_E.step()

            pretrain_emb_costs.append(E_loss0.item())
            if i % 100 == 0:
                print(f'Pre-train Epoch: {epoch+1}/{PRE_TRAIN_EPOCHS} | Batch: {i}/{len(train_loader)} | Recon Loss: {E_loss0:.4f}')

    # =============================================================================
    # Supervisione: Allenamento supervisionato del Supervisor
    # =============================================================================
    supervisor_costs = []
    for epoch in range(EPOCHS):
        for i, (real_samples, seq_lens) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            seq_lens = seq_lens.to(device)

            generator.train()
            supervisor.train()
            optimizer_G.zero_grad()

            # Generazione delle embedding dei dati reali
            H = embedder(real_samples, seq_lens).detach()
            H_hat_Supervise = supervisor(H, seq_lens)
            G_loss_S = mse_loss(H[:, 1:, :], H_hat_Supervise[:, :-1, :])
            G_loss_S.backward()
            optimizer_G.step()

            supervisor_costs.append(G_loss_S.item())
            if i % 100 == 0:
                print(f'Supervisor Train Epoch: {epoch+1}/{EPOCHS} | Batch: {i}/{len(train_loader)} | Supervisor Loss: {G_loss_S:.4f}')

    # =============================================================================
    # Training combinato: Generator, Embedder/Recovery, Discriminator
    # =============================================================================
    generator_losses = []
    discriminator_losses = []
    comb_emb_costs = []
    comb_supervisor_costs = []

    for epoch in range(EPOCHS):
        for i, (real_samples, seq_lens) in enumerate(train_loader):
            # Modalità training per tutti i modelli coinvolti
            discriminator.train()
            generator.train()
            supervisor.train()
            embedder.train()
            recovery.train()

            real_samples = real_samples.to(device)
            seq_lens = seq_lens.to(device)

            # Allenamento del Generator (2 volte per ogni update del Discriminator)
            for _ in range(2):
                optimizer_G.zero_grad()
                Z_mb = random_generator(len(real_samples), hidden_dim_generator, seq_lens, seq_len)
                H = embedder(real_samples, seq_lens).detach()
                E_hat = generator(Z_mb, seq_lens)
                H_hat = supervisor(E_hat, seq_lens)
                X_hat = recovery(H_hat, seq_lens).detach()

                # Output del Discriminator
                Y_fake = discriminator(H_hat, seq_lens).detach()
                Y_fake_e = discriminator(E_hat, seq_lens).detach()

                # Calcolo della loss per il Generator
                G_loss_U = bce_loss(torch.ones_like(Y_fake), Y_fake)
                G_loss_U_e = bce_loss(torch.ones_like(Y_fake_e), Y_fake_e)
                G_loss_S = mse_loss(H[:, 1:, :], H_hat[:, :-1, :])
                # Variazioni statistiche tra i dati reali e ricostruiti
                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var(real_samples, dim=0) + 1e-6) -
                                                   torch.sqrt(torch.var(X_hat, dim=0) + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs(torch.mean(real_samples, dim=0) - torch.mean(X_hat, dim=0)))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = G_loss_U + G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
                G_loss.backward()
                optimizer_G.step()

                generator_losses.append(G_loss.item())
                comb_supervisor_costs.append(G_loss_S.item())

                # Allenamento combinato di Embedder & Recovery
                optimizer_E.zero_grad()
                X_tilde = recovery(H, seq_lens)
                E_loss_T0 = mse_loss(X_tilde, real_samples)
                E_loss_T0.backward()
                optimizer_E.step()
                comb_emb_costs.append(E_loss_T0.item())

            # Allenamento del Discriminator
            optimizer_D.zero_grad()
            H = embedder(real_samples, seq_lens).detach()
            E_hat = generator(Z_mb, seq_lens).detach()
            H_hat = supervisor(E_hat, seq_lens).detach()
            X_hat = recovery(H_hat, seq_lens).detach()

            Y_real = discriminator(H, seq_lens)
            Y_fake = discriminator(H_hat, seq_lens)
            Y_fake_e = discriminator(E_hat, seq_lens)

            D_loss_real = bce_loss(Y_real, torch.ones_like(Y_real))
            D_loss_fake = bce_loss(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = bce_loss(Y_fake_e, torch.zeros_like(Y_fake_e))
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

            D_loss.backward()
            optimizer_D.step()
            discriminator_losses.append(D_loss.item())

            if i % 100 == 0:
                print(f'Combined Train Epoch: {epoch+1}/{EPOCHS} | Batch: {i}/{len(train_loader)}')
                print(f'  Discriminator Loss: {D_loss:.4f}')
                print(f'  Generator Loss: {G_loss:.4f} | Embedder Loss: {E_loss_T0:.4f} | Supervisor Loss: {G_loss_S:.4f}')


if __name__ == '__main__':
    main()