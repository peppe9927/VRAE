import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Importa il modello VRAE dal package models
from models import VRAE

# Parametri di training e dimensioni dei dati
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

SENSOR_DIM = 1       # dimensione dei dati sensor per ogni timestep
NUM_ROUTES = 30        # numero di route possibili
ROUTE_EMB_DIM = 4     # dimensione dell'embedding per le route
NUM_HIDDEN = 64       # dimensione dello stato nascosto della LSTM
NUM_LATENT = 16       # dimensione dello spazio latente
SEQ_LEN = 20          # lunghezza delle sequenze

device = torch.device("mps" if torch.cuda.is_available() else "cpu")


def loss_function(time_recon, route_recon, sensor_true, route_true, z_mean, z_log_var):
    """
    Calcola la loss combinata per il VRAE:
      - Ricostruzione dei dati sensor: Mean Squared Error (MSE)
      - Ricostruzione delle route: Negative Log Likelihood (NLL) calcolato su log-probabilità
      - Termini di regolarizzazione: KL Divergence
    La loss viene normalizzata per il batch size.
    """
    # 1. Ricostruzione sensor: MSE
    mse_loss = F.mse_loss(time_recon, sensor_true, reduction='sum')

    # 2. Ricostruzione route:
    # route_recon ha forma (batch, seq_len, num_routes) e contiene già le probabilità (softmax)
    # Per calcolare il Negative Log Likelihood, trasformiamo i dati in forma 2D.
    batch_size, seq_len, num_routes = route_recon.shape
    route_recon_flat = route_recon.view(-1, num_routes)    # (batch*seq_len, num_routes)
    route_true_flat = route_true.view(-1)                  # (batch*seq_len)
    
    # Fissiamo un piccolo epsilon per evitare log(0)
    nll_loss = F.nll_loss(torch.log(route_recon_flat + 1e-9), route_true_flat, reduction='sum')
    
    # 3. KL Divergence tra N(z_mean, exp(z_log_var)) e N(0,1)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    
    total_loss = (mse_loss + nll_loss + kl_loss) / batch_size
    return total_loss, mse_loss / batch_size, nll_loss / batch_size, kl_loss / batch_size


def main():
    # Istanzia il modello VRAE e spostalo sul device (CPU/GPU)
    model = VRAE(
        sensor_dim=SENSOR_DIM,
        num_routes=NUM_ROUTES,
        route_emb_dim=ROUTE_EMB_DIM,
        num_hidden=NUM_HIDDEN,
        num_latent=NUM_LATENT
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Esempio: creazione di un dataset fittizio
    num_samples = 1000
    # Dati sensor: tensore di forma (num_samples, SEQ_LEN, SENSOR_DIM)
    sensor_data = torch.randn(num_samples, SEQ_LEN, SENSOR_DIM)
    # Route: tensore di forma (num_samples, SEQ_LEN) contenente interi da 0 a NUM_ROUTES-1
    route_data = torch.randint(0, NUM_ROUTES, (num_samples, SEQ_LEN))
    
    # Crea un DataLoader per gestire i batch
    dataset = TensorDataset(sensor_data, route_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Loop di training
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for sensor_batch, route_batch in dataloader:
            sensor_batch = sensor_batch.to(device)
            route_batch = route_batch.to(device)
            
            optimizer.zero_grad()
            # Esegui forward pass
            z_mean, z_log_var, z, sensor_recon, route_recon = model(sensor_batch, route_batch)
            # Calcola la loss
            loss, mse, nll, kl = loss_function(sensor_recon, route_recon, sensor_batch, route_batch, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch: {epoch+1} - Loss: {avg_loss:.4f}")
    
    # Salva il modello
    torch.save(model.state_dict(), "vrae_model.pth")
    print("Modello salvato in 'vrae_model.pth'.")


if __name__ == "__main__":
    main()