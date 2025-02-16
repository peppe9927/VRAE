import torch
import torch.nn.functional as F

def loss_function_vrae(time_recon, route_recon, sensor_true, route_true, z_mean, z_log_var):
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
    return total_loss
