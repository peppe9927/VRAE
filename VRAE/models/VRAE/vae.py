import torch
import torch.nn as nn
from .encoder import Encoder
from .autoregressive_decoder import AutoRegressiveDecoder

class VRAE(nn.Module):
    def __init__(self, sensor_dim, num_routes, route_emb_dim, num_hidden, num_latent):
        """
        Combina l'Encoder e il Decoder per formare il Variational Recurrent Autoencoder (VRAE).
        Parametri:
          sensor_dim   : dimensione dei dati sensor per timestep.
          num_routes   : numero di categorie possibili per le route.
          route_emb_dim: dimensione dell'embedding per le route.
          num_hidden   : dimensione dello stato nascosto della LSTM.
          num_latent   : dimensione dello spazio latente.
        """
        super(VRAE, self).__init__()
        self.encoder = Encoder(sensor_dim, num_routes, route_emb_dim, num_hidden, num_latent)
        self.decoder = AutoRegressiveDecoder(sensor_dim, num_routes, num_hidden, num_latent)
    
    def forward(self, routes, times, lengths):
        """
        Parametri:
          sensor_seq: (batch, seq_len, sensor_dim)
          route_seq : (batch, seq_len)
        Restituisce:
          z_mean, z_log_var, z, sensor_recon, route_recon
        """
        z_mean, z_log_var, z = self.encoder(routes,times, lengths) # z è la sequenza encoded
        seq_len = lengths.max().item()
        sensor_recon, route_recon = self.decoder(z, seq_len)
        return z_mean, z_log_var, z, sensor_recon, route_recon