import torch
import torch.nn as nn
from models.cvrae.cvrae_enc import ConditionalVariationalEncoder
from models.cvrae.cvrae_dec import ConditionalVariationalDecoder

class CVRAE(nn.Module):
    def __init__(self, sensor_dim, num_routes, route_emb_dim, num_hidden_lstm, num_hidden, num_latent, num_classes, device):
        """
        Combina l'Encoder e il Decoder per formare il Variational Recurrent Autoencoder (VRAE).
        Parametri:
          sensor_dim   : dimensione dei dati sensor per timestep.
          num_routes   : numero di categorie possibili per le route.
          route_emb_dim: dimensione dell'embedding per le route.
          num_hidden   : dimensione dello stato nascosto della LSTM.
          num_latent   : dimensione dello spazio latente.
        """
        super(CVRAE, self).__init__()
        
        self.encoder = ConditionalVariationalEncoder(
                                num_classes, 
                                sensor_dim, 
                                num_routes,
                                route_emb_dim,
                                num_hidden_lstm,
                                num_hidden,
                                num_latent,
                                device
                                )
        
        self.decoder = ConditionalVariationalDecoder(
                                sensor_dim,
                                num_routes,
                                num_hidden,
                                num_latent,
                                num_classes
                                )
        self.num_classes = num_classes
    def forward(self, routes, times, lengths, labels, device):
        """
        Parametri:
          sensor_seq: (batch, seq_len, sensor_dim)
          route_seq : (batch, seq_len)
        Restituisce:
          z_mean, z_log_var, z, sensor_recon, route_recon
        """
        z_mean, z_log_var, z = self.encoder(routes,times, lengths, labels, device) # z è la sequenza encoded
        seq_len = lengths.max().item()
        sensor_recon, route_recon = self.decoder(z, seq_len, labels, device)
        return z_mean, z_log_var, z, sensor_recon, route_recon