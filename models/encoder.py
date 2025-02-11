import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, sensor_dim, num_routes, route_emb_dim, num_hidden, num_latent):
        """
        Parametri:
          sensor_dim   : dimensione dei dati sensor per ogni timestep.
          num_routes   : numero di categorie possibili per le route.
          route_emb_dim: dimensione dell'embedding per le route.
          num_hidden   : dimensione complessiva dello stato nascosto (dopo fusione dei branch).
          num_latent   : dimensione dello spazio latente.
        """
        super(Encoder, self).__init__()
        
        # prende in input il batch che ha dimensione BATCH, MAX_SEQ_LEN, DIMENSIONE

        # embedding per le route, prende in input un tensore di dimensione (batch, seq_len) e restituisce un tensore di dimensione (batch, seq_len, route_emb_dim)
        self.route_embedding = nn.Embedding(num_routes, route_emb_dim, padding_idx=0) 

        # Branch per i dati sensor: utilizzo di un LSTM
        self.sensor_lstm = nn.LSTM(sensor_dim+route_emb_dim, num_hidden, batch_first=True)
                
        # Fusione dei due branch: il vettore concatenato avrà dimensione num_hidden
        self.linear_layer = nn.Linear(num_hidden, num_hidden)
        
        # Calcolo dei parametri dello spazio latente
        self.z_mean = nn.Linear(num_hidden, num_latent)
        self.z_log_var = nn.Linear(num_hidden, num_latent)
    
    def reparameterize(self, z_mean, z_log_var):
        """Campiona z in modo differenziabile usando il trucco della ri-parametrizzazione."""
        eps = torch.randn_like(z_mean)
        z = z_mean + eps * torch.exp(z_log_var / 2)
        return z

    def forward(self, X):
        """
        Parametri:
          sensor_seq: tensore di forma (batch, seq_len, sensor_dim)
          route_seq : tensore di forma (batch, seq_len) contenente indici per le route
        Restituisce:
          z_mean, z_log_var, z
        """
        # Elaborazione del ramo sensor
        route_emb = self.route_embedding(X['routes'])  # calcola prima la sequenza di embedding
        
        combined = torch.cat([X['times'], route_emb], dim=-1)  # unisci le sequenze

        packed_input = pack_padded_sequence(combined, X['lengths'], batch_first=True)

        sensor_out, (h_sensor, _) = self.sensor_lstm(packed_input) # passale a un lstm
        h_final = h_sensor[-1]  # forma: [batch, num_hidden]

        # Concatenazione delle rappresentazioni
        output = F.leaky_relu(self.linear_layer(h_final))
        
        # Calcolo dei parametri latenti
        z_mean = self.z_mean(output)
        z_log_var = self.z_log_var(output)
        z = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, z