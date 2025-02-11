import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self, sensor_dim, num_routes, num_hidden, num_latent):
        """
        Parametri:
          sensor_dim : dimensione dei dati sensor per ogni timestep.
          num_routes : numero di categorie per le route.
          num_hidden : dimensione dello stato nascosto della LSTM decoder.
          num_latent : dimensione dello spazio latente.
        """
        super(Decoder, self).__init__()
        
        # Decoder LSTM: prende in input il vettore latente ripetuto per ogni timestep
        self.decoder_lstm = nn.LSTM(num_latent, num_hidden, batch_first=True)
        
        # Output per ricostruire i dati sensor
        self.sensor_out = nn.Linear(num_hidden, sensor_dim) # va quindi da num_hidden a 1
        # Output per ricostruire le route (logits, da trasformare con softmax)
        self.route_out = nn.Linear(num_hidden, num_routes) # va da num_hidden a 30
    
    def forward(self, z, seq_len):
        """
        Modalità training in forward parallelo.
        
        Parametri:
          z      : vettore latente di forma (batch, num_latent)
          seq_len: lunghezza fissa della sequenza da generare
          
        Restituisce:
          times_recon : ricostruzione dei tempi (batch, seq_len, sensor_dim)
          route_recon : distribuzione sulle route (batch, seq_len, num_routes)
        """
        # Espande il vettore latente lungo la dimensione temporale
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, num_latent)
        # Passa la sequenza nella LSTM
        decoder_out, _ = self.decoder_lstm(z_expanded)       # (batch, seq_len, num_hidden)
        
        # Proiezione per il riconoscimento dei tempi
        times_recon = self.sensor_out(decoder_out)           # (batch, seq_len, sensor_dim)
        # Proiezione per le route, con softmax per ottenere la distribuzione di probabilità
        route_logits = self.route_out(decoder_out)           # (batch, seq_len, num_routes)
        route_recon = F.softmax(route_logits, dim=-1)
        
        return times_recon, route_recon
        
        