# python
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('VRAE/VRAE')
from utils import to_onehot

class ConditionalVariationalDecoder(nn.Module):
    def __init__(self,
                 sensor_dim: int,
                 num_routes:int,
                 num_hidden:int,
                 num_latent:int,
                 num_classes:int):
        """
        Parametri:
          sensor_dim : dimensione dei dati sensor per ogni timestep.
          num_routes : numero di categorie per le route.
          num_hidden : dimensione dello stato nascosto della LSTM decoder.
          num_latent : dimensione dello spazio latente.
        """
        super(ConditionalVariationalDecoder, self).__init__()
        
        # Decoder LSTM: prende in input un vettore latente (differenziato ad ogni step grazie all'autoregressione)
        self.decoder_lstm = nn.LSTM(num_latent+num_classes, num_hidden, batch_first=True) # adding num_classes for the condition
        self.num_classes = num_classes
        # Output per ricostruire i dati sensor
        self.sensor_out = nn.Linear(num_hidden, sensor_dim)
        # Output per ricostruire le route (logits, da trasformare con softmax)
        self.route_out = nn.Linear(num_hidden, num_routes)
        
        # Token di partenza appreso (start token) per l'autoregressione
        self.start_input = nn.Parameter(torch.randn(num_latent))
        # Proiezione per mappare l'output sensor all'ingresso latente per il passo successivo
        self.out2latent = nn.Linear(sensor_dim, num_latent)

    def forward(self, z, seq_len, targets, device):
        """
        Decodifica "parallela" (non autoregressiva) come implementato precedentemente.
        """

        onehot_targets = to_onehot(targets, self.num_classes, device) # convert the targets to one-hot encoding
        z = torch.cat((z, onehot_targets), dim=-1) # concatenate the one-hot encoded condition to the latent vector
        
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, num_latent+num_classes)
        decoder_out, _ = self.decoder_lstm(z_expanded)       # (batch, seq_len, num_hidden)
        
        sensor_recon = self.sensor_out(decoder_out) 
        route_logits = self.route_out(decoder_out)
        route_recon = F.softmax(route_logits, dim=-1)
        return sensor_recon, route_recon