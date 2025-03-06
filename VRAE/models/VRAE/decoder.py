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
        super(Decoder, self).__init__() # Inizializza la classe base
        
        self.decoder_lstm = nn.LSTM(num_latent, num_hidden, batch_first=True)
        self.sensor_out = nn.Linear(num_hidden, sensor_dim)
        self.route_out = nn.Linear(num_hidden, num_routes)
        # Inserisci qui lo start input (un parametro appreso)
        self.start_input = nn.Parameter(torch.randn(num_latent))
        self.out2latent = nn.Linear(sensor_dim, num_latent)
    
    def forward(self, z, seq_lens): # z è il vettore latente, seq_lens è la lunghezza della sequenza
        """
        Modalità training in forward parallelo.
        
        Parametri:
          z      : vettore latente di forma (batch, num_latent)
          seq_len: lunghezza fissa della sequenza da generare
          
        Restituisce:
          times_recon : ricostruzione dei tempi (batch, seq_len, sensor_dim)
          route_recon : distribuzione sulle route (batch, seq_len, num_routes)
        """
        # Estrae le dimensioni del batch
        batch = z.size(0) 
        max_seq_len = max(seq_lens) if not torch.is_tensor(seq_lens) else int(seq_lens.max())
        # Espande il vettore latente lungo la dimensione temporale
        z_expanded = z.unsqueeze(1).repeat(1, max_seq_len, 1)
        # Packa la sequenza usando le lunghezze reali
        packed_input = nn.utils.rnn.pack_padded_sequence(z_expanded, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.decoder_lstm(packed_input) # LSTM parallela
        
        # Unpacka l'output per ottenere un tensore di dimensione (batch, max_seq_len, num_hidden)
        decoder_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_seq_len)
        
        # Proiezioni agli output
        times_recon = self.sensor_out(decoder_out)
        route_logits = self.route_out(decoder_out)
        route_recon = F.softmax(route_logits, dim=-1)
        
        return times_recon, route_recon
    
    def autoregressive_forward(self, z, seq_lens, teacher_inputs=None, teacher_forcing_ratio=0.5):
      """
      Modalità autoregressiva con teacher forcing.
      
      Parametri:
        z                 : vettore latente, forma (batch, num_latent)
        seq_lens          : sequenza o lista con la lunghezza reale per ogni esempio
        teacher_inputs    : tensore target per i dati sensor (batch, max_seq_len, sensor_dim)
                            da usare come input per teacher forcing (opzionale)
        teacher_forcing_ratio : probabilità (0.0-1.0) di usare il teacher forcing ad ogni timestep
        
      Restituisce:
        times_outputs : predizioni dei tempi (batch, max_seq_len, sensor_dim)
        route_outputs : predizioni delle route (batch, max_seq_len, num_routes)
      """
      batch = z.size(0)
      max_seq_len = max(seq_lens) if not torch.is_tensor(seq_lens) else int(seq_lens.max())
      device = z.device
      # Inizializza lo stato nascosto
      h = torch.zeros(1, batch, self.decoder_lstm.hidden_size, device=device)
      c = torch.zeros(1, batch, self.decoder_lstm.hidden_size, device=device)
      # Il primo input è lo start token, replicato per il batch
      input_t = self.start_input.unsqueeze(0).expand(batch, 1, -1)  # (batch, 1, num_latent)
      
      outputs_sensor = []
      outputs_routes = []
      
      for t in range(max_seq_len):
          # Condiziona l'input corrente con il vettore latente
          current_input = input_t + z.unsqueeze(1)  # (batch, 1, num_latent)
          out, (h, c) = self.decoder_lstm(current_input, (h, c))  # out: (batch, 1, num_hidden)
          
          sensor_t = self.sensor_out(out)  # (batch, 1, sensor_dim)
          route_logits = self.route_out(out)  # (batch, 1, num_routes)
          route_t = F.softmax(route_logits, dim=-1)
          
          outputs_sensor.append(sensor_t)
          outputs_routes.append(route_t)
          
          # Teacher forcing: se teacher_inputs è fornito e si verifica la condizione,
          # usa il target come input per il passo successivo.
          if teacher_inputs is not None and t < max_seq_len - 1:
              if torch.rand(1).item() < teacher_forcing_ratio:
                  # Usa il teacher input per il prossimo timestep.
                  # Assumendo che teacher_inputs abbia dimensione (batch, max_seq_len, sensor_dim)
                  next_input = teacher_inputs[:, t, :].unsqueeze(1)
                  input_t = self.out2latent(next_input)  # Proiettato nello spazio latente
              else:
                  input_t = self.out2latent(sensor_t)
          else:
              input_t = self.out2latent(sensor_t)
      
      # Concatena gli output lungo la dimensione temporale
      times_outputs = torch.cat(outputs_sensor, dim=1)  # (batch, max_seq_len, sensor_dim)
      route_outputs = torch.cat(outputs_routes, dim=1)    # (batch, max_seq_len, num_routes)
      
      # Ritorna i tensori completi; in fase di loss si utilizzerà teacher forcing per mascherare i pad.
      return times_outputs, route_outputs